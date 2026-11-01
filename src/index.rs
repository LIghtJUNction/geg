use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, OptionalExtension, params};

use crate::{
    cache::Cache,
    input,
    search::{Hit, Response, Searcher, Stats, insert_hit},
};

const INDEX_FORMAT: &str = "geg-index-v2-single-vector-store";

pub fn default_path() -> Result<PathBuf> {
    Ok(std::env::current_dir()?.join(".geg/index.sqlite3"))
}

fn open(path: &Path) -> Result<Connection> {
    crate::cache::prepare_database(path)?;
    let db = Connection::open(path)?;
    db.busy_timeout(Duration::from_secs(30))?;
    db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA cache_size=-2048; PRAGMA mmap_size=0;
        CREATE TABLE IF NOT EXISTS meta_v1 (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS files_v1 (path TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS vectors_v1 (key BLOB PRIMARY KEY, vector BLOB NOT NULL) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS chunks_v2 (path TEXT NOT NULL, line INTEGER NOT NULL, text TEXT NOT NULL, vector_key BLOB NOT NULL);
        CREATE INDEX IF NOT EXISTS chunks_vector_key ON chunks_v2(vector_key);")?;
    Ok(db)
}

fn decode(bytes: &[u8], dimensions: usize) -> Result<Vec<f32>> {
    ensure!(
        bytes.len() == dimensions * 4,
        "invalid index vector length; rebuild the index"
    );
    let vector: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect();
    ensure!(
        vector.iter().all(|v| v.is_finite()),
        "non-finite index vector; rebuild the index"
    );
    Ok(vector)
}

impl Searcher<'_> {
    pub fn build_index(&mut self, paths: &[PathBuf], output: &Path) -> Result<Stats> {
        ensure!(
            paths.iter().all(|p| p != Path::new("-")),
            "indexing requires files/directories, not stdin"
        );
        ensure!(
            !self.cache.shares_path(output)?,
            "index and global cache must use different database paths"
        );
        self.progress.begin("building repository index");
        let mut db = open(output)?;
        let output = fs::canonicalize(output)?;
        let mut wal = output.as_os_str().to_os_string();
        wal.push("-wal");
        let mut shm = output.as_os_str().to_os_string();
        shm.push("-shm");
        let transaction = db.transaction()?;
        transaction.execute("DELETE FROM chunks_v2", [])?;
        transaction.execute("DELETE FROM files_v1", [])?;
        let mut stats = Stats::default();
        input::visit_files(paths, self.args.no_ignore, self.args.hidden, |path| {
            let absolute = fs::canonicalize(path)?;
            if absolute == output || absolute.as_os_str() == wal || absolute.as_os_str() == shm {
                return Ok(());
            }
            self.read_file(&absolute, |this, reader, label| {
                transaction.execute("INSERT OR IGNORE INTO files_v1 VALUES (?1)", [label])?;
                input::chunks(reader, this.args.chunk_lines, |line, text| {
                    if !input::searchable(text) {
                        return Ok(());
                    }
                    let prompt = format!("title: none | text: {text}");
                    let prompt = if this.args.ignore_case {
                        prompt.to_lowercase()
                    } else {
                        prompt
                    };
                    let key = Cache::key(&this.model_hash, this.dimensions, &prompt);
                    let local: Option<Vec<u8>> = transaction
                        .prepare_cached("SELECT vector FROM vectors_v1 WHERE key=?1")?
                        .query_row([key.as_bytes().as_slice()], |row| row.get(0))
                        .optional()?;
                    if let Some(bytes) = local {
                        decode(&bytes, this.dimensions)?;
                        stats.cache_hits += 1;
                    } else {
                        let vector = if let Some(vector) = this.cached_vector(&key)? {
                            stats.cache_hits += 1;
                            vector
                        } else {
                            this.embed_uncached(&prompt, &mut stats)?
                        };
                        let bytes: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
                        transaction
                            .prepare_cached("INSERT INTO vectors_v1 VALUES (?1, ?2)")?
                            .execute(params![key.as_bytes().as_slice(), bytes])?;
                    }
                    transaction
                        .prepare_cached("INSERT INTO chunks_v2 VALUES (?1, ?2, ?3, ?4)")?
                        .execute(params![
                            label,
                            i64::try_from(line)?,
                            text,
                            key.as_bytes().as_slice()
                        ])?;
                    stats.chunks += 1;
                    this.progress.update(&stats);
                    Ok(())
                })
            })?;
            stats.files += 1;
            Ok(())
        })?;
        for (key, value) in [
            ("model", self.model_hash.to_hex().to_string()),
            ("dimensions", self.dimensions.to_string()),
            ("ignore_case", self.args.ignore_case.to_string()),
            ("embedding_format", crate::cache::FORMAT.to_owned()),
            ("format", INDEX_FORMAT.to_owned()),
        ] {
            transaction.execute(
                "INSERT OR REPLACE INTO meta_v1 VALUES (?1, ?2)",
                params![key, value],
            )?;
        }
        // Prune removed/changed chunks, duplicate content shares one vector locally too.
        transaction.execute(
            "DELETE FROM vectors_v1 WHERE key NOT IN (SELECT vector_key FROM chunks_v2)",
            [],
        )?;
        transaction.execute_batch("DROP TABLE IF EXISTS chunks_v1;")?;
        transaction.commit()?;
        db.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
        // A failure before commit never removes the user's last cached generation.
        self.cache.remove_indexed(&output)?;
        self.progress.finish(&stats);
        Ok(stats)
    }

    pub fn search_index(&mut self, query: &str, path: &Path) -> Result<Response> {
        self.validate_query(query)?;
        self.progress
            .begin("reading index and matching semantic chunks");
        let db = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .with_context(|| {
                format!(
                    "opening index {}; run `geg index <paths>` to build/rebuild it",
                    path.display()
                )
            })?;
        db.busy_timeout(Duration::from_secs(30))?;
        db.execute_batch("PRAGMA cache_size=-2048; PRAGMA mmap_size=0; BEGIN;")?;
        // Validate format first so legacy indexes get a clear rebuild message.
        for (key, expected) in [
            ("format", INDEX_FORMAT.to_owned()),
            ("model", self.model_hash.to_hex().to_string()),
            ("dimensions", self.dimensions.to_string()),
            ("ignore_case", self.args.ignore_case.to_string()),
            ("embedding_format", crate::cache::FORMAT.to_owned()),
        ] {
            let actual: Option<String> = db
                .query_row("SELECT value FROM meta_v1 WHERE key=?1", [key], |r| {
                    r.get(0)
                })
                .optional()?;
            ensure!(
                actual.as_deref() == Some(&expected),
                "index {key} mismatch; rebuild with `geg index <paths>` and the current model/settings"
            );
        }
        let mut stats = Stats::default();
        let query_vector = self.query_vector(query, &mut stats)?;
        let mut hits = Vec::new();
        let mut counts = BTreeMap::new();
        for path in db
            .prepare("SELECT path FROM files_v1")?
            .query_map([], |r| r.get::<_, String>(0))?
        {
            counts.insert(path?, 0u64);
        }
        stats.files = counts.len() as u64;
        let mut statement = db.prepare("SELECT c.path, c.line, c.text, v.vector FROM chunks_v2 c JOIN vectors_v1 v ON v.key=c.vector_key")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let text: String = row.get(2)?;
            if !input::searchable(&text) {
                continue;
            }
            let bytes: Vec<u8> = row.get(3)?;
            let vector = decode(&bytes, self.dimensions)?;
            let score = crate::search::best_score(&query_vector, &vector);
            stats.chunks += 1;
            self.progress.update(&stats);
            let path: String = row.get(0)?;
            let count = counts.entry(path.clone()).or_default();
            if self.accepts(score) && *count < self.args.max_count.unwrap_or(u64::MAX) {
                let text = crate::search::select_lines(
                    &text,
                    self.args.max_count.unwrap_or(u64::MAX) - *count,
                );
                *count += text.lines().count() as u64;
                insert_hit(
                    &mut hits,
                    Hit {
                        path,
                        line: u64::try_from(row.get::<_, i64>(1)?)?,
                        text,
                        score,
                    },
                    self.limit(),
                );
            }
        }
        self.progress.finish(&stats);
        Ok(Response {
            query: query.to_owned(),
            hits,
            counts,
            stats,
        })
    }
}
