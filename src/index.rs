use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, params};

use crate::{
    input,
    search::{Hit, Response, Searcher, Stats, insert_hit},
};

pub fn default_path(cache: &Path, paths: &[PathBuf]) -> Result<PathBuf> {
    let mut hash = blake3::Hasher::new();
    for path in paths {
        hash.update(fs::canonicalize(path)?.to_string_lossy().as_bytes());
        hash.update(&[0]);
    }
    Ok(cache
        .join("indexes")
        .join(format!("{}.sqlite3", &hash.finalize().to_hex()[..16])))
}

fn open(path: &Path) -> Result<Connection> {
    crate::cache::prepare_database(path)?;
    let db = Connection::open(path)?;
    db.busy_timeout(Duration::from_secs(30))?;
    db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA cache_size=-2048; PRAGMA mmap_size=0;
        CREATE TABLE IF NOT EXISTS meta_v1 (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS files_v1 (path TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS chunks_v1 (path TEXT NOT NULL, line INTEGER NOT NULL, text TEXT NOT NULL, vector BLOB NOT NULL);")?;
    Ok(db)
}

impl Searcher<'_> {
    pub fn build_index(&mut self, paths: &[PathBuf], output: &Path) -> Result<Stats> {
        ensure!(
            paths.iter().all(|p| p != Path::new("-")),
            "indexing requires files/directories, not stdin"
        );
        let mut db = open(output)?;
        let output = fs::canonicalize(output)?;
        let mut wal = output.as_os_str().to_os_string();
        wal.push("-wal");
        let mut shm = output.as_os_str().to_os_string();
        shm.push("-shm");
        let transaction = db.transaction()?;
        transaction.execute("DELETE FROM chunks_v1", [])?;
        transaction.execute("DELETE FROM files_v1", [])?;
        let mut stats = Stats::default();
        input::visit_files(paths, |path| {
            // Never index the output database or its SQLite sidecars.
            let absolute = fs::canonicalize(path)?;
            if absolute == output || absolute.as_os_str() == wal || absolute.as_os_str() == shm {
                return Ok(());
            }
            self.read_file(&absolute, |this, reader, label| {
                transaction.execute("INSERT OR IGNORE INTO files_v1 VALUES (?1)", [label])?;
                input::chunks(reader, this.args.chunk_lines, |line, text| {
                    if text.trim().is_empty() {
                        return Ok(());
                    }
                    let vector = this.vector(&format!("title: none | text: {text}"), &mut stats)?;
                    let bytes: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
                    transaction
                        .prepare_cached("INSERT INTO chunks_v1 VALUES (?1, ?2, ?3, ?4)")?
                        .execute(params![label, i64::try_from(line)?, text, bytes])?;
                    stats.chunks += 1;
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
            ("format", crate::cache::FORMAT.to_owned()),
        ] {
            transaction.execute(
                "INSERT OR REPLACE INTO meta_v1 VALUES (?1, ?2)",
                params![key, value],
            )?;
        }
        transaction.commit()?;
        db.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
        Ok(stats)
    }

    pub fn search_index(&mut self, query: &str, path: &Path) -> Result<Response> {
        self.validate_query(query)?;
        let db = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .with_context(|| format!("opening index {}", path.display()))?;
        db.busy_timeout(Duration::from_secs(30))?;
        db.execute_batch("PRAGMA cache_size=-2048; PRAGMA mmap_size=0;")?;
        // Keep metadata and chunks in the same snapshot during concurrent rebuilds.
        db.execute_batch("BEGIN;")?;
        for (key, expected) in [
            ("model", self.model_hash.to_hex().to_string()),
            ("dimensions", self.dimensions.to_string()),
            ("ignore_case", self.args.ignore_case.to_string()),
            ("format", crate::cache::FORMAT.to_owned()),
        ] {
            let actual: String =
                db.query_row("SELECT value FROM meta_v1 WHERE key=?1", [key], |r| {
                    r.get(0)
                })?;
            ensure!(
                actual == expected,
                "index {key} mismatch; rebuild with the current model/settings"
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
        let mut statement = db.prepare("SELECT path, line, text, vector FROM chunks_v1")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let bytes: Vec<u8> = row.get(3)?;
            ensure!(
                bytes.len() == self.dimensions * 4,
                "invalid index vector length; rebuild the index"
            );
            let vector: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
                .collect();
            ensure!(
                vector.iter().all(|v| v.is_finite()),
                "non-finite index vector; rebuild the index"
            );
            let score = crate::search::best_score(&query_vector, &vector);
            stats.chunks += 1;
            let path: String = row.get(0)?;
            let count = counts.entry(path.clone()).or_default();
            if self.accepts(score) && *count < self.args.max_count.unwrap_or(u64::MAX) {
                let text: String = row.get(2)?;
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
        Ok(Response {
            query: query.to_owned(),
            hits,
            counts,
            stats,
        })
    }
}
