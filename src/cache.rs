use std::{fs, path::Path, time::Duration};

use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, OptionalExtension, params};

pub(crate) const FORMAT: &str = "geg-embedding-v1-llama4-0.9-mean-512";

pub struct Cache(Option<Connection>);

/// New databases may contain sensitive source excerpts; restrict Unix permissions.
pub(crate) fn prepare_database(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    match options.open(path) {
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(error) => Err(error.into()),
    }
}

impl Cache {
    pub fn open(path: Option<&Path>) -> Result<Self> {
        let Some(path) = path else {
            return Ok(Self(None));
        };
        prepare_database(path)?;
        let db = Connection::open(path)?;
        db.busy_timeout(Duration::from_secs(30))?;
        db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
            PRAGMA cache_size=-2048; PRAGMA mmap_size=0;
            CREATE TABLE IF NOT EXISTS vectors_v1 (key BLOB PRIMARY KEY, vector BLOB NOT NULL) WITHOUT ROWID;")?;
        Ok(Self(Some(db)))
    }

    /// Borrow project-owned vectors without copying them back into the global cache.
    pub fn open_index(path: &Path) -> Result<Self> {
        if !path.is_file() {
            return Ok(Self(None));
        }
        let db = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        db.busy_timeout(Duration::from_secs(30))?;
        let current: bool = db.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='vectors_v1')",
            [],
            |r| r.get(0),
        )?;
        if !current {
            return Ok(Self(None));
        }
        db.execute_batch("PRAGMA cache_size=-2048; PRAGMA mmap_size=0;")?;
        Ok(Self(Some(db)))
    }

    pub fn key(model: &blake3::Hash, dimensions: usize, text: &str) -> blake3::Hash {
        let mut hash = blake3::Hasher::new();
        // Changing inference settings/prompt processing requires a new namespace.
        hash.update(FORMAT.as_bytes());
        hash.update(&[0]);
        hash.update(model.as_bytes());
        hash.update(&(dimensions as u64).to_le_bytes());
        hash.update(text.as_bytes());
        hash.finalize()
    }

    pub fn get(&self, key: &blake3::Hash, dimensions: usize) -> Result<Option<Vec<f32>>> {
        let Some(db) = &self.0 else {
            return Ok(None);
        };
        let bytes: Option<Vec<u8>> = db
            .prepare_cached("SELECT vector FROM vectors_v1 WHERE key=?1")?
            .query_row([key.as_bytes().as_slice()], |r| r.get(0))
            .optional()?;
        let Some(bytes) = bytes else {
            return Ok(None);
        };
        ensure!(
            bytes.len() == dimensions * 4,
            "invalid cached vector length; remove the cache file"
        );
        let vector: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
            .collect();
        ensure!(
            vector.iter().all(|v| v.is_finite()),
            "invalid cached vector values; remove the cache file"
        );
        Ok(Some(vector))
    }

    pub fn shares_path(&self, path: &Path) -> Result<bool> {
        let Some(db) = &self.0 else {
            return Ok(false);
        };
        let Some(current) = db.path() else {
            return Ok(false);
        };
        // Resolve existing files so symlink aliases cannot make the index delete itself.
        Ok(path.exists() && fs::canonicalize(current)? == fs::canonicalize(path)?)
    }

    /// Transfer ownership of document vectors only after the index commit succeeds.
    pub fn remove_indexed(&self, index: &Path) -> Result<()> {
        let Some(db) = &self.0 else {
            return Ok(());
        };
        ensure!(
            !self.shares_path(index)?,
            "index and global cache must use different database paths"
        );
        db.execute(
            "ATTACH DATABASE ?1 AS geg_index",
            [index.to_str().context("index path is not UTF-8")?],
        )?;
        let result = db.execute(
            "DELETE FROM main.vectors_v1 WHERE key IN (SELECT key FROM geg_index.vectors_v1)",
            [],
        );
        let detached = db.execute("DETACH DATABASE geg_index", []);
        result?;
        detached?;
        Ok(())
    }

    pub fn put(&self, key: &blake3::Hash, vector: &[f32]) -> Result<()> {
        if let Some(db) = &self.0 {
            let bytes: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
            db.prepare_cached("INSERT OR REPLACE INTO vectors_v1(key, vector) VALUES (?1, ?2)")?
                .execute(params![key.as_bytes().as_slice(), bytes])?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_roundtrip_and_invalidation() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("cache.sqlite3");
        let model = blake3::hash(b"model");
        let key = Cache::key(&model, 2, "title: none | text: hello");
        let cache = Cache::open(Some(&path))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(fs::metadata(&path)?.permissions().mode() & 0o777, 0o600);
        }
        assert!(cache.get(&key, 2)?.is_none());
        cache.put(&key, &[0.6, 0.8])?;
        drop(cache);
        let cache = Cache::open(Some(&path))?;
        assert_eq!(cache.get(&key, 2)?, Some(vec![0.6, 0.8]));
        assert_ne!(key, Cache::key(&model, 3, "title: none | text: hello"));
        assert_ne!(key, Cache::key(&model, 2, "title: none | text: changed"));
        assert_ne!(
            key,
            Cache::key(
                &blake3::hash(b"other model"),
                2,
                "title: none | text: hello"
            )
        );
        assert!(cache.get(&key, 3).is_err());
        Ok(())
    }

    #[test]
    fn disabled_cache() -> Result<()> {
        let cache = Cache::open(None)?;
        let key = Cache::key(&blake3::hash(b"x"), 2, "a");
        cache.put(&key, &[1.0, 0.0])?;
        assert!(cache.get(&key, 2)?.is_none());
        Ok(())
    }
}
