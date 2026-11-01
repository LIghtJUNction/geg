use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Result, ensure};
use rusqlite::{Connection, OpenFlags};
use serde::Serialize;

use crate::cli::{Args, CacheCommand};

fn file_bytes(path: &Path) -> Result<u64> {
    match fs::symlink_metadata(path) {
        Ok(meta) => Ok(if meta.is_file() { meta.len() } else { 0 }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(error) => Err(error.into()),
    }
}

fn database_bytes(path: &Path) -> Result<u64> {
    let mut bytes = file_bytes(path)?;
    for suffix in ["-wal", "-shm"] {
        let mut sidecar = path.as_os_str().to_os_string();
        sidecar.push(suffix);
        bytes += file_bytes(Path::new(&sidecar))?;
    }
    Ok(bytes)
}

// Don't follow symlinks out of the cache tree, and don't read file contents.
fn directory_bytes(path: &Path) -> Result<u64> {
    if !path.exists() {
        return Ok(0);
    }
    let mut bytes = 0u64;
    let mut directories = vec![path.to_owned()];
    while let Some(directory) = directories.pop() {
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let meta = fs::symlink_metadata(entry.path())?;
            if meta.is_dir() {
                directories.push(entry.path());
            } else if meta.is_file() {
                bytes = bytes.saturating_add(meta.len());
            }
        }
    }
    Ok(bytes)
}

pub fn total_bytes(root: &Path, database: &Path) -> Result<u64> {
    let mut bytes = directory_bytes(root)?;
    let absolute = if database.is_absolute() {
        database.to_owned()
    } else {
        std::env::current_dir()?.join(database)
    };
    if !absolute.starts_with(root) {
        bytes = bytes.saturating_add(database_bytes(database)?);
    }
    Ok(bytes)
}

pub fn warn(root: &Path, path: &Path, args: &Args) {
    if args.cache_warning_gb == 0 || args.quiet || args.no_messages {
        return;
    }
    match total_bytes(root, path) {
        Ok(bytes) if bytes > u64::from(args.cache_warning_gb) * 1024 * 1024 * 1024 => {
            eprintln!(
                "geg: global cache is {:.2} GiB (warning threshold: {} GiB). Run `geg cache info` to inspect it or `geg cache clean` to clear global vectors; models and project indexes are kept.",
                bytes as f64 / 1073741824.0,
                args.cache_warning_gb
            );
        }
        Err(error) => eprintln!("geg: cannot measure global cache size: {error}"),
        _ => {}
    }
}

fn open_existing(path: &Path, write: bool) -> Result<Option<Connection>> {
    if !path.exists() {
        return Ok(None);
    }
    let flags = if write {
        OpenFlags::SQLITE_OPEN_READ_WRITE
    } else {
        OpenFlags::SQLITE_OPEN_READ_ONLY
    };
    let db = Connection::open_with_flags(path, flags)?;
    db.busy_timeout(Duration::from_secs(30))?;
    let tables: Vec<String> = db
        .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")?
        .query_map([], |row| row.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    ensure!(
        tables.len() == 1 && tables[0] == "vectors_v1",
        "refusing cache management: {} is not a standalone global vector cache (project indexes and other databases are protected)",
        path.display()
    );
    Ok(Some(db))
}

#[derive(Serialize)]
struct Info {
    global_directory: PathBuf,
    vector_cache: PathBuf,
    global_bytes: u64,
    vector_bytes: u64,
    vectors: i64,
    warning_limit_bytes: u64,
}

pub fn run(
    action: &CacheCommand,
    root: &Path,
    path: &Path,
    args: &Args,
    out: &mut impl Write,
) -> Result<()> {
    match action {
        CacheCommand::Info => {
            let db = open_existing(path, false)?;
            let vectors = if let Some(db) = db {
                db.query_row("SELECT count(*) FROM vectors_v1", [], |row| {
                    row.get::<_, i64>(0)
                })?
            } else {
                0
            };
            let info = Info {
                global_directory: root.to_owned(),
                vector_cache: path.to_owned(),
                global_bytes: total_bytes(root, path)?,
                vector_bytes: database_bytes(path)?,
                vectors,
                warning_limit_bytes: u64::from(args.cache_warning_gb) * 1024 * 1024 * 1024,
            };
            if args.json {
                serde_json::to_writer(&mut *out, &info)?;
                writeln!(out)?;
            } else {
                writeln!(
                    out,
                    "Global cache: {}\nVector cache: {}\nTotal: {:.2} MiB; vector database: {:.2} MiB; vectors: {}\nWarning threshold: {} GiB\nModels and project indexes are not removed by `cache clean`.",
                    root.display(),
                    path.display(),
                    info.global_bytes as f64 / 1048576.0,
                    info.vector_bytes as f64 / 1048576.0,
                    vectors,
                    args.cache_warning_gb
                )?;
            }
        }
        CacheCommand::Clean => {
            let before = database_bytes(path)?;
            let removed = if let Some(db) = open_existing(path, true)? {
                let removed = db.execute("DELETE FROM vectors_v1", [])?;
                db.execute_batch("VACUUM; PRAGMA wal_checkpoint(TRUNCATE);")?;
                removed
            } else {
                0
            };
            let reclaimed = before.saturating_sub(database_bytes(path)?);
            if args.json {
                serde_json::to_writer(
                    &mut *out,
                    &serde_json::json!({"removed_vectors":removed,"reclaimed_bytes":reclaimed,"models_kept":true,"project_indexes_kept":true}),
                )?;
                writeln!(out)?;
            } else {
                writeln!(
                    out,
                    "Removed {removed} global vectors; reclaimed {:.2} MiB. Models and project indexes kept.",
                    reclaimed as f64 / 1048576.0
                )?;
            }
        }
    }
    Ok(())
}
