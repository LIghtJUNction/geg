//! Predictable user/project configuration; never executes project code.
use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use clap::{ArgMatches, parser::ValueSource};
use directories::BaseDirs;
use serde::Deserialize;

use crate::cli::{Args, Command};

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    model: Option<PathBuf>,
    cache: Option<PathBuf>,
    index: Option<PathBuf>,
    top: Option<u32>,
    threshold: Option<f32>,
    chunk_lines: Option<u32>,
    dimensions: Option<u16>,
    threads: Option<u32>,
    code: Option<bool>,
    jsonl: Option<bool>,
    format: Option<String>,
    no_progress: Option<bool>,
    device: Option<String>,
    exhaustive: Option<bool>,
    candidates: Option<u32>,
    no_cache: Option<bool>,
    cache_warning_gb: Option<u32>,
    no_index_hint: Option<bool>,
    ignore_case: Option<bool>,
    no_ignore: Option<bool>,
    hidden: Option<bool>,
}

fn project_config(start: &Path, home: Option<&Path>) -> Option<PathBuf> {
    for directory in start.ancestors() {
        if Some(directory) == home {
            break;
        }
        let path = directory.join(".gegrc.toml");
        if path.is_file() {
            return Some(path);
        }
        if directory.join(".git").exists() {
            break;
        }
    }
    None
}

fn resolve(base: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

pub fn apply(args: &mut Args, matches: &ArgMatches) -> Result<()> {
    if args.no_config {
        return Ok(());
    }
    let cwd = std::env::current_dir()?;
    let mut paths = Vec::new();
    if let Some(path) = &args.config {
        paths.push(resolve(&cwd, path.clone()));
    } else {
        let home = BaseDirs::new().map(|base| base.home_dir().to_owned());
        if let Some(home) = &home {
            let path = home.join(".gegrc.toml");
            if path.is_file() {
                paths.push(path);
            }
        }
        if let Some(path) = project_config(&cwd, home.as_deref()) {
            paths.push(path);
        }
    }
    for path in paths {
        apply_file(args, matches, &path)
            .with_context(|| format!("configuration {}", path.display()))?;
    }
    Ok(())
}

fn apply_file(args: &mut Args, matches: &ArgMatches, path: &Path) -> Result<()> {
    use std::io::Read;
    let mut text = String::new();
    fs::File::open(path)?
        .take(65537)
        .read_to_string(&mut text)?;
    ensure!(text.len() <= 65536, "configuration exceeds 64 KiB");
    let config: Config = toml::from_str(&text)?;
    // Validate even overridden fields so mistakes don't silently survive.
    ensure!(
        config.top.is_none_or(|v| (1..=10000).contains(&v)),
        "top must be in 1..=10000"
    );
    ensure!(
        config
            .threshold
            .is_none_or(|v| v.is_finite() && (-1.0..=1.0).contains(&v)),
        "threshold must be finite and in -1..=1"
    );
    ensure!(
        config.chunk_lines.is_none_or(|v| (1..=128).contains(&v)),
        "chunk_lines must be in 1..=128"
    );
    ensure!(
        config
            .dimensions
            .is_none_or(|v| [128, 256, 512, 768].contains(&v)),
        "dimensions must be 128, 256, 512 or 768"
    );
    ensure!(
        config.threads.is_none_or(|v| (1..=256).contains(&v)),
        "threads must be in 1..=256"
    );
    let explicit = |id| {
        matches!(
            matches.value_source(id),
            Some(ValueSource::CommandLine | ValueSource::EnvVariable)
        )
    };
    let base = path
        .parent()
        .context("configuration has no parent directory")?;
    macro_rules! value {
        ($($field:ident),* $(,)?) => { $(if !explicit(stringify!($field)) {
            if let Some(value) = config.$field { args.$field = value; }
        })* };
    }
    ensure!(
        config.cache_warning_gb.is_none_or(|v| v <= 1048576),
        "cache_warning_gb must be in 0..=1048576"
    );
    value!(
        top,
        threshold,
        chunk_lines,
        code,
        no_cache,
        no_index_hint,
        ignore_case,
        no_ignore,
        hidden,
        no_progress,
        cache_warning_gb,
        exhaustive,
        candidates
    );
    ensure!(
        config.candidates.is_none_or(|v| (1..=512).contains(&v)),
        "candidates must be in 1..=512"
    );
    if let Some(value) = config.device {
        ensure!(
            ["auto", "gpu", "cpu"].contains(&value.as_str()),
            "device must be auto, gpu or cpu"
        );
        if !explicit("device") {
            args.device = value;
        }
    }
    if !explicit("json")
        && let Some(value) = config.jsonl
    {
        args.json = value;
    }
    if let Some(value) = config.format {
        ensure!(
            ["md", "grep"].contains(&value.as_str()),
            "format must be md or grep"
        );
        if !explicit("format") {
            args.format = value;
        }
    }
    if !explicit("threads")
        && let Some(value) = config.threads
    {
        args.threads = Some(value);
    }
    if !explicit("dimensions")
        && let Some(value) = config.dimensions
    {
        args.dimensions = value.to_string();
    }
    if !explicit("model")
        && let Some(value) = config.model
    {
        args.model = Some(resolve(base, value));
    }
    if !explicit("cache")
        && let Some(value) = config.cache
    {
        args.cache = Some(resolve(base, value));
    }
    if let Some(index) = config.index {
        let index = resolve(base, index);
        args.index_path = Some(index.clone());
        match &mut args.command {
            Some(Command::Index { output, .. }) => {
                let supplied = matches
                    .subcommand_matches("index")
                    .and_then(|m| m.value_source("output"))
                    == Some(ValueSource::CommandLine);
                if !supplied {
                    *output = Some(index);
                }
            }
            Some(Command::Query { index: value, .. }) => {
                let supplied = matches
                    .subcommand_matches("query")
                    .and_then(|m| m.value_source("index"))
                    == Some(ValueSource::CommandLine);
                if !supplied {
                    *value = index;
                }
            }
            _ => {}
        }
    }
    Ok(())
}
