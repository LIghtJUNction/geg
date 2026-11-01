//! Offline semantic grep with a local EmbeddingGemma GGUF model.
mod cache;
mod cache_management;
mod candidates;
mod cli;
mod config;
mod embedding;
mod index;
mod input;
mod progress;
mod search;
mod tui;

use std::{
    io::{self, BufRead, Read, Write},
    path::PathBuf,
    process::ExitCode,
};

use anyhow::{Context, Result, ensure};
use clap::{CommandFactory, FromArgMatches};
use directories::ProjectDirs;

use crate::{
    cache::Cache,
    cli::{Args, Command},
    search::Searcher,
};

fn cache_dir() -> Result<PathBuf> {
    Ok(ProjectDirs::from("", "", "geg")
        .context("cannot locate user cache directory; set --cache and --model")?
        .cache_dir()
        .to_owned())
}

fn run(args: &mut Args, matches: &clap::ArgMatches) -> Result<u8> {
    if !matches!(args.command, Some(Command::Skill)) {
        config::apply(args, matches)?;
    }
    normalize_args(args)?;
    if matches!(args.command, Some(Command::Skill)) {
        io::stdout()
            .lock()
            .write_all(include_str!("../skills/geg/SKILL.md").as_bytes())?;
        return Ok(0);
    }
    ensure!(
        args.threshold.is_finite() && (-1.0..=1.0).contains(&args.threshold),
        "--threshold must be finite and between -1 and 1"
    );
    let defaults = cache_dir()?;
    let model = args
        .model
        .clone()
        .unwrap_or_else(|| defaults.join("models/embeddinggemma-2-UD-Q4_K_XL.gguf"));
    let cache_path = args
        .cache
        .clone()
        .unwrap_or_else(|| defaults.join("vectors.sqlite3"));
    if let Some(Command::Cache { action }) = &args.command {
        let stdout = io::stdout();
        let mut out = io::BufWriter::new(stdout.lock());
        cache_management::run(action, &defaults, &cache_path, args, &mut out)?;
        out.flush()?;
        return Ok(0);
    }
    if args.tui {
        let selected = tui::run(args, model, cache_path)?;
        if let Some(hit) = selected {
            let counts = std::collections::BTreeMap::from([(hit.path.clone(), 1)]);
            let response = search::Response {
                query: String::new(),
                hits: vec![hit],
                counts,
                stats: search::Stats::default(),
            };
            let mut out = io::BufWriter::new(io::stdout().lock());
            print_response(&mut out, &response, args)?;
            out.flush()?;
            return Ok(0);
        }
        return Ok(1);
    }
    let cache = Cache::open(if args.no_cache {
        None
    } else {
        Some(&cache_path)
    })?;
    let mut searcher = Searcher::new(args, model, cache)?;
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    if let Some(command) = &args.command {
        return match command {
            Command::Skill | Command::Cache { .. } => {
                unreachable!("command handled before model setup")
            }
            Command::Index { paths, output } => {
                let path = if let Some(path) = output {
                    path.clone()
                } else {
                    index::default_path()?
                };
                let stats = searcher.build_index(paths, &path)?;
                serde_json::to_writer(
                    &mut out,
                    &serde_json::json!({"index": path, "stats": stats}),
                )?;
                writeln!(out)?;
                out.flush()?;
                Ok(0)
            }
            Command::Query { query, index } => {
                let response = searcher.search_index(query, index)?;
                let matched = response.matched(args);
                print_response(&mut out, &response, args)?;
                out.flush()?;
                Ok(u8::from(!matched))
            }
        };
    }
    if args.interactive {
        ensure!(
            !args.paths.is_empty() && args.paths.iter().all(|p| p != std::path::Path::new("-")),
            "--interactive requires file/directory paths after '--'"
        );
        let mut reader = io::stdin().lock();
        let mut line = Vec::new();
        loop {
            line.clear();
            // Bound protocol frames too, not just file chunks.
            let n = std::io::Read::by_ref(&mut reader)
                .take(1026)
                .read_until(b'\n', &mut line)?;
            if n == 0 {
                break;
            }
            ensure!(n <= 1025, "interactive query line exceeds 1024 bytes");
            let query = std::str::from_utf8(&line)?.trim_end_matches(['\n', '\r']);
            match searcher.search(query) {
                Ok(response) => serde_json::to_writer(&mut out, &response)?,
                Err(error) => serde_json::to_writer(
                    &mut out,
                    &serde_json::json!({"query": query, "error": format!("{error:#}")}),
                )?,
            }
            writeln!(out)?;
            out.flush()?;
        }
        return Ok(0);
    }
    let query = args
        .query
        .as_deref()
        .context("provide a query or a subcommand (see --help)")?;
    let response = searcher.search(query)?;
    let matched = response.matched(args);
    print_response(&mut out, &response, args)?;
    out.flush()?;
    Ok(u8::from(!matched))
}

fn print_response(out: &mut impl Write, response: &search::Response, args: &Args) -> Result<()> {
    if args.quiet {
        return Ok(());
    }
    let filename = args.with_filename || (!args.no_filename && response.counts.len() > 1);
    if args.files_with_matches || args.files_without_match || args.count {
        for (path, &count) in &response.counts {
            if args.files_with_matches || args.files_without_match {
                if (args.files_with_matches && count > 0)
                    || (args.files_without_match && count == 0)
                {
                    writeln!(out, "{}", safe_text(path))?;
                }
            } else if filename {
                writeln!(out, "{}:{count}", safe_text(path))?;
            } else {
                writeln!(out, "{count}")?;
            }
        }
        return Ok(());
    }
    if !args.json && args.format == "md" {
        if !response.hits.is_empty() {
            writeln!(out, "| path | line | score | text |")?;
            writeln!(out, "| --- | ---: | ---: | --- |")?;
            for hit in &response.hits {
                writeln!(
                    out,
                    "| {} | {} | {:.4} | {} |",
                    markdown_cell(&hit.path),
                    hit.line,
                    hit.score,
                    markdown_cell(hit.text.trim_end())
                )?;
                if args.line_buffered {
                    out.flush()?;
                }
            }
        }
        return Ok(());
    }
    for hit in &response.hits {
        if args.json {
            serde_json::to_writer(&mut *out, hit)?;
            writeln!(out)?;
            if args.line_buffered {
                out.flush()?;
            }
        } else {
            for (offset, line) in hit.text.lines().enumerate() {
                if filename {
                    write!(out, "{}:", safe_text(&hit.path))?;
                }
                if args.line_number {
                    write!(out, "{}:", hit.line + offset as u64)?;
                }
                if args.score {
                    write!(out, "{:.4}:", hit.score)?;
                }
                writeln!(out, "{}", safe_text(line))?;
                if args.line_buffered {
                    out.flush()?;
                }
            }
        }
    }
    Ok(())
}

fn markdown_cell(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '|' => escaped.push_str("&#124;"),
            '`' => escaped.push_str("&#96;"),
            '\\' => escaped.push_str("&#92;"),
            '*' => escaped.push_str("&#42;"),
            '_' => escaped.push_str("&#95;"),
            '[' => escaped.push_str("&#91;"),
            ']' => escaped.push_str("&#93;"),
            '\n' => escaped.push_str("<br>"),
            '\r' => {}
            '\t' => escaped.push_str("    "),
            c if c.is_control() => escaped.extend(c.escape_default()),
            c => escaped.push(c),
        }
    }
    escaped
}

fn safe_text(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for c in text.chars() {
        if c.is_control() && c != '\t' {
            escaped.extend(c.escape_default());
        } else {
            escaped.push(c);
        }
    }
    escaped
}

fn normalize_args(args: &mut Args) -> Result<()> {
    if args.command.is_some() {
        return Ok(());
    }
    if args.tui {
        if args.paths.is_empty()
            && args
                .query
                .as_ref()
                .is_some_and(|query| std::path::Path::new(query).exists())
            && let Some(path) = args.query.take()
        {
            args.paths.push(PathBuf::from(path));
        }
        if args.paths.is_empty() {
            args.paths.push(PathBuf::from("."));
        }
    }
    if args.interactive {
        ensure!(
            args.patterns.is_empty() && args.pattern_files.is_empty(),
            "--interactive cannot be combined with -e/-f"
        );
        if let Some(path) = args.query.take() {
            args.paths.insert(0, PathBuf::from(path));
        }
    } else if !args.patterns.is_empty() || !args.pattern_files.is_empty() {
        if let Some(path) = args.query.take() {
            args.paths.insert(0, PathBuf::from(path));
        }
        let mut patterns = args.patterns.clone();
        for path in &args.pattern_files {
            let mut text = String::new();
            if path == std::path::Path::new("-") {
                io::Read::take(io::stdin().lock(), 65537).read_to_string(&mut text)?;
            } else {
                io::Read::take(std::fs::File::open(path)?, 65537).read_to_string(&mut text)?;
            }
            ensure!(text.len() <= 65536, "query file exceeds 64 KiB");
            patterns.extend(text.lines().map(str::to_owned));
        }
        args.query = Some(patterns.join("\n"));
    }
    if args.paths.is_empty() && (args.recursive || args.dereference_recursive) {
        args.paths.push(PathBuf::from("."));
    }
    Ok(())
}

fn main() -> ExitCode {
    let matches = Args::command().get_matches();
    let mut args = match Args::from_arg_matches(&matches) {
        Ok(args) => args,
        Err(error) => error.exit(),
    };
    let result = run(&mut args, &matches);
    if result.is_ok()
        && !matches!(args.command, Some(Command::Skill))
        && let Ok(root) = cache_dir()
    {
        let path = args
            .cache
            .clone()
            .unwrap_or_else(|| root.join("vectors.sqlite3"));
        cache_management::warn(&root, &path, &args);
    }
    match result {
        Ok(code) => ExitCode::from(code),
        Err(error)
            if error
                .downcast_ref::<io::Error>()
                .is_some_and(|e| e.kind() == io::ErrorKind::BrokenPipe) =>
        {
            ExitCode::SUCCESS
        }
        Err(error) => {
            if !args.no_messages {
                eprintln!("geg: {error:#}");
            }
            ExitCode::from(2)
        }
    }
}
