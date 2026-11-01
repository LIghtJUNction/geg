use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader, IsTerminal},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use ignore::WalkBuilder;
use serde::Serialize;

use crate::{
    cache::Cache,
    cli::Args,
    embedding::{self, Engine},
    input,
};

// One suggestion per process; never pre-walk the tree just to count files.
const INDEX_HINT_FILES: u64 = 100;

#[derive(Debug, Serialize)]
pub struct Hit {
    pub path: String,
    pub line: u64,
    pub score: f32,
    pub text: String,
}

#[derive(Default, Serialize)]
pub struct Stats {
    pub cache_hits: u64,
    pub embedded: u64,
    pub chunks: u64,
    pub files: u64,
}

#[derive(Serialize)]
pub struct Response {
    pub query: String,
    pub hits: Vec<Hit>,
    pub counts: BTreeMap<String, u64>,
    pub stats: Stats,
}

impl Response {
    pub fn matched(&self, args: &Args) -> bool {
        if args.files_without_match {
            self.counts.values().any(|&n| n == 0)
        } else {
            self.counts.values().any(|&n| n > 0)
        }
    }
}

pub struct Searcher<'a> {
    pub(crate) args: &'a Args,
    model_path: PathBuf,
    pub(crate) model_hash: blake3::Hash,
    pub(crate) dimensions: usize,
    cache: Cache,
    engine: Option<Engine>,
    hint_emitted: bool,
}

impl<'a> Searcher<'a> {
    pub fn new(args: &'a Args, model_path: PathBuf, cache: Cache) -> Result<Self> {
        // Hash contents, not just mtime: replacing a model cannot reuse stale vectors.
        let mut file = File::open(&model_path).with_context(|| {
            format!(
                "opening model {}; set --model/GEG_MODEL to a local EmbeddingGemma 2 GGUF",
                model_path.display()
            )
        })?;
        let mut hash = blake3::Hasher::new();
        hash.update_reader(&mut file)?;
        Ok(Self {
            args,
            model_path,
            model_hash: hash.finalize(),
            dimensions: args.dimensions.parse()?,
            cache,
            engine: None,
            hint_emitted: false,
        })
    }

    pub(crate) fn vector(&mut self, text: &str, stats: &mut Stats) -> Result<Vec<f32>> {
        let lowered;
        let text = if self.args.ignore_case {
            lowered = text.to_lowercase();
            &lowered
        } else {
            text
        };
        let key = Cache::key(&self.model_hash, self.dimensions, text);
        if let Some(vector) = self.cache.get(&key, self.dimensions)? {
            stats.cache_hits += 1;
            return Ok(vector);
        }
        if self.engine.is_none() {
            let threads = self.args.threads.unwrap_or_else(|| {
                std::thread::available_parallelism().map_or(1, |n| n.get().min(8) as u32)
            });
            self.engine = Some(Engine::load(&self.model_path, threads, self.dimensions)?);
        }
        let vector = self
            .engine
            .as_mut()
            .context("embedding engine not initialized")?
            .embed(text)?;
        self.cache.put(&key, &vector)?;
        stats.embedded += 1;
        Ok(vector)
    }

    pub(crate) fn validate_query(&self, query: &str) -> Result<()> {
        ensure!(!query.trim().is_empty(), "query must not be empty");
        ensure!(
            query.len() <= 64 * 1024,
            "combined queries must not exceed 64 KiB"
        );
        for query in query.lines() {
            ensure!(
                !query.trim().is_empty() && query.len() <= 1024,
                "each query must contain 1..1024 UTF-8 bytes"
            );
        }
        Ok(())
    }

    pub(crate) fn query_vector(&mut self, query: &str, stats: &mut Stats) -> Result<Vec<Vec<f32>>> {
        let task = if self.args.code {
            "code retrieval"
        } else {
            "search result"
        };
        query
            .lines()
            .map(|query| self.vector(&format!("task: {task} | query: {query}"), stats))
            .collect()
    }

    pub(crate) fn accepts(&self, score: f32) -> bool {
        (score >= self.args.threshold) != self.args.invert_match
    }

    pub(crate) fn limit(&self) -> usize {
        self.args.top as usize
    }

    pub fn search(&mut self, query: &str) -> Result<Response> {
        self.validate_query(query)?;
        let mut stats = Stats::default();
        let query_vector = self.query_vector(query, &mut stats)?;
        let mut hits = Vec::new();
        let mut counts = BTreeMap::new();
        let paths = self.args.paths.clone();
        if paths.is_empty() {
            ensure!(
                !std::io::stdin().is_terminal(),
                "provide a file/directory or pipe text to stdin"
            );
            self.scan(
                std::io::stdin().lock(),
                &self.args.label.clone(),
                &query_vector,
                &mut hits,
                &mut counts,
                &mut stats,
            )?;
        }
        for path in paths {
            if path == Path::new("-") {
                self.scan(
                    std::io::stdin().lock(),
                    &self.args.label.clone(),
                    &query_vector,
                    &mut hits,
                    &mut counts,
                    &mut stats,
                )?;
            } else if path.is_dir() {
                for entry in WalkBuilder::new(&path)
                    .follow_links(self.args.dereference_recursive)
                    .build()
                {
                    let entry = entry?;
                    if entry.file_type().is_some_and(|kind| kind.is_file()) {
                        self.scan_file(
                            entry.path(),
                            &query_vector,
                            &mut hits,
                            &mut counts,
                            &mut stats,
                        )?;
                    }
                }
            } else {
                self.scan_file(&path, &query_vector, &mut hits, &mut counts, &mut stats)?;
            }
        }
        Ok(Response {
            query: query.to_owned(),
            hits,
            counts,
            stats,
        })
    }

    pub(crate) fn read_file(
        &mut self,
        path: &Path,
        consume: impl FnOnce(&mut Self, BufReader<File>, &str) -> Result<()>,
    ) -> Result<()> {
        let mut reader = BufReader::new(
            File::open(path).with_context(|| format!("opening {}", path.display()))?,
        );
        // Skip binary files before inference; stdin is explicit text.
        if reader.fill_buf()?.contains(&0) {
            return Ok(());
        }
        consume(self, reader, &path.to_string_lossy())
            .with_context(|| format!("searching {}", path.display()))
    }

    fn scan_file(
        &mut self,
        path: &Path,
        query: &[Vec<f32>],
        hits: &mut Vec<Hit>,
        counts: &mut BTreeMap<String, u64>,
        stats: &mut Stats,
    ) -> Result<()> {
        stats.files += 1;
        if stats.files >= INDEX_HINT_FILES
            && !self.hint_emitted
            && !self.args.no_index_hint
            && !self.args.quiet
        {
            eprintln!(
                "geg: searched {INDEX_HINT_FILES} files; for repeated searches build an explicit index: geg index <root>, then geg query <query> --index <printed-path> (已扫描较多文件，建议显式建立索引加速后续搜索)"
            );
            self.hint_emitted = true;
        }
        self.read_file(path, |this, reader, label| {
            this.scan(reader, label, query, hits, counts, stats)
        })
    }

    fn scan(
        &mut self,
        reader: impl BufRead,
        path: &str,
        query: &[Vec<f32>],
        hits: &mut Vec<Hit>,
        counts: &mut BTreeMap<String, u64>,
        stats: &mut Stats,
    ) -> Result<()> {
        counts.entry(path.to_owned()).or_default();
        input::chunks(reader, self.args.chunk_lines, |line, text| {
            if text.trim().is_empty() || counts[path] >= self.args.max_count.unwrap_or(u64::MAX) {
                return Ok(());
            }
            stats.chunks += 1;
            let vector = self.vector(&format!("title: none | text: {text}"), stats)?;
            let score = best_score(query, &vector);
            if self.accepts(score) {
                let remaining = self.args.max_count.unwrap_or(u64::MAX) - counts[path];
                let text = select_lines(text, remaining);
                *counts.get_mut(path).context("missing file count")? += text.lines().count() as u64;
                insert_hit(
                    hits,
                    Hit {
                        path: path.to_owned(),
                        line,
                        score,
                        text,
                    },
                    self.limit(),
                );
            }
            Ok(())
        })
    }
}

pub(crate) fn best_score(queries: &[Vec<f32>], vector: &[f32]) -> f32 {
    queries
        .iter()
        .map(|q| embedding::cosine(q, vector))
        .fold(-1.0, f32::max)
}

pub(crate) fn select_lines(text: &str, max: u64) -> String {
    let mut end = 0;
    for part in text
        .split_inclusive('\n')
        .take(usize::try_from(max).unwrap_or(usize::MAX))
    {
        end += part.len();
    }
    text[..end].to_owned()
}

pub(crate) fn insert_hit(hits: &mut Vec<Hit>, hit: Hit, top: usize) {
    let index = hits.partition_point(|h| {
        h.score.total_cmp(&hit.score).is_gt()
            || (h.score == hit.score && (&h.path, h.line) <= (&hit.path, hit.line))
    });
    if index < top {
        hits.insert(index, hit);
        if hits.len() > top {
            hits.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_is_bounded_ranked_and_stable() {
        let mut hits = Vec::new();
        for (line, score) in [(3, 0.8), (2, 0.8), (1, 0.2), (4, 0.9), (5, -0.5)] {
            insert_hit(
                &mut hits,
                Hit {
                    path: "x".into(),
                    line,
                    score,
                    text: "x".into(),
                },
                3,
            );
            assert!(hits.len() <= 3);
        }
        assert_eq!(hits.iter().map(|h| h.line).collect::<Vec<_>>(), [4, 2, 3]);
    }

    #[test]
    fn max_lines_preserves_bytes() {
        assert_eq!(select_lines("a\nb\nc", 2), "a\nb\n");
        assert_eq!(select_lines("你好", 1), "你好");
        assert_eq!(select_lines("a", 0), "");
    }
}
