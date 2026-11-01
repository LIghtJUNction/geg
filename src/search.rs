use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader, IsTerminal},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use serde::Serialize;

use crate::{
    cache::Cache,
    cli::Args,
    embedding::{self, Engine},
    input,
};

// One suggestion per process; never pre-walk the tree just to count files.
const INDEX_HINT_FILES: u64 = 100;

#[derive(Debug, Clone, Serialize)]
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
    pub candidate_chunks: u64,
    pub skipped_files: u64,
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
    pub(crate) cache: Cache,
    index_cache: Cache,
    engine: Option<Engine>,
    pub(crate) revision: Option<(std::sync::Arc<std::sync::atomic::AtomicU64>, u64)>,
    hint_emitted: bool,
    pub(crate) progress: crate::progress::Progress,
}

impl<'a> Searcher<'a> {
    pub fn new(args: &'a Args, model_path: PathBuf, cache: Cache) -> Result<Self> {
        let mut progress =
            crate::progress::Progress::new(!args.quiet && !args.no_messages && !args.no_progress);
        progress.begin("verifying local model identity");
        // Hash contents, not just mtime: replacing a model cannot reuse stale vectors.
        let mut file = File::open(&model_path).with_context(|| {
            format!(
                "local model missing/unreadable at {}; geg does not download models. Prepare a text-only EmbeddingGemma 2 GGUF and set --model/GEG_MODEL (see README)",
                model_path.display()
            )
        })?;
        let mut hash = blake3::Hasher::new();
        hash.update_reader(&mut file)?;
        let index_path = args
            .index_path
            .clone()
            .unwrap_or(crate::index::default_path()?);
        let index_cache = if args.no_cache {
            Cache::open(None)?
        } else {
            Cache::open_index(&index_path)?
        };
        Ok(Self {
            args,
            model_path,
            model_hash: hash.finalize(),
            dimensions: args.dimensions.parse()?,
            cache,
            index_cache,
            engine: None,
            revision: None,
            hint_emitted: false,
            progress,
        })
    }

    pub(crate) fn set_revision(
        &mut self,
        revision: std::sync::Arc<std::sync::atomic::AtomicU64>,
        expected: u64,
    ) {
        self.revision = Some((revision, expected));
    }

    pub(crate) fn check_cancelled(&self) -> Result<()> {
        if let Some((revision, expected)) = &self.revision {
            ensure!(
                revision.load(std::sync::atomic::Ordering::Relaxed) == *expected,
                "search superseded by a newer query"
            );
        }
        Ok(())
    }

    pub(crate) fn cached_vector(&self, key: &blake3::Hash) -> Result<Option<Vec<f32>>> {
        if let Some(vector) = self.cache.get(key, self.dimensions)? {
            return Ok(Some(vector));
        }
        self.index_cache.get(key, self.dimensions)
    }

    pub(crate) fn vector(&mut self, text: &str, stats: &mut Stats) -> Result<Vec<f32>> {
        self.check_cancelled()?;
        let lowered;
        let text = if self.args.ignore_case {
            lowered = text.to_lowercase();
            &lowered
        } else {
            text
        };
        let key = Cache::key(&self.model_hash, self.dimensions, text);
        if let Some(vector) = self.cached_vector(&key)? {
            stats.cache_hits += 1;
            self.progress.update(stats);
            return Ok(vector);
        }
        let vector = self.embed_uncached(text, stats)?;
        self.cache.put(&key, &vector)?;
        Ok(vector)
    }

    fn ensure_engine(&mut self) -> Result<()> {
        if self.engine.is_none() {
            self.progress.loading(&self.model_path);
            let threads = self.args.threads.unwrap_or_else(|| {
                std::thread::available_parallelism().map_or(1, |n| n.get().min(8) as u32)
            });
            let engine = Engine::load(
                &self.model_path,
                threads,
                self.dimensions,
                &self.args.device,
            )?;
            self.progress.device(engine.device);
            self.engine = Some(engine);
        }
        Ok(())
    }

    pub(crate) fn vectors(
        &mut self,
        prompts: &[String],
        stats: &mut Stats,
    ) -> Result<Vec<Vec<f32>>> {
        let prompts: Vec<String> = prompts
            .iter()
            .map(|text| {
                if self.args.ignore_case {
                    text.to_lowercase()
                } else {
                    text.clone()
                }
            })
            .collect();
        let keys: Vec<_> = prompts
            .iter()
            .map(|text| Cache::key(&self.model_hash, self.dimensions, text))
            .collect();
        let mut output = Vec::with_capacity(prompts.len());
        let mut missing = Vec::new();
        for (i, key) in keys.iter().enumerate() {
            let vector = self.cached_vector(key)?;
            if vector.is_some() {
                stats.cache_hits += 1;
            } else {
                missing.push(i);
            }
            output.push(vector);
        }
        if !missing.is_empty() {
            self.check_cancelled()?;
            self.ensure_engine()?;
            let texts: Vec<_> = missing.iter().map(|&i| prompts[i].as_str()).collect();
            let vectors = self
                .engine
                .as_mut()
                .context("embedding engine not initialized")?
                .embed_many(&texts)?;
            for (i, vector) in missing.into_iter().zip(vectors) {
                self.cache.put(&keys[i], &vector)?;
                output[i] = Some(vector);
                stats.embedded += 1;
            }
        }
        self.progress.update(stats);
        output
            .into_iter()
            .map(|vector| vector.context("model returned too few embeddings"))
            .collect()
    }

    pub(crate) fn embed_uncached(&mut self, text: &str, stats: &mut Stats) -> Result<Vec<f32>> {
        self.ensure_engine()?;
        let vector = self
            .engine
            .as_mut()
            .context("embedding engine not initialized")?
            .embed(text)?;
        stats.embedded += 1;
        self.progress.update(stats);
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
        if !self.args.exhaustive
            && !self.args.count
            && !self.args.files_with_matches
            && !self.args.files_without_match
            && !self.args.invert_match
            && self.args.max_count.is_none()
        {
            return self.search_candidates(query);
        }
        self.progress
            .begin("reading files and matching semantic chunks");
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
                for entry in input::walker(
                    &path,
                    self.args.no_ignore,
                    self.args.hidden,
                    self.args.dereference_recursive,
                )
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
        self.progress.finish(&stats);
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
        self.note_file(stats);
        self.read_file(path, |this, reader, label| {
            this.scan(reader, label, query, hits, counts, stats)
        })
    }

    pub(crate) fn note_file(&mut self, stats: &mut Stats) {
        stats.files += 1;
        if stats.files >= INDEX_HINT_FILES
            && !self.hint_emitted
            && !self.args.no_index_hint
            && !self.args.quiet
            && !self.args.no_messages
        {
            eprintln!(
                "geg: searched {INDEX_HINT_FILES} files; for repeated searches build an explicit index: geg index <root>, then geg query <query> --index <printed-path> (已扫描较多文件，建议显式建立索引加速后续搜索)"
            );
            self.hint_emitted = true;
        }
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
            if !input::searchable(text) || counts[path] >= self.args.max_count.unwrap_or(u64::MAX) {
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
