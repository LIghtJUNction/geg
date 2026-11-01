//! Bounded first-pass retrieval for unindexed repositories. Not exhaustive semantic search.
use crate::{
    input,
    search::{Hit, Response, Searcher, Stats, best_score, insert_hit},
};
use anyhow::{Result, ensure};
use std::{
    cmp::{Ordering, Reverse},
    collections::{BTreeMap, BTreeSet, BinaryHeap},
    fs::File,
    io::{BufRead, BufReader, IsTerminal},
    path::Path,
    sync::Mutex,
};

struct Candidate {
    path: String,
    line: u64,
    text: String,
    lexical: f32,
}
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.lexical
            .total_cmp(&other.lexical)
            .then_with(|| other.path.cmp(&self.path))
            .then_with(|| other.line.cmp(&self.line))
    }
}
struct Selection {
    terms: Vec<String>,
    fuzzy: Vec<nucleo::pattern::Atom>,
    matcher: nucleo::Matcher,
    chunks: BinaryHeap<Reverse<Candidate>>,
    limit: usize,
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32, 0x3400..=0x9fff | 0x20000..=0x2ffff)
}

fn terms(query: &str) -> Vec<String> {
    let query = query.to_lowercase();
    let mut terms = BTreeSet::new();
    for token in query.split(|c: char| !c.is_alphanumeric()) {
        if token.chars().count() >= 2 {
            terms.insert(token.to_owned());
        }
    }
    for token in query.split(|c: char| !c.is_ascii_alphanumeric()) {
        if token.len() >= 2 {
            terms.insert(token.to_owned());
        }
    }
    let chars: Vec<_> = query.chars().collect();
    for pair in chars.windows(2) {
        if pair.iter().all(|&c| is_cjk(c)) {
            terms.insert(pair.iter().collect());
        }
    }
    terms.into_iter().collect()
}

impl Selection {
    fn new(terms: Vec<String>, limit: usize) -> Self {
        let fuzzy = terms
            .iter()
            .filter(|term| term.is_ascii() && term.len() <= 32)
            .map(|term| {
                nucleo::pattern::Atom::new(
                    term,
                    nucleo::pattern::CaseMatching::Ignore,
                    nucleo::pattern::Normalization::Smart,
                    nucleo::pattern::AtomKind::Fuzzy,
                    false,
                )
            })
            .collect();
        Self {
            terms,
            fuzzy,
            matcher: nucleo::Matcher::new(nucleo::Config::DEFAULT.match_paths()),
            chunks: BinaryHeap::with_capacity(limit),
            limit,
        }
    }
    fn offer(&mut self, candidate: Candidate) {
        if self.chunks.len() < self.limit {
            self.chunks.push(Reverse(candidate));
        } else if self.chunks.peek().is_some_and(|worst| candidate > worst.0) {
            *self.chunks.peek_mut().expect("nonempty candidate heap") = Reverse(candidate);
        }
    }
    fn ordered(self) -> Vec<Candidate> {
        let mut chunks: Vec<_> = self.chunks.into_iter().map(|item| item.0).collect();
        chunks.sort_unstable_by(|a, b| b.cmp(a));
        chunks
    }
    fn collect(
        &mut self,
        reader: impl BufRead,
        path: &str,
        chunk_lines: u32,
        stats: &mut Stats,
    ) -> Result<()> {
        let lower_path = path.to_lowercase();
        // fzf-style boundary/camelCase/consecutive/gap scoring is computed once per path,
        // never with a full dynamic-programming matrix for every source-code chunk.
        let mut scratch = Vec::new();
        let path_text = nucleo::Utf32Str::new(path, &mut scratch);
        let fuzzy_bonus: f32 = self
            .fuzzy
            .iter()
            .filter_map(|atom| atom.score(path_text, &mut self.matcher))
            .map(|score| (f32::from(score) / 64.0).min(1.5))
            .sum();
        input::chunks(reader, chunk_lines, |line, text| {
            if !input::searchable(text) {
                return Ok(());
            }
            stats.chunks += 1;
            let lower = text.to_lowercase();
            let lexical: f32 = self
                .terms
                .iter()
                .map(|term| {
                    let weight = (term.chars().count() as f32).sqrt();
                    weight
                        * (lower.matches(term.as_str()).take(3).count() as f32
                            + if lower_path.contains(term) { 2.0 } else { 0.0 })
                })
                .sum::<f32>();
            let lexical = (lexical + fuzzy_bonus) / (1.0 + text.len() as f32 / 2048.0);
            let worth_keeping = self.chunks.len() < self.limit
                || self.chunks.peek().is_some_and(|worst| {
                    lexical.total_cmp(&worst.0.lexical).is_gt()
                        || (lexical == worst.0.lexical
                            && (path, line) < (worst.0.path.as_str(), worst.0.line))
                });
            if worth_keeping {
                self.offer(Candidate {
                    path: path.to_owned(),
                    line,
                    text: text.to_owned(),
                    lexical,
                });
            }
            Ok(())
        })
    }
}

impl Searcher<'_> {
    pub fn search_candidates(&mut self, query: &str) -> Result<Response> {
        self.progress
            .begin("scanning text and selecting candidates (no full-repository embedding)");
        let mut stats = Stats::default();
        let mut selection = Selection::new(terms(query), self.args.candidates as usize);
        let paths = self.args.paths.clone();
        if paths.is_empty() {
            ensure!(
                !std::io::stdin().is_terminal(),
                "provide a file/directory or pipe text to stdin"
            );
            selection.collect(
                std::io::stdin().lock(),
                &self.args.label,
                self.args.chunk_lines,
                &mut stats,
            )?;
        }
        for path in paths {
            if path == Path::new("-") {
                selection.collect(
                    std::io::stdin().lock(),
                    &self.args.label,
                    self.args.chunk_lines,
                    &mut stats,
                )?;
            } else if path.is_dir() {
                self.collect_parallel(&path, &mut selection, &mut stats)?;
            } else {
                self.collect_file(&path, &mut selection, &mut stats, false)?;
            }
        }
        stats.candidate_chunks = selection.chunks.len() as u64;
        if stats.chunks > stats.candidate_chunks && !self.args.quiet && !self.args.no_messages {
            eprintln!(
                "geg: fast search: scanned {} chunks in {} files, semantically reranking {} lexical candidates. This is not exhaustive; use --exhaustive or a full index for broader semantic recall.",
                stats.chunks, stats.files, stats.candidate_chunks
            );
        }
        if stats.skipped_files > 0 && !self.args.no_messages && !self.args.quiet {
            eprintln!(
                "geg: skipped the unsupported remainder of {} files (invalid UTF-8/binary/oversized lines or read errors); explicit file inputs report errors instead",
                stats.skipped_files
            );
        }
        let mut counts = BTreeMap::new();
        let mut hits = Vec::new();
        let chunks = selection.ordered();
        if !chunks.is_empty() {
            let query_vector = self.query_vector(query, &mut stats)?;
            let prompts: Vec<_> = chunks
                .iter()
                .map(|candidate| format!("title: none | text: {}", candidate.text))
                .collect();
            let vectors = self.vectors(&prompts, &mut stats)?;
            for (candidate, vector) in chunks.into_iter().zip(vectors) {
                let score = best_score(&query_vector, &vector);
                let count = counts.entry(candidate.path.clone()).or_insert(0);
                if self.accepts(score) {
                    *count += candidate.text.lines().count() as u64;
                    insert_hit(
                        &mut hits,
                        Hit {
                            path: candidate.path,
                            line: candidate.line,
                            text: candidate.text,
                            score,
                        },
                        self.limit(),
                    );
                }
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

    fn collect_parallel(
        &mut self,
        path: &Path,
        selection: &mut Selection,
        stats: &mut Stats,
    ) -> Result<()> {
        // dust's useful property: thread-local collection, one merge per worker,
        // not one contended lock or another metadata lookup per source chunk.
        struct Local<'a> {
            selection: Selection,
            stats: Stats,
            merged: &'a Mutex<(Selection, Stats, Option<anyhow::Error>)>,
        }
        impl Drop for Local<'_> {
            fn drop(&mut self) {
                let mut merged = self.merged.lock().expect("candidate merge poisoned");
                for candidate in self.selection.chunks.drain() {
                    merged.0.offer(candidate.0);
                }
                merged.1.files += self.stats.files;
                merged.1.chunks += self.stats.chunks;
                merged.1.skipped_files += self.stats.skipped_files;
            }
        }
        let merged = Mutex::new((
            Selection::new(selection.terms.clone(), selection.limit),
            Stats::default(),
            None,
        ));
        let chunk_lines = self.args.chunk_lines;
        let cancellation = self.revision.clone();
        let terms = selection.terms.clone();
        let limit = selection.limit;
        let mut walker = input::walker(
            path,
            self.args.no_ignore,
            self.args.hidden,
            self.args.dereference_recursive,
        );
        walker.threads(std::thread::available_parallelism().map_or(1, |n| n.get().min(4)));
        walker.build_parallel().run(|| {
            let mut local = Local {
                selection: Selection::new(terms.clone(), limit),
                stats: Stats::default(),
                merged: &merged,
            };
            let cancellation = cancellation.clone();
            Box::new(move |entry| {
                if cancellation.as_ref().is_some_and(|(revision, expected)| {
                    revision.load(std::sync::atomic::Ordering::Relaxed) != *expected
                }) {
                    return ignore::WalkState::Quit;
                }
                let entry = match entry {
                    Ok(entry) => entry,
                    Err(error) => {
                        local.merged.lock().expect("candidate merge poisoned").2 =
                            Some(error.into());
                        return ignore::WalkState::Quit;
                    }
                };
                if !entry.file_type().is_some_and(|kind| kind.is_file()) {
                    return ignore::WalkState::Continue;
                }
                local.stats.files += 1;
                let mut read = || -> Result<()> {
                    let mut reader = BufReader::new(File::open(entry.path())?);
                    if reader.fill_buf()?.contains(&0) {
                        return Ok(());
                    }
                    local.selection.collect(
                        reader,
                        &entry.path().to_string_lossy(),
                        chunk_lines,
                        &mut local.stats,
                    )
                };
                if (read)().is_err() {
                    local.stats.skipped_files += 1;
                }
                ignore::WalkState::Continue
            })
        });
        self.check_cancelled()?;
        let (collected, local_stats, error) =
            merged.into_inner().expect("candidate merge poisoned");
        if let Some(error) = error {
            return Err(error);
        }
        for candidate in collected.chunks {
            selection.offer(candidate.0);
        }
        stats.chunks += local_stats.chunks;
        stats.skipped_files += local_stats.skipped_files;
        for _ in 0..local_stats.files {
            self.note_file(stats);
        }
        self.progress.update(stats);
        Ok(())
    }

    fn collect_file(
        &mut self,
        path: &Path,
        selection: &mut Selection,
        stats: &mut Stats,
        allow_skip: bool,
    ) -> Result<()> {
        self.check_cancelled()?;
        self.note_file(stats);
        let lines = self.args.chunk_lines;
        let result = self.read_file(path, |_, reader, label| {
            selection.collect(reader, label, lines, stats)
        });
        if allow_skip && result.is_err() {
            stats.skipped_files += 1;
            Ok(())
        } else {
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_heap_matches_full_sort_with_stable_ties() {
        let mut selection = Selection::new(vec![], 7);
        let make = |i: u64| Candidate {
            path: format!("p{:03}", 99 - i),
            line: i,
            text: String::new(),
            lexical: (i % 4) as f32,
        };
        for i in 0..100 {
            selection.offer(make(i));
        }
        let mut expected: Vec<_> = (0..100).map(make).collect();
        expected.sort_unstable_by(|a, b| b.cmp(a));
        let actual = selection.ordered();
        assert_eq!(actual.len(), 7);
        assert!(actual.iter().zip(expected).all(|(a, b)| a == &b));
    }
    #[test]
    fn fuzzy_path_score_finds_abbreviated_camel_case() -> Result<()> {
        let mut selection = Selection::new(terms("reqrt"), 1);
        let mut stats = Stats::default();
        for path in ["src/aaa.rs", "src/requestRetry.rs"] {
            selection.collect("same body\n".as_bytes(), path, 1, &mut stats)?;
        }
        assert_eq!(selection.ordered()[0].path, "src/requestRetry.rs");
        Ok(())
    }
    #[test]
    fn chinese_bigrams_and_ascii_identifiers_are_searchable() {
        let tokens = terms("失败请求的重试逻辑 HTTP_retry");
        assert!(tokens.contains(&"重试".to_owned()));
        assert!(tokens.contains(&"retry".to_owned()));
        let mut selection = Selection::new(tokens, 1);
        let mut stats = Stats::default();
        selection
            .collect(
                "unrelated cats\n# 失败请求的重试逻辑\n".as_bytes(),
                "a.py",
                1,
                &mut stats,
            )
            .unwrap();
        assert_eq!(selection.ordered()[0].line, 2);
        assert_eq!(stats.chunks, 2);
    }
}
