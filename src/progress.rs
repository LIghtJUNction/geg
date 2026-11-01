//! Status stays on stderr; never mix it into Markdown/JSONL results.
use crate::search::Stats;
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    io::IsTerminal,
    path::Path,
    time::{Duration, Instant},
};

pub struct Progress {
    bar: Option<ProgressBar>,
    enabled: bool,
    inferred: bool,
    last_report: Instant,
}

impl Progress {
    pub fn new(enabled: bool) -> Self {
        let bar = if enabled && std::io::stderr().is_terminal() {
            let bar = ProgressBar::new_spinner();
            bar.set_style(
                ProgressStyle::with_template("{spinner} geg [{elapsed_precise}] {msg}")
                    .expect("static progress template"),
            );
            bar.enable_steady_tick(Duration::from_millis(100));
            Some(bar)
        } else {
            None
        };
        Self {
            bar,
            enabled,
            inferred: false,
            last_report: Instant::now(),
        }
    }
    pub fn begin(&mut self, stage: &str) {
        if let Some(bar) = &self.bar {
            bar.reset();
            bar.set_message(stage.to_owned());
            bar.enable_steady_tick(Duration::from_millis(100));
        }
    }
    pub fn loading(&mut self, path: &Path) {
        self.inferred = true;
        let message = format!("loading local model {} (no download)", path.display());
        if let Some(bar) = &self.bar {
            bar.set_message(message);
        } else if self.enabled {
            eprintln!("geg: {message}");
        }
        self.last_report = Instant::now();
    }
    pub fn device(&self, device: &str) {
        let message = format!("using {device} inference");
        if let Some(bar) = &self.bar {
            bar.set_message(message);
        } else if self.enabled {
            eprintln!("geg: {message}");
        }
    }
    pub fn update(&mut self, stats: &Stats) {
        if self.bar.is_none()
            && (!self.enabled
                || !self.inferred
                || self.last_report.elapsed() < Duration::from_secs(2))
        {
            return;
        }
        let message = format!(
            "searching · {} files · {} chunks · {} embedded · {} cache hits",
            stats.files, stats.chunks, stats.embedded, stats.cache_hits
        );
        if let Some(bar) = &self.bar {
            bar.set_message(message);
        } else if self.enabled
            && self.inferred
            && self.last_report.elapsed() >= Duration::from_secs(2)
        {
            eprintln!("geg: {message}");
            self.last_report = Instant::now();
        }
    }
    pub fn finish(&mut self, stats: &Stats) {
        if let Some(bar) = &self.bar {
            bar.finish_and_clear();
        } else if self.enabled && self.inferred {
            eprintln!(
                "geg: done · {} files · {} chunks · {} embedded · {} cache hits",
                stats.files, stats.chunks, stats.embedded, stats.cache_hits
            );
        }
        self.inferred = false;
    }
}
impl Drop for Progress {
    fn drop(&mut self) {
        if let Some(bar) = &self.bar {
            bar.finish_and_clear();
        }
    }
}
