//! fzf-style picker: asynchronous fuzzy snapshots, latest-query-wins semantic work.
use crate::{
    cache::Cache,
    cli::Args,
    input,
    search::{Hit, Searcher},
};
use anyhow::{Result, ensure};
use crossterm::{
    cursor::{Hide, MoveTo, Show},
    event::{
        self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEventKind,
        KeyModifiers,
    },
    execute, queue,
    style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor},
    terminal::{
        self, BeginSynchronizedUpdate, Clear, ClearType, EndSynchronizedUpdate,
        EnterAlternateScreen, LeaveAlternateScreen,
    },
};
use nucleo::{
    Config, Matcher, Nucleo, Utf32Str,
    pattern::{CaseMatching, Normalization},
};
use std::{
    collections::BTreeSet,
    io::{self, IsTerminal, Read, Write},
    path::PathBuf,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
use unicode_width::UnicodeWidthChar;

const DEBOUNCE: Duration = Duration::from_millis(200);
const MAX_ITEMS: u32 = 100_000;

// Like fzf's EventBox: overwrite pending work rather than queue every keystroke.
struct Mailbox<T> {
    pending: Mutex<Option<T>>,
    ready: Condvar,
    closed: AtomicBool,
}
impl<T> Mailbox<T> {
    fn new() -> Self {
        Self {
            pending: Mutex::new(None),
            ready: Condvar::new(),
            closed: AtomicBool::new(false),
        }
    }
    fn put(&self, value: T) {
        *self.pending.lock().expect("query mailbox poisoned") = Some(value);
        self.ready.notify_one();
    }
    fn take(&self) -> Option<T> {
        let mut pending = self.pending.lock().expect("query mailbox poisoned");
        while pending.is_none() && !self.closed.load(Ordering::Relaxed) {
            pending = self.ready.wait(pending).expect("query mailbox poisoned");
        }
        if self.closed.load(Ordering::Relaxed) {
            None
        } else {
            pending.take()
        }
    }
    fn close(&self) {
        let _lock = self.pending.lock().expect("query mailbox poisoned");
        self.closed.store(true, Ordering::Relaxed);
        self.ready.notify_all();
    }
}
struct Terminal;
impl Terminal {
    fn enter() -> Result<Self> {
        terminal::enable_raw_mode()?;
        // RAII also restores raw mode if entering the alternate screen fails.
        let guard = Self;
        execute!(
            io::stderr(),
            EnterAlternateScreen,
            EnableBracketedPaste,
            Hide
        )?;
        Ok(guard)
    }
}
impl Drop for Terminal {
    fn drop(&mut self) {
        let _ = execute!(
            io::stderr(),
            EndSynchronizedUpdate,
            ResetColor,
            SetAttribute(Attribute::Reset),
            Show,
            DisableBracketedPaste,
            LeaveAlternateScreen
        );
        let _ = terminal::disable_raw_mode();
    }
}
struct Shutdown {
    mailbox: Arc<Mailbox<(u64, String)>>,
    revision: Arc<AtomicU64>,
    workers: Vec<thread::JoinHandle<()>>,
}
impl Drop for Shutdown {
    fn drop(&mut self) {
        self.revision.fetch_add(1, Ordering::Relaxed);
        self.mailbox.close();
        // ggml/Vulkan context destruction must finish before process-global
        // backend destructors run. Never detach a live inference thread.
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

enum Message {
    InventoryDone(bool),
    Result(u64, std::result::Result<Vec<Hit>, String>),
}

pub fn run(args: &Args, model: PathBuf, cache_path: PathBuf) -> Result<Option<Hit>> {
    ensure!(
        io::stdin().is_terminal() && io::stderr().is_terminal(),
        "the finder needs an interactive terminal; use --interactive for stdin JSON queries"
    );
    ensure!(
        args.paths
            .iter()
            .all(|path| path != std::path::Path::new("-")),
        "--tui requires files/directories, not stdin"
    );
    let mut fuzzy = Nucleo::<Hit>::new(Config::DEFAULT.match_paths(), Arc::new(|| {}), Some(2), 1);
    let injector = fuzzy.injector();
    let (sender, receiver) = mpsc::channel();
    let mailbox = Arc::new(Mailbox::<(u64, String)>::new());
    let revision = Arc::new(AtomicU64::new(1));
    let mut shutdown = Shutdown {
        mailbox: mailbox.clone(),
        revision: revision.clone(),
        workers: Vec::new(),
    };
    // Declared last so terminal state is restored before waiting on inference.
    let _terminal = Terminal::enter()?;
    let inventory_args = args.clone();
    let inventory_sender = sender.clone();
    let inventory_mailbox = mailbox.clone();
    shutdown.workers.push(thread::spawn(move || {
        let mut seen = BTreeSet::new();
        'roots: for path in &inventory_args.paths {
            for entry in input::walker(
                path,
                inventory_args.no_ignore,
                inventory_args.hidden,
                inventory_args.dereference_recursive,
            )
            .build()
            {
                if inventory_mailbox.closed.load(Ordering::Relaxed) {
                    return;
                }
                let Ok(entry) = entry else { continue };
                if !entry.file_type().is_some_and(|kind| kind.is_file()) {
                    continue;
                }
                let path = entry.path().to_string_lossy().into_owned();
                if !seen.insert(path.clone()) {
                    continue;
                }
                injector.push(
                    Hit {
                        path,
                        line: 1,
                        score: 0.0,
                        text: String::new(),
                    },
                    |item, columns| columns[0] = item.path.as_str().into(),
                );
                if injector.injected_items() >= MAX_ITEMS {
                    break 'roots;
                }
            }
        }
        let _ = inventory_sender.send(Message::InventoryDone(
            injector.injected_items() >= MAX_ITEMS,
        ));
    }));
    let mut worker_args = args.clone();
    worker_args.no_progress = true;
    worker_args.no_messages = true;
    worker_args.no_index_hint = true;
    let worker_mailbox = mailbox.clone();
    let worker_revision = revision.clone();
    shutdown.workers.push(thread::spawn(move || {
        // Do not load weights for file browsing. Model/context survive all queries.
        let mut searcher = None;
        while let Some((id, query)) = worker_mailbox.take() {
            if worker_revision.load(Ordering::Relaxed) != id {
                continue;
            }
            if searcher.is_none() {
                let initialization = Cache::open(if worker_args.no_cache {
                    None
                } else {
                    Some(&cache_path)
                })
                .and_then(|cache| Searcher::new(&worker_args, model.clone(), cache));
                match initialization {
                    Ok(value) => searcher = Some(value),
                    Err(error) => {
                        let _ = sender.send(Message::Result(id, Err(format!("{error:#}"))));
                        continue;
                    }
                }
            }
            if let Some(searcher) = &mut searcher {
                searcher.set_revision(worker_revision.clone(), id);
                let result = searcher
                    .search(&query)
                    .map(|response| response.hits)
                    .map_err(|error| format!("{error:#}"));
                if worker_revision.load(Ordering::Relaxed) == id {
                    let _ = sender.send(Message::Result(id, result));
                }
            }
        }
    }));
    let mut edit = Edit::new(args.query.clone().unwrap_or_default());
    fuzzy.pattern.reparse(
        0,
        &edit.text,
        CaseMatching::Smart,
        Normalization::Smart,
        false,
    );
    let mut id = 1;
    let mut pending = (!edit.text.is_empty()).then(Instant::now);
    let mut semantic: Option<Vec<Hit>> = None;
    let mut selected = 0;
    let mut preview = false;
    let mut preview_cache = (String::new(), 0, String::new());
    let mut inventory_done = false;
    let mut status =
        String::from("scanning files · type to search · Tab preview · Enter select · Esc exit");
    let mut matcher = Matcher::new(Config::DEFAULT.match_paths());
    let mut dirty = true;
    let mut rendered = Instant::now() - Duration::from_secs(1);
    loop {
        let tick = fuzzy.tick(5);
        dirty |= tick.changed;
        while let Ok(message) = receiver.try_recv() {
            match message {
                Message::InventoryDone(capped) => {
                    inventory_done = true;
                    if edit.text.is_empty() {
                        status = if capped {
                            "file list capped at 100000; narrow the input paths"
                        } else {
                            "fuzzy file picker · Tab preview · Enter select · Esc exit"
                        }
                        .into();
                    }
                }
                Message::Result(generation, result) if generation == id => match result {
                    Ok(hits) => {
                        status = format!(
                            "{} semantic results · Tab preview · Enter select · Esc exit",
                            hits.len()
                        );
                        semantic = Some(hits);
                        selected = 0;
                    }
                    Err(error) => {
                        status = error;
                    }
                },
                _ => {}
            }
            dirty = true;
        }
        if pending.is_some_and(|time| time.elapsed() >= DEBOUNCE) {
            mailbox.put((id, edit.text.clone()));
            pending = None;
            status = "semantic search: scanning candidates / loading local model / GPU inference…"
                .into();
            dirty = true;
        }
        let count = semantic
            .as_ref()
            .map_or(fuzzy.snapshot().matched_item_count() as usize, Vec::len);
        selected = selected.min(count.saturating_sub(1));
        let (_, height) = terminal::size()?;
        let page = usize::from(height.saturating_sub(4).max(1));
        let start = selected / page * page;
        let items: Vec<Hit> = if let Some(hits) = &semantic {
            hits.iter().skip(start).take(page).cloned().collect()
        } else {
            fuzzy
                .snapshot()
                .matched_items(start as u32..(start + page).min(count) as u32)
                .map(|item| item.data.clone())
                .collect()
        };
        if dirty && rendered.elapsed() >= Duration::from_millis(33) {
            if preview
                && let Some(hit) = items.get(selected.saturating_sub(start))
                && (preview_cache.0 != hit.path || preview_cache.1 != hit.line)
            {
                preview_cache = (
                    hit.path.clone(),
                    hit.line,
                    if hit.text.is_empty() {
                        read_preview(&hit.path)
                    } else {
                        hit.text.clone()
                    },
                );
            }
            draw(
                &edit,
                &items,
                selected.saturating_sub(start),
                count,
                fuzzy.snapshot().item_count() as usize,
                &status,
                &preview_cache,
                preview,
                semantic.is_some(),
                &mut matcher,
            )?;
            dirty =
                !inventory_done || tick.running || (semantic.is_none() && !edit.text.is_empty());
            rendered = Instant::now();
        }
        if !event::poll(Duration::from_millis(16))? {
            continue;
        }
        let mut changed = false;
        match event::read()? {
            Event::Resize(..) => dirty = true,
            Event::Paste(text) => changed = edit.insert(&text),
            Event::Key(key) if key.kind != KeyEventKind::Release => {
                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                match key.code {
                    KeyCode::Esc => return Ok(None),
                    KeyCode::Char('c') if ctrl => return Ok(None),
                    KeyCode::Enter => {
                        if let Some(mut hit) = items.get(selected.saturating_sub(start)).cloned() {
                            if hit.text.is_empty() {
                                hit.text = read_preview(&hit.path);
                            }
                            return Ok(Some(hit));
                        }
                    }
                    KeyCode::Tab | KeyCode::BackTab => preview = !preview,
                    KeyCode::Up | KeyCode::Char('p') if key.code == KeyCode::Up || ctrl => {
                        selected = (selected + 1).min(count.saturating_sub(1))
                    }
                    KeyCode::Down | KeyCode::Char('n') if key.code == KeyCode::Down || ctrl => {
                        selected = selected.saturating_sub(1)
                    }
                    KeyCode::PageUp => selected = (selected + page).min(count.saturating_sub(1)),
                    KeyCode::PageDown => selected = selected.saturating_sub(page),
                    KeyCode::Left => edit.left(),
                    KeyCode::Right => edit.right(),
                    KeyCode::Home => edit.cursor = 0,
                    KeyCode::End => edit.cursor = edit.text.len(),
                    KeyCode::Char('a') if ctrl => edit.cursor = 0,
                    KeyCode::Char('e') if ctrl => edit.cursor = edit.text.len(),
                    KeyCode::Char('u') if ctrl => {
                        changed = edit.cursor > 0;
                        edit.text.drain(..edit.cursor);
                        edit.cursor = 0;
                    }
                    KeyCode::Backspace => changed = edit.backspace(),
                    KeyCode::Delete => {
                        let previous = edit.cursor;
                        edit.right();
                        if edit.cursor != previous {
                            edit.text.drain(previous..edit.cursor);
                            edit.cursor = previous;
                            changed = true;
                        }
                    }
                    KeyCode::Char(c) if !ctrl && !key.modifiers.contains(KeyModifiers::ALT) => {
                        changed = edit.insert(&c.to_string())
                    }
                    _ => {}
                }
                dirty = true;
            }
            _ => {}
        }
        if changed {
            id = revision.fetch_add(1, Ordering::Relaxed) + 1;
            selected = 0;
            semantic = None;
            fuzzy.pattern.reparse(
                0,
                &edit.text,
                CaseMatching::Smart,
                Normalization::Smart,
                false,
            );
            pending = (!edit.text.trim().is_empty()).then(Instant::now);
            status = if pending.is_some() {
                "fuzzy results now · semantic reranking after typing pauses"
            } else {
                "fuzzy file picker · Tab preview · Enter select · Esc exit"
            }
            .into();
            dirty = true;
        }
    }
}

fn read_preview(path: &str) -> String {
    let mut bytes = Vec::new();
    match std::fs::File::open(path).and_then(|file| file.take(65536).read_to_end(&mut bytes)) {
        Ok(_) if !bytes.contains(&0) => String::from_utf8_lossy(&bytes)
            .lines()
            .take(100)
            .collect::<Vec<_>>()
            .join("\n"),
        _ => "(binary or unreadable file)".into(),
    }
}
fn clean(text: &str) -> String {
    text.chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect()
}
fn clipped(text: &str, width: usize) -> String {
    let mut used = 0;
    text.chars()
        .take_while(|&c| {
            used += c.width().unwrap_or(0);
            used <= width
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn draw(
    edit: &Edit,
    items: &[Hit],
    selected: usize,
    count: usize,
    total: usize,
    status: &str,
    preview: &(String, u64, String),
    show_preview: bool,
    semantic: bool,
    matcher: &mut Matcher,
) -> Result<()> {
    let (width, height) = terminal::size()?;
    if width < 8 || height < 5 {
        return Ok(());
    }
    let mut out = io::stderr().lock();
    queue!(
        out,
        BeginSynchronizedUpdate,
        Hide,
        MoveTo(0, 0),
        Clear(ClearType::All)
    )?;
    let list_width = if show_preview && width >= 80 {
        width / 2
    } else {
        width
    };
    let bottom = height - 4;
    let pattern =
        nucleo::pattern::Pattern::parse(&edit.text, CaseMatching::Smart, Normalization::Smart);
    let mut scratch = Vec::new();
    let mut indices = Vec::new();
    for (index, item) in items.iter().enumerate().take(usize::from(bottom + 1)) {
        let y = bottom - index as u16;
        let label = clean(&format!(
            "{}:{}{}",
            item.path,
            item.line,
            if semantic {
                format!("  {:.3}", item.score)
            } else {
                String::new()
            }
        ));
        let label = clipped(&label, usize::from(list_width.saturating_sub(3)));
        indices.clear();
        let _ = pattern.indices(Utf32Str::new(&label, &mut scratch), matcher, &mut indices);
        indices.sort_unstable();
        indices.dedup();
        queue!(
            out,
            MoveTo(0, y),
            SetForegroundColor(if index == selected {
                Color::Magenta
            } else {
                Color::DarkGrey
            }),
            Print(if index == selected { "> " } else { "  " })
        )?;
        for (offset, c) in label.chars().enumerate() {
            queue!(
                out,
                SetForegroundColor(if indices.binary_search(&(offset as u32)).is_ok() {
                    Color::Cyan
                } else if index == selected {
                    Color::White
                } else {
                    Color::Grey
                }),
                Print(c)
            )?;
        }
    }
    if list_width < width {
        let x = list_width + 1;
        queue!(
            out,
            MoveTo(x, 0),
            SetForegroundColor(Color::Cyan),
            Print(clipped(
                &clean(&preview.0),
                usize::from(width.saturating_sub(x))
            ))
        )?;
        for (offset, line) in preview.2.lines().take(usize::from(height - 5)).enumerate() {
            let label = format!("{:>4} {}", preview.1 + offset as u64, clean(line));
            queue!(
                out,
                MoveTo(x, offset as u16 + 1),
                SetForegroundColor(Color::Grey),
                Print(clipped(&label, usize::from(width.saturating_sub(x))))
            )?;
        }
    }
    queue!(
        out,
        MoveTo(0, height - 3),
        SetForegroundColor(Color::DarkGrey),
        Print("─".repeat(usize::from(width))),
        MoveTo(0, height - 2),
        SetForegroundColor(Color::Yellow),
        Print(clipped(
            &format!("{count}/{total}  {}", clean(status)),
            usize::from(width)
        ))
    )?;
    // Scroll the input horizontally to keep a Unicode cursor visible.
    let room = usize::from(width - 3);
    let mut start = edit.cursor;
    let mut used = 0;
    for (index, c) in edit.text[..edit.cursor].char_indices().rev() {
        let next = used + c.width().unwrap_or(0);
        if next > room {
            break;
        }
        used = next;
        start = index;
    }
    let visible = clipped(&edit.text[start..], usize::from(width - 2));
    queue!(
        out,
        MoveTo(0, height - 1),
        SetForegroundColor(Color::Cyan),
        Print("> "),
        SetForegroundColor(Color::White),
        Print(visible),
        MoveTo((2 + used) as u16, height - 1),
        Show,
        ResetColor,
        EndSynchronizedUpdate
    )?;
    out.flush()?;
    Ok(())
}

struct Edit {
    text: String,
    cursor: usize,
}
impl Edit {
    fn new(text: String) -> Self {
        let text: String = text.chars().filter(|c| !c.is_control()).collect();
        Self {
            cursor: text.len(),
            text,
        }
    }
    fn left(&mut self) {
        if let Some((index, _)) = self.text[..self.cursor].char_indices().next_back() {
            self.cursor = index;
        }
    }
    fn right(&mut self) {
        if let Some(c) = self.text[self.cursor..].chars().next() {
            self.cursor += c.len_utf8();
        }
    }
    fn insert(&mut self, text: &str) -> bool {
        let remaining = 1024_usize.saturating_sub(self.text.len());
        let mut inserted = String::new();
        for c in text.chars().filter(|c| !c.is_control()) {
            if inserted.len() + c.len_utf8() > remaining {
                break;
            }
            inserted.push(c);
        }
        if inserted.is_empty() {
            return false;
        }
        self.text.insert_str(self.cursor, &inserted);
        self.cursor += inserted.len();
        true
    }
    fn backspace(&mut self) -> bool {
        let end = self.cursor;
        self.left();
        if self.cursor == end {
            false
        } else {
            self.text.drain(self.cursor..end);
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shutdown_wakes_and_joins_workers() {
        let mailbox = Arc::new(Mailbox::<(u64, String)>::new());
        let revision = Arc::new(AtomicU64::new(1));
        let finished = Arc::new(AtomicBool::new(false));
        let m = mailbox.clone();
        let done = finished.clone();
        let worker = thread::spawn(move || {
            assert!(m.take().is_none());
            done.store(true, Ordering::Relaxed);
        });
        drop(Shutdown {
            mailbox,
            revision: revision.clone(),
            workers: vec![worker],
        });
        assert!(finished.load(Ordering::Relaxed));
        assert_eq!(revision.load(Ordering::Relaxed), 2);
    }
    #[test]
    fn latest_query_replaces_pending_work() {
        let mailbox = Mailbox::new();
        mailbox.put("old");
        mailbox.put("new");
        assert_eq!(mailbox.take(), Some("new"));
        mailbox.close();
        assert_eq!(mailbox.take(), None);
    }
    #[test]
    fn utf8_editing_and_paste_are_bounded() {
        let mut edit = Edit::new("中文".into());
        edit.left();
        assert!(edit.insert("检索\u{1b}\n"));
        assert_eq!(edit.text, "中检索文");
        assert!(edit.backspace());
        assert_eq!(edit.text, "中检文");
        edit.insert(&"长".repeat(1000));
        assert!(edit.text.len() <= 1024);
    }
    #[test]
    fn terminal_controls_and_wide_characters_are_safe() {
        assert_eq!(clean("a\u{1b}[31m"), "a [31m");
        assert_eq!(clipped("中文abc", 5), "中文a");
    }
}
