use std::{
    io::{BufRead, Read},
    path::{Path, PathBuf},
};

use anyhow::{Result, bail};

/// Bounds text held for one embedding, independently of total input size.
const CHUNK_BYTES: usize = 1024;
const MAX_LINE_BYTES: u64 = 64 * 1024;

pub fn chunks(
    mut reader: impl BufRead,
    lines_per_chunk: u32,
    mut emit: impl FnMut(u64, &str) -> Result<()>,
) -> Result<()> {
    let mut line = Vec::new();
    let mut text = String::with_capacity(CHUNK_BYTES);
    let mut line_number = 0;
    let mut start = 1;
    let mut count = 0;
    loop {
        line.clear();
        let n = reader
            .by_ref()
            .take(MAX_LINE_BYTES + 1)
            .read_until(b'\n', &mut line)?;
        if n == 0 {
            break;
        }
        line_number += 1;
        if n as u64 > MAX_LINE_BYTES {
            bail!("line {line_number} exceeds 64 KiB; pre-split long/minified lines");
        }
        if line.contains(&0) {
            bail!("binary input (NUL byte) at line {line_number}");
        }
        let value = std::str::from_utf8(&line)?;
        if !text.is_empty() && text.len() + value.len() > CHUNK_BYTES {
            emit(start, &text)?;
            text.clear();
            count = 0;
        }
        let mut rest = value;
        while !rest.is_empty() {
            if text.is_empty() {
                start = line_number;
            }
            let mut end = (CHUNK_BYTES - text.len()).min(rest.len());
            while !rest.is_char_boundary(end) {
                end -= 1;
            }
            if end == 0 {
                emit(start, &text)?;
                text.clear();
                count = 0;
                continue;
            }
            text.push_str(&rest[..end]);
            rest = &rest[end..];
            if !rest.is_empty() {
                emit(start, &text)?;
                text.clear();
                count = 0;
            }
        }
        count += 1;
        if count >= lines_per_chunk {
            emit(start, &text)?;
            text.clear();
            count = 0;
        }
    }
    if !text.is_empty() {
        emit(start, &text)?;
    }
    Ok(())
}

pub fn visit_files(
    paths: &[PathBuf],
    no_ignore: bool,
    hidden: bool,
    mut visit: impl FnMut(&Path) -> Result<()>,
) -> Result<()> {
    for path in paths {
        if path.is_dir() {
            for entry in walker(path, no_ignore, hidden, false).build() {
                let entry = entry?;
                if entry.file_type().is_some_and(|kind| kind.is_file()) {
                    visit(entry.path())?;
                }
            }
        } else {
            visit(path)?;
        }
    }
    Ok(())
}

/// Punctuation-only fragments (for example a closing brace) have no semantic content.
pub fn searchable(text: &str) -> bool {
    text.chars().any(char::is_alphanumeric)
}

/// Shared traversal rules for direct search and explicit indexing.
pub fn walker(
    path: &Path,
    no_ignore: bool,
    hidden: bool,
    follow_links: bool,
) -> ignore::WalkBuilder {
    let mut builder = ignore::WalkBuilder::new(path);
    builder
        .require_git(false)
        .follow_links(follow_links)
        .hidden(!hidden)
        .ignore(!no_ignore)
        .git_ignore(!no_ignore)
        .git_global(!no_ignore)
        .git_exclude(!no_ignore)
        .filter_entry(|entry| entry.file_name() != ".geg" && entry.file_name() != ".git");
    builder
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unicode_split_preserves_bytes_and_lines() -> Result<()> {
        let input = format!("alpha\n{}\nend", "你好🦀".repeat(600));
        let mut got = Vec::new();
        chunks(input.as_bytes(), 8, |line, text| {
            assert!(text.len() <= CHUNK_BYTES);
            got.push((line, text.to_owned()));
            Ok(())
        })?;
        assert_eq!(
            got.iter().map(|(_, t)| t.as_str()).collect::<String>(),
            input
        );
        assert_eq!(got[0].0, 1);
        assert_eq!(got[1].0, 2);
        Ok(())
    }

    #[test]
    fn groups_and_final_line() -> Result<()> {
        let mut got = Vec::new();
        chunks(&b"a\nb\nc"[..], 2, |line, text| {
            got.push((line, text.to_owned()));
            Ok(())
        })?;
        assert_eq!(got, [(1, "a\nb\n".into()), (3, "c".into())]);
        Ok(())
    }

    #[test]
    fn rejects_binary_invalid_utf8_and_oversize() {
        for input in [vec![0], vec![255], vec![b'x'; MAX_LINE_BYTES as usize + 1]] {
            assert!(chunks(input.as_slice(), 8, |_, _| Ok(())).is_err());
        }
    }
}
