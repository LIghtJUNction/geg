//! Human-readable result cards; color is opt-in for redirected output.
use crate::{cli::Args, search::Response};
use anyhow::Result;
use std::io::{self, IsTerminal, Write};

pub fn print(out: &mut impl Write, response: &Response, args: &Args) -> Result<()> {
    let colored = args.color == "always"
        || (args.color == "auto"
            && io::stdout().is_terminal()
            && std::env::var_os("NO_COLOR").is_none());
    let style = |code: &'static str| if colored { code } else { "" };
    let reset = style("\x1b[0m");
    writeln!(
        out,
        "{}{} result(s){} · {}",
        style("\x1b[1m"),
        response.hits.len(),
        reset,
        crate::safe_text(&response.query)
    )?;
    for (rank, hit) in response.hits.iter().enumerate() {
        let lines = hit.text.lines().count().max(1);
        let last = hit.line + lines as u64 - 1;
        writeln!(
            out,
            "\n{}{}. {}{}:{}–{}{}  {}similarity {:.4}{}",
            style("\x1b[1;36m"),
            rank + 1,
            crate::safe_text(&hit.path),
            reset,
            hit.line,
            last,
            reset,
            style("\x1b[33m"),
            hit.score,
            reset
        )?;
        let width = last.to_string().len();
        for (offset, line) in hit.text.lines().enumerate() {
            writeln!(
                out,
                "  {}{:>width$} │{} {}",
                style("\x1b[2m"),
                hit.line + offset as u64,
                reset,
                crate::safe_text(line)
            )?;
            if args.line_buffered {
                out.flush()?;
            }
        }
    }
    Ok(())
}
