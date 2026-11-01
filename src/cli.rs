use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Print the bundled agent skill (no model required)
    Skill,
    /// Build/update a persistent repository index (no .geg directory needed)
    Index {
        #[arg(required = true)]
        paths: Vec<PathBuf>,
        /// Index database path (defaults to the user cache directory)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Search an existing index without reading source files
    Query {
        query: String,
        #[arg(long)]
        index: PathBuf,
    },
}

/// Geg Embeds Gemma — local semantic grep. No network requests, ever.
#[derive(Debug, Parser)]
#[command(
    version,
    about,
    disable_help_flag = true,
    subcommand_negates_reqs = true,
    subcommand_precedence_over_arg = true,
    after_help = "Examples:\n  geg 'release a resource' src/\n  cat notes.txt | geg 'travel plans'\n  printf 'authentication\\nretry logic\\n' | geg --interactive src/\n\nExit codes: 0 = matches, 1 = no matches, 2 = error.\nModel: text-only EmbeddingGemma 2 GGUF; set GEG_MODEL or --model."
)]
pub struct Args {
    #[command(subcommand)]
    pub command: Option<Command>,
    #[arg(long, action = clap::ArgAction::Help)]
    help: Option<bool>,
    /// Suppress normal output (exit status still reports matches)
    #[arg(short = 'q', long, alias = "silent", global = true)]
    pub quiet: bool,
    /// Suppress the large-repository indexing suggestion
    #[arg(long, global = true)]
    pub no_index_hint: bool,
    /// Suppress error messages (errors still exit with status 2)
    #[arg(short = 's', long = "no-messages", global = true)]
    pub no_messages: bool,
    /// Supply a semantic query; repeat for OR matching
    #[arg(short = 'e', long = "regexp", action = clap::ArgAction::Append)]
    pub patterns: Vec<String>,
    /// Read semantic queries from a UTF-8 file, one per line
    #[arg(short = 'f', long = "file", action = clap::ArgAction::Append)]
    pub pattern_files: Vec<PathBuf>,
    /// Recursively search directories (defaults to '.' if no files)
    #[arg(short = 'r', long = "recursive")]
    pub recursive: bool,
    /// Recurse and follow symbolic links
    #[arg(short = 'R', long = "dereference-recursive")]
    pub dereference_recursive: bool,
    /// Lowercase queries and documents before semantic embedding
    #[arg(short = 'i', long = "ignore-case", global = true)]
    pub ignore_case: bool,
    /// Select chunks below the semantic threshold instead
    #[arg(short = 'v', long = "invert-match", global = true)]
    pub invert_match: bool,
    /// Prefix selected lines with their line number
    #[arg(short = 'n', long = "line-number", global = true)]
    pub line_number: bool,
    #[arg(
        short = 'H',
        long = "with-filename",
        overrides_with = "no_filename",
        global = true
    )]
    pub with_filename: bool,
    #[arg(
        short = 'h',
        long = "no-filename",
        overrides_with = "with_filename",
        global = true
    )]
    pub no_filename: bool,
    /// Print selected line counts per file (before top-k truncation)
    #[arg(short = 'c', long, global = true)]
    pub count: bool,
    #[arg(
        short = 'l',
        long = "files-with-matches",
        overrides_with = "files_without_match",
        global = true
    )]
    pub files_with_matches: bool,
    #[arg(
        short = 'L',
        long = "files-without-match",
        overrides_with = "files_with_matches",
        global = true
    )]
    pub files_without_match: bool,
    /// Maximum selected lines per file
    #[arg(short = 'm', long = "max-count", global = true)]
    pub max_count: Option<u64>,
    /// Prefix human-readable matches with cosine similarity
    #[arg(long, global = true)]
    pub score: bool,
    /// Filename label for stdin
    #[arg(long, default_value = "(standard input)")]
    pub label: String,
    /// Flush output after every selected line
    #[arg(long, global = true)]
    pub line_buffered: bool,
    /// Semantic query (omit in --interactive mode)
    #[arg(required_unless_present_any = ["interactive", "patterns", "pattern_files"])]
    pub query: Option<String>,
    /// Files/directories; '-' or no paths reads stdin
    pub paths: Vec<PathBuf>,
    /// Local GGUF weights (never downloaded automatically)
    #[arg(long, env = "GEG_MODEL", global = true)]
    pub model: Option<PathBuf>,
    /// SQLite vector cache location
    #[arg(long, env = "GEG_CACHE", global = true)]
    pub cache: Option<PathBuf>,
    /// Disable persistent vector caching
    #[arg(long, global = true)]
    pub no_cache: bool,
    /// Keep model/context resident; read one query per stdin line, emit JSON responses
    #[arg(long)]
    pub interactive: bool,
    /// Emit one JSON object per match (interactive mode always emits JSON responses)
    #[arg(long, global = true)]
    pub json: bool,
    /// Maximum number of ranked matches
    #[arg(short = 'k', long, global = true, default_value_t = 10, value_parser = clap::value_parser!(u32).range(1..=10000))]
    pub top: u32,
    /// Minimum cosine similarity, from -1 to 1
    #[arg(
        long,
        global = true,
        default_value_t = 0.15,
        allow_hyphen_values = true
    )]
    pub threshold: f32,
    /// Lines per chunk; long lines are split further at UTF-8 boundaries
    #[arg(long, global = true, default_value_t = 8, value_parser = clap::value_parser!(u32).range(1..=128))]
    pub chunk_lines: u32,
    /// Matryoshka embedding dimension: 128, 256, 512, 768
    #[arg(long, global = true, default_value = "256", value_parser = ["128", "256", "512", "768"])]
    pub dimensions: String,
    /// Use the model's code retrieval query instruction
    #[arg(long, global = true)]
    pub code: bool,
    /// CPU inference threads (defaults to available parallelism, capped at 8)
    #[arg(short = 't', long, global = true, value_parser = clap::value_parser!(u32).range(1..=256))]
    pub threads: Option<u32>,
}
