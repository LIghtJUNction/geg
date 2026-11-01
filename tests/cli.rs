use std::{
    fs,
    io::Write,
    path::Path,
    process::{Command, Output, Stdio},
};

use rusqlite::{Connection, params};
use tempfile::TempDir;

struct Fixture {
    dir: TempDir,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model.gguf"), b"test-model-not-real").unwrap();
        let db = Connection::open(dir.path().join("cache.db")).unwrap();
        db.execute_batch(
            "CREATE TABLE vectors_v1(key BLOB PRIMARY KEY, vector BLOB NOT NULL) WITHOUT ROWID",
        )
        .unwrap();
        Self { dir }
    }

    fn seed(&self, text: &str, positive: bool) {
        let mut hash = blake3::Hasher::new();
        hash.update(b"geg-embedding-v1-llama4-0.9-mean-512\0");
        hash.update(blake3::hash(b"test-model-not-real").as_bytes());
        hash.update(&128u64.to_le_bytes());
        hash.update(text.as_bytes());
        let key = hash.finalize();
        let mut vector = vec![0.0f32; 128];
        vector[usize::from(!positive)] = 1.0;
        let bytes: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
        Connection::open(self.dir.path().join("cache.db"))
            .unwrap()
            .execute(
                "INSERT OR REPLACE INTO vectors_v1 VALUES (?1, ?2)",
                params![key.as_bytes().as_slice(), bytes],
            )
            .unwrap();
    }

    fn command(&self) -> Command {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_geg"));
        cmd.current_dir(self.dir.path())
            .env("HOME", self.dir.path())
            .env("XDG_CACHE_HOME", self.dir.path().join("user-cache"))
            .env_remove("GEG_DEVICE")
            .env("GEG_MODEL", self.dir.path().join("model.gguf"))
            .env("GEG_CACHE", self.dir.path().join("cache.db"))
            .args([
                "--dimensions",
                "128",
                "--chunk-lines",
                "1",
                "--format",
                "grep",
                "--no-progress",
            ]);
        cmd
    }

    fn run(&self, args: &[&str]) -> Output {
        self.command().args(args).output().unwrap()
    }
    fn write(&self, name: &str, text: &str) {
        fs::write(self.dir.path().join(name), text).unwrap();
    }
}

fn stdout(output: &Output) -> String {
    assert_ne!(
        output.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout.clone()).unwrap()
}

#[test]
fn help_version_and_unsupported_flags() {
    let f = Fixture::new();
    let out = f.run(&["--help"]);
    assert!(out.status.success());
    assert!(stdout(&out).contains("--no-filename"));
    assert!(f.run(&["-V"]).status.success());
    assert_eq!(
        stdout(&f.run(&["skill"])),
        include_str!("../skills/geg/SKILL.md")
    );
    assert!(
        Command::new(env!("CARGO_BIN_EXE_geg"))
            .arg("skill")
            .env("GEG_MODEL", "does-not-exist")
            .output()
            .unwrap()
            .status
            .success()
    );
    assert_eq!(f.run(&["-P", "query"]).status.code(), Some(2));
}

#[test]
fn headless_startup_prints_grep_help_without_a_model() {
    let f = Fixture::new();
    let out = f.run(&[]);
    assert!(out.status.success());
    let text = stdout(&out);
    assert!(text.contains("grep-style syntax") && text.contains("geg --help"));
    assert!(!text.contains("| path |") && !text.contains('\u{1b}'));
    assert!(out.stderr.is_empty());
    assert_eq!(f.run(&["--tui"]).status.code(), Some(2));
    assert!(!stdout(&f.run(&["--help"])).contains("--tui"));
}

#[test]
fn pretty_output_color_policy_and_jsonl_precedence() {
    let f = Fixture::new();
    f.seed("task: search result | query: login", true);
    f.seed("title: none | text: security\n", true);
    fs::write(f.dir.path().join("a.txt"), "security\n").unwrap();
    let plain = f.run(&["--pretty", "login", "a.txt"]);
    let text = stdout(&plain);
    assert!(text.contains("result(s)") && text.contains("a.txt") && text.contains("similarity"));
    assert!(text.contains("│ security") && !text.contains('\u{1b}'));
    let colored = f.run(&["--pretty", "--color", "always", "login", "a.txt"]);
    assert!(stdout(&colored).contains("\u{1b}[1;36m"));
    let never = f.run(&["--format", "pretty", "--color", "never", "login", "a.txt"]);
    assert!(!stdout(&never).contains('\u{1b}'));
    let json = f.run(&["--pretty", "--color", "always", "--jsonl", "login", "a.txt"]);
    let hit: serde_json::Value = serde_json::from_str(&stdout(&json)).unwrap();
    assert_eq!(hit["path"], "a.txt");
}

#[test]
fn parallel_shortlist_is_deterministic_and_respects_ignores() {
    let f = Fixture::new();
    fs::create_dir(f.dir.path().join("src")).unwrap();
    for i in 0..60 {
        f.write(&format!("src/case_{i:03}.txt"), "retry request\n");
    }
    f.write("src/ignored.txt", "retry request\n");
    f.write(".gitignore", "src/ignored.txt\n");
    f.seed("task: search result | query: retry", true);
    f.seed("title: none | text: retry request\n", true);
    let expected: Vec<_> = (0..7).map(|i| format!("src/case_{i:03}.txt")).collect();
    for _ in 0..3 {
        let out = stdout(&f.run(&["--candidates", "7", "--top", "7", "--jsonl", "retry", "src"]));
        let paths: Vec<String> = out
            .lines()
            .map(|line| {
                serde_json::from_str::<serde_json::Value>(line).unwrap()["path"]
                    .as_str()
                    .unwrap()
                    .to_owned()
            })
            .collect();
        assert_eq!(paths, expected);
    }
}

#[test]
fn grep_outputs_patterns_and_exit_codes() {
    let f = Fixture::new();
    f.write("a.txt", "security\nflowers\n");
    f.seed("task: search result | query: login", true);
    f.seed("task: search result | query: auth", true);
    f.seed("title: none | text: security\n", true);
    f.seed("title: none | text: flowers\n", false);
    assert_eq!(
        stdout(&f.run(&["-nH", "login", "a.txt"])),
        "a.txt:1:security\n"
    );
    assert_eq!(stdout(&f.run(&["-nh", "login", "a.txt"])), "1:security\n");
    assert_eq!(stdout(&f.run(&["-vc", "login", "a.txt"])), "1\n");
    assert_eq!(stdout(&f.run(&["-l", "login", "a.txt"])), "a.txt\n");
    assert_eq!(f.run(&["-L", "login", "a.txt"]).status.code(), Some(1));
    assert_eq!(stdout(&f.run(&["-q", "login", "a.txt"])), "");
    assert_eq!(stdout(&f.run(&["-m", "0", "-c", "login", "a.txt"])), "0\n");
    assert_eq!(
        stdout(&f.run(&["-n", "-e", "login", "-e", "auth", "a.txt"])),
        "1:security\n"
    );
    f.write("queries.txt", "login\nauth\n");
    assert_eq!(
        stdout(&f.run(&["-n", "-f", "queries.txt", "a.txt"])),
        "1:security\n"
    );
    assert_eq!(
        f.run(&["--threshold", "NaN", "login", "a.txt"])
            .status
            .code(),
        Some(2)
    );
    assert_eq!(f.run(&["login", "missing.txt"]).status.code(), Some(2));
    let suppressed = f.run(&["-ns", "login", "missing.txt"]);
    assert_eq!(suppressed.status.code(), Some(2));
    assert!(suppressed.stderr.is_empty());
}

#[test]
fn stdin_and_interactive_protocol() {
    let f = Fixture::new();
    f.seed("task: search result | query: login", true);
    f.seed("title: none | text: security\n", true);
    let mut child = f
        .command()
        .args(["-nH", "login", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"security\n")
        .unwrap();
    assert_eq!(
        stdout(&child.wait_with_output().unwrap()),
        "(standard input):1:security\n"
    );
    f.write("a.txt", "security\n");
    let mut child = f
        .command()
        .args(["--interactive", "a.txt"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"login\nlogin\n")
        .unwrap();
    let out = stdout(&child.wait_with_output().unwrap());
    let lines: Vec<_> = out.lines().collect();
    assert_eq!(lines.len(), 2);
    for line in lines {
        let value: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(value["stats"]["embedded"], 0);
        assert_eq!(value["hits"][0]["text"], "security\n");
    }
}

#[test]
fn explicit_index_is_incremental_atomic_and_independent_of_sources() {
    let f = Fixture::new();
    f.write("a.txt", "security\n");
    f.seed("title: none | text: security\n", true);
    f.seed("task: search result | query: login", true);
    for _ in 0..2 {
        let out = f.run(&["index", "a.txt", "--output", "index.db"]);
        let value: serde_json::Value = serde_json::from_str(&stdout(&out)).unwrap();
        assert_eq!(value["stats"]["embedded"], 0);
        assert_eq!(value["stats"]["cache_hits"], 1);
    }
    // A failed update must leave the previous committed generation intact.
    assert_eq!(
        f.run(&["index", "missing.txt", "--output", "index.db"])
            .status
            .code(),
        Some(2)
    );
    fs::remove_file(f.dir.path().join("a.txt")).unwrap();
    assert_eq!(
        stdout(&f.run(&["query", "login", "--index", "index.db", "-nH"])),
        format!(
            "{}:1:security\n",
            fs::canonicalize(f.dir.path())
                .unwrap()
                .join("a.txt")
                .display()
        )
    );
    assert_eq!(
        f.run(&["query", "login", "--index", "index.db", "-i"])
            .status
            .code(),
        Some(2)
    );
    assert!(!f.dir.path().join(".geg").exists());
}

#[test]
fn many_files_hint_is_stderr_only_once_and_suppressible() {
    let f = Fixture::new();
    fs::create_dir(f.dir.path().join("src")).unwrap();
    for i in 0..105 {
        f.write(&format!("src/{i}.txt"), "security\n");
    }
    f.seed("title: none | text: security\n", true);
    f.seed("task: search result | query: login", true);
    let out = f.run(&["--jsonl", "login", "src"]);
    assert!(out.status.success(), "{:?}", out);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(err.matches("geg index").count(), 1);
    assert!(!stdout(&out).contains("geg index"));
    assert!(
        f.run(&["--exhaustive", "--no-index-hint", "login", "src"])
            .stderr
            .is_empty()
    );
}

#[test]
fn no_match_files_and_full_counts_are_not_top_k_limited() {
    let f = Fixture::new();
    f.write("a.txt", &"security\n".repeat(25));
    f.write("b.txt", "flowers\n");
    f.seed("title: none | text: security\n", true);
    f.seed("title: none | text: flowers\n", false);
    f.seed("task: search result | query: login", true);
    assert_eq!(
        stdout(&f.run(&["-c", "-k", "1", "login", "a.txt", "b.txt"])),
        "a.txt:25\nb.txt:0\n"
    );
    assert_eq!(
        stdout(&f.run(&["-L", "login", "a.txt", "b.txt"])),
        "b.txt\n"
    );
}

#[test]
fn local_default_index_and_gitignore_work_without_a_git_directory() {
    let f = Fixture::new();
    fs::create_dir(f.dir.path().join("src")).unwrap();
    f.write(".gitignore", "src/ignored.txt\n");
    f.write("src/kept.txt", "security\n");
    f.write("src/ignored.txt", "flowers\n");
    f.seed("title: none | text: security\n", true);
    f.seed("title: none | text: flowers\n", false);
    f.seed("task: search result | query: login", true);
    let direct = stdout(&f.run(&["-cH", "login", "src"]));
    assert!(direct.contains("kept.txt:1"), "{direct}");
    assert!(!direct.contains("ignored.txt"), "{direct}");
    let value: serde_json::Value =
        serde_json::from_str(&stdout(&f.run(&["index", "src"]))).unwrap();
    assert_eq!(value["stats"]["files"], 1);
    assert!(f.dir.path().join(".geg/index.sqlite3").is_file());
    let out = stdout(&f.run(&["query", "login", "--jsonl"]));
    assert_eq!(out.lines().count(), 1);
    assert!(out.contains("kept.txt"));
    let direct = stdout(&f.run(&["--no-ignore", "-cH", "login", "src"]));
    assert!(direct.contains("ignored.txt:0"), "{direct}");
}

#[test]
fn markdown_has_one_schema_and_escapes_source_content() {
    let f = Fixture::new();
    f.write("a.txt", "a | <script> `code`\n}\nsecurity\n");
    f.seed("task: search result | query: login", true);
    f.seed("title: none | text: a | <script> `code`\n", true);
    f.seed("title: none | text: security\n", true);
    // The punctuation-only chunk is deliberately not seeded: it must not invoke the model.
    let out = stdout(&f.run(&["--format", "md", "login", "a.txt"]));
    assert_eq!(out.matches("| path | line | score | text |").count(), 1);
    assert_eq!(out.lines().count(), 4);
    assert!(
        out.contains("a &#124; &lt;script&gt; &#96;code&#96;"),
        "{out}"
    );
    let jsonl = stdout(&f.run(&["--jsonl", "login", "a.txt"]));
    assert_eq!(jsonl.lines().count(), 2);
    for line in jsonl.lines() {
        assert!(serde_json::from_str::<serde_json::Value>(line).is_ok());
    }
}

#[test]
fn dotfile_config_cli_env_precedence_and_validation() {
    let f = Fixture::new();
    f.write("a.txt", "security\nflowers\n");
    f.seed("task: search result | query: login", true);
    f.seed("task: code retrieval | query: login", true);
    f.seed("title: none | text: security\n", true);
    f.seed("title: none | text: flowers\n", false);
    f.write(
        ".gegrc.toml",
        "model = 'not-the-model.gguf'\ncode = true\njsonl = true\ntop = 1\nthreshold = 0.8\n",
    );
    let out = stdout(&f.run(&["login", "a.txt"]));
    let hit: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
    assert_eq!(hit["text"], "security\n");
    // CLI overrides TOML, and GEG_MODEL overrides the bogus configured model.
    assert_eq!(
        stdout(&f.run(&["--threshold", "-1", "-k", "2", "login", "a.txt"]))
            .lines()
            .count(),
        2
    );
    assert_eq!(
        stdout(&f.run(&["--no-config", "login", "a.txt"])),
        "security\n"
    );
    f.write(".gegrc.toml", "typo = true\n");
    let out = f.run(&["login", "a.txt"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("unknown field"));
    assert!(f.run(&["--no-config", "login", "a.txt"]).status.success());
    f.write(".gegrc.toml", "chunk_lines = 0\n");
    assert_eq!(f.run(&["login", "a.txt"]).status.code(), Some(2));
}

#[test]
fn configured_index_paths_are_relative_to_config_and_cli_can_override() {
    let f = Fixture::new();
    f.write("a.txt", "security\n");
    f.seed("task: search result | query: login", true);
    f.seed("title: none | text: security\n", true);
    f.write(".gegrc.toml", "index = '.geg/custom.sqlite3'\n");
    assert!(f.run(&["index", "a.txt"]).status.success());
    assert!(f.dir.path().join(".geg/custom.sqlite3").is_file());
    assert!(f.run(&["query", "login"]).status.success());
    assert!(
        f.run(&["index", "a.txt", "--output", "explicit.db"])
            .status
            .success()
    );
    assert!(f.dir.path().join("explicit.db").is_file());
    assert!(
        f.run(&["query", "login", "--index", "explicit.db"])
            .status
            .success()
    );
}

#[test]
fn index_owns_one_vector_copy_and_cache_clean_protects_indexes_and_models() {
    let f = Fixture::new();
    f.write("a.txt", "security\nsecurity\n");
    f.seed("task: search result | query: login", true);
    f.seed("title: none | text: security\n", true);
    assert!(f.run(&["index", "a.txt"]).status.success());
    let global = Connection::open(f.dir.path().join("cache.db")).unwrap();
    assert_eq!(
        global
            .query_row("SELECT count(*) FROM vectors_v1", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        1
    );
    let index = Connection::open(f.dir.path().join(".geg/index.sqlite3")).unwrap();
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM vectors_v1", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM chunks_v2", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        2
    );
    drop(index);
    drop(global);
    let clean: serde_json::Value =
        serde_json::from_str(&stdout(&f.run(&["cache", "clean", "--jsonl"]))).unwrap();
    assert_eq!(clean["removed_vectors"], 1);
    assert!(f.dir.path().join("model.gguf").is_file());
    assert!(f.dir.path().join(".geg/index.sqlite3").is_file());
    assert_eq!(
        f.run(&["--cache", ".geg/index.sqlite3", "cache", "clean"])
            .status
            .code(),
        Some(2)
    );
    f.seed("task: search result | query: login", true);
    assert!(f.run(&["query", "login"]).status.success());
    assert!(f.run(&["index", "a.txt"]).status.success());
}

#[test]
fn hidden_search_still_excludes_git_and_geg() {
    let f = Fixture::new();
    fs::create_dir_all(f.dir.path().join("src/.git")).unwrap();
    fs::create_dir_all(f.dir.path().join("src/.geg")).unwrap();
    f.write("src/a.txt", "security\n");
    f.write("src/.git/secret", "never-embed-this\n");
    f.write("src/.geg/secret", "never-embed-this\n");
    f.seed("task: search result | query: login", true);
    f.seed("title: none | text: security\n", true);
    assert_eq!(
        stdout(&f.run(&["--hidden", "--no-ignore", "-cH", "login", "src"])),
        "src/a.txt:1\n"
    );
    assert!(
        f.run(&["--hidden", "--no-ignore", "index", "src"])
            .status
            .success()
    );
}

#[test]
fn cache_info_needs_no_model_and_size_warning_stays_on_stderr() {
    let f = Fixture::new();
    fs::remove_file(f.dir.path().join("model.gguf")).unwrap();
    let info: serde_json::Value =
        serde_json::from_str(&stdout(&f.run(&["cache", "info", "--jsonl"]))).unwrap();
    let root = Path::new(info["global_directory"].as_str().unwrap());
    fs::create_dir_all(root.join("models")).unwrap();
    // Sparse file tests the threshold without allocating a GiB of actual disk blocks.
    fs::File::create(root.join("models/large.gguf"))
        .unwrap()
        .set_len(1073741825)
        .unwrap();
    let out = f.run(&["--cache-warning-gb", "1", "cache", "info", "--jsonl"]);
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("geg cache clean"));
    assert!(serde_json::from_str::<serde_json::Value>(stdout(&out).trim()).is_ok());
    assert!(
        f.run(&["--cache-warning-gb", "0", "cache", "info"])
            .stderr
            .is_empty()
    );
    assert!(f.run(&["cache", "clean"]).status.success());
    assert!(root.join("models/large.gguf").is_file());
}

#[test]
#[ignore = "requires local EmbeddingGemma 2 GGUF; set GEG_TEST_MODEL"]
fn real_model_semantic_smoke() {
    let model = std::env::var("GEG_TEST_MODEL").expect("set GEG_TEST_MODEL");
    assert!(Path::new(&model).is_file());
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("demo.txt");
    fs::write(
        &file,
        "A cat sleeps on the warm sofa.\nPasswords are hashed with argon2 before storage.\n",
    )
    .unwrap();
    for dimensions in ["128", "256", "512", "768"] {
        let output = Command::new(env!("CARGO_BIN_EXE_geg"))
            .env("GEG_MODEL", &model)
            .env("GEG_CACHE", dir.path().join("cache.db"))
            .args([
                "--jsonl",
                "--chunk-lines",
                "1",
                "--dimensions",
                dimensions,
                "--threshold",
                "-1",
                "password security",
            ])
            .arg(&file)
            .output()
            .unwrap();
        let result = stdout(&output);
        let hit: serde_json::Value = serde_json::from_str(result.lines().next().unwrap()).unwrap();
        assert_eq!(hit["line"], 2, "dimensions={dimensions}: {result}");
    }
}
