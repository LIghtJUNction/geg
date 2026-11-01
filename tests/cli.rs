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
            .env("GEG_MODEL", self.dir.path().join("model.gguf"))
            .env("GEG_CACHE", self.dir.path().join("cache.db"))
            .args(["--dimensions", "128", "--chunk-lines", "1"]);
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
    let out = f.run(&["--json", "login", "src"]);
    assert!(out.status.success(), "{:?}", out);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(err.matches("geg index").count(), 1);
    assert!(!stdout(&out).contains("geg index"));
    assert!(
        f.run(&["--no-index-hint", "login", "src"])
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
                "--json",
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
