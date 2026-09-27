//! chelis#2558: the cache decoders adopt the producer's effect and linearity
//! results instead of rerunning those checkers over a cached library. That is
//! sound because a library that fails either checker is never written, and a
//! source edit moves the cache key. These tests lock both halves from the
//! command line: a linearity violation in a library module the entry reaches
//! fails `chelis test`, `chelis check` and `chelis eval --file` on a cold cache
//! and after a warm one, leaves no cache entry behind, and restoring the source
//! hits the entry written before the edit.

use assert_cmd::Command;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use tempfile::{TempDir, tempdir};

const VALID_OPS: &str = "module Mylib.Ops\nexport (settle)\n\
     def settle(x: tensor[2, f32]) -> tensor[2, f32] = realize(x)\n";

/// `settle` consumes `x` with `realize` and then reads it again.
const INVALID_OPS: &str = "module Mylib.Ops\nexport (settle)\n\
     def settle(x: tensor[2, f32]) -> tensor[2, f32] = {\n  y = realize(x)\n  add(x, y)\n}\n";

const VIOLATION: &str = "already consumed by realize";

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, contents).expect("write file");
}

/// `app` with a path dependency on `mylib`, whose `settle` every entry reaches:
/// the package module, its test file, and the evaluated file all call it.
fn package(ops: &str) -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join("app");
    let version = env!("CARGO_PKG_VERSION");
    write_file(
        &root.join("reef.toml"),
        &format!(
            "[package]\nname = \"app\"\nversion = \"0.1.0\"\ncompiler = \"={version}\"\n\
             module_prefix = \"App\"\n\n[dependencies]\nmylib = {{ path = \"./mylib\" }}\n"
        ),
    );
    write_file(
        &root.join("reef.lock"),
        &format!(
            "[package]\nname = \"app\"\nversion = \"0.1.0\"\n\n[[dependencies]]\n\
             name = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={version}\"\n\
             archive_sha256 = \"\"\nshell_sha256 = \"\"\n\n[dependencies.source]\n\
             kind = \"path\"\npath = \"./mylib\"\n"
        ),
    );
    write_file(
        &root.join("mylib/reef.toml"),
        &format!(
            "[package]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={version}\"\n\
             module_prefix = \"Mylib\"\n"
        ),
    );
    write_file(&root.join("mylib/src/ops.ch"), ops);
    write_file(
        &root.join("src/main.ch"),
        "module App.Main\nimport Mylib.Ops (settle)\n\
         def run() -> tensor[2, f32] = settle(to_tensor([1.0f32, 2.0f32]))\n",
    );
    write_file(
        &root.join("tests/settle.ch"),
        "module App.Tests.Settle\nimport Mylib.Ops (settle)\n\
         def test_settle() -> unit ! { Test } = \
         test_assert_eq_tensor(settle(to_tensor([1.0f32, 2.0f32])), \
         to_tensor([1.0f32, 2.0f32]), \"settle\")\n",
    );
    write_file(
        &root.join("probe.ch"),
        "import Mylib.Ops (settle)\nout = settle(to_tensor([1.0f32, 2.0f32]))\n",
    );
    (dir, root)
}

fn chelis(root: &Path, home: &Path, args: &[&str]) -> (bool, String) {
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(root)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", home.join("reef-home"))
        .env("XDG_CACHE_HOME", home.join("xdg"))
        .args(args)
        .output()
        .expect("run chelis");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    (output.status.success(), text)
}

/// Every checked-library cache entry under the two cache roots -- compiled
/// contexts (`.ctx`) and typecheck sub-contexts (`.tc`) -- with its
/// modification time. The prepared reef-graph cache is not a checked artifact
/// and is left out.
fn cache_files(home: &Path) -> BTreeMap<PathBuf, SystemTime> {
    fn walk(dir: &Path, out: &mut BTreeMap<PathBuf, SystemTime>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path
                .extension()
                .is_some_and(|ext| ext == "ctx" || ext == "tc")
            {
                let modified = entry
                    .metadata()
                    .and_then(|metadata| metadata.modified())
                    .expect("cache file mtime");
                out.insert(path, modified);
            }
        }
    }
    let mut files = BTreeMap::new();
    walk(&home.join("reef-home"), &mut files);
    walk(&home.join("xdg"), &mut files);
    files
}

fn compiled_contexts(files: &BTreeMap<PathBuf, SystemTime>) -> Vec<&PathBuf> {
    files
        .keys()
        .filter(|path| path.extension().is_some_and(|ext| ext == "ctx"))
        .collect()
}

const ENTRIES: [&[&str]; 3] = [
    &["test", "tests/"],
    &["check", "src/main.ch"],
    &["eval", "--file", "probe.ch"],
];

fn assert_every_entry_rejects(root: &Path, home: &Path, when: &str) {
    for args in ENTRIES {
        let (ok, output) = chelis(root, home, args);
        assert!(
            !ok,
            "`chelis {}` must reject the reached violation {when}: {output}",
            args.join(" ")
        );
        assert!(
            output.contains(VIOLATION),
            "`chelis {}` must report the linearity violation {when}: {output}",
            args.join(" ")
        );
    }
}

#[test]
fn a_reached_linearity_violation_fails_on_a_cold_cache_and_writes_no_context() {
    let (dir, root) = package(INVALID_OPS);
    let home = dir.path().join("home");

    assert_every_entry_rejects(&root, &home, "on a cold cache");
    let files = cache_files(&home);
    assert!(
        compiled_contexts(&files).is_empty(),
        "a rejected library must not be written as a compiled context: {files:?}"
    );
}

#[test]
fn a_reached_linearity_violation_fails_after_a_warm_cache_and_the_restored_source_hits() {
    let (dir, root) = package(VALID_OPS);
    let home = dir.path().join("home");
    let ops = root.join("mylib/src/ops.ch");

    for args in ENTRIES {
        let (ok, output) = chelis(&root, &home, args);
        assert!(ok, "`chelis {}` must pass: {output}", args.join(" "));
    }
    let warm = cache_files(&home);
    assert_eq!(
        compiled_contexts(&warm).len(),
        1,
        "the valid package must leave exactly one compiled context: {warm:?}"
    );

    write_file(&ops, INVALID_OPS);
    assert_every_entry_rejects(&root, &home, "after a warm cache");
    assert_eq!(
        cache_files(&home),
        warm,
        "the edited, rejected sources must neither add nor rewrite a cache entry"
    );

    write_file(&ops, VALID_OPS);
    for args in ENTRIES {
        let (ok, output) = chelis(&root, &home, args);
        assert!(
            ok,
            "`chelis {}` must pass once the source is restored: {output}",
            args.join(" ")
        );
    }
    assert_eq!(
        cache_files(&home),
        warm,
        "the restored source must hit the entries written before the edit, not rebuild them"
    );
}
