//! chelis#2558: a compiled context links only the chelis-std modules that
//! its package and the entries its command runs import. Entries that reach
//! past the package's own imports keep working: a loose `eval --file`
//! snippet and a `chelis test` of a directory outside `tests/`. Contexts
//! linked for different sets of modules are cached apart, so a cached
//! package-only context never serves an entry that needs more.

use assert_cmd::Command;
use predicates::str::contains;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

const MAIN_MODULE: &str = "module Probe.Main
export (main)
def main() -> i64 = cast(1, i64)
";

const SMOKE_TEST: &str = "module Probe.Tests.Smoke
def test_ok() -> unit = test_assert(true, \"ok\")
";

const JSON_SNIPPET: &str = "import Std.Io.Json (try_parse_json)
bench = match try_parse_json(\"{}\") with {
  | Some(_) => cast(1, i64)
  | None => cast(0, i64)
}
";

const JSON_TEST: &str = "module Probe.Other.JsonTest
import Std.Io.Json (try_parse_json)
import Std.Test (assert_true)
export (test_json)
def test_json() -> unit ! { Test } =
  assert_true(match try_parse_json(\"{}\") with {
    | Some(_) => true
    | None => false
  }, \"json\")
";

fn write(path: &Path, contents: &str) {
    fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    fs::write(path, contents).expect("write fixture");
}

/// A package whose own modules and `tests/` import no chelis-std module.
/// Its manifest names the bundled runtime so chelis-std is in its graph.
fn package() -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join("probe");
    write(
        &root.join("reef.toml"),
        &format!(
            "[package]\nname = \"probe\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"Probe\"\n\n[dependencies]\nchelis-std = {{ version = \"{}\" }}\n",
            env!("CARGO_PKG_VERSION"),
            chelis_reef::compiler_bundled_chelis_std_version(),
        ),
    );
    write(&root.join("src/main.ch"), MAIN_MODULE);
    write(&root.join("tests/smoke.ch"), SMOKE_TEST);
    write(&root.join("bench.ch"), JSON_SNIPPET);
    write(&root.join("other/json_test.ch"), JSON_TEST);
    (dir, root)
}

fn chelis(root: &Path) -> Command {
    let mut command = Command::cargo_bin("chelis").expect("binary");
    command
        .current_dir(root)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env(
            "CHELIS_REEF_HOME",
            root.parent().expect("package parent").join("reef-home"),
        );
    command
}

fn compiled_contexts(root: &Path) -> usize {
    let dir = root
        .parent()
        .expect("parent")
        .join("reef-home/.cache/compiled");
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|entry| entry.path().extension().and_then(|e| e.to_str()) == Some("ctx"))
                .count()
        })
        .unwrap_or(0)
}

/// The package-only context is compiled and cached first; the snippet that
/// imports `Std.Io.Json` must then get its own context rather than the
/// cached one, and the package module must still hit its own afterwards.
#[test]
fn a_loose_eval_snippet_reaches_past_the_package_imports() {
    let (_dir, root) = package();
    for _ in 0..2 {
        chelis(&root)
            .args(["eval", "--file", "src/main.ch"])
            .assert()
            .success()
            .stdout("main = 1\n");
        chelis(&root)
            .args(["eval", "--file", "bench.ch"])
            .assert()
            .success()
            .stdout("bench = 1\n");
    }
    assert_eq!(
        compiled_contexts(&root),
        2,
        "the package-only and the Std.Io.Json contexts are cached apart"
    );
    chelis(&root)
        .args(["check", "src/main.ch"])
        .assert()
        .success();
}

#[test]
fn chelis_test_of_a_directory_outside_tests_reaches_past_the_package_imports() {
    let (_dir, root) = package();
    for _ in 0..2 {
        chelis(&root)
            .arg("test")
            .assert()
            .success()
            .stdout(contains("1 passed, 0 failed"));
        chelis(&root)
            .args(["test", "other"])
            .assert()
            .success()
            .stdout(contains("1 passed, 0 failed"));
    }
}
