//! PP4 / chelis#1264: module scope is exact at every checker and native-test
//! entry boundary. A unique terminal name elsewhere in the linked package, or
//! a declaration in a batched sibling test file, is never an implicit import.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Output;
use tempfile::{TempDir, tempdir};

const CHECK_ERRORS_EXIT_CODE: i32 = 2;

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent directory");
    }
    fs::write(path, contents).expect("write fixture");
}

fn make_package(name: &str) -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join(name);
    fs::create_dir_all(root.join("src")).expect("create src");
    fs::create_dir_all(root.join("tests")).expect("create tests");
    write_file(
        &root.join("reef.toml"),
        &format!(
            "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n\
             compiler = \"={ver}\"\nmodule_prefix = \"Scope\"\n",
            ver = chelis_compiler_api::COMPILER_VERSION,
        ),
    );
    (dir, root)
}

fn run(root: &Path, args: &[&str]) -> Output {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(root)
        .args(args)
        .output()
        .expect("run chelis")
}

fn run_check(root: &Path, entry: &Path) -> (Output, Value) {
    let output = run(root, &["check", entry.to_str().expect("utf8 path")]);
    let json = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|error| panic!("check output must be JSON: {error}\n{output:?}"));
    (output, json)
}

fn errors(report: &Value) -> &[Value] {
    report["errors"].as_array().expect("errors array")
}

fn has_error(report: &Value, kind: &str, identifier: &str) -> bool {
    errors(report).iter().any(|error| {
        error["kind"] == kind
            && error["message"]
                .as_str()
                .is_some_and(|message| message.contains(identifier))
    })
}

fn assert_check_name_error(output: &Output, report: &Value, kind: &str, identifier: &str) {
    assert_eq!(
        output.status.code(),
        Some(CHECK_ERRORS_EXIT_CODE),
        "name errors must use check exit 2: {output:?}"
    );
    assert!(
        report["components"]["names"].as_f64().expect("names score") < 1.0,
        "name error must lower the names component: {report}"
    );
    assert!(
        report["score"].as_f64().expect("fitness score") < 1.0,
        "name error must make fitness imperfect: {report}"
    );
    assert!(
        has_error(report, kind, identifier),
        "expected {kind} for {identifier}: {report}"
    );
    assert_eq!(
        report["unresolved_names"],
        serde_json::json!([identifier]),
        "each name diagnostic contributes one ordered unresolved name: {report}"
    );
}

fn write_value_provider(root: &Path) {
    write_file(
        &root.join("src/provider.ch"),
        "module Scope.Provider\n\
         export (borrowed)\n\
         def borrowed() -> tensor[1, f32] = to_tensor([7.0])\n",
    );
}

fn write_value_consumer(root: &Path, body: &str) -> PathBuf {
    let entry = root.join("src/consumer.ch");
    write_file(
        &entry,
        &format!("module Scope.Consumer\nexport (main)\n{body}\n"),
    );
    entry
}

fn assert_eval_and_build_reject_unbound(root: &Path, entry: &Path, identifier: &str) {
    let eval = run(
        root,
        &["eval", "--file", entry.to_str().expect("utf8 path")],
    );
    assert!(
        !eval.status.success(),
        "eval must reject an unimported value: {eval:?}"
    );
    let eval_stderr = String::from_utf8_lossy(&eval.stderr);
    assert!(
        eval_stderr.contains("unbound variable") && eval_stderr.contains(identifier),
        "eval must preserve the UnboundVariable reason: {eval_stderr}"
    );

    let out = root.join("out");
    let build = run(
        root,
        &[
            "build",
            entry.to_str().expect("utf8 path"),
            "--output",
            out.to_str().expect("utf8 path"),
        ],
    );
    assert!(
        !build.status.success(),
        "build must reject an unimported value: {build:?}"
    );
    let build_stderr = String::from_utf8_lossy(&build.stderr);
    assert!(
        build_stderr.contains("unbound variable") && build_stderr.contains(identifier),
        "build must preserve the UnboundVariable reason: {build_stderr}"
    );
}

fn assert_all_value_lanes_pass(root: &Path, entry: &Path) {
    let (check, report) = run_check(root, entry);
    assert!(check.status.success(), "check failed: {check:?}");
    assert_eq!(report["score"], 1.0, "check must be perfect: {report}");
    assert!(errors(&report).is_empty(), "check errors: {report}");
    assert_eq!(report["unresolved_names"], serde_json::json!([]));

    let eval = run(
        root,
        &["eval", "--file", entry.to_str().expect("utf8 path")],
    );
    assert!(eval.status.success(), "eval failed: {eval:?}");

    let out = root.join("out-positive");
    let build = run(
        root,
        &[
            "build",
            entry.to_str().expect("utf8 path"),
            "--output",
            out.to_str().expect("utf8 path"),
        ],
    );
    assert!(build.status.success(), "build failed: {build:?}");
}

#[test]
fn unimported_value_rejects_check_eval_and_build_while_imports_pass() {
    let (_dir, root) = make_package("pp4-value-scope");
    write_value_provider(&root);

    let entry = write_value_consumer(&root, "def main() -> tensor[1, f32] = borrowed()");
    let (check, report) = run_check(&root, &entry);
    assert_check_name_error(&check, &report, "UnboundVariable", "borrowed");
    assert_eval_and_build_reject_unbound(&root, &entry, "borrowed");

    write_file(
        &entry,
        "module Scope.Consumer\n\
         import Scope.Provider (borrowed)\n\
         export (main)\n\
         def main() -> tensor[1, f32] = borrowed()\n",
    );
    assert_all_value_lanes_pass(&root, &entry);

    write_file(
        &entry,
        "module Scope.Consumer\n\
         import Scope.Provider\n\
         export (main)\n\
         def main() -> tensor[1, f32] = Scope.Provider.borrowed()\n",
    );
    assert_all_value_lanes_pass(&root, &entry);
}

#[test]
fn unrelated_exporters_cannot_change_an_unbound_value_verdict() {
    let (_dir, root) = make_package("pp4-exporter-invariance");
    let entry = write_value_consumer(&root, "def main() -> i64 = absent()");

    for provider in [
        None,
        Some("module Scope.Other\nexport (absent)\ndef absent() -> i64 = cast(1, i64)\n"),
        Some("module Scope.Other\nexport (renamed)\ndef renamed() -> i64 = cast(1, i64)\n"),
    ] {
        let path = root.join("src/other.ch");
        match provider {
            Some(source) => write_file(&path, source),
            None => {
                let _ = fs::remove_file(&path);
            }
        }
        let (check, report) = run_check(&root, &entry);
        assert_check_name_error(&check, &report, "UnboundVariable", "absent");
    }

    write_file(
        &root.join("src/second.ch"),
        "module Scope.Second\nexport (absent)\ndef absent() -> i64 = cast(2, i64)\n",
    );
    write_file(
        &root.join("src/other.ch"),
        "module Scope.Other\nexport (absent)\ndef absent() -> i64 = cast(1, i64)\n",
    );
    let (check, report) = run_check(&root, &entry);
    assert_check_name_error(&check, &report, "UnboundVariable", "absent");
    assert_eval_and_build_reject_unbound(&root, &entry, "absent");
}

#[test]
fn unknown_constructor_is_counted_as_an_unresolved_name() {
    let (_dir, root) = make_package("pp4-constructor-fitness");
    write_file(
        &root.join("src/model.ch"),
        "module Scope.Model\n\
         export (Payload)\n\
         type Payload = | Payload { value: i64 }\n",
    );
    let entry = root.join("src/consumer.ch");
    write_file(
        &entry,
        "module Scope.Consumer\n\
         def make() = Payload { value: cast(1, i64) }\n",
    );
    let (check, report) = run_check(&root, &entry);
    assert_check_name_error(&check, &report, "UnknownConstructor", "Payload");

    write_file(
        &root.join("src/other.ch"),
        "module Scope.Other\n\
         export (Payload)\n\
         type Payload = | Payload { other: i64 }\n",
    );
    let (check, report) = run_check(&root, &entry);
    assert_check_name_error(&check, &report, "UnknownConstructor", "Payload");

    write_file(
        &entry,
        "module Scope.Consumer\n\
         import Scope.Model (Payload)\n\
         def make() = Payload { value: cast(1, i64) }\n",
    );
    let (check, report) = run_check(&root, &entry);
    assert!(
        check.status.success(),
        "imported constructor failed: {check:?}"
    );
    assert_eq!(report["components"]["names"], 1.0);
    assert_eq!(report["unresolved_names"], serde_json::json!([]));
    assert!(errors(&report).is_empty(), "imported constructor: {report}");
}

#[test]
fn unrelated_exporters_cannot_change_a_positional_constructor_kind() {
    let (_dir, root) = make_package("pp4-positional-constructor-invariance");
    let entry = root.join("src/consumer.ch");
    write_file(
        &entry,
        "module Scope.Consumer\n\
         def make() = Token(cast(1, i64))\n",
    );

    let (check, report) = run_check(&root, &entry);
    assert_check_name_error(&check, &report, "UnknownConstructor", "Token");

    for (path, source) in [
        (
            "src/model.ch",
            "module Scope.Model\n\
             export (Token)\n\
             type Envelope = | Token(i64)\n",
        ),
        (
            "src/other.ch",
            "module Scope.Other\n\
             export (Token)\n\
             type OtherEnvelope = | Token(i64)\n",
        ),
        (
            "src/third.ch",
            "module Scope.Third\n\
             export (Token)\n\
             type ThirdEnvelope = | Token(i64)\n",
        ),
    ] {
        write_file(&root.join(path), source);
        let (check, report) = run_check(&root, &entry);
        assert_check_name_error(&check, &report, "UnknownConstructor", "Token");
    }

    write_file(
        &entry,
        "module Scope.Consumer\n\
         import Scope.Model (Token)\n\
         def make() = Token(cast(1, i64))\n",
    );
    let (check, report) = run_check(&root, &entry);
    assert!(
        check.status.success(),
        "imported positional constructor failed: {check:?}"
    );
    assert_eq!(report["components"]["names"], 1.0);
    assert_eq!(report["unresolved_names"], serde_json::json!([]));
    assert!(
        errors(&report).is_empty(),
        "imported positional constructor: {report}"
    );
}

fn batch_rows(output: &Output) -> Vec<(String, String, String)> {
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter_map(|value| {
            Some((
                value.get("file")?.as_str()?.to_string(),
                value.get("test")?.as_str()?.to_string(),
                value.get("status")?.as_str()?.to_string(),
            ))
        })
        .collect()
}

fn has_batch_fallback(output: &Output) -> bool {
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .any(|value| value.get("batch_fallback").is_some_and(Value::is_object))
}

fn run_test_mode(root: &Path, mode: &str) -> Output {
    run(root, &["test", "--json", "--batch-mode", mode, "tests/"])
}

#[test]
fn auto_batch_cannot_borrow_a_sibling_declaration() {
    let (_dir, root) = make_package("pp4-batch-negative");
    write_file(
        &root.join("src/main.ch"),
        "module Scope.Main\ndef noop() -> bool = true\n",
    );
    write_file(
        &root.join("tests/a_unimported.ch"),
        "module Scope.Tests.A\n\
         def test_unimported() -> unit = test_assert(sibling_value(), \"must be in scope\")\n",
    );
    write_file(
        &root.join("tests/b_provider.ch"),
        "module Scope.Tests.B\n\
         def sibling_value() -> bool = true\n\
         def test_provider() -> unit = test_assert(sibling_value(), \"local binding\")\n",
    );

    let auto = run_test_mode(&root, "auto");
    let file = run_test_mode(&root, "file");
    assert_eq!(
        auto.status.code(),
        file.status.code(),
        "auto={auto:?}\nfile={file:?}"
    );
    assert_eq!(
        batch_rows(&auto),
        batch_rows(&file),
        "auto={auto:?}\nfile={file:?}"
    );
    assert_eq!(auto.status.code(), Some(1), "unbound suite must fail");
    assert!(
        has_batch_fallback(&auto),
        "the batch rejection must use the visible per-file fallback: {auto:?}"
    );
}

#[test]
fn imported_and_unrelated_sibling_batch_controls_stay_green() {
    let (_dir, root) = make_package("pp4-batch-positive");
    write_file(
        &root.join("src/helpers.ch"),
        "module Scope.Helpers\nexport (allowed)\ndef allowed() -> bool = true\n",
    );
    write_file(
        &root.join("tests/a_imported.ch"),
        "module Scope.Tests.A\n\
         import Scope.Helpers (allowed)\n\
         def test_imported() -> unit = test_assert(allowed(), \"imported\")\n",
    );
    write_file(
        &root.join("tests/b_unrelated.ch"),
        "module Scope.Tests.B\n\
         def unrelated() -> bool = true\n\
         def test_unrelated() -> unit = test_assert(unrelated(), \"local\")\n",
    );

    let auto = run_test_mode(&root, "auto");
    let file = run_test_mode(&root, "file");
    assert!(auto.status.success(), "auto batch failed: {auto:?}");
    assert!(file.status.success(), "file mode failed: {file:?}");
    assert_eq!(
        batch_rows(&auto),
        batch_rows(&file),
        "auto={auto:?}\nfile={file:?}"
    );
    assert!(
        !has_batch_fallback(&auto),
        "the valid isolated batch should stay on its shared fast path: {auto:?}"
    );
}

#[test]
fn lexical_locals_and_builtins_remain_exact_positive_controls() {
    let (_dir, root) = make_package("pp4-exact-controls");
    let entry = root.join("src/main.ch");
    write_file(
        &entry,
        "module Scope.Main\n\
         export (main)\n\
         def main() -> tensor[1, f32] = {\n\
           local = to_tensor([2.0])\n\
           add(local, to_tensor([3.0]))\n\
         }\n",
    );
    assert_all_value_lanes_pass(&root, &entry);
}
