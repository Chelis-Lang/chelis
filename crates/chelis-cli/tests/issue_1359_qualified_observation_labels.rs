//! chelis#1359: [05-OBS-8] dotted descendants retain their originating
//! root's name in both evaluator and compiled observation output.

use assert_cmd::Command;
use serde_json::Value;
use std::path::PathBuf;
use tempfile::{TempDir, tempdir};

#[path = "common/mod.rs"]
mod common;

const EXACT_REPRO: &str = r#"
type Box =
  | Box { text: string }

gadt = Box { text: "abcd" }
"#;

const NESTED_PRODUCTS: &str = r#"
type Inner =
  | Inner { inner__value: string }

type Outer =
  | Outer { field__name: string, inner: Inner, pair: (i32, string) }

record_root =
  Outer {
    field__name: "top",
    inner: Inner { inner__value: "abcd" },
    pair: (cast(7, i32), "tail"),
  }

root__tuple = (cast(1, i32), (cast(2, i32), cast(3, i32)))
"#;

fn eval_text(source: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("program.ch");
    common::write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().expect("utf-8 path")])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    String::from_utf8(output).expect("eval stdout is UTF-8")
}

fn eval_json(source: &str) -> Value {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("program.ch");
    common::write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "eval",
            "--file",
            path.to_str().expect("utf-8 path"),
            "--json",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    serde_json::from_slice(&output).expect("eval --json emits one JSON document")
}

fn reef_package(source: &str) -> (TempDir, PathBuf, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join("obs-labels");
    let reef_home = dir.path().join("reef-home");
    std::fs::create_dir_all(root.join("src")).expect("create package source directory");
    std::fs::create_dir_all(&reef_home).expect("create reef home");
    common::write_file(
        &root.join("reef.toml"),
        &format!(
            r#"[package]
name = "obs-labels"
version = "0.1.0"
compiler = "={}"
module_prefix = "App"
"#,
            env!("CARGO_PKG_VERSION")
        ),
    );
    common::write_file(
        &root.join("src/main.ch"),
        &format!("module App.Main\n\n{source}"),
    );
    (dir, root, reef_home)
}

fn eval_reef_text(root: &std::path::Path, reef_home: &std::path::Path) -> String {
    let entry = root.join("src/main.ch");
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(root)
        .args(["eval", "--file", entry.to_str().expect("utf-8 path")])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    String::from_utf8(output).expect("reef eval stdout is UTF-8")
}

fn text_labels(stdout: &str) -> Vec<&str> {
    stdout
        .lines()
        .map(|line| {
            line.split_once(" = ")
                .unwrap_or_else(|| panic!("observation is not `name = value`: {line:?}"))
                .0
        })
        .collect()
}

fn json_names<'a>(json: &'a Value, field: &str) -> Vec<&'a str> {
    json[field]
        .as_array()
        .unwrap_or_else(|| panic!("{field} is not an array: {json}"))
        .iter()
        .map(|entry| {
            entry["name"]
                .as_str()
                .unwrap_or_else(|| panic!("{field} entry has no string name: {entry}"))
        })
        .collect()
}

#[test]
fn record_field_reproducer_is_byte_identical_across_eval_and_c() {
    let eval = eval_text(EXACT_REPRO);
    let compiled = common::build_and_run(EXACT_REPRO, "qualified_record_root");

    assert_eq!(eval, "gadt.text = abcd\n");
    assert_eq!(compiled, eval, "eval and C observation bytes must agree");
    assert!(!eval.lines().any(|line| line.starts_with("text = ")));
}

#[test]
fn lexical_leading_underscores_remain_authored_root_text() {
    const SOURCE: &str = "__eval_result = cast(7, i32)\n";
    let eval = eval_text(SOURCE);
    let compiled = common::build_and_run(SOURCE, "authored_leading_underscores");

    assert_eq!(eval, "__eval_result = 7\n");
    assert_eq!(compiled, eval, "eval and C lexical root bytes must agree");
}

#[test]
fn nested_record_and_tuple_descendants_keep_their_originating_root() {
    let eval = eval_text(NESTED_PRODUCTS);
    let compiled = common::build_and_run(NESTED_PRODUCTS, "nested_qualified_roots");
    let expected = [
        "record_root.field__name",
        "record_root.inner.inner__value",
        "record_root.pair.0",
        "record_root.pair.1",
        "root__tuple.0",
        "root__tuple.1.0",
        "root__tuple.1.1",
    ];

    assert_eq!(
        compiled, eval,
        "nested eval/C output must be byte-identical"
    );
    assert_eq!(text_labels(&eval), expected);
    assert!(
        !text_labels(&eval).iter().any(|name| matches!(
            *name,
            "name" | "value" | "inner__value" | "tuple.0" | "tuple.1.0" | "tuple.1.1"
        )),
        "authored repeated underscores and originating roots must remain intact: {eval}"
    );
}

#[test]
fn reef_linker_qualification_is_removed_without_rewriting_authored_underscores() {
    let (_dir, root, reef_home) = reef_package(NESTED_PRODUCTS);
    let evaluated = eval_reef_text(&root, &reef_home);
    let compiled = common::build_and_run_app(&reef_home, &root, "main");
    let expected = [
        "record_root.field__name",
        "record_root.inner.inner__value",
        "record_root.pair.0",
        "record_root.pair.1",
        "root__tuple.0",
        "root__tuple.1.0",
        "root__tuple.1.1",
    ];

    assert_eq!(
        compiled, evaluated,
        "reef eval and C output must be byte-identical after linker dequalification"
    );
    assert_eq!(text_labels(&evaluated), expected);
    assert!(
        !evaluated.contains("pkg__") && !evaluated.contains("Pkg__"),
        "private linker qualification must not leak into package output: {evaluated}"
    );
}

#[test]
fn text_labels_equal_json_roots_and_manifest_entries_in_order() {
    let text = eval_text(NESTED_PRODUCTS);
    let json = eval_json(NESTED_PRODUCTS);
    let labels = text_labels(&text);
    let roots = json_names(&json, "roots");
    let manifest = json_names(&json["manifest"], "entries");

    assert_eq!(labels, roots, "text output must use structured root names");
    assert_eq!(roots, manifest, "realized roots must equal manifest order");
    assert_eq!(json["manifest"]["requires_main"], true);
    assert!(
        labels.iter().all(|name| name.contains('.')),
        "every fixed-product leaf must remain a dotted root descendant: {labels:?}"
    );
}
