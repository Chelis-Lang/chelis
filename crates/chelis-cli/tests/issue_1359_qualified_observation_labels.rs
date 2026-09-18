//! chelis#1359: [05-OBS-8] dotted descendants retain their originating
//! root's name in both evaluator and compiled observation output.

use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

const EXACT_REPRO: &str = r#"
type Box =
  | Box { text: string }

gadt = Box { text: "abcd" }
"#;

const NESTED_PRODUCTS: &str = r#"
type Inner =
  | Inner { value: string }

type Outer =
  | Outer { inner: Inner, pair: (i32, string) }

gadt =
  Outer {
    inner: Inner { value: "abcd" },
    pair: (cast(7, i32), "tail"),
  }

tuple_root = (cast(1, i32), (cast(2, i32), cast(3, i32)))
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
fn nested_record_and_tuple_descendants_keep_their_originating_root() {
    let eval = eval_text(NESTED_PRODUCTS);
    let compiled = common::build_and_run(NESTED_PRODUCTS, "nested_qualified_roots");
    let expected = [
        "gadt.inner.value",
        "gadt.pair.0",
        "gadt.pair.1",
        "tuple_root.0",
        "tuple_root.1.0",
        "tuple_root.1.1",
    ];

    assert_eq!(
        compiled, eval,
        "nested eval/C output must be byte-identical"
    );
    assert_eq!(text_labels(&eval), expected);
    assert!(
        !text_labels(&eval)
            .iter()
            .any(|name| matches!(*name, "value" | "pair.0" | "pair.1" | "root.0")),
        "record and tuple suffixes must not lose their originating root: {eval}"
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
