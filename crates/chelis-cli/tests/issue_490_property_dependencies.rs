//! chelis#490: `chelis prove --json` exposes each property's dependency /
//! attached-target set so an admission policy can confirm the property
//! mentions the export it claims to constrain WITHOUT regex-scanning the body.
//!
//! The `dependencies` field is the set of FREE top-level symbols the
//! property's preconditions + body reference (the quantifier params and any
//! `let`/lambda/match binders are excluded). It is always present on a
//! property record (an empty array means the body references only its own
//! params). The deep unit coverage (scoping, shadowing, lambda, block,
//! precondition union) lives in
//! `chelis-prove/src/property_runner/tests.rs`; these are the CLI-surface
//! shape locks.

use assert_cmd::Command;
use serde_json::Value;
use tempfile::{TempDir, tempdir};

fn write_prop(source: &str) -> TempDir {
    let dir = tempdir().expect("tempdir");
    std::fs::write(dir.path().join("prop.ch"), source).expect("write property");
    dir
}

fn property_records(output: &[u8]) -> Vec<Value> {
    String::from_utf8_lossy(output)
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str::<Value>(line).expect("json line"))
        .filter(|record| record.get("kind").and_then(Value::as_str) == Some("property"))
        .collect()
}

/// A property whose body calls a top-level def carries that def in its
/// `dependencies`; a property referencing only its quantifier param carries
/// an empty `dependencies`. Both records ALWAYS have the key.
#[test]
fn prove_json_property_record_carries_dependencies() {
    let dir = write_prop(
        "def helper(x: f32) -> f32 = x * x\n\
         @property uses_helper forall(x: f32):\n  (helper(x) >= 0.0)\n\
         @property only_param forall(x: f32):\n  (x * x >= 0.0)\n",
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--json",
            "--tier",
            "auto",
            "--samples",
            "4",
        ])
        .output()
        .expect("run prove");

    let props = property_records(&output.stdout);
    let uses_helper = props
        .iter()
        .find(|r| r["name"] == "uses_helper")
        .expect("uses_helper record");
    let only_param = props
        .iter()
        .find(|r| r["name"] == "only_param")
        .expect("only_param record");

    // Both records carry the key (a consumer can rely on its presence).
    assert!(
        uses_helper.get("dependencies").is_some(),
        "every property record carries `dependencies`: {uses_helper:?}"
    );
    assert!(
        only_param.get("dependencies").is_some(),
        "every property record carries `dependencies`: {only_param:?}"
    );

    assert_eq!(
        uses_helper["dependencies"],
        serde_json::json!(["helper"]),
        "the body's call to `helper` is the property's attached target"
    );
    // NEGATIVE twin: a property referencing only its param has an EMPTY set —
    // a param is never an attached target.
    assert_eq!(
        only_param["dependencies"],
        serde_json::json!([]),
        "a body referencing only its param has no attached target"
    );
}

/// A property that calls a `Std`-imported function lists the resolved symbol
/// as a dependency: an admission policy keying on the export sees the edge.
/// (Bundled chelis-std is always available, no package setup needed.)
#[test]
fn prove_json_dependencies_list_imported_symbol() {
    let dir = write_prop(
        "import Std.Scalar (abs)\n\
         @property abs_nonneg forall(x: f32):\n  (abs(x) >= 0.0)\n",
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--json",
            "--tier",
            "auto",
            "--samples",
            "4",
        ])
        .output()
        .expect("run prove");

    let props = property_records(&output.stdout);
    let record = props
        .iter()
        .find(|r| r["name"] == "abs_nonneg")
        .expect("abs_nonneg record");
    let deps = record["dependencies"]
        .as_array()
        .expect("dependencies is an array");
    // The dependency is the resolved symbol; `abs` appears as the leaf name of
    // the resolved (possibly linker-internal) symbol an admission policy
    // matches against.
    assert!(
        deps.iter()
            .filter_map(Value::as_str)
            .any(|name| name.contains("abs")),
        "the imported `abs` must appear as a dependency edge, got {deps:?}"
    );
}
