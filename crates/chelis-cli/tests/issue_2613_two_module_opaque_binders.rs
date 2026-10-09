//! A property can generate opaque invariant binders owned by two Reef modules.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

const LEFT: &str = "module TwoMod.Left
export (WideLeft, make_left, left_value)
@opaque
@invariant(x) ((x.value >= -1000.0f32) && (x.value <= 1000.0f32))
type WideLeft =
  | WideLeft { value: f32 }
def make_left() -> WideLeft = WideLeft { value: 0.0f32 }
def left_value(x: WideLeft) -> f32 = x.value
";

const RIGHT: &str = "module TwoMod.Deep.Right
export (WideRight, make_right, right_value)
@opaque
@invariant(x) ((x.value >= -1000.0f32) && (x.value <= 1000.0f32))
type WideRight =
  | WideRight { value: f32 }
def make_right() -> WideRight = WideRight { value: 0.0f32 }
def right_value(x: WideRight) -> f32 = x.value
";

const MAIN_TWO: &str = "module TwoMod.Main
import TwoMod.Left (WideLeft, left_value)
import TwoMod.Deep.Right (WideRight, right_value)
@property joint forall(a: WideLeft, b: WideRight):
  ((left_value(a) >= -1000.0f32) && (right_value(b) <= 1000.0f32))
@property joint_false forall(a: WideLeft, b: WideRight):
  (left_value(a) >= 2000.0f32)
";

const SAME_MODULE: &str = "module TwoMod.Left
export (WideLeft, WideRight, make_left, make_right, left_value, right_value)
@opaque
@invariant(x) ((x.value >= -1000.0f32) && (x.value <= 1000.0f32))
type WideLeft =
  | WideLeft { value: f32 }
@opaque
@invariant(x) ((x.value >= -1000.0f32) && (x.value <= 1000.0f32))
type WideRight =
  | WideRight { value: f32 }
def make_left() -> WideLeft = WideLeft { value: 0.0f32 }
def make_right() -> WideRight = WideRight { value: 0.0f32 }
def left_value(x: WideLeft) -> f32 = x.value
def right_value(x: WideRight) -> f32 = x.value
";

const MAIN_SAME: &str = "module TwoMod.Main
import TwoMod.Left (WideLeft, WideRight, left_value, right_value)
@property joint forall(a: WideLeft, b: WideRight):
  ((left_value(a) >= -1000.0f32) && (right_value(b) <= 1000.0f32))
@property joint_false forall(a: WideLeft, b: WideRight):
  (left_value(a) >= 2000.0f32)
";

fn package(name: &str, files: &[(&str, &str)]) -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join(name);
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "reef",
            "init",
            name,
            "--module-prefix",
            "TwoMod",
            "--output",
        ])
        .arg(&root)
        .assert()
        .success();
    for (relative, source) in files {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
        fs::write(path, source).expect("write source");
    }
    (dir, root)
}

fn prove(root: &Path) -> (i32, Vec<Value>) {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(root)
        .args([
            "prove",
            "src/main.ch",
            "--tier",
            "fuzz-only",
            "--samples",
            "16",
            "--seed",
            "1",
            "--json",
        ])
        .output()
        .expect("prove runs");
    let records = String::from_utf8(output.stdout)
        .expect("utf8")
        .lines()
        .map(|line| serde_json::from_str(line).expect("JSON record"))
        .collect();
    (output.status.code().expect("exit code"), records)
}

fn property<'a>(records: &'a [Value], name: &str) -> &'a Value {
    records
        .iter()
        .find(|record| record["kind"] == "property" && record["name"] == name)
        .unwrap_or_else(|| panic!("missing property {name}: {records:#?}"))
}

fn assert_true_false(records: &[Value]) {
    let good = property(records, "joint");
    let bad = property(records, "joint_false");
    assert_eq!(good["status"], "passed", "{good:#}");
    assert_eq!(good["composite_verdict"], "fuzz_validated", "{good:#}");
    assert_eq!(bad["status"], "failed", "{bad:#}");
    assert!(bad["counterexample"].is_object(), "{bad:#}");
    let assumptions = good["assumptions"].as_array().expect("assumptions");
    assert_eq!(
        assumptions.len(),
        2,
        "both binders need invariants: {good:#}"
    );
}

#[test]
fn two_defining_modules_match_the_same_module_true_and_false_control() {
    let (_two_dir, two) = package(
        "two-module-binders",
        &[
            ("src/left.ch", LEFT),
            ("src/deep/right.ch", RIGHT),
            ("src/main.ch", MAIN_TWO),
        ],
    );
    let (_same_dir, same) = package(
        "same-module-binders",
        &[("src/left.ch", SAME_MODULE), ("src/main.ch", MAIN_SAME)],
    );
    let (two_code, two_records) = prove(&two);
    let (same_code, same_records) = prove(&same);
    assert_true_false(&same_records);
    assert_eq!(two_code, same_code, "two modules: {two_records:#?}");
    assert_true_false(&two_records);
    for name in ["joint", "joint_false"] {
        for field in ["status", "composite_verdict", "samples"] {
            assert_eq!(
                property(&two_records, name)[field],
                property(&same_records, name)[field],
                "{name} {field}"
            );
        }
    }
}

#[test]
fn generated_value_helper_is_fresh_against_the_linked_terminal() {
    let right = RIGHT.replace(
        "def make_right()",
        "def chelis_value_probe() -> f32 = 1.0f32\ndef make_right()",
    );
    let (_dir, root) = package(
        "two-module-helper-collision",
        &[
            ("src/left.ch", LEFT),
            ("src/deep/right.ch", &right),
            ("src/main.ch", MAIN_TWO),
        ],
    );
    let (code, records) = prove(&root);
    assert_eq!(code, 1, "{records:#?}");
    assert_true_false(&records);
}

#[test]
fn authored_source_cannot_call_the_generated_value_helper() {
    let forged = MAIN_TWO.replace(
        "@property joint forall",
        "def forged() -> WideRight = chelis_value_probe\n@property joint forall",
    );
    let (_dir, root) = package(
        "two-module-helper-forged",
        &[
            ("src/left.ch", LEFT),
            ("src/deep/right.ch", RIGHT),
            ("src/main.ch", &forged),
        ],
    );
    let (code, records) = prove(&root);
    assert_ne!(code, 0, "{records:#?}");
    assert!(
        records
            .iter()
            .any(|record| record["kind"] == "error" && record["stage"] == "check"),
        "authored source must be checked before generated helpers exist: {records:#?}"
    );
}

#[test]
fn user_construction_outside_the_second_defining_module_stays_forbidden() {
    let forged = MAIN_TWO.replace(
        "@property joint forall",
        "def forged() -> WideRight = WideRight { value: 5.0f32 }\n@property joint forall",
    );
    let (_dir, root) = package(
        "two-module-forged",
        &[
            ("src/left.ch", LEFT),
            ("src/deep/right.ch", RIGHT),
            ("src/main.ch", &forged),
        ],
    );
    let (code, records) = prove(&root);
    assert_ne!(code, 0, "{records:#?}");
    let reason = records
        .iter()
        .find(|record| record["kind"] == "error" && record["stage"] == "check")
        .and_then(|record| record["reason"].as_str())
        .unwrap_or_else(|| panic!("missing check failure: {records:#?}"));
    assert!(reason.contains("outside its defining module"), "{reason}");
}
