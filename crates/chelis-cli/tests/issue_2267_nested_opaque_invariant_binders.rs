//! Package-level acceptance for qualified and nested opaque invariant binders.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use tempfile::tempdir;

const TYPES: &str = "module Nested.Left
export (WideLeft, make_left, left_value)
@opaque
@invariant(x) ((x.value >= -1000.0f32) && (x.value <= 1000.0f32))
type WideLeft =
  | WideLeft { value: f32 }
def make_left() -> WideLeft = WideLeft { value: 0.0f32 }
def left_value(x: WideLeft) -> f32 = x.value
";

const MAIN: &str = "module Nested.Main
import Nested.Left (WideLeft, left_value)
type Wrapper =
  | Wrapper { inner: WideLeft }
@property qualified forall(p: Nested.Left.WideLeft):
  (left_value(p) >= -1000.0f32)
@property nested forall(w: Wrapper):
  (left_value(w.inner) >= -1000.0f32)
";

fn property<'a>(records: &'a [Value], name: &str) -> &'a Value {
    records
        .iter()
        .find(|record| record["kind"] == "property" && record["name"] == name)
        .unwrap_or_else(|| panic!("missing {name}: {records:#?}"))
}

#[test]
fn qualified_and_nested_package_binders_discharge_their_invariants() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join("nested-binders");
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "reef",
            "init",
            "nested-binders",
            "--module-prefix",
            "Nested",
            "--output",
        ])
        .arg(&root)
        .assert()
        .success();
    fs::write(root.join("src/left.ch"), TYPES).expect("write type module");
    fs::write(root.join("src/main.ch"), MAIN).expect("write property module");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&root)
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
        .collect::<Vec<Value>>();
    // The qualified direct binder is a green control on linked packages.
    // The nested record must receive the same invariant assumption.
    for name in ["qualified", "nested"] {
        let result = property(&records, name);
        assert_eq!(result["status"], "passed", "{result:#}");
        assert_eq!(result["composite_verdict"], "fuzz_validated", "{result:#}");
        assert_eq!(
            result["assumptions"].as_array().expect("assumptions").len(),
            1,
            "the contained opaque invariant is injected once: {result:#}"
        );
    }
    assert_eq!(output.status.code(), Some(0), "{records:#?}");
}
