//! chelis#2416: `chelis prove` in a Reef package runs a property over an
//! invariant-carrying opaque binder inside the type's defining module, as it
//! does for a bare file. The reachable-slice check that `prove` performs
//! lists the defining module's exported producers, as `chelis check` does.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

const SAME_MODULE: &str = "module Qpopq.Main
export (clamp_prob, prob_value)
@opaque
@invariant(p) ((p.value >= 0.0) && (p.value <= 1.0))
type Probability =
  | Probability { value: f32 }
def clamp_prob(x: f32) -> Probability = Probability { value: if (x >= 0.0) then if (x <= 1.0) then x else 1.0 else 0.0 }
def prob_value(p: Probability) -> f32 = p.value
@property bounded forall(p: Probability):
  (prob_value(p) <= 1.0)
";

const TYPES_MODULE: &str = "module Qpopq.Types
export (Probability, clamp_prob, prob_value)
@opaque
@invariant(p) ((p.value >= 0.0) && (p.value <= 1.0))
type Probability =
  | Probability { value: f32 }
def clamp_prob(x: f32) -> Probability = Probability { value: if (x >= 0.0) then if (x <= 1.0) then x else 1.0 else 0.0 }
def prob_value(p: Probability) -> f32 = p.value
";

/// The binder's type comes from another module, and the producer is not
/// imported, so the property's reachable slice does not reference it.
const CROSS_MODULE_BINDER: &str = "module Qpopq.Main
import Qpopq.Types (Probability, prob_value)
export (value_of)
def value_of(p: Probability) -> f32 = prob_value(p)
@property bounded forall(p: Probability):
  (prob_value(p) <= 1.0)
";

/// A qualified binder spelling resolves through the package linker.
const QUALIFIED_BINDER: &str = "module Qpopq.Main
import Qpopq.Types (prob_value)
export (half)
def half() -> f32 = 0.5
@property bounded forall(p: Qpopq.Types.Probability):
  (prob_value(p) <= 1.0)
";

/// A genuine construction outside the defining module.
const OUTSIDE_CONSTRUCTION: &str = "module Qpopq.Main
import Qpopq.Types (Probability, prob_value)
export (forged)
def forged() -> f32 = prob_value(Probability { value: 2.0 })
";

const PRODUCER_LISTING: &str = "record construction of opaque type `Probability` outside its \
     defining module `Qpopq.Types`; exported producers of `Qpopq.Types`: \
     clamp_prob: (f32) -> Probability";

fn package(files: &[(&str, &str)]) -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join("qpopq");
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "reef",
            "init",
            "qpopq",
            "--module-prefix",
            "Qpopq",
            "--output",
        ])
        .arg(&root)
        .assert()
        .success();
    for (relative, source) in files {
        fs::write(root.join(relative), source).expect("write source");
    }
    (dir, root)
}

/// Run `chelis prove --json` on `file` and return the exit code and the
/// parsed NDJSON records.
fn prove(cwd: &Path, file: &str) -> (i32, Vec<Value>) {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(cwd)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["prove", "--json", "--samples", "30", file])
        .output()
        .expect("run prove");
    let records = String::from_utf8(output.stdout)
        .expect("utf8 stdout")
        .lines()
        .map(|line| serde_json::from_str(line).expect("NDJSON record"))
        .collect();
    (output.status.code().expect("exit code"), records)
}

fn property<'a>(records: &'a [Value], name: &str) -> &'a Value {
    records
        .iter()
        .find(|record| record["kind"] == "property" && record["name"] == name)
        .unwrap_or_else(|| panic!("no property `{name}` record in {records:#?}"))
}

fn assert_injected_and_validated(records: &[Value]) {
    let bounded = property(records, "bounded");
    assert_eq!(bounded["status"], "passed", "{bounded:#}");
    assert_eq!(
        bounded["composite_verdict"], "fuzz_validated",
        "{bounded:#}"
    );
    let names = bounded["assumptions"]
        .as_array()
        .expect("assumptions")
        .iter()
        .map(|assumption| assumption["name"].as_str().expect("name"))
        .collect::<Vec<_>>();
    assert_eq!(names, ["invariant:Probability:binder:p"], "{bounded:#}");
}

#[test]
fn package_property_over_same_module_opaque_binder_is_fuzz_validated() {
    let (_dir, root) = package(&[("src/main.ch", SAME_MODULE)]);
    let (code, records) = prove(&root, "src/main.ch");
    assert_eq!(code, 0, "{records:#?}");
    assert_injected_and_validated(&records);
}

#[test]
fn bare_file_property_over_opaque_binder_is_unchanged() {
    let dir = tempdir().expect("tempdir");
    fs::write(
        dir.path().join("bare.ch"),
        SAME_MODULE.replace("module Qpopq.Main", "module Bare"),
    )
    .expect("write bare file");
    let (code, records) = prove(dir.path(), "bare.ch");
    assert_eq!(code, 0, "{records:#?}");
    assert_injected_and_validated(&records);
}

#[test]
fn package_property_over_cross_module_binder_is_fuzz_validated() {
    let (_dir, root) = package(&[
        ("src/types.ch", TYPES_MODULE),
        ("src/main.ch", CROSS_MODULE_BINDER),
    ]);
    let (code, records) = prove(&root, "src/main.ch");
    assert_eq!(code, 0, "{records:#?}");
    assert_injected_and_validated(&records);
}

#[test]
fn package_property_over_qualified_binder_is_fuzz_validated() {
    let (_dir, root) = package(&[
        ("src/types.ch", TYPES_MODULE),
        ("src/main.ch", QUALIFIED_BINDER),
    ]);
    let (code, records) = prove(&root, "src/main.ch");
    assert_eq!(code, 0, "{records:#?}");
    assert_injected_and_validated(&records);
}

/// Negative control: a construction outside the defining module is still
/// refused, and `prove` lists the exported producer that is absent from the
/// entry module's reachable slice, matching `chelis check`.
#[test]
fn outside_module_construction_is_refused_with_its_exported_producer() {
    let (_dir, root) = package(&[
        ("src/types.ch", TYPES_MODULE),
        ("src/main.ch", OUTSIDE_CONSTRUCTION),
    ]);

    let (code, records) = prove(&root, "src/main.ch");
    assert_ne!(code, 0, "{records:#?}");
    let check_error = records
        .iter()
        .find(|record| record["kind"] == "error" && record["stage"] == "check")
        .unwrap_or_else(|| panic!("no check error record in {records:#?}"));
    let reason = check_error["reason"].as_str().expect("reason");
    assert!(
        reason.contains(PRODUCER_LISTING),
        "prove must list the exported producer: {reason}"
    );

    let check = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&root)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", "src/main.ch"])
        .output()
        .expect("run check");
    let stdout = String::from_utf8_lossy(&check.stdout);
    assert!(!check.status.success(), "{stdout}");
    assert!(stdout.contains(PRODUCER_LISTING), "{stdout}");
}
