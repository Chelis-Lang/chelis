//! Acceptance surface for chelis#3236: a property binder or module name
//! spelled like a prover-generated solver symbol (`__erf_abs_0`,
//! `__contract_std_normal_cdf_0`, `__contract_std_quantile_0`) stays its own
//! variable. A false property of that shape is never reported as proved, and
//! the valid controls keep their verdicts.
//!
//! The certified-envelope and contract lanes discharge through cvc5, so every
//! test needs the `smt` feature.

#![cfg(feature = "smt")]

use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

fn write(path: &std::path::Path, contents: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create fixture directory");
    }
    std::fs::write(path, contents).expect("write fixture");
}

fn prove(path: &std::path::Path, tier_args: &[&str]) -> (Vec<Value>, String) {
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .arg("prove")
        .arg(path)
        .args(tier_args)
        .arg("--json")
        .output()
        .expect("run prove");
    let transcript = format!(
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let records = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|record| record["kind"] == "property")
        .collect();
    (records, transcript)
}

fn record<'a>(records: &'a [Value], name: &str, transcript: &str) -> &'a Value {
    records
        .iter()
        .find(|record| record["name"] == name)
        .unwrap_or_else(|| panic!("no record for {name}\n{transcript}"))
}

const AUTO: &[&str] = &["--tier", "auto", "--samples", "16", "--seed", "1"];
const SMT_ONLY: &[&str] = &["--tier", "smt-only", "--samples", "0"];

/// A false property must not pass and must carry no `proven_*` badge.
fn assert_not_proved(record: &Value, transcript: &str) {
    assert_ne!(record["status"], "passed", "{record}\n{transcript}");
    let verdict = record["composite_verdict"].as_str().unwrap_or_default();
    assert!(
        !verdict.starts_with("proven"),
        "a false property must not be certified: {record}\n{transcript}"
    );
}

/// A refutation must report the user binder as its own model variable.
fn assert_counterexample_keeps_binder(record: &Value, binder: &str, transcript: &str) {
    assert_eq!(record["status"], "failed", "{record}\n{transcript}");
    let counterexample = record["counterexample"]
        .as_object()
        .unwrap_or_else(|| panic!("a refutation carries a counterexample: {record}"));
    assert!(
        counterexample.contains_key(binder),
        "the user binder is its own model variable: {record}\n{transcript}"
    );
}

const ENVELOPE_PROPERTIES: &str = "module Probe.Collision
@property forged forall(x: f32, __erf_abs_0: f32) where x >= 0.0f32, x <= 1.0f32:
  ((erf(x) <= 0.9f32) && (__erf_abs_0 <= 0.9f32))
@property control forall(x: f32) where x >= 0.0f32, x <= 1.0f32:
  (erf(x) <= 0.9f32)
";

const ENVELOPE_FREE_NAME: &str = "module Probe.Free
__erf_abs_0 = 5.0f32
@property forged_free forall(x: f32) where x >= 0.0f32, x <= 1.0f32:
  ((erf(x) <= 0.9f32) && (__erf_abs_0 <= 0.9f32))
";

#[test]
fn envelope_binder_collision_is_refuted_and_control_keeps_its_proof() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("collision.ch");
    write(&path, ENVELOPE_PROPERTIES);

    let (records, transcript) = prove(&path, AUTO);
    let forged = record(&records, "forged", &transcript);
    assert_not_proved(forged, &transcript);
    assert_eq!(forged["status"], "failed", "{forged}\n{transcript}");
    let control = record(&records, "control", &transcript);
    assert_eq!(control["status"], "passed", "{control}\n{transcript}");
    assert_eq!(
        control["composite_verdict"], "proven_modulo_certified_envelope",
        "{control}\n{transcript}"
    );

    let (records, transcript) = prove(&path, SMT_ONLY);
    assert_not_proved(record(&records, "forged", &transcript), &transcript);
    let control = record(&records, "control", &transcript);
    assert_eq!(control["status"], "passed", "{control}\n{transcript}");
    assert_eq!(
        control["composite_verdict"], "proven_modulo_certified_envelope",
        "{control}\n{transcript}"
    );
}

#[test]
fn envelope_module_name_collision_is_not_proved() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("free.ch");
    write(&path, ENVELOPE_FREE_NAME);
    for tier in [AUTO, SMT_ONLY] {
        let (records, transcript) = prove(&path, tier);
        assert_not_proved(record(&records, "forged_free", &transcript), &transcript);
    }
}

#[test]
fn normal_cdf_contract_binder_collision_is_refuted_and_control_keeps_its_proof() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join("myapp");
    write(
        &root.join("reef.toml"),
        &format!(
            r#"[package]
name = "myapp"
version = "0.1.0"
compiler = "={}"
module_prefix = "App"

[dependencies]
chelis-std = {{ version = "0.4.0" }}
"#,
            env!("CARGO_PKG_VERSION")
        ),
    );
    let entry = root.join("src/proofs.ch");
    write(
        &entry,
        r#"module App.Proofs
import Std.Contracts (normal_cdf)
@property forged_cdf forall(x: f64, __contract_std_normal_cdf_0: f64):
  ((normal_cdf(x) <= 1.0f64) && (__contract_std_normal_cdf_0 <= 1.0f64))
  with contract = "std.normal_cdf.range"
@property control_cdf forall(x: f64):
  (normal_cdf(x) <= 1.0f64)
  with contract = "std.normal_cdf.range"
"#,
    );
    for tier in [AUTO, SMT_ONLY] {
        let (records, transcript) = prove(&entry, tier);
        let forged = record(&records, "forged_cdf", &transcript);
        assert_not_proved(forged, &transcript);
        assert_counterexample_keeps_binder(forged, "__contract_std_normal_cdf_0", &transcript);
        let control = record(&records, "control_cdf", &transcript);
        assert_eq!(control["status"], "passed", "{control}\n{transcript}");
        assert_eq!(
            control["composite_verdict"], "proven_modulo_fuzz_validated_contract",
            "{control}\n{transcript}"
        );
    }
}

const NAUTILUS_REEF: &str =
    include_str!("../../../examples/nautilus_quantile_contract/fixtures/nautilus/reef.toml");
const NAUTILUS_STATS: &str =
    include_str!("../../../examples/nautilus_quantile_contract/fixtures/nautilus/src/stats.ch");

#[test]
fn quantile_contract_binder_collision_is_refuted_and_control_keeps_its_proof() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join("risk-model");
    write(
        &root.join("reef.toml"),
        &format!(
            r#"[package]
name = "risk-model"
version = "0.1.0"
compiler = "={}"
module_prefix = "Risk"

[dependencies]
nautilus = {{ path = "../nautilus" }}
"#,
            env!("CARGO_PKG_VERSION")
        ),
    );
    write(&dir.path().join("nautilus/reef.toml"), NAUTILUS_REEF);
    write(&dir.path().join("nautilus/src/stats.ch"), NAUTILUS_STATS);
    let entry = root.join("src/proofs.ch");
    write(
        &entry,
        r#"module Risk.Proofs
import Nautilus.Stats (quantile_vec)
def observations() -> tensor[3, f32] = (to_tensor([3.0, 1.0, 2.0]) : tensor[3, f32])
@property forged_quantile forall(p: f32, q: f32, __contract_std_quantile_0: f32, __contract_std_quantile_1: f32) where 0.0 <= p, p <= q, q <= 1.0:
  ((quantile_vec(observations(), p) <= quantile_vec(observations(), q)) && (__contract_std_quantile_0 <= __contract_std_quantile_1))
  with contract = "std.quantile.monotonicity"
@property control_quantile forall(p: f32, q: f32) where 0.0 <= p, p <= q, q <= 1.0:
  (quantile_vec(observations(), p) <= quantile_vec(observations(), q))
  with contract = "std.quantile.monotonicity"
"#,
    );
    let (records, transcript) = prove(&entry, SMT_ONLY);
    let forged = record(&records, "forged_quantile", &transcript);
    assert_not_proved(forged, &transcript);
    assert_counterexample_keeps_binder(forged, "__contract_std_quantile_0", &transcript);
    let control = record(&records, "control_quantile", &transcript);
    assert_eq!(control["status"], "passed", "{control}\n{transcript}");
    assert_eq!(control["proof_tier"], "smt", "{control}\n{transcript}");
}
