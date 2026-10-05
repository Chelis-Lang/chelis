//! Acceptance surface for chelis#979.
//!
//! The fixture is a real Reef dependency named `nautilus`, exporting the
//! released `Nautilus.Stats.quantile_vec` implementation.  The proof must bind
//! the linker's internal symbol, not an author-spelled lookalike.

#![cfg(all(feature = "chelis-prove", feature = "smt"))]

use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

fn write(path: &std::path::Path, contents: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create fixture directory");
    }
    std::fs::write(path, contents).expect("write fixture");
}

fn property_records(stdout: &[u8]) -> Vec<Value> {
    String::from_utf8_lossy(stdout)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|record| record["kind"] == "property")
        .collect()
}

const NAUTILUS_REEF: &str =
    include_str!("../../../examples/nautilus_quantile_contract/fixtures/nautilus/reef.toml");
const NAUTILUS_STATS: &str =
    include_str!("../../../examples/nautilus_quantile_contract/fixtures/nautilus/src/stats.ch");

fn package(property: &str) -> tempfile::TempDir {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join("risk-model");
    let dep = dir.path().join("nautilus");
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
    write(&dep.join("reef.toml"), NAUTILUS_REEF);
    write(&dep.join("src/stats.ch"), NAUTILUS_STATS);
    write(&root.join("src/proofs.ch"), property);
    dir
}

fn prove(dir: &tempfile::TempDir) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "prove",
            dir.path()
                .join("risk-model/src/proofs.ch")
                .to_str()
                .expect("utf8 path"),
            "--json",
            "--tier",
            "smt-only",
        ])
        .output()
        .expect("run prove")
}

#[test]
fn linked_nautilus_quantile_monotonicity_consumes_named_contract() {
    let dir = package(
        r#"module Risk.Proofs
import Nautilus.Stats (quantile_vec)
def observations() -> tensor[3, f32] =
  (to_tensor([3.0, 1.0, 2.0], f32) : tensor[3, f32])
@property quantiles_are_monotone forall(p: f32, q: f32)
  where 0.0 <= p, p <= q, q <= 1.0:
  quantile_vec(observations(), p) <= quantile_vec(observations(), q)
  with contract = "std.quantile.monotonicity"
"#,
    );
    let output = prove(&dir);
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let records = property_records(&output.stdout);
    assert_eq!(records.len(), 1, "records={records:?}");
    assert_eq!(records[0]["status"], "passed");
    assert_eq!(records[0]["proof_tier"], "smt");
    assert!(records[0]["assumptions"].as_array().is_some_and(|items| {
        items.iter().any(|item| {
            item["name"] == "std.quantile.monotonicity"
                && item["discharge"]["evidence"]["implementation"] == "Nautilus.Stats.quantile_vec"
        })
    }));
}

#[test]
fn linked_nautilus_quantile_checks_and_evaluates_normally() {
    let dir = package(
        r#"module Risk.Proofs
import Nautilus.Stats (quantile_vec)
def observations() -> tensor[3, f32] =
  (to_tensor([3.0, 1.0, 2.0], f32) : tensor[3, f32])
median = quantile_vec(observations(), 0.5)
"#,
    );
    let path = dir.path().join("risk-model/src/proofs.ch");
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().expect("utf8 path")])
        .assert()
        .success();
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "eval",
            "--json",
            "--file",
            path.to_str().expect("utf8 path"),
        ])
        .output()
        .expect("eval linked quantile");
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout).expect("eval JSON");
    let median = result["roots"]
        .as_array()
        .and_then(|roots| roots.iter().find(|root| root["name"] == "median"))
        .expect("median root");
    let value = median["value"]["value"]
        .as_f64()
        .or_else(|| median["value"]["value"]["data"][0].as_f64())
        .expect("scalar median value");
    assert!((value - 2.0).abs() <= f32::EPSILON as f64, "median={value}");
}

#[test]
fn contract_does_not_turn_a_wrong_quantile_claim_green() {
    let dir = package(
        r#"module Risk.Proofs
import Nautilus.Stats (quantile_vec)
def observations() -> tensor[3, f32] =
  (to_tensor([3.0, 1.0, 2.0], f32) : tensor[3, f32])
@property wrong_direction forall(p: f32, q: f32)
  where 0.0 <= p, p < q, q <= 1.0:
  quantile_vec(observations(), q) < quantile_vec(observations(), p)
  with contract = "std.quantile.monotonicity"
"#,
    );
    let output = prove(&dir);
    assert_eq!(output.status.code(), Some(1));
    let records = property_records(&output.stdout);
    assert_eq!(records.len(), 1, "records={records:?}");
    assert_eq!(records[0]["status"], "failed");
}

#[test]
fn local_or_linker_shaped_spoof_cannot_receive_nautilus_contract() {
    let dir = tempdir().expect("tempdir");
    write(
        &dir.path().join("spoof.ch"),
        r#"def pkg__nautilus__Nautilus__Stats__quantile_vec(v: &tensor[3, f32], q: f32) -> f32 = q
def observations() -> tensor[3, f32] =
  (to_tensor([3.0, 1.0, 2.0], f32) : tensor[3, f32])
@property spoof forall(p: f32, q: f32) where 0.0 <= p, p <= q, q <= 1.0:
  pkg__nautilus__Nautilus__Stats__quantile_vec(observations(), p)
    <= pkg__nautilus__Nautilus__Stats__quantile_vec(observations(), q)
  with contract = "std.quantile.monotonicity"
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args([
            "prove",
            dir.path().join("spoof.ch").to_str().expect("utf8 path"),
            "--json",
            "--tier",
            "smt-only",
        ])
        .output()
        .expect("run prove");
    assert_ne!(output.status.code(), Some(0));
    assert!(
        property_records(&output.stdout)
            .iter()
            .all(|record| record["status"] != "passed")
    );
}

#[test]
fn requested_quantile_contract_without_trusted_call_is_unsupported() {
    let dir = tempdir().expect("tempdir");
    write(
        &dir.path().join("local.ch"),
        r#"@property unrelated forall(p: f32, q: f32) where p <= q:
  p <= q
  with contract = "std.quantile.monotonicity"
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args([
            "prove",
            dir.path().join("local.ch").to_str().expect("utf8 path"),
            "--json",
            "--tier",
            "smt-only",
        ])
        .output()
        .expect("run prove");
    assert_eq!(output.status.code(), Some(2));
    let records = property_records(&output.stdout);
    assert_eq!(records[0]["status"], "unsupported");
    assert!(
        records[0]["reason"]
            .as_str()
            .is_some_and(|reason| reason.contains("Nautilus.Stats.quantile_vec"))
    );
}

#[test]
fn linker_symbol_with_wrong_signature_cannot_receive_nautilus_contract() {
    let dir = package(
        r#"module Risk.Proofs
import Nautilus.Stats (quantile_vec)
def observations() -> tensor[3, f32] =
  (to_tensor([3.0, 1.0, 2.0], f32) : tensor[3, f32])
@property wrong_signature forall(p: f32, q: f32)
  where 0.0 <= p, p <= q, q <= 1.0:
  quantile_vec(observations(), p) <= quantile_vec(observations(), q)
  with contract = "std.quantile.monotonicity"
"#,
    );
    write(
        &dir.path().join("nautilus/src/stats.ch"),
        r#"module Nautilus.Stats
export (quantile_vec)
def quantile_vec[n](v: &tensor[n, f32], q: f32) -> f64 = cast(q, f64)
"#,
    );

    let output = prove(&dir);
    assert_eq!(output.status.code(), Some(2));
    let records = property_records(&output.stdout);
    assert_eq!(records[0]["status"], "unsupported");
}

#[test]
fn concrete_tensor_shape_cannot_receive_generic_nautilus_contract() {
    let dir = package(
        r#"module Risk.Proofs
import Nautilus.Stats (quantile_vec)
def observations() -> tensor[3, f32] =
  (to_tensor([3.0, 1.0, 2.0], f32) : tensor[3, f32])
@property concrete_shape_spoof forall(p: f32, q: f32)
  where 0.0 <= p, p <= q, q <= 1.0:
  quantile_vec(observations(), p) <= quantile_vec(observations(), q)
  with contract = "std.quantile.monotonicity"
"#,
    );
    write(
        &dir.path().join("nautilus/src/stats.ch"),
        r#"module Nautilus.Stats
export (quantile_vec)
def quantile_vec(v: &tensor[3, f32], q: f32) -> f32 = neg(q)
"#,
    );

    let output = prove(&dir);
    assert_eq!(output.status.code(), Some(2));
    let records = property_records(&output.stdout);
    assert_eq!(records.len(), 1, "records={records:?}");
    assert_eq!(records[0]["status"], "unsupported");
}

#[test]
fn wrong_rank_tensor_cannot_receive_nautilus_contract() {
    let dir = package(
        r#"module Risk.Proofs
import Nautilus.Stats (quantile_vec)
def observations() -> tensor[2, 2, f32] =
  (to_tensor([[3.0, 1.0], [2.0, 4.0]], f32) : tensor[2, 2, f32])
@property wrong_rank_spoof forall(p: f32, q: f32)
  where 0.0 <= p, p <= q, q <= 1.0:
  quantile_vec(observations(), p) <= quantile_vec(observations(), q)
  with contract = "std.quantile.monotonicity"
"#,
    );
    write(
        &dir.path().join("nautilus/src/stats.ch"),
        r#"module Nautilus.Stats
export (quantile_vec)
def quantile_vec[n](v: &tensor[n, n, f32], q: f32) -> f32 = neg(q)
"#,
    );

    let output = prove(&dir);
    assert_eq!(output.status.code(), Some(2));
    let records = property_records(&output.stdout);
    assert_eq!(records.len(), 1, "records={records:?}");
    assert_eq!(records[0]["status"], "unsupported");
}

#[test]
fn quantile_contract_does_not_couple_different_datasets() {
    let dir = package(
        r#"module Risk.Proofs
import Nautilus.Stats (quantile_vec)
def observations_a() -> tensor[3, f32] =
  (to_tensor([3.0, 1.0, 2.0], f32) : tensor[3, f32])
def observations_b() -> tensor[3, f32] =
  (to_tensor([30.0, 10.0, 20.0], f32) : tensor[3, f32])
@property unrelated_datasets forall(p: f32, q: f32)
  where 0.0 <= p, p <= q, q <= 1.0:
  quantile_vec(observations_a(), p) <= quantile_vec(observations_b(), q)
  with contract = "std.quantile.monotonicity"
"#,
    );
    let output = prove(&dir);
    assert_eq!(output.status.code(), Some(2));
    let records = property_records(&output.stdout);
    assert_eq!(records[0]["status"], "unsupported");
    assert!(
        records[0]["reason"]
            .as_str()
            .is_some_and(|reason| reason.contains("same compiler-bound dataset"))
    );
}
