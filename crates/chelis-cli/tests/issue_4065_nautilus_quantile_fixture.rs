//! Acceptance surface for chelis#4065.
//!
//! `examples/nautilus_quantile_contract` vendors `Nautilus.Stats.quantile_vec`.
//! Its lower order-statistic index is `q * (n - 1)`, which is fractional for
//! almost every `q`. A checked `cast` of a fractional float to `i64` traps
//! `Domain` under [04-NUM-14], so the index must use the truncating cast
//! [05-OP-6]. These tests evaluate the vendored function on the example's
//! five-element data set, where sorted `[1, 2, 3, 5, 8]` gives `pos = 4q`, and
//! pin the linear-interpolation value `x[lo] + frac * (x[lo + 1] - x[lo])`
//! bit for bit. Each expected value is the f32 result of that formula with
//! every operation rounded to f32, the order `quantile_vec` evaluates it in.

use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

fn write(path: &std::path::Path, contents: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create fixture directory");
    }
    std::fs::write(path, contents).expect("write fixture");
}

const NAUTILUS_REEF: &str =
    include_str!("../../../examples/nautilus_quantile_contract/fixtures/nautilus/reef.toml");
const NAUTILUS_STATS: &str =
    include_str!("../../../examples/nautilus_quantile_contract/fixtures/nautilus/src/stats.ch");
const EXAMPLE_PROOFS: &str =
    include_str!("../../../examples/nautilus_quantile_contract/src/proofs.ch");

/// A package laid out like the example: the vendored Nautilus fixture as a
/// path dependency and `source` as `src/proofs.ch`.
fn package(source: &str) -> tempfile::TempDir {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join("quantile-contract");
    let dep = dir.path().join("nautilus");
    write(
        &root.join("reef.toml"),
        &format!(
            r#"[package]
name = "quantile-contract"
version = "0.1.0"
compiler = "={}"
module_prefix = "QuantileContract"

[dependencies]
nautilus = {{ path = "../nautilus" }}
"#,
            env!("CARGO_PKG_VERSION")
        ),
    );
    write(&dep.join("reef.toml"), NAUTILUS_REEF);
    write(&dep.join("src/stats.ch"), NAUTILUS_STATS);
    write(&root.join("src/proofs.ch"), source);
    dir
}

fn proofs_path(dir: &tempfile::TempDir) -> std::path::PathBuf {
    dir.path().join("quantile-contract/src/proofs.ch")
}

/// `quantile_vec` over the example's observations at the level spelled `q`.
fn quantile_source(q: &str) -> String {
    format!(
        r#"module QuantileContract.Proofs
import Nautilus.Stats (quantile_vec)
def observations() -> tensor[5, f32] = (to_tensor([8.0, 1.0, 5.0, 3.0, 2.0], f32) : tensor[5, f32])
level = quantile_vec(observations(), {q})
"#
    )
}

fn eval(dir: &tempfile::TempDir) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "eval",
            "--json",
            "--file",
            proofs_path(dir).to_str().expect("utf8 path"),
        ])
        .output()
        .expect("eval linked quantile")
}

/// Evaluates `quantile_vec` at `q` and asserts the f32 result has `bits`.
fn assert_quantile(q: &str, bits: u32) {
    let dir = package(&quantile_source(q));
    let output = eval(&dir);
    assert!(
        output.status.success(),
        "q={q}\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout).expect("eval JSON");
    let level = result["roots"]
        .as_array()
        .and_then(|roots| roots.iter().find(|root| root["name"] == "level"))
        .expect("level root");
    assert_eq!(
        level["value"],
        serde_json::json!({
            "type": "scalar",
            "value": { "dtype": "f32", "bits": format!("{bits:08x}") },
        }),
        "q={q} expected {} level={level}",
        f32::from_bits(bits)
    );
}

#[test]
fn fractional_index_interpolates_between_order_statistics() {
    // q = 0.3: pos = 1.2000000477, lo = 1, frac = 0.2000000477, so the result
    // is 2 + frac * (3 - 2) = f32 2.2.
    assert_quantile("0.3f32", 0x400c_cccd);
    // q = 0.9: pos = 3.5999999046, lo = 3, frac = 0.5999999046, so the result
    // is 5 + frac * (8 - 5) = 6.7999997139, one ulp below f32 6.8.
    assert_quantile("0.9f32", 0x40d9_9999);
}

#[test]
fn integral_index_selects_the_order_statistic_exactly() {
    // q = 0 and q = 1 are the minimum and maximum; at q = 1 the upper index
    // clamps to the last element. q = 0.25 and q = 0.5 land on index 1 and 2.
    assert_quantile("0.0f32", 1.0f32.to_bits());
    assert_quantile("0.25f32", 2.0f32.to_bits());
    assert_quantile("0.5f32", 3.0f32.to_bits());
    assert_quantile("1.0f32", 8.0f32.to_bits());
}

#[test]
fn out_of_range_level_clamps_to_the_extremes() {
    assert_quantile("-0.5f32", 1.0f32.to_bits());
    assert_quantile("1.5f32", 8.0f32.to_bits());
}

#[test]
fn nan_level_traps_domain_rather_than_returning_a_value() {
    // A NaN level survives the clamp, so `pos` is NaN. Truncating a
    // non-finite value has no integer meaning and traps `Domain` [05-OP-6].
    let dir = package(&quantile_source("div(0.0f32, 0.0f32)"));
    let output = eval(&dir);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "stdout={}\nstderr={stderr}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(stderr.contains("numeric trap: domain"), "stderr={stderr}");
}

#[cfg(feature = "chelis-prove")]
#[test]
fn example_property_passes_the_fuzz_tier_on_the_implementation() {
    // `--tier fuzz-only` evaluates the bound implementation at sampled
    // levels, so a trap at any fractional index fails the property.
    let dir = package(EXAMPLE_PROOFS);
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env_remove("CHELIS_BEACON_BIN")
        .args([
            "prove",
            proofs_path(&dir).to_str().expect("utf8 path"),
            "--json",
            "--tier",
            "fuzz-only",
        ])
        .output()
        .expect("run prove");
    let transcript = format!(
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.status.success(), "{transcript}");
    let records: Vec<Value> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|record| record["kind"] == "property")
        .collect();
    assert_eq!(records.len(), 1, "{transcript}");
    assert_eq!(records[0]["name"], "example_quantiles_are_monotone");
    assert_eq!(records[0]["status"], "passed", "{transcript}");
    assert_eq!(records[0]["proof_tier"], "fuzz", "{transcript}");
}
