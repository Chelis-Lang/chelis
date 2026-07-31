//! chelis#977: constraint-directed fuzz generation for narrow scalar guards.
#![cfg(feature = "smt")]

use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

fn prove(source: &str, seed: u64) -> (i32, Value) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("risk.ch");
    std::fs::write(&path, source).expect("write fixture");
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "prove",
            path.to_str().expect("utf-8 path"),
            "--json",
            "--tier",
            "fuzz-only",
            "--samples",
            "32",
            "--seed",
            &seed.to_string(),
        ])
        .output()
        .expect("run prove");
    let record = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|value| value["kind"] == "property")
        .unwrap_or_else(|| {
            panic!(
                "property record missing: {}",
                String::from_utf8_lossy(&output.stdout)
            )
        });
    (output.status.code().unwrap_or(-1), record)
}

const NARROW_VAR_GUARD: &str = "module Risk.Guards
@property confidence_tail_order forall(alpha1: f32, alpha2: f32)
where (alpha1 > 0.99), (alpha1 < alpha2), (alpha2 < 1.0):
  ((1.0 - alpha2) < (1.0 - alpha1))
";

#[test]
fn narrow_var_es_order_is_non_vacuous_and_reproducible_across_seeds() {
    for seed in [0, 1, 42, 9_999] {
        let (code, first) = prove(NARROW_VAR_GUARD, seed);
        let (_, second) = prove(NARROW_VAR_GUARD, seed);
        assert_eq!(code, 0, "seed {seed}: {first}");
        let mut first_repro = first.clone();
        let mut second_repro = second.clone();
        first_repro
            .as_object_mut()
            .expect("record")
            .remove("source");
        second_repro
            .as_object_mut()
            .expect("record")
            .remove("source");
        assert_eq!(
            first_repro, second_repro,
            "fixed seed must reproduce all non-path evidence"
        );
        assert_eq!(first["status"], "passed");
        assert_eq!(first["composite_verdict"], "fuzz_validated");
        assert_eq!(first["sampling_method"], "constraint_directed");
        assert_eq!(first["accepted_samples"], 32);
        assert_eq!(first["attempted_samples"], 32);
        assert_eq!(first["rejected_samples"], 0);
        assert_eq!(
            first["assumptions"][0]["non_vacuity"]["evidence"]["sampling_method"],
            "constraint_directed"
        );
    }
}

#[test]
fn corrupt_var_es_order_finds_an_in_domain_witness() {
    let corrupt = NARROW_VAR_GUARD.replace(
        "((1.0 - alpha2) < (1.0 - alpha1))",
        "((1.0 - alpha1) < (1.0 - alpha2))",
    );
    let (code, record) = prove(&corrupt, 42);
    assert_eq!(code, 1, "{record}");
    assert_eq!(record["status"], "failed");
    assert_eq!(record["sampling_method"], "constraint_directed");
    let alpha1 = record["counterexample"]["alpha1"]
        .as_f64()
        .expect("alpha1 witness");
    let alpha2 = record["counterexample"]["alpha2"]
        .as_f64()
        .expect("alpha2 witness");
    assert!(0.99 < alpha1 && alpha1 < alpha2 && alpha2 < 1.0, "{record}");
}

#[test]
fn inconsistent_scalar_guard_fails_closed_without_sampling() {
    let source = "module Risk.Invalid
@property impossible forall(alpha: f32)
where (alpha > 0.99), (alpha < 0.5):
  (alpha == alpha)
";
    let (code, record) = prove(source, 7);
    assert_eq!(code, 2, "{record}");
    assert_eq!(record["status"], "unsupported");
    assert_ne!(record["composite_verdict"], "fuzz_validated");
    assert_eq!(record["sampling_method"], "constraint_directed");
    assert_eq!(record["accepted_samples"], 0);
    assert!(
        record["reason"]
            .as_str()
            .unwrap_or_default()
            .contains("inconsistent")
    );
}

#[test]
fn unsupported_scalar_guard_shape_fails_closed_without_sampling() {
    let source = "module Risk.Unsupported
@property coupled_sum forall(alpha1: f32, alpha2: f32)
where ((alpha1 + alpha2) < 1.0):
  (alpha1 == alpha1)
";
    let (code, record) = prove(source, 7);
    assert_eq!(code, 2, "{record}");
    assert_eq!(record["status"], "unsupported");
    assert_ne!(record["composite_verdict"], "fuzz_validated");
    assert_eq!(record["sampling_method"], "constraint_directed");
    assert_eq!(record["accepted_samples"], 0);
    assert!(
        record["reason"]
            .as_str()
            .unwrap_or_default()
            .contains("unsupported")
    );
}
