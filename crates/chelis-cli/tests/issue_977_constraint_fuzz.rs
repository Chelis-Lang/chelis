//! chelis#977: constraint-directed fuzz generation for narrow scalar guards.
use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

fn prove(source: &str, seed: u64) -> (i32, Value) {
    prove_with_kind(source, seed, false)
}

fn prove_with_attempt_limit(source: &str, samples: &str, max_attempts: &str) -> (i32, Value) {
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
            samples,
            "--max-attempts",
            max_attempts,
            "--seed",
            "42",
        ])
        .output()
        .expect("run prove");
    let record = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|value| value["kind"] == "property")
        .expect("property record");
    (output.status.code().unwrap_or(-1), record)
}

fn prove_with_kind(source: &str, seed: u64, deep: bool) -> (i32, Value) {
    let dir = tempdir().expect("tempdir");
    let surf_path = dir.path().join("risk.ch");
    std::fs::write(&surf_path, source).expect("write fixture");
    let path = if deep {
        let deep_path = dir.path().join("risk.dp");
        let output = Command::cargo_bin("chelis")
            .expect("chelis binary")
            .args(["deep", surf_path.to_str().expect("utf-8 path")])
            .output()
            .expect("desugar fixture");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        std::fs::write(&deep_path, output.stdout).expect("write deep fixture");
        deep_path
    } else {
        surf_path
    };
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

#[test]
fn default_non_smt_build_uses_the_shared_constraint_sampler() {
    let (code, record) = prove(NARROW_VAR_GUARD, 42);
    assert_eq!(code, 0, "{record}");
    assert_eq!(record["sampling_method"], "constraint_directed");
    assert_eq!(record["accepted_samples"], 32);
    assert_eq!(record["attempted_samples"], 32);
    assert_eq!(record["rejected_samples"], 0);
}

#[test]
fn f32_and_f64_interval_bounds_are_not_clipped_to_the_uniform_range() {
    for (ty, suffix) in [("f32", ""), ("f64", "f64")] {
        let source = format!(
            "module Risk.Outside\n@property outside forall(x: {ty})\nwhere (x > 20.0{suffix}), (x < 30.0{suffix}):\n  (x == x)\n"
        );
        let (code, record) = prove(&source, 5);
        assert_eq!(code, 0, "{ty}: {record}");
        assert_eq!(record["sampling_method"], "constraint_directed");
        assert_eq!(record["accepted_samples"], 32);
        assert_eq!(record["rejected_samples"], 0);
    }
}

#[test]
fn exhausted_run_preserves_the_accepted_sample_count() {
    let (code, record) = prove_with_attempt_limit(NARROW_VAR_GUARD, "8", "3");
    assert_eq!(code, 3, "{record}");
    assert_eq!(record["status"], "error");
    assert_eq!(record["proof_tier"], "none");
    assert_eq!(record["accepted_samples"], 3);
    assert_eq!(record["attempted_samples"], 3);
    assert_eq!(record["rejected_samples"], 0);
    assert_eq!(record["samples"], 0, "terminal verdict remains non-green");
}

#[test]
fn wide_finite_f64_interval_is_representable_without_span_overflow() {
    let source = "module Risk.Wide64
@property wide forall(x: f64)
where (x > -1e308f64), (x < 1e308f64):
  (x == x)
";
    let (code, record) = prove(source, 29);
    assert_eq!(code, 0, "{record}");
    assert_eq!(record["accepted_samples"], 32);
    assert_eq!(record["rejected_samples"], 0);
}

#[test]
fn one_sided_extreme_intervals_clamp_windows_to_the_binder_range() {
    for (module, ty, lower, upper) in [
        ("High32", "f32", "3.3e38", ""),
        ("Low32", "f32", "", "-3.3e38"),
        ("High64", "f64", "1.7e308f64", ""),
        ("Low64", "f64", "", "-1.7e308f64"),
    ] {
        let guard = if lower.is_empty() {
            format!("x < {upper}")
        } else {
            format!("x > {lower}")
        };
        let source = format!(
            "module Risk.{module}\n@property extreme forall(x: {ty})\nwhere ({guard}):\n  (x == x)\n"
        );
        let (code, record) = prove(&source, 31);
        assert_eq!(code, 0, "{module}: {record}");
        assert_eq!(record["sampling_method"], "constraint_directed");
        assert_eq!(record["accepted_samples"], 32);
        assert_eq!(record["attempted_samples"], 32);
        assert_eq!(record["rejected_samples"], 0);
    }
}

#[test]
fn negative_literals_and_reversed_comparisons_are_directed() {
    let source = "module Risk.Negative
@property negative forall(x: f32)
where (-1.0 < x), (0.0 > x):
  (x < 0.0)
";
    let (code, record) = prove(source, 11);
    assert_eq!(code, 0, "{record}");
    assert_eq!(record["sampling_method"], "constraint_directed");
    assert_eq!(record["accepted_samples"], 32);
    assert_eq!(record["rejected_samples"], 0);
}

#[test]
fn non_strict_order_allows_equal_singleton_domains() {
    for (ty, suffix) in [("f32", ""), ("f64", "f64")] {
        let source = format!(
            "module Risk.Equal\n@property equal forall(x: {ty}, y: {ty})\nwhere (x >= 1.0{suffix}), (x <= 1.0{suffix}), (y >= 1.0{suffix}), (y <= 1.0{suffix}), (x <= y):\n  (x == y)\n"
        );
        let (code, record) = prove(&source, 13);
        assert_eq!(code, 0, "{ty}: {record}");
        assert_eq!(record["accepted_samples"], 32);
        assert_eq!(record["rejected_samples"], 0);
    }
}

#[test]
fn narrow_f64_interval_uses_representable_successors_not_fixed_epsilon() {
    let source = "module Risk.Narrow64
@property narrow forall(x: f64, y: f64)
where (x > 0.9999999999999997f64), (x < y), (y < 1.0000000000000002f64):
  (x < y)
";
    let (code, record) = prove(source, 17);
    assert_eq!(code, 0, "{record}");
    assert_eq!(record["accepted_samples"], 32);
    assert_eq!(record["attempted_samples"], 32);
    assert_eq!(record["rejected_samples"], 0);
}

#[test]
fn narrow_f32_interval_uses_representable_successors() {
    let source = "module Risk.Narrow32
@property narrow forall(x: f32, y: f32)
where (x > 0.9999998), (x < y), (y < 1.0000002):
  (x < y)
";
    let (code, record) = prove(source, 19);
    assert_eq!(code, 0, "{record}");
    assert_eq!(record["accepted_samples"], 32);
    assert_eq!(record["attempted_samples"], 32);
    assert_eq!(record["rejected_samples"], 0);
}

#[test]
fn interval_without_enough_machine_values_fails_closed() {
    let source = "module Risk.NoSlots
@property no_slots forall(x: f32, y: f32)
where (x > 0.9999999), (x < y), (y < 1.0):
  (x < y)
";
    let (code, record) = prove(source, 21);
    assert_eq!(code, 2, "{record}");
    assert_eq!(record["status"], "unsupported");
    assert_eq!(record["sampling_method"], "constraint_directed");
    assert_eq!(record["accepted_samples"], 0);
    assert_ne!(record["composite_verdict"], "fuzz_validated");
}

#[test]
fn deep_constraint_sampler_matches_surf_evidence() {
    let (surf_code, mut surf) = prove_with_kind(NARROW_VAR_GUARD, 23, false);
    let (deep_code, mut deep) = prove_with_kind(NARROW_VAR_GUARD, 23, true);
    assert_eq!(surf_code, 0, "{surf}");
    assert_eq!(deep_code, 0, "{deep}");
    for record in [&mut surf, &mut deep] {
        record.as_object_mut().expect("record").remove("source");
        record.as_object_mut().expect("record").remove("goal");
    }
    assert_eq!(surf, deep);
}
