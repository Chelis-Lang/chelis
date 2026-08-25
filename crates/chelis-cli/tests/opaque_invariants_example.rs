//! W6 acceptance oracle for the worked opaque-invariants examples.
//!
//! Two executable files (Example Corpus Policy):
//!
//! - `examples/opaque_invariants.ch` (executable): the `Probability`
//!   unit-interval type. Every part runs clean -- `fmt --check`, `check`
//!   (score 1), `eval`/`build` (no owed roots), and `prove` (three SMT-tier
//!   producer obligations plus an injected property).
//! - `examples/opaque_invariants_simplex.ch` (executable): the `Simplex`
//!   tolerance-band type with a `sum`-over-a-tensor-field invariant. The
//!   invariant predicate is declaration metadata consumed only by `chelis
//!   prove`; it is never lowered to runtime IR, so the runtime IR audit skips
//!   it and the file now `eval`/`build`s cleanly. Its top-level `eps` value is
//!   an automatic owed root under [05-OBS-7]. Its producer obligation
//!   discharges at Tier C (fuzz) and a
//!   `Simplex` binder is served by constructor-based generation without
//!   starving (the D-STARVE acceptance probe). Promoted from
//!   `examples/illustrative/` once that audit stopped rejecting the
//!   declaration metadata.
//!
//! This pins both files on the RFC surface (`opaque_invariants_rfc.md`
//! D-PRODUCER, D-OBLIG, D-TIERB, D-INJECT, D-STARVE): the exact obligation
//! names, statuses, and proof tiers the docs page transcribes.
//!
//! The obligation surface compiles only under the `smt` feature (the Tier B
//! lowering lives in the optional `chelis-prove` dependency), so the prove
//! assertions are `#[cfg(feature = "smt")]`; the `check`-clean assertions run
//! unconditionally. Run the full oracle with:
//!   `cargo nextest run -p chelis-cli --features smt
//!     --test opaque_invariants_example`
//! with `LD_LIBRARY_PATH` set to the uv python lib (see AGENTS.md).

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;
use std::path::PathBuf;
use tempfile::tempdir;

fn example_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(rel)
        .canonicalize()
        .unwrap_or_else(|_| panic!("{rel} should exist"))
}

fn probability_example() -> PathBuf {
    example_path("../../examples/opaque_invariants.ch")
}

fn simplex_example() -> PathBuf {
    example_path("../../examples/opaque_invariants_simplex.ch")
}

fn run_json_check(path: &PathBuf) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .arg("check")
        .arg(path)
        .output()
        .expect("run check");
    assert!(
        output.status.success(),
        "check exited non-zero: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("check emits JSON")
}

fn assert_check_clean(path: &PathBuf) {
    let json = run_json_check(path);
    assert_eq!(json["score"].as_f64().unwrap(), 1.0, "{path:?} score is 1");
    assert_eq!(
        json["errors"].as_array().unwrap().len(),
        0,
        "{path:?} has no check errors"
    );
    assert_eq!(
        json["untyped_nodes"].as_u64().unwrap(),
        0,
        "{path:?} every node is typed"
    );
    assert_eq!(
        json["unresolved_names"].as_array().unwrap().len(),
        0,
        "{path:?} no unresolved names"
    );
}

/// Both worked examples survive `chelis check` cleanly: score 1, no errors,
/// no untyped nodes, no unresolved names. This runs without the `smt`
/// feature. `check` also enforces the style gate (`fmt --check` plus the
/// blocking lint set), so a green check certifies canonical formatting too.
#[test]
fn both_examples_check_clean_with_score_one() {
    assert_check_clean(&probability_example());
    assert_check_clean(&simplex_example());
}

/// `eval --file` succeeds on both examples. `Probability` has no owed roots,
/// so it emits the def-only warning and produces no value. `Simplex` has the
/// top-level `eps` binding, which [05-OBS-7] requires it to realize.
/// This is the regression guard for the runtime IR audit fix: before it, the
/// `Simplex` tensor-field invariant tripped `assert_ir_typed`
/// ("shape-sensitive IR app nodes must carry explicit type metadata before
/// lowering") because the audit walked the declaration metadata.
fn assert_eval_without_roots(path: &PathBuf) {
    Command::cargo_bin("chelis")
        .expect("binary")
        .arg("eval")
        .arg("--file")
        .arg(path)
        .assert()
        .success()
        .stdout(predicate::str::is_empty())
        .stderr(predicate::str::contains(
            "input contains only def declarations; nothing to evaluate",
        ));
}

#[test]
fn both_examples_eval_clean_with_manifested_roots() {
    assert_eval_without_roots(&probability_example());
    Command::cargo_bin("chelis")
        .expect("binary")
        .arg("eval")
        .arg("--file")
        .arg(simplex_example())
        .assert()
        .success()
        .stdout("eps = 0.0001\n")
        .stderr(predicate::str::is_empty());
}

/// `build` (default C target) succeeds on both examples. `Probability` emits
/// an object without an observation entry point, while `Simplex` emits an
/// executable entry point for `eps`. Its tensor-field invariant predicate is
/// declaration metadata for `chelis prove`; it is never lowered to runtime IR.
fn assert_build_clean(path: &PathBuf) {
    let out_dir = tempdir().expect("tempdir");
    Command::cargo_bin("chelis")
        .expect("binary")
        .arg("build")
        .arg(path)
        .arg("-o")
        .arg(out_dir.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("Wrote"));
}

#[test]
fn both_examples_build_clean() {
    assert_build_clean(&probability_example());
    assert_build_clean(&simplex_example());
}

#[cfg(feature = "smt")]
mod prove_oracle {
    use super::*;

    /// Run `chelis prove --json` on `path`, returning (exit, records).
    fn prove_json(path: &PathBuf) -> (i32, Vec<Value>) {
        let output = Command::cargo_bin("chelis")
            .expect("binary")
            .arg("prove")
            .arg(path)
            .arg("--json")
            .output()
            .expect("run prove");
        let code = output.status.code().unwrap_or(-1);
        let records = String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .collect();
        (code, records)
    }

    fn obligation<'a>(records: &'a [Value], name: &str) -> &'a Value {
        records
            .iter()
            .find(|r| {
                r.get("kind").and_then(Value::as_str) == Some("obligation")
                    && r.get("name").and_then(Value::as_str) == Some(name)
            })
            .unwrap_or_else(|| panic!("obligation `{name}` is present"))
    }

    fn property<'a>(records: &'a [Value], name: &str) -> &'a Value {
        records
            .iter()
            .find(|r| {
                r.get("kind").and_then(Value::as_str) == Some("property")
                    && r.get("name").and_then(Value::as_str) == Some(name)
            })
            .unwrap_or_else(|| panic!("property `{name}` is present"))
    }

    fn summary(records: &[Value]) -> &Value {
        records
            .iter()
            .find(|r| r.get("kind").and_then(Value::as_str) == Some("summary"))
            .expect("summary record")
    }

    fn obligation_count(records: &[Value]) -> usize {
        records
            .iter()
            .filter(|r| r.get("kind").and_then(Value::as_str) == Some("obligation"))
            .count()
    }

    /// The oracle for the executable example: the three `Probability`
    /// producers discharge at SMT tier over the reals, and the injected
    /// property passes.
    #[test]
    fn probability_example_prove_discharges_every_obligation() {
        let (code, records) = prove_json(&probability_example());
        assert_eq!(code, 0, "prove succeeds (exit 0)");
        assert_eq!(obligation_count(&records), 3, "three producer obligations");

        // The guard-then-Option base constructor plus two update-shaped
        // producers (the D-SOUND inductive step). All discharge at SMT tier.
        for (name, producer) in [
            ("invariant:Probability:probability", "probability"),
            ("invariant:Probability:scale", "scale"),
            ("invariant:Probability:combine", "combine"),
        ] {
            let ob = obligation(&records, name);
            assert_eq!(ob["status"], "passed", "{name} passes");
            assert_eq!(ob["proof_tier"], "smt", "{name} proves at SMT tier");
            assert_eq!(ob["obligation_kind"], "invariant_producer");
            assert_eq!(ob["source_type"], "Probability");
            assert_eq!(ob["producer"], producer);
            assert_eq!(ob["arith_model"], "real", "SMT proofs are over the reals");
        }

        // The injected property relies on assumption injection (D-INJECT) of
        // the Probability invariant on its binder.
        assert_eq!(
            property(&records, "prob_value_in_unit_interval")["status"],
            "passed"
        );

        let s = summary(&records);
        assert_eq!(s["obligations"], 3, "three producer obligations");
        assert_eq!(s["total"], 4, "one property + three obligations");
        assert_eq!(s["passed"], 4, "all four pass");
        assert_eq!(s["failed"], 0);
        assert_eq!(s["errors"], 0);
        assert_eq!(s["unsupported"], 0);
    }

    /// The oracle for the illustrative Simplex example: the producer
    /// obligation discharges at Tier C (fuzz, no arith_model), and the
    /// Simplex-binder property is served by constructor-based generation
    /// without starving (the D-STARVE acceptance probe).
    #[test]
    fn simplex_example_prove_discharges_obligation_and_generates_binder() {
        let (code, records) = prove_json(&simplex_example());
        assert_eq!(code, 0, "prove succeeds (exit 0)");
        assert_eq!(obligation_count(&records), 1, "one producer obligation");

        let ob = obligation(&records, "invariant:Simplex:make_simplex");
        assert_eq!(ob["status"], "passed");
        assert_eq!(ob["proof_tier"], "fuzz", "Tier C (sum-over-tensor body)");
        assert_eq!(ob["source_type"], "Simplex");
        assert_eq!(ob["producer"], "make_simplex");
        assert!(
            ob.get("arith_model").is_none(),
            "Tier C obligation carries no arith_model"
        );

        // The D-STARVE probe: the tolerance band is measure-near-zero under
        // independent component sampling, so a `Simplex` binder is served by
        // constructor-based generation. Generation succeeding (a "passed"
        // property, not "unsupported"/starvation) is the assertion.
        assert_eq!(
            property(&records, "simplex_binder_is_generated")["status"],
            "passed",
            "Simplex binder generated without starving"
        );

        let s = summary(&records);
        assert_eq!(s["obligations"], 1);
        assert_eq!(s["passed"], 2, "one property + one obligation");
        assert_eq!(s["errors"], 0);
        assert_eq!(s["unsupported"], 0);
    }

    /// Determinism: two runs on the executable example produce identical
    /// records.
    #[test]
    fn probability_example_prove_is_deterministic() {
        let (_c1, r1) = prove_json(&probability_example());
        let (_c2, r2) = prove_json(&probability_example());
        assert_eq!(r1, r2, "prove is deterministic on the example");
    }
}
