//! W3+W4 acceptance oracle: derived producer obligations through
//! `chelis prove` (RFC D-PRODUCER, D-OBLIG, D-TIERB, D-PARITY, D-STARVE).
//!
//! These tests exercise the OBLIGATION surface, which is built only under
//! the `smt` feature (the obligation collection + Tier B lowering live in
//! the optional `chelis-prove` dependency, activated by `smt`). Every test
//! is `#[cfg(feature = "smt")]`. Without the feature the obligation path is
//! not compiled, mirroring the existing Tier B gating.
//!
//! Run: `cargo nextest run -p chelis-cli --features smt
//!   --test prove_invariant_obligations`
//! with `LD_LIBRARY_PATH` set to the uv python lib (see AGENTS.md).
//!
//! `CHELIS_STYLE_GATE_DISABLE=1` is set so the ad-hoc opaque/invariant
//! fixtures are not blocked by the opaque-domain-construction lint before
//! the prove pipeline runs (their construction discipline is the checker's
//! and the lint's own concern, covered elsewhere).
#![cfg(feature = "smt")]

use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

/// Run `chelis prove --json` on `source`, returning (exit_code, records).
fn prove_json(source: &str, extra: &[&str]) -> (i32, Vec<Value>) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("mod.ch");
    std::fs::write(&path, source).expect("write");
    let mut cmd = Command::cargo_bin("chelis").expect("binary");
    cmd.env("CHELIS_STYLE_GATE_DISABLE", "1");
    cmd.arg("prove").arg(&path).arg("--json");
    for a in extra {
        cmd.arg(a);
    }
    let output = cmd.output().expect("run");
    let code = output.status.code().unwrap_or(-1);
    let records = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .collect();
    (code, records)
}

fn obligations(records: &[Value]) -> Vec<&Value> {
    records
        .iter()
        .filter(|r| r.get("kind").and_then(Value::as_str) == Some("obligation"))
        .collect()
}

fn summary(records: &[Value]) -> &Value {
    records
        .iter()
        .find(|r| r.get("kind").and_then(Value::as_str) == Some("summary"))
        .expect("summary record")
}

const FLAGSHIP: &str = "module Stats.Prob
export (probability)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Option[Probability] =
  if x >= 0.0 && x <= 1.0 then Some(Probability { value: x }) else None
";

#[test]
fn flagship_guarded_option_proves_at_smt_tier() {
    let (code, records) = prove_json(FLAGSHIP, &[]);
    assert_eq!(code, 0, "flagship obligation passes");
    let obs = obligations(&records);
    assert_eq!(obs.len(), 1);
    let ob = obs[0];
    assert_eq!(ob["name"], "invariant:Probability:probability");
    assert_eq!(ob["status"], "passed");
    assert_eq!(ob["proof_tier"], "smt");
    assert_eq!(ob["obligation_kind"], "invariant_producer");
    assert_eq!(ob["source_type"], "Probability");
    assert_eq!(ob["producer"], "probability");
    assert_eq!(ob["arith_model"], "real");
    assert_eq!(summary(&records)["obligations"], 1);
}

#[test]
fn clamping_constructor_proves_at_smt_tier() {
    let source = "module Stats.Prob
export (clamp_prob)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def clamp_prob(x: f32) -> Probability =
  Probability { value: if x >= 0.0 then (if x <= 1.0 then x else 1.0) else 0.0 }
";
    let (code, records) = prove_json(source, &[]);
    assert_eq!(code, 0);
    let obs = obligations(&records);
    assert_eq!(obs[0]["status"], "passed");
    assert_eq!(obs[0]["proof_tier"], "smt");
}

#[test]
fn non_validating_constructor_fails_with_counterexample() {
    let source = "module Stats.Prob
export (bad_prob)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def bad_prob(x: f32) -> Option[Probability] =
  if x >= 0.0 then Some(Probability { value: x }) else None
";
    let (code, records) = prove_json(source, &[]);
    assert_eq!(code, 1, "violating constructor fails => exit 1");
    let obs = obligations(&records);
    assert_eq!(obs[0]["status"], "failed");
    assert!(obs[0].get("counterexample").is_some(), "has counterexample");
}

#[test]
fn none_always_passes_vacuously() {
    // A producer that always returns None: the obligation is vacuously
    // true (None branch => true). Documented + asserted (D-OBLIG).
    let source = "module Stats.Prob
export (never)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def never(x: f32) -> Option[Probability] = None
";
    let (code, records) = prove_json(source, &[]);
    assert_eq!(code, 0, "None-always passes vacuously");
    assert_eq!(obligations(&records)[0]["status"], "passed");
}

#[test]
fn covered_or_rejected_list_container_errors() {
    let source = "module Stats.Prob
export (many)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def many(x: f32) -> List[Probability] = Cons(Probability { value: x }, Nil)
";
    let (code, records) = prove_json(source, &[]);
    assert_eq!(code, 3, "unsupported container => error => exit 3");
    let obs = obligations(&records);
    assert_eq!(obs[0]["status"], "error");
    let reason = obs[0]["reason"].as_str().unwrap();
    assert!(reason.contains("many"), "names the producer: {reason}");
    assert!(reason.contains("List"), "names the container: {reason}");
}

#[test]
fn unannotated_producer_has_obligation_via_inferred_return() {
    // No `-> Probability`; the inferred return must place it in the set.
    let source = "module Stats.Prob
export (probability)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def probability(x: f32) = Probability { value: x }
";
    let (_code, records) = prove_json(source, &[]);
    let obs = obligations(&records);
    assert_eq!(obs.len(), 1, "inferred-return producer is in the set");
    assert_eq!(obs[0]["producer"], "probability");
}

#[test]
fn caller_receives_signature_is_declaration_error() {
    let source = "module Stats.Prob
export (with_prob)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def with_prob(f: (Probability) -> f32) -> f32 = f(default_prob())
def default_prob() -> Probability = Probability { value: 0.0 }
";
    let (code, records) = prove_json(source, &[]);
    assert_eq!(code, 3, "caller-receives sig => declaration error");
    let obs = obligations(&records);
    assert!(
        obs.iter()
            .any(|o| o["status"] == "error"
                && o["reason"].as_str().unwrap_or("").contains("with_prob")),
        "names the offending function"
    );
}

#[test]
fn module_receives_record_param_is_legal() {
    let source = "module Stats.Prob
export (read_prob)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def read_prob(p: Probability) -> f32 = prob_value(p)
def prob_value(p: Probability) -> f32 = p.value
";
    let (code, records) = prove_json(source, &[]);
    assert_eq!(code, 0, "module-receives is legal, no obligation error");
    // read_prob returns f32 (not a producer) => no obligation record.
    assert!(obligations(&records).is_empty());
}

#[test]
fn no_export_means_zero_obligations() {
    let source = "module Stats.Prob
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Probability = Probability { value: x }
";
    let (code, records) = prove_json(source, &[]);
    assert_eq!(code, 0);
    assert!(
        obligations(&records).is_empty(),
        "no export => no producers"
    );
    assert_eq!(summary(&records)["obligations"], 0);
}

#[test]
fn non_exported_violating_constructor_yields_zero_obligation_records() {
    // The bad constructor is NOT exported; it must not produce ANY
    // obligation record (count kind:"obligation" == 0).
    let source = "module Stats.Prob
export (good)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def good(x: f32) -> Probability = Probability { value: if x >= 0.0 then (if x <= 1.0 then x else 1.0) else 0.0 }
def bad(x: f32) -> Probability = Probability { value: x }
";
    let (_code, records) = prove_json(source, &[]);
    let obs = obligations(&records);
    // Exactly one obligation (good); bad is unexported => no record.
    assert_eq!(obs.len(), 1);
    assert_eq!(obs[0]["producer"], "good");
    assert!(!obs.iter().any(|o| o["producer"] == "bad"));
}

#[test]
fn only_invariant_glob_selects_obligations_excludes_user_property() {
    let source = "module Stats.Prob
export (probability)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Option[Probability] =
  if x >= 0.0 && x <= 1.0 then Some(Probability { value: x }) else None
@property user_prop forall(y: f32) where y >= 0.0:
  y >= 0.0
";
    let (_code, records) = prove_json(source, &["--only", "invariant:*"]);
    let obs = obligations(&records);
    assert_eq!(obs.len(), 1, "only the obligation is selected");
    // No user-property record should appear.
    let props: Vec<_> = records
        .iter()
        .filter(|r| r.get("kind").and_then(Value::as_str) == Some("property"))
        .collect();
    assert!(
        props.is_empty(),
        "user property excluded by invariant:* glob"
    );
}

#[test]
fn only_user_property_excludes_obligations() {
    let source = "module Stats.Prob
export (probability)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Option[Probability] =
  if x >= 0.0 && x <= 1.0 then Some(Probability { value: x }) else None
@property user_prop forall(y: f32) where y >= 0.0:
  y >= 0.0
";
    let (_code, records) = prove_json(source, &["--only", "user_prop"]);
    assert!(
        obligations(&records).is_empty(),
        "user_prop selector excludes obligations"
    );
}

#[test]
fn tuple_position_obligation_is_discovered() {
    // Option[(Probability, f32)] decomposes; the tuple component carries
    // the obligation.
    let source = "module Stats.Prob
export (mk)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def mk(x: f32) -> Option[(Probability, f32)] =
  if x >= 0.0 && x <= 1.0 then Some((Probability { value: x }, x)) else None
";
    let (_code, records) = prove_json(source, &[]);
    let obs = obligations(&records);
    assert_eq!(obs.len(), 1, "tuple-carrying producer is an obligation");
    assert_eq!(obs[0]["producer"], "mk");
}

#[test]
fn same_seed_is_deterministic() {
    // Two runs with the same seed produce identical obligation records.
    let source = FLAGSHIP;
    let (_c1, r1) = prove_json(source, &["--seed", "7"]);
    let (_c2, r2) = prove_json(source, &["--seed", "7"]);
    assert_eq!(obligations(&r1), obligations(&r2));
}
