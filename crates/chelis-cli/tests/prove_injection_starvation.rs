//! W4 injection + starvation acceptance suite (RFC D-INJECT, D-STARVE,
//! D-SOUND, D-TIERB). Covers the rows deferred from W3+W4: assumption
//! injection into user properties, update-shaped / tensor-input / simplex
//! producer obligations, and the generator-starvation classifier.
//!
//! Built only under the `smt` feature (the obligation/generation machinery
//! lives in the optional `chelis-prove` dependency, activated by `smt`).
//! Run: `cargo nextest run -p chelis-cli --features smt
//!   --test prove_injection_starvation` with `LD_LIBRARY_PATH` set to the
//! uv python lib (see AGENTS.md).
//!
//! `CHELIS_STYLE_GATE_DISABLE=1` is set so the ad-hoc opaque/invariant
//! fixtures are not blocked by the opaque-domain-construction lint.
#![cfg(feature = "smt")]

use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

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

fn property(records: &[Value], name: &str) -> Option<Value> {
    records
        .iter()
        .find(|r| {
            r.get("kind").and_then(Value::as_str) == Some("property")
                && r.get("name").and_then(Value::as_str) == Some(name)
        })
        .cloned()
}

fn obligation(records: &[Value], producer: &str) -> Option<Value> {
    records
        .iter()
        .find(|r| {
            r.get("kind").and_then(Value::as_str) == Some("obligation")
                && r.get("producer").and_then(Value::as_str) == Some(producer)
        })
        .cloned()
}

const PROB_DEFS: &str = "module Stats.Prob
export (clamp_prob)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def clamp_prob(x: f32) -> Probability =
  Probability { value: if x >= 0.0 then (if x <= 1.0 then x else 1.0) else 0.0 }
def prob_value(p: Probability) -> f32 = p.value
";

// ── D-INJECT: user-property assumption injection ──────────────────

#[test]
fn property_true_only_under_invariant_passes() {
    // prob_value(p) <= 1.0 holds because injection generates only valid
    // Probabilities (value in [0,1]).
    let source =
        format!("{PROB_DEFS}@property bounded forall(p: Probability):\n  prob_value(p) <= 1.0\n");
    let (_code, records) = prove_json(&source, &["--samples", "30", "--only", "bounded"]);
    let prop = property(&records, "bounded").expect("property record");
    assert_eq!(
        prop["status"], "passed",
        "true-under-invariant passes: {prop}"
    );
    assert_eq!(prop["proof_tier"], "fuzz");
}

#[test]
fn false_property_fails_even_under_injection() {
    // prob_value(p) <= 0.5 is false for valid Probabilities in (0.5, 1].
    // Injection generates valid binders; it does not make a false
    // property true.
    let source = format!(
        "{PROB_DEFS}@property too_strong forall(p: Probability):\n  prob_value(p) <= 0.5\n"
    );
    let (code, records) = prove_json(&source, &["--samples", "80", "--only", "too_strong"]);
    let prop = property(&records, "too_strong").expect("property record");
    assert_eq!(prop["status"], "failed", "false property fails: {prop}");
    assert!(prop.get("counterexample").is_some());
    assert_eq!(code, 1);
}

#[test]
fn invariant_free_opaque_binder_is_not_injected() {
    // An opaque type with NO @invariant: injection does not apply, so the
    // same property is NOT verified as passing (the binder is unsupported
    // for injection). Test-lock: injection is scoped to invariant-carrying
    // types only.
    let source = "module M.Plain
export (mk)
@opaque
type Token =
  | Token { value: f32 }
def mk(x: f32) -> Token = Token { value: x }
def token_value(t: Token) -> f32 = t.value
@property tok_bounded forall(t: Token):
  token_value(t) <= 1.0
";
    let (code, records) = prove_json(source, &["--samples", "30", "--only", "tok_bounded"]);
    let prop = property(&records, "tok_bounded").expect("property record");
    assert_ne!(
        prop["status"], "passed",
        "invariant-free opaque binder must not be injected (so it cannot pass): {prop}"
    );
    assert_eq!(code, 2, "unsupported binder => exit 2");
}

#[test]
fn precondition_composes_with_injected_invariant() {
    // A user precondition `prob_value(p) >= 0.3` restricts further; the
    // body `prob_value(p) >= 0.3` then holds on the filtered samples.
    let source = format!(
        "{PROB_DEFS}@property lower_band forall(p: Probability) where prob_value(p) >= 0.3:\n  prob_value(p) >= 0.3\n"
    );
    let (_code, records) = prove_json(&source, &["--samples", "20", "--only", "lower_band"]);
    let prop = property(&records, "lower_band").expect("property record");
    assert_eq!(prop["status"], "passed", "precondition composes: {prop}");
}

#[test]
fn precondition_other_direction_filters_then_body_fails() {
    // Precondition `prob_value(p) >= 0.6` admits only the upper band; the
    // body `prob_value(p) <= 0.5` is then false on every admitted sample.
    let source = format!(
        "{PROB_DEFS}@property contradiction forall(p: Probability) where prob_value(p) >= 0.6:\n  prob_value(p) <= 0.5\n"
    );
    let (code, records) = prove_json(&source, &["--samples", "40", "--only", "contradiction"]);
    let prop = property(&records, "contradiction").expect("property record");
    assert_eq!(prop["status"], "failed", "filtered body fails: {prop}");
    assert_eq!(code, 1);
}

#[test]
fn unsatisfiable_injection_precondition_terminates_not_hangs() {
    // The precondition `prob_value(p) >= 2.0` CONTRADICTS the injected
    // invariant `0 <= value <= 1`: no invariant-valid binder can ever satisfy
    // it, so the outer rejection loop accepts ZERO samples. Without the
    // attempt cap (RT #4) this spins forever; with it the run TERMINATES and
    // reports the generator-exhausted error. Unlike the in-file unit model,
    // this drives the real `prove_with_injection` loop end-to-end through the
    // `chelis` binary, so it fails (hangs to the harness timeout) if the cap
    // or the post-loop exhaustion return is reverted (re-review RT2).
    let source = format!(
        "{PROB_DEFS}@property starves forall(p: Probability) where prob_value(p) >= 2.0:\n  prob_value(p) <= 1.0\n"
    );
    let (code, records) = prove_json(&source, &["--samples", "5", "--only", "starves"]);
    let prop = property(&records, "starves").expect("property record");
    assert_eq!(
        prop["status"], "error",
        "an unsatisfiable injection precondition must exhaust to an error, not hang or pass: {prop}"
    );
    assert_eq!(code, 3, "injection exhaustion => error => exit 3");
}

#[test]
fn same_seed_injection_is_deterministic() {
    let source =
        format!("{PROB_DEFS}@property bounded forall(p: Probability):\n  prob_value(p) <= 1.0\n");
    let (_c1, r1) = prove_json(
        &source,
        &["--samples", "10", "--seed", "5", "--only", "bounded"],
    );
    let (_c2, r2) = prove_json(
        &source,
        &["--samples", "10", "--seed", "5", "--only", "bounded"],
    );
    // Compare the deterministic verification fields, not the `source.file`
    // path (each run uses a fresh tempdir, so the absolute path differs while
    // the verification result is identical).
    let strip = |mut p: Value| {
        if let Some(obj) = p.as_object_mut() {
            obj.remove("source");
        }
        p
    };
    let p1 = strip(property(&r1, "bounded").expect("p1").clone());
    let p2 = strip(property(&r2, "bounded").expect("p2").clone());
    assert_eq!(p1, p2);
}

// ── D-SOUND: update-shaped producer obligations ───────────────────

#[test]
fn update_shaped_producer_passes_under_input_assumption() {
    // scale_down preserves [0,1] GIVEN the input is in [0,1]. The Tier B
    // input-invariant injection makes it provable at smt.
    let source = "module Stats.Prob
export (scale_down)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def prob_value(p: Probability) -> f32 = p.value
def scale_down(p: Probability, k: f32) -> Probability =
  Probability { value: if k >= 0.0 then (if k <= 1.0 then prob_value(p) * k else prob_value(p)) else prob_value(p) }
";
    let (code, records) = prove_json(source, &[]);
    let ob = obligation(&records, "scale_down").expect("update-shaped obligation");
    assert_eq!(ob["status"], "passed", "update preserves invariant: {ob}");
    assert_eq!(ob["proof_tier"], "smt");
    assert_eq!(code, 0);
}

#[test]
fn update_shaped_violating_twin_fails() {
    let source = "module Stats.Prob
export (bad_scale)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def prob_value(p: Probability) -> f32 = p.value
def bad_scale(p: Probability, k: f32) -> Probability = Probability { value: prob_value(p) + k }
";
    let (code, records) = prove_json(source, &[]);
    let ob = obligation(&records, "bad_scale").expect("violating update obligation");
    assert_eq!(ob["status"], "failed", "unguarded update fails: {ob}");
    assert!(ob.get("counterexample").is_some());
    assert_eq!(code, 1);
}

// ── D-STARVE: tensor-input + simplex obligations ──────────────────

const SIMPLEX_DEFS: &str = "module Stats.Simplex
export (make_simplex)
@opaque
@invariant(p) sum(p.weights) >= 1.0 - eps && sum(p.weights) <= 1.0 + eps
type Simplex =
  | Simplex { weights: tensor[3, f32] }
def eps() -> f32 = 0.01
def make_simplex(a: f32, b: f32, c: f32) -> Simplex =
  { s = abs(a) + abs(b) + abs(c) + 0.001;
    Simplex { weights: to_tensor([abs(a) / s, abs(b) / s, (abs(c) + 0.001) / s]) } }
";

#[test]
fn simplex_obligation_is_supported_and_passes() {
    // The tolerance-band simplex obligation (sum over a tensor field, with
    // a module constant) is SUPPORTED (no longer "unsupported") and the
    // normalizing producer passes. The named D-STARVE acceptance probe.
    let (code, records) = prove_json(SIMPLEX_DEFS, &["--samples", "15"]);
    let ob = obligation(&records, "make_simplex").expect("simplex obligation");
    assert_eq!(
        ob["status"], "passed",
        "simplex obligation supported + passes: {ob}"
    );
    assert_eq!(ob["proof_tier"], "fuzz");
    assert_ne!(ob["status"], "unsupported");
    assert_eq!(code, 0);
}

#[test]
fn non_normalizing_simplex_producer_fails() {
    let source = "module Stats.Simplex
export (bad_simplex)
@opaque
@invariant(p) sum(p.weights) >= 1.0 - eps && sum(p.weights) <= 1.0 + eps
type Simplex =
  | Simplex { weights: tensor[3, f32] }
def eps() -> f32 = 0.01
def bad_simplex(a: f32, b: f32, c: f32) -> Simplex = Simplex { weights: to_tensor([a, b, c]) }
";
    let (code, records) = prove_json(source, &["--samples", "40"]);
    let ob = obligation(&records, "bad_simplex").expect("bad simplex obligation");
    assert_eq!(
        ob["status"], "failed",
        "non-normalizing producer fails: {ob}"
    );
    assert!(ob.get("counterexample").is_some());
    assert_eq!(code, 1);
}

// ── CR-1: Option/tuple inner position applied for tensor fields ────

const SIMPLEX_OPT_TUPLE_HEAD: &str = "module Stats.Simplex
@opaque
@invariant(p) sum(p.weights) >= 1.0 - eps && sum(p.weights) <= 1.0 + eps
type Simplex =
  | Simplex { weights: tensor[3, f32] }
def eps() -> f32 = 0.01
def normed(a: f32, b: f32, c: f32) -> tensor[3, f32] =
  { s = abs(a) + abs(b) + abs(c) + 0.001;
    to_tensor([abs(a) / s, abs(b) / s, (abs(c) + 0.001) / s]) }
";

#[test]
fn cr1_option_of_tuple_validates_the_produced_tensor_invariant() {
    // CR-1 HIGH: for a TENSOR-field opaque type, an Option[(Simplex, f32)]
    // producer must actually validate the produced Simplex's invariant.
    // The bug bound the match var to the whole tuple and read record fields
    // off it. A VALID producer (normalized weights) must pass.
    let source = format!(
        "{SIMPLEX_OPT_TUPLE_HEAD}export (mk)\ndef mk(a: f32, b: f32, c: f32) -> Option[(Simplex, f32)] =\n  Some((Simplex {{ weights: normed(a, b, c) }}, a))\n"
    );
    let (code, records) = prove_json(&source, &["--samples", "15"]);
    let ob = obligation(&records, "mk").expect("option-of-tuple obligation");
    assert_eq!(
        ob["status"], "passed",
        "valid Option[(Simplex,f32)] producer validates the inner Simplex: {ob}"
    );
    assert_eq!(code, 0);
}

#[test]
fn cr1_option_of_tuple_violating_producer_fails() {
    // The negative twin: an Option[(Simplex, f32)] whose Simplex is NOT
    // normalized must FAIL -- the inner tuple component's invariant is
    // actually checked (the bug never checked it).
    let source = format!(
        "{SIMPLEX_OPT_TUPLE_HEAD}export (mk)\ndef mk(a: f32, b: f32, c: f32) -> Option[(Simplex, f32)] =\n  Some((Simplex {{ weights: to_tensor([a, b, c]) }}, a))\n"
    );
    let (code, records) = prove_json(&source, &["--samples", "40"]);
    let ob = obligation(&records, "mk").expect("violating option-of-tuple obligation");
    assert_eq!(
        ob["status"], "failed",
        "violating inner Simplex must be caught: {ob}"
    );
    assert_eq!(code, 1);
}

// ── CR-4/CR-6: a legitimate NaN representation is not read as None ──

#[test]
fn cr4_some_record_with_nan_field_does_not_pass_vacuously() {
    // CR-4/CR-6 HIGH (fail-open): a producer returning Some(record) whose
    // tensor field legitimately contains NaN was misread as None (the NaN
    // sentinel) and passed VACUOUSLY. A NaN representation must NOT
    // vacuously satisfy the band invariant -- NaN comparisons are false, so
    // the obligation must FAIL (not pass).
    let source = "module Stats.Simplex
export (nan_prod)
@opaque
@invariant(p) sum(p.weights) >= 1.0 - eps && sum(p.weights) <= 1.0 + eps
type Simplex =
  | Simplex { weights: tensor[3, f32] }
def eps() -> f32 = 0.01
def nan_prod(a: f32) -> Option[Simplex] =
  Some(Simplex { weights: to_tensor([0.0 / 0.0, 0.5, 0.5]) })
";
    let (code, records) = prove_json(source, &["--samples", "10"]);
    let ob = obligation(&records, "nan_prod").expect("nan obligation");
    assert_ne!(
        ob["status"], "passed",
        "Some(record-with-NaN) must NOT vacuously pass: {ob}"
    );
    // It is a real produced value that violates the invariant => failed.
    assert_eq!(
        ob["status"], "failed",
        "NaN representation fails the invariant: {ob}"
    );
    assert_eq!(code, 1);
}

#[test]
fn simplex_binder_under_injection_is_not_starved() {
    // The simplex as a property binder: constructor-based generation
    // (tier 2) serves the equality-constrained band WITHOUT starving, so
    // the property's binder is generated and the property is verified
    // (does not report a generator-starvation diagnostic). The body is a
    // structural predicate over the binder (a Simplex always has a
    // well-defined first-vs-rest comparison), exercising binder
    // generation, not a specific arithmetic claim.
    let source = format!(
        "{SIMPLEX_DEFS}def has_simplex(p: Simplex) -> bool = is_simplex(p)\ndef is_simplex(p: Simplex) -> bool = true\n@property generated forall(p: Simplex):\n  has_simplex(p)\n"
    );
    let (code, records) = prove_json(&source, &["--samples", "8", "--only", "generated"]);
    let prop = property(&records, "generated").expect("property record");
    assert_eq!(
        prop["status"], "passed",
        "simplex binder served by constructor generation, not starved: {prop}"
    );
    let reason = prop.get("reason").and_then(Value::as_str).unwrap_or("");
    assert!(
        !reason.contains("starvation"),
        "binder must not starve: {reason}"
    );
    assert_eq!(code, 0);
}

#[test]
fn rt3_f3_tight_scalar_band_binder_is_served_by_constructor_generation() {
    // RT3-F3 MEDIUM: a tight SCALAR-field band (value in [0.5, 0.5005])
    // false-starved because constructor read_produced_field only handled
    // Tensor results -- a scalar field access yields Float64, which was
    // dropped (0/200). `norm` always returns value=0.5 (in band), so the
    // property must verify, not report unsupported.
    let source = "module M
export (norm, prob_value)
@opaque
@invariant(p) p.value >= 0.5 && p.value <= 0.5005
type T = | T { value: f32 }
def norm(x: f32) -> T = T { value: 0.5 }
def prob_value(p: T) -> f32 = p.value
@property w forall(p: T):
  prob_value(p) <= 0.5005
";
    let (code, records) = prove_json(source, &["--samples", "8", "--only", "w"]);
    let prop = property(&records, "w").expect("property record");
    assert_eq!(
        prop["status"], "passed",
        "scalar-field band served by constructor generation, not starved: {prop}"
    );
    let reason = prop.get("reason").and_then(Value::as_str).unwrap_or("");
    assert!(!reason.contains("starvation"), "must not starve: {reason}");
    assert_eq!(code, 0);
}

// ── D-STARVE: the starvation classifier ───────────────────────────

#[test]
fn tight_equality_invariant_starves_with_shape_classified_diagnostic() {
    // An exact `==` over a float field has a measure-zero satisfying set:
    // rejection starves and (with no producer that hits it exactly)
    // constructor generation starves too => Unsupported exit 2 with the
    // equality-atoms shape diagnostic.
    let source = "module M.Exact
@opaque
@invariant(p) p.value == 0.5
type Exact =
  | Exact { value: f32 }
def mk(x: f32) -> Exact = Exact { value: x }
def exact_value(p: Exact) -> f32 = p.value
@property always forall(p: Exact):
  exact_value(p) >= 0.0
";
    let (code, records) = prove_json(source, &["--samples", "10", "--only", "always"]);
    let prop = property(&records, "always").expect("property record");
    assert_eq!(prop["status"], "unsupported", "exact == starves: {prop}");
    let reason = prop["reason"].as_str().unwrap_or("");
    assert!(reason.contains("starvation"), "names starvation: {reason}");
    assert!(
        reason.contains("equality-atoms"),
        "shape classified: {reason}"
    );
    assert!(reason.contains("Tier B"), "recommends Tier B: {reason}");
    assert_eq!(code, 2);
}

#[test]
fn min_rate_zero_disables_classifier_legacy_error_path() {
    // With --invariant-min-rate 0.0 the starvation classifier is disabled;
    // a starving binder reports Error (legacy exhaustion), not Unsupported.
    let source = "module M.Exact
@opaque
@invariant(p) p.value == 0.5
type Exact =
  | Exact { value: f32 }
def mk(x: f32) -> Exact = Exact { value: x }
def exact_value(p: Exact) -> f32 = p.value
@property always forall(p: Exact):
  exact_value(p) >= 0.0
";
    let (code, records) = prove_json(
        source,
        &[
            "--samples",
            "10",
            "--invariant-min-rate",
            "0.0",
            "--only",
            "always",
        ],
    );
    let prop = property(&records, "always").expect("property record");
    assert_eq!(
        prop["status"], "error",
        "min-rate 0.0 => legacy exhaustion error: {prop}"
    );
    assert_eq!(code, 3);
}

#[test]
fn generous_floor_passes_for_a_wide_band() {
    // A wide band ([0,1]) has ~10% acceptance, well above a low floor.
    let source =
        format!("{PROB_DEFS}@property bounded forall(p: Probability):\n  prob_value(p) <= 1.0\n");
    let (code, records) = prove_json(
        &source,
        &[
            "--samples",
            "20",
            "--invariant-min-rate",
            "0.001",
            "--only",
            "bounded",
        ],
    );
    let prop = property(&records, "bounded").expect("property record");
    assert_eq!(
        prop["status"], "passed",
        "wide band passes the floor: {prop}"
    );
    assert_eq!(code, 0);
}
