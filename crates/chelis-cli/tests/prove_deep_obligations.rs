//! Coverage hole (opaque-types Finding #5): `prove_deep_file` ran ZERO
//! derived producer-obligation verification, while `prove_surf_file` runs the
//! shared obligation engine. So a `.dp` module declaring an `@opaque` type
//! with a declared `@invariant` and an UNSOUND producer passed `chelis prove`
//! silently under `--features smt` -- no failed obligation, no error, exit 0.
//! The fix wires the SAME `chelis_prove::obligation_engine` Deep-program entry
//! (`run_module_obligations`, the one the Surf source entry reaches after
//! desugaring) into `prove_deep_file`.
//!
//! These tests exercise the OBLIGATION surface, built only under the `smt`
//! feature (the obligation collection + Tier B lowering live in the optional
//! `chelis-prove` dependency, activated by `smt`). Every test is
//! `#[cfg(feature = "smt")]`. Without the feature the obligation path is not
//! compiled, mirroring `prove_invariant_obligations.rs`.
//!
//! Run: `cargo nextest run -p chelis-cli --features smt
//!   --test prove_deep_obligations`
//! with `LD_LIBRARY_PATH` set to the uv python lib (see AGENTS.md).
//!
//! The `.dp` fixtures are produced by running `chelis deep` on a hand-authored
//! `.ch` opaque-invariant module, so the Deep `deftype` metadata shape is
//! exactly what the desugarer emits (the form `collect_opaque_invariants`
//! reads). `CHELIS_STYLE_GATE_DISABLE=1` is set on the prove run so the ad-hoc
//! opaque/invariant fixture is not blocked by the opaque-domain-construction
//! lint before the prove pipeline runs.
#![cfg(feature = "smt")]

use assert_cmd::Command;
use serde_json::Value;
use tempfile::{TempDir, tempdir};

/// Lower a `.ch` opaque-invariant module to canonical Deep via `chelis deep`
/// and write it as `mod.dp` in a fresh tempdir. Returns the dir (kept alive)
/// and the `.dp` path. Producing the `.dp` from the desugarer (rather than
/// hand-authoring the metadata) guarantees the `deftype` `opaque`/`invariant`
/// metadata is byte-shaped exactly as the engine's collector expects.
fn deep_fixture_from_surf(surf: &str) -> (TempDir, std::path::PathBuf) {
    let dir = tempdir().expect("tempdir");
    let ch_path = dir.path().join("mod.ch");
    std::fs::write(&ch_path, surf).expect("write .ch");

    let mut deep = Command::cargo_bin("chelis").expect("binary");
    deep.env("CHELIS_STYLE_GATE_DISABLE", "1");
    deep.arg("deep").arg(&ch_path);
    let out = deep.output().expect("run chelis deep");
    assert!(
        out.status.success(),
        "chelis deep failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let deep_source = String::from_utf8(out.stdout).expect("utf8 deep");

    let dp_path = dir.path().join("mod.dp");
    std::fs::write(&dp_path, &deep_source).expect("write .dp");
    (dir, dp_path)
}

/// Run `chelis prove --json` on the `.dp` produced from `surf`, returning
/// (exit_code, records).
fn prove_deep_json(surf: &str, extra: &[&str]) -> (i32, Vec<Value>) {
    let (_dir, dp_path) = deep_fixture_from_surf(surf);
    let mut cmd = Command::cargo_bin("chelis").expect("binary");
    cmd.env("CHELIS_STYLE_GATE_DISABLE", "1");
    cmd.arg("prove").arg(&dp_path).arg("--json");
    for a in extra {
        cmd.arg(a);
    }
    let output = cmd.output().expect("run prove");
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

// A guarded Option constructor whose guard exactly matches the invariant:
// every produced `Some(Probability { value })` satisfies `0 <= value <= 1`.
const SOUND_SURF: &str = "module Stats.Prob
export (probability)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Option[Probability] =
  if x >= 0.0 && x <= 1.0 then Some(Probability { value: x }) else None
";

// An UNSOUND constructor: it constructs `Probability { value: x }` for any
// `x >= 0.0`, so an `x > 1.0` escapes the upper bound of the invariant.
const UNSOUND_SURF: &str = "module Stats.Prob
export (bad_prob)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def bad_prob(x: f32) -> Option[Probability] =
  if x >= 0.0 then Some(Probability { value: x }) else None
";

#[test]
fn deep_sound_producer_obligation_passes() {
    // The whole point of the fix: a `.dp` opaque-invariant module's producer
    // obligation is actually RUN on the Deep surface. A sound producer must
    // yield a PASSED obligation at the smt tier, exit 0, and a summary that
    // counts the obligation -- byte-identically to the `.ch` surface.
    let (code, records) = prove_deep_json(SOUND_SURF, &[]);
    assert_eq!(code, 0, "sound producer obligation passes => exit 0");
    let obs = obligations(&records);
    assert_eq!(
        obs.len(),
        1,
        "the Deep surface now runs exactly one producer obligation: {records:?}"
    );
    let ob = obs[0];
    assert_eq!(ob["status"], "passed");
    assert_eq!(ob["proof_tier"], "smt");
    assert_eq!(ob["obligation_kind"], "invariant_producer");
    assert_eq!(ob["source_type"], "Probability");
    assert_eq!(ob["producer"], "probability");
    assert_eq!(ob["name"], "invariant:Probability:probability");
    assert_eq!(ob["arith_model"], "real");
    // The summary counts the obligation exactly as the `.ch` surface does.
    assert_eq!(summary(&records)["obligations"], 1);
}

#[test]
fn deep_smt_obligation_discloses_real_arithmetic_qualifier() {
    // F1 disclosure parity: the `.dp` producer-obligation record must carry the
    // SAME structured disclosure as the `.ch` surface (c1 in
    // prove_invariant_obligations.rs). An all-SMT obligation is proved over the
    // REALS, so the green token is `proven_modulo_real_arithmetic` AND the
    // qualifiers[] array discloses `real_arithmetic`. The `.dp` obligation
    // emitter previously omitted `qualifiers`, shipping a strictly weaker
    // disclosure than the equivalent `.ch` obligation (chelis#422 D2).
    let (code, records) = prove_deep_json(SOUND_SURF, &[]);
    assert_eq!(code, 0);
    let ob = obligations(&records)[0];
    assert_eq!(ob["status"], "passed");
    assert_eq!(ob["proof_tier"], "smt");
    assert_eq!(ob["composite_verdict"], "proven_modulo_real_arithmetic");
    assert_eq!(
        ob["qualifiers"],
        serde_json::json!(["real_arithmetic"]),
        "the .dp obligation must disclose real_arithmetic byte-identically to the .ch surface: {ob}"
    );
}

#[test]
fn deep_fuzz_only_obligation_discloses_fuzz_base_qualifier() {
    // F1 disclosure parity (fuzz tier, mirrors c2). A fuzz-only obligation BASE
    // is empirically validated, never proven: the token is `fuzz_validated` and
    // the qualifiers[] array discloses `fuzz_base`. The `.dp` obligation
    // emitter must surface that disclosure too.
    let (code, records) = prove_deep_json(
        SOUND_SURF,
        &["--tier", "fuzz-only", "--samples", "8", "--seed", "7"],
    );
    assert_eq!(code, 0);
    let ob = obligations(&records)[0];
    assert_eq!(ob["status"], "passed");
    assert_eq!(ob["proof_tier"], "fuzz");
    assert_eq!(ob["composite_verdict"], "fuzz_validated");
    // The fuzz-only base discloses `fuzz_base`; the non-vacuity check is also
    // discharged by fuzz, so the union additionally carries `fuzz`. Pin the
    // full sorted set so the disclosure is locked exactly, and assert the
    // `fuzz_base` base caveat is present (the F1 contract).
    assert_eq!(
        ob["qualifiers"],
        serde_json::json!(["fuzz", "fuzz_base"]),
        "the .dp fuzz-only obligation must disclose the full caveat set: {ob}"
    );
    assert!(
        ob["qualifiers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|q| q == "fuzz_base"),
        "the fuzz-only base caveat fuzz_base must be disclosed: {ob}"
    );
}

#[test]
fn deep_unsound_producer_obligation_fails() {
    // The soundness hole: WITHOUT the fix this `.dp` ran no obligation, so a
    // violating producer passed silently (exit 0, zero obligation records).
    // WITH the fix the obligation is disproved => failed obligation, exit 1.
    let (code, records) = prove_deep_json(UNSOUND_SURF, &[]);
    assert_eq!(
        code, 1,
        "an invariant-violating producer on the Deep surface fails => exit 1: {records:?}"
    );
    let obs = obligations(&records);
    assert_eq!(
        obs.len(),
        1,
        "the violating producer surfaces exactly one obligation: {records:?}"
    );
    assert_eq!(obs[0]["status"], "failed");
    assert_eq!(obs[0]["producer"], "bad_prob");
    assert_eq!(obs[0]["source_type"], "Probability");
    assert!(
        obs[0].get("counterexample").is_some(),
        "a failed obligation carries a counterexample: {}",
        obs[0]
    );
    assert_eq!(summary(&records)["obligations"], 1);
    assert_eq!(summary(&records)["failed"], 1);
}

#[test]
fn deep_user_property_path_not_regressed_by_obligations() {
    // The existing Deep user-@property path must still run alongside the new
    // obligation path. A `.dp` with BOTH an opaque-invariant producer AND a
    // user `@property` should report the user property record AND the
    // obligation record, without the obligation wiring clobbering the
    // property surface (do-not-regress requirement).
    let surf = "module Stats.Prob
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
    let (code, records) = prove_deep_json(surf, &[]);
    assert_eq!(code, 0, "sound producer + true user property => exit 0");
    // The obligation is present.
    let obs = obligations(&records);
    assert_eq!(obs.len(), 1, "obligation still runs: {records:?}");
    assert_eq!(obs[0]["status"], "passed");
    // The user property record is also present (path not regressed).
    let props: Vec<_> = records
        .iter()
        .filter(|r| {
            r.get("kind").and_then(Value::as_str) == Some("property")
                && r.get("name").and_then(Value::as_str) == Some("user_prop")
        })
        .collect();
    assert_eq!(
        props.len(),
        1,
        "the user @property path still emits its record: {records:?}"
    );
    assert_eq!(props[0]["status"], "passed");
}

/// Re-review RT1: a TYPE-BROKEN `.dp` module that declares NO opaque invariant
/// must be reported as an Error (exit 3), exactly as the `.ch` form is. The
/// deep obligation path type-checks the whole module unconditionally and
/// surfaces a type-check failure, so `chelis prove foo.dp` can no longer
/// silently pass a module `chelis prove foo.ch` rejects (the first fix
/// short-circuited the no-invariant case to Passed; this pins the parity).
#[test]
fn type_broken_deep_module_with_no_invariant_errors_not_silent_pass() {
    // `bad` returns f32 from an int32 body: a hard type error, and there is no
    // opaque type / invariant in sight.
    let surf = "module M\nexport (bad)\ndef bad(x: int32) -> f32 = x\n";
    let (code, records) = prove_deep_json(surf, &[]);
    assert_eq!(
        code, 3,
        "a type-broken deep module must exit 3 (Error), not silently pass: {records:?}"
    );
    assert!(
        records
            .iter()
            .any(|r| r["kind"] == "error" && r["stage"] == "check"),
        "the type-check failure must be surfaced as a check-error record: {records:?}"
    );
}
