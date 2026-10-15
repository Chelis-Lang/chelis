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
//! This required SMT target also owns canonical numeric literal values and
//! typed integer input domains on both Surf and Deep (the proof_numeric_values module).
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
    // `bad` returns f32 from an i32 body: a hard type error, and there is no
    // opaque type / invariant in sight.
    let surf = "module M\nexport (bad)\ndef bad(x: i32) -> f32 = x\n";
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

// Numeric literal/domain parity shares this required, feature-enabled SMT owner.
mod proof_numeric_values {
    //! Shared Surf/Deep acceptance run by the required prove_deep_obligations SMT target.
    use assert_cmd::Command;
    use serde_json::Value;

    fn prove(source: &str, deep: bool) -> (i32, Vec<Value>) {
        prove_with_tier(source, deep, "smt-only")
    }

    fn prove_with_tier(source: &str, deep: bool, tier: &str) -> (i32, Vec<Value>) {
        let directory = tempfile::tempdir().unwrap();
        let surf = directory.path().join("numeric.ch");
        std::fs::write(&surf, source).unwrap();
        let formatted = Command::cargo_bin("chelis")
            .unwrap()
            .args(["fmt", "--inplace"])
            .arg(&surf)
            .output()
            .unwrap();
        assert!(formatted.status.success(), "{formatted:?}");
        let input = if deep {
            let output = Command::cargo_bin("chelis")
                .unwrap()
                .arg("deep")
                .arg(&surf)
                .env_remove("CHELIS_STYLE_GATE_DISABLE")
                .output()
                .unwrap();
            assert!(output.status.success(), "{output:?}");
            let path = directory.path().join("numeric.dp");
            std::fs::write(&path, output.stdout).unwrap();
            path
        } else {
            surf
        };
        let output = Command::cargo_bin("chelis")
            .unwrap()
            .arg("prove")
            .arg(input)
            .args(["--json", "--tier", tier, "--seed", "37"])
            .env_remove("CHELIS_STYLE_GATE_DISABLE")
            .output()
            .unwrap();
        let records: Vec<Value> = String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert!(
            !records.is_empty(),
            "{source}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        (output.status.code().unwrap(), records)
    }
    fn property<'a>(records: &'a [Value], name: &str) -> &'a Value {
        records
            .iter()
            .find(|r| r["kind"] == "property" && r["name"] == name)
            .unwrap_or_else(|| panic!("missing {name}: {records:?}"))
    }
    fn mutate_one_definition(source: &str, original: &str, replacement: &str) -> String {
        assert_eq!(source.matches(original).count(), 1);
        let mutant = source.replacen(original, replacement, 1);
        assert_eq!(mutant.matches(replacement).count(), 1);
        assert_eq!(
            source.split_once("@property").unwrap().1,
            mutant.split_once("@property").unwrap().1
        );
        assert_eq!(mutant.replace(replacement, original), source);
        assert_eq!(
            source
                .lines()
                .zip(mutant.lines())
                .filter(|(left, right)| left != right)
                .count(),
            1
        );
        mutant
    }

    fn exact_integer(value: &Value) -> i64 {
        if let Some(value) = value.as_i64() {
            return value;
        }
        let text = value.as_str().unwrap();
        if let Some(magnitude) = text.strip_prefix("(- ").and_then(|s| s.strip_suffix(')')) {
            i64::try_from(-magnitude.trim().parse::<i128>().unwrap()).unwrap()
        } else {
            text.parse().unwrap()
        }
    }

    #[test]
    fn typed_integer_domains_cast_minima_and_implementation_mutants_match_on_surf_and_deep() {
        for (dtype, minimum, maximum) in [
            ("i8", i8::MIN as i64, i8::MAX as i64),
            ("i16", i16::MIN as i64, i16::MAX as i64),
            ("i32", i32::MIN as i64, i32::MAX as i64),
            ("i64", i64::MIN, i64::MAX),
        ] {
            let source = format!(
                "def value(x: {dtype}) -> {dtype} = x\ndef minimum(dummy: {dtype}) -> {dtype} = cast({minimum}, {dtype})\n@property lower forall(x: {dtype}):\n  value(x) >= minimum(x)\n@property upper forall(x: {dtype}):\n  value(x) <= cast({maximum}, {dtype})\n@property at_minimum forall(x: {dtype}) where x == cast({minimum}, {dtype}):\n  value(x) == x\n@property strict_minimum forall(x: {dtype}):\n  value(x) > cast({minimum}, {dtype})\n"
            );
            for deep in [false, true] {
                let (code, records) = prove(&source, deep);
                assert_eq!(code, 1, "{records:?}"); // only the strict bound is false
                for name in ["lower", "upper", "at_minimum"] {
                    let record = property(&records, name);
                    assert_eq!(record["status"], "passed", "{record}");
                    assert_eq!(record["proof_tier"], "smt", "{record}");
                    assert_eq!(record["qualifiers"], serde_json::json!(["real_arithmetic"]));
                }
                assert_eq!(
                    property(&records, "lower")["assumptions"],
                    serde_json::json!([])
                );
                let guarded = property(&records, "at_minimum");
                assert_eq!(guarded["assumptions"].as_array().unwrap().len(), 1);
                assert_eq!(
                    guarded["assumptions"][0]["name"],
                    "preconditions:at_minimum"
                );
                assert_eq!(
                    guarded["assumptions"][0]["non_vacuity"]["evidence"]["assumption_count"],
                    1
                );
                assert_eq!(
                    guarded["assumptions"][0]["non_vacuity"]["evidence"]["intrinsic_domain_count"],
                    2
                );
                let strict = property(&records, "strict_minimum");
                assert_eq!(strict["status"], "failed", "{strict}");
                assert_eq!(exact_integer(&strict["counterexample"]["x"]), minimum);
                let original_definition = format!("def value(x: {dtype}) -> {dtype} = x\n");
                let mutant_definition =
                    format!("def value(x: {dtype}) -> {dtype} = x + cast(1, {dtype})\n");
                let mutant =
                    mutate_one_definition(&source, &original_definition, &mutant_definition);
                let (_, records) = prove(&mutant, deep);
                let failure = property(&records, "at_minimum");
                assert_eq!(failure["status"], "failed", "{failure}");
                assert_eq!(failure["proof_tier"], "smt");
                assert_eq!(exact_integer(&failure["counterexample"]["x"]), minimum);
            }
        }
    }

    #[test]
    fn nonminimum_casts_float_rounding_sentinel_and_nested_substitution_keep_values() {
        let source = "def inner(dummy: i32) -> i32 = cast(-3, i32)\ndef outer(x: i32) -> i32 = inner(x)\n@property nearby forall(x: i32) where x == cast(-3, i32):\n  outer(x) == x\n@property rounding forall(x: f32):\n  cast(16777217, f32) == 16777216f32\n@property rounding_false forall(x: f32):\n  cast(16777217, f32) != 16777216f32\n@property decimal forall(x: f64):\n  cast(1.1, f64) == 1.1f64\n@property sentinel forall(x: i64):\n  -9223372036854775808i64 < 0i64\n@property sentinel_false forall(x: i64):\n  -9223372036854775808i64 > 0i64\n";
        for deep in [false, true] {
            let (code, records) = prove(source, deep);
            assert_eq!(code, 1, "{records:?}");
            for name in ["nearby", "rounding", "decimal", "sentinel"] {
                assert_eq!(property(&records, name)["status"], "passed", "{records:?}");
                assert_eq!(property(&records, name)["proof_tier"], "smt");
            }
            for name in ["rounding_false", "sentinel_false"] {
                assert_eq!(property(&records, name)["status"], "failed", "{records:?}");
            }
            let mutant = mutate_one_definition(
                source,
                "def inner(dummy: i32) -> i32 = cast(-3, i32)\n",
                "def inner(dummy: i32) -> i32 = cast(-2, i32)\n",
            );
            let (_, records) = prove(&mutant, deep);
            assert_eq!(
                property(&records, "nearby")["status"],
                "failed",
                "{records:?}"
            );
        }
    }

    #[test]
    fn typed_domains_reject_vacuous_guards_without_hiding_unbounded_intermediate_arithmetic() {
        let source = "@property impossible forall(x: i64) where x < cast(-9223372036854775808, i64):\n  true\n@property arithmetic forall(x: i8):\n  x + 1i8 > x\n@property computed_argument forall(x: i8):\n  identity(x + 1i8) > x\ndef identity(x: i8) -> i8 = x\n";
        for deep in [false, true] {
            let (code, records) = prove(source, deep);
            assert_ne!(code, 0);
            let invalid = property(&records, "impossible");
            assert_eq!(invalid["status"], "unsupported", "{invalid}");
            assert!(
                invalid["reason"]
                    .as_str()
                    .unwrap()
                    .contains("non_vacuity_invalid")
            );
            for name in ["arithmetic", "computed_argument"] {
                let record = property(&records, name);
                assert_eq!(record["status"], "passed", "{record}");
                assert_eq!(record["composite_verdict"], "proven_modulo_real_arithmetic");
            }
        }
    }

    #[test]
    fn declaration_context_literals_keep_smt_surf_deep_and_runtime_verdicts() {
        let source = "sig standalone: i32 -> f64\ndef declared(dummy: i32) -> f64 = 1.1\ndef standalone(dummy: i32) = 1.1\ndef wide(dummy: i32) -> i64 = 3000000000\ndef integer_float(dummy: i32) -> f64 = 16777217\ndef local(dummy: i32) -> f64 = { \n  value: f64 = 1.1\n  value\n}\ndef minimum(dummy: i32) -> i8 = -128\ndef minimum64(dummy: i32) -> i64 = -9223372036854775808\ndef nested(x: i32) -> f64 = declared(x)\n@property declared_true forall(x: i32):\n  declared(x) == 1.1f64\n@property declared_false forall(x: i32):\n  declared(x) != 1.1f64\n@property standalone_true forall(x: i32):\n  standalone(x) == 1.1f64\n@property local_true forall(x: i32):\n  local(x) == 1.1f64\n@property local_false forall(x: i32):\n  local(x) != 1.1f64\n@property wide_true forall(x: i32):\n  wide(x) == 3000000000i64\n@property wide_false forall(x: i32):\n  wide(x) != 3000000000i64\n@property integer_float_true forall(x: i32):\n  integer_float(x) == 16777217f64\n@property minimum_true forall(x: i32):\n  minimum(x) == cast(-128, i8)\n@property minimum64_true forall(x: i32):\n  minimum64(x) == cast(-9223372036854775808, i64)\n@property nested_true forall(x: i32):\n  nested(x) == 1.1f64\n";
        for deep in [false, true] {
            for tier in ["smt-only", "fuzz-only"] {
                let (code, records) = prove_with_tier(source, deep, tier);
                assert_eq!(code, 1, "{records:?}");
                for name in [
                    "declared_true",
                    "standalone_true",
                    "wide_true",
                    "integer_float_true",
                    "minimum_true",
                    "minimum64_true",
                    "nested_true",
                ] {
                    let record = property(&records, name);
                    assert_eq!(record["status"], "passed", "{record}");
                    assert_eq!(
                        record["proof_tier"],
                        if tier == "smt-only" { "smt" } else { "fuzz" },
                        "{record}"
                    );
                    assert_eq!(record["assumptions"], serde_json::json!([]));
                }
                // Canonical Deep retains the typed local's ascription Block, which
                // its SMT arithmetic fragment does not lower. Surf lowers the
                // same source, and both surfaces execute it in the runtime tier.
                for name in ["local_true", "local_false"] {
                    let record = property(&records, name);
                    if deep && tier == "smt-only" {
                        assert_eq!(record["status"], "unsupported", "{record}");
                        assert_eq!(
                            record["reason"], "property does not lower to Tier B (smt-only)",
                            "{record}"
                        );
                        assert_eq!(record["samples"], 0, "{record}");
                    } else {
                        assert_eq!(
                            record["status"],
                            if name == "local_true" {
                                "passed"
                            } else {
                                "failed"
                            },
                            "{record}"
                        );
                        assert_eq!(
                            record["proof_tier"],
                            if tier == "smt-only" { "smt" } else { "fuzz" },
                            "{record}"
                        );
                        assert_eq!(record["assumptions"], serde_json::json!([]));
                    }
                }
                for name in ["declared_false", "wide_false"] {
                    let record = property(&records, name);
                    assert_eq!(record["status"], "failed", "{record}");
                    assert_eq!(
                        record["proof_tier"],
                        if tier == "smt-only" { "smt" } else { "fuzz" },
                        "{record}"
                    );
                }
            }
        }
    }

    #[test]
    fn surf_gradient_literals_keep_smt_and_runtime_verdicts_and_implementation_mutants() {
        let source = "def coefficient(x: f32) -> f32 = x * 1.1f32\ndef negative_coefficient(x: f32) -> f32 = x * -1.1f32\ndef helper(x: f32) -> f32 = coefficient(x)\ndef tiny(x: f32) -> f32 = x * 1e-45f32\ndef underflow(x: f32) -> f32 = x * 1e-46f32\ndef wide_coefficient(x: f64) -> f64 = x * 1.1f64\n@property gradient_true forall(x: f32):\n  grad(coefficient, wrt=x)(x) == 1.1f32\n@property gradient_false forall(x: f32):\n  grad(coefficient, wrt=x)(x) != 1.1f32\n@property negative_true forall(x: f32):\n  grad(negative_coefficient, wrt=x)(x) == -1.1f32\n@property negative_false forall(x: f32):\n  grad(negative_coefficient, wrt=x)(x) != -1.1f32\n@property nested_true forall(x: f32):\n  grad(helper, wrt=x)(x) == 1.1f32\n@property inline_true forall(x: f32):\n  grad(fn (v: f32) -> v * 1.1f32, wrt=v)(x) == 1.1f32\n@property tiny_true forall(x: f32):\n  grad(tiny, wrt=x)(x) == 1e-45f32\n@property tiny_false forall(x: f32):\n  grad(tiny, wrt=x)(x) != 1e-45f32\n@property underflow_true forall(x: f32):\n  grad(underflow, wrt=x)(x) == 0.0f32\n@property underflow_false forall(x: f32):\n  grad(underflow, wrt=x)(x) != 0.0f32\n@property wide_true forall(x: f64):\n  grad(wide_coefficient, wrt=x)(x) == 1.1f64\n@property wide_false forall(x: f64):\n  grad(wide_coefficient, wrt=x)(x) != 1.1f64\n";
        for tier in ["smt-only", "fuzz-only"] {
            let (code, records) = prove_with_tier(source, false, tier);
            assert_eq!(code, 1, "{records:?}");
            for name in [
                "gradient_true",
                "gradient_false",
                "negative_true",
                "negative_false",
                "nested_true",
                "inline_true",
                "tiny_true",
                "tiny_false",
                "underflow_true",
                "underflow_false",
                "wide_true",
                "wide_false",
            ] {
                let record = property(&records, name);
                assert_eq!(
                    record["status"],
                    if name.ends_with("_false") {
                        "failed"
                    } else {
                        "passed"
                    },
                    "{record}"
                );
                assert_eq!(
                    record["proof_tier"],
                    if tier == "smt-only" { "smt" } else { "fuzz" },
                    "{record}"
                );
                if tier == "smt-only" {
                    assert_eq!(record["samples"], 0, "{record}");
                }
            }
            let mutant = mutate_one_definition(
                source,
                "def coefficient(x: f32) -> f32 = x * 1.1f32\n",
                "def coefficient(x: f32) -> f32 = x * 1.2f32\n",
            );
            let (code, records) = prove_with_tier(&mutant, false, tier);
            assert_eq!(code, 1, "{records:?}");
            for name in ["gradient_true", "nested_true"] {
                assert_eq!(property(&records, name)["status"], "failed", "{records:?}");
            }
            assert_eq!(property(&records, "gradient_false")["status"], "passed");
        }
    }

    #[test]
    fn canonical_guard_plan_preserves_authored_fuzz_evaluation_at_an_exact_f64_interval() {
        let source = "def declared(dummy: i32) -> f64 = 1.1\n@property exact_interval forall(x: f64) where x >= declared(0), x <= 1.1f64:\n  x == 1.1f64\n";
        for deep in [false, true] {
            let (code, records) = prove_with_tier(source, deep, "fuzz-only");
            assert_eq!(code, 0, "{records:?}");
            let record = property(&records, "exact_interval");
            assert_eq!(record["status"], "passed", "{record}");
            assert_eq!(record["proof_tier"], "fuzz", "{record}");
            assert_eq!(record["sampling_method"], "constraint_directed");
            assert_eq!(record["accepted_samples"], 100);
            assert_eq!(record["rejected_samples"], 0);
            assert!(record["goal"].as_str().unwrap().contains(if deep {
                "declared"
            } else {
                "declared(0)"
            }));
        }
    }
}
