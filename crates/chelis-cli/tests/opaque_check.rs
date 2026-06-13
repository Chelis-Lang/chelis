//! W1 checker-enforced opacity: CLI acceptance suite (RFC D-CHECK).
//!
//! Written RED-FIRST per AGENTS.md spec-first development: the
//! rejection tests were committed before the enforcement existed and
//! verified to fail (`chelis check` scored the violations 1 with an
//! empty error array). Baseline tests (inside-module pass, fmt
//! round-trip) pass before and after.
//!
//! `CHELIS_STYLE_GATE_DISABLE=1` is used per its documented purpose:
//! these fixtures synthesize ad-hoc Surf/Deep to exercise CHECKER
//! behavior, and the `opaque-domain-construction` lint (the
//! defense-in-depth half, D-LINT) would otherwise block the same
//! constructions at the style gate before the checker runs. The lint
//! half keeps its own coverage in
//! crates/chelis-lint/src/rules/opaque_domain_construction.rs.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

// ── Fixtures ─────────────────────────────────────────────────────

/// Canonical (`chelis fmt`-stable) defining-module Surf fixture.
const PROB_MODULE_CH: &str = "module Stats.Prob
export (probability, prob_value)
@opaque
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Probability = Probability { value: x }
def prob_value(p: Probability) -> f32 = p.value
";

/// Two-module Deep check unit: the defining module plus an
/// out-of-module record construction. Surf admits one `module` per
/// file, so the single-file multi-module encoding is Deep.
const VIOLATION_DP: &str = r#"(module {}
  stats.prob
  (export {} probability)
  (deftype {opaque: true}
    Probability
    ()
    (variant {} Probability (field {} value (t-prim {} f32))))
  (defsig {} probability (t-fn {} (t-prim {} f32) (t-adt {} Probability)))
  (def {}
    probability
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (record {} Probability (kv {} value (var {} x))))))
(module {}
  agent.strategy
  (def {}
    bad
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (record {} Probability (kv {} value (var {} x))))))
"#;

/// Defining module only: everything stays in-module and clean.
const CLEAN_DP: &str = r#"(module {}
  stats.prob
  (export {} probability)
  (deftype {opaque: true}
    Probability
    ()
    (variant {} Probability (field {} value (t-prim {} f32))))
  (defsig {} probability (t-fn {} (t-prim {} f32) (t-adt {} Probability)))
  (def {}
    probability
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (record {} Probability (kv {} value (var {} x))))))
"#;

const EXPECTED_DP_VIOLATION_MSG: &str = "in def `bad`: record construction of opaque type \
     `Probability` outside its defining module `stats.prob`; exported producers of \
     `stats.prob`: probability: (f32) -> Probability";

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent dir");
    }
    fs::write(path, contents).expect("write fixture file");
}

fn chelis() -> Command {
    let mut cmd = Command::cargo_bin("chelis").expect("chelis binary");
    cmd.env("CHELIS_STYLE_GATE_DISABLE", "1");
    cmd
}

fn check_json(stdout: &[u8]) -> Value {
    let text = String::from_utf8(stdout.to_vec()).expect("utf8 stdout");
    serde_json::from_str(&text)
        .unwrap_or_else(|err| panic!("chelis check output must be JSON: {err}\n{text}"))
}

fn errors_of(json: &Value) -> Vec<Value> {
    json.get("errors")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("errors array missing: {json}"))
        .clone()
}

fn opaque_violations(errors: &[Value]) -> Vec<&Value> {
    errors
        .iter()
        .filter(|e| e.get("kind").and_then(Value::as_str) == Some("OpaqueTypeViolation"))
        .collect()
}

/// Scaffold a dependency-free reef package named `opq` with
/// `module_prefix = "Demo"` and the given `src/` files.
fn reef_package(files: &[(&str, &str)]) -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let pkg = dir.path().join("opq");
    write_file(
        &pkg.join("reef.toml"),
        &format!(
            "[package]\nname = \"opq\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"Demo\"\n",
            env!("CARGO_PKG_VERSION")
        ),
    );
    for (rel, contents) in files {
        write_file(&pkg.join("src").join(rel), contents);
    }
    (dir, pkg)
}

const REEF_TYPES_CH: &str = "module Demo.Types
export (probability)
@opaque
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Probability = Probability { value: x }
macro forge_prob(x) = Probability { value: x }
";

/// Defining module for the sixth-rejection reef surface (RT-1 F1):
/// exported producer/reader plus an UNEXPORTED `raw_make` whose
/// signature mentions the opaque type.
const REEF_SIXTH_TYPES_CH: &str = "module Demo.Types
export (probability, prob_value)
@opaque
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Probability = Probability { value: x }
def prob_value(p: Probability) -> f32 = p.value
def raw_make(x: f32) -> Probability = Probability { value: x }
";

fn assert_reef_sixth_rejection(json: &Value) {
    let errors = errors_of(json);
    let violations = opaque_violations(&errors);
    assert_eq!(
        violations.len(),
        1,
        "expected exactly one OpaqueTypeViolation, got: {errors:?}"
    );
    let msg = violations[0]
        .get("message")
        .and_then(Value::as_str)
        .expect("violation message");
    assert!(
        msg.contains("reference to unexported binding"),
        "message must name the sixth-rejection action: {msg}"
    );
    assert!(
        msg.contains("raw_make"),
        "message must name the unexported binding: {msg}"
    );
    assert!(
        msg.contains("Probability"),
        "message must name the opaque type: {msg}"
    );
}

// ── `.dp` lexical encoding through `chelis check` ────────────────

#[test]
fn check_dp_rejects_out_of_module_record_construction() {
    let dir = tempdir().expect("tempdir");
    let dp = dir.path().join("violation.dp");
    write_file(&dp, VIOLATION_DP);

    let output = chelis()
        .args(["check", dp.to_str().unwrap()])
        .assert()
        .code(2)
        .get_output()
        .stdout
        .clone();
    let json = check_json(&output);
    let errors = errors_of(&json);
    let violations = opaque_violations(&errors);
    assert_eq!(
        violations.len(),
        1,
        "expected exactly one OpaqueTypeViolation, got: {errors:?}"
    );
    assert_eq!(
        violations[0].get("message").and_then(Value::as_str),
        Some(EXPECTED_DP_VIOLATION_MSG),
        "pinned violation message mismatch: {errors:?}"
    );
    assert_eq!(errors.len(), 1, "no cascade errors expected: {errors:?}");
}

#[test]
fn check_dp_passes_inside_defining_module() {
    let dir = tempdir().expect("tempdir");
    let dp = dir.path().join("clean.dp");
    write_file(&dp, CLEAN_DP);

    let output = chelis()
        .args(["check", dp.to_str().unwrap()])
        .assert()
        .code(0)
        .get_output()
        .stdout
        .clone();
    let json = check_json(&output);
    assert_eq!(
        json.get("score").and_then(Value::as_f64),
        Some(1.0),
        "inside-module program must score 1: {json}"
    );
    assert!(
        errors_of(&json).is_empty(),
        "inside-module program must have no errors: {json}"
    );
}

// ── Reef package encoding (two files + reef.toml) ────────────────

fn assert_reef_violation(json: &Value) {
    let errors = errors_of(json);
    let violations = opaque_violations(&errors);
    assert_eq!(
        violations.len(),
        1,
        "expected exactly one OpaqueTypeViolation, got: {errors:?}"
    );
    // The reef encoding renders the package-linked module identity
    // (internal-name stem); the byte-exact message contract is pinned
    // on the lexical encoding in chelis-types/tests/opaque_types.rs.
    let msg = violations[0]
        .get("message")
        .and_then(Value::as_str)
        .expect("violation message");
    assert!(
        msg.contains("record construction of opaque type"),
        "message must name the action: {msg}"
    );
    assert!(
        msg.contains("Probability"),
        "message must name the type: {msg}"
    );
    assert!(
        msg.contains("Types"),
        "message must name the defining module: {msg}"
    );
    assert!(
        msg.contains("probability"),
        "message must enumerate the exported producers: {msg}"
    );
}

#[test]
fn check_reef_rejects_out_of_module_construction() {
    let (_dir, pkg) = reef_package(&[
        ("types.ch", REEF_TYPES_CH),
        (
            "main.ch",
            "module Demo.Main
import Demo.Types (probability)
def sneak(x: f32) -> f32 = {
  p = Probability { value: x }
  x
}
",
        ),
    ]);
    let output = chelis()
        .current_dir(&pkg)
        .args(["check", pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .code(2)
        .get_output()
        .stdout
        .clone();
    assert_reef_violation(&check_json(&output));
}

#[test]
fn check_reef_rejects_out_of_module_construction_with_stdlib_cache_disabled() {
    // Same rejection with the stdlib typecheck cache switched off
    // (CHELIS_STDLIB_CACHE_DISABLE=1), so the enforcement cannot
    // depend on cache state.
    let (_dir, pkg) = reef_package(&[
        ("types.ch", REEF_TYPES_CH),
        (
            "main.ch",
            "module Demo.Main
import Demo.Types (probability)
def sneak(x: f32) -> f32 = {
  p = Probability { value: x }
  x
}
",
        ),
    ]);
    let output = chelis()
        .env("CHELIS_STDLIB_CACHE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["check", pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .code(2)
        .get_output()
        .stdout
        .clone();
    assert_reef_violation(&check_json(&output));
}

#[test]
fn check_reef_passes_via_exported_producer() {
    let (_dir, pkg) = reef_package(&[
        ("types.ch", REEF_TYPES_CH),
        (
            "main.ch",
            "module Demo.Main
import Demo.Types (probability)
def fine(x: f32) -> f32 = {
  p = probability(x)
  x
}
",
        ),
    ]);
    let output = chelis()
        .current_dir(&pkg)
        .args(["check", pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .code(0)
        .get_output()
        .stdout
        .clone();
    let json = check_json(&output);
    assert!(
        errors_of(&json).is_empty(),
        "producer-mediated construction must pass: {json}"
    );
}

#[test]
fn check_reef_macro_from_defining_module_expanded_outside_rejected() {
    // Macro call-site attribution (survey section 3): `forge_prob` is
    // defined (unexported) in Demo.Types; the same-package expansion
    // lands the record construction inside Demo.Macuser's def and is
    // checked under the CALLER's module.
    let (_dir, pkg) = reef_package(&[
        ("types.ch", REEF_TYPES_CH),
        (
            "macuser.ch",
            "module Demo.Macuser
import Demo.Types (forge_prob)
def sneak(x: f32) -> f32 = {
  p = forge_prob(x)
  x
}
",
        ),
    ]);
    let output = chelis()
        .current_dir(&pkg)
        .args(["check", pkg.join("src/macuser.ch").to_str().unwrap()])
        .assert()
        .code(2)
        .get_output()
        .stdout
        .clone();
    assert_reef_violation(&check_json(&output));
}

// ── Sixth rejection on the reef package surface (RT-1 F1) ────────

#[test]
fn check_reef_rejects_unexported_producer_reference_bare_call() {
    // RT-1 F1 primary probe: `attack.ch` imports only the exported
    // reader and calls the UNEXPORTED `raw_make` by its bare name
    // (un-imported cross-module reference). The sixth rejection must
    // fire -- W1 fail-opened here because reef leaves un-imported
    // references at their bare terminal name while the opacity
    // metadata is keyed by the internal (mangled) name.
    let (_dir, pkg) = reef_package(&[
        ("types.ch", REEF_SIXTH_TYPES_CH),
        (
            "attack.ch",
            "module Demo.Attack
import Demo.Types (prob_value)
def attack(x: f32) -> f32 = prob_value(raw_make(x))
",
        ),
    ]);
    let output = chelis()
        .current_dir(&pkg)
        .args(["check", pkg.join("src/attack.ch").to_str().unwrap()])
        .assert()
        .code(2)
        .get_output()
        .stdout
        .clone();
    assert_reef_sixth_rejection(&check_json(&output));
}

#[test]
fn check_reef_rejects_unexported_producer_reference_imported() {
    // RT-1 F1 import-shape variant: same-package imports admit
    // non-exported names, so `attack.ch` can `import` the unexported
    // `raw_make`. The reference is then rewritten to the internal
    // name; the sixth rejection must still fire (attribution must be
    // independent of import shape).
    let (_dir, pkg) = reef_package(&[
        ("types.ch", REEF_SIXTH_TYPES_CH),
        (
            "attack.ch",
            "module Demo.Attack
import Demo.Types (prob_value, raw_make)
def attack(x: f32) -> f32 = prob_value(raw_make(x))
",
        ),
    ]);
    let output = chelis()
        .current_dir(&pkg)
        .args(["check", pkg.join("src/attack.ch").to_str().unwrap()])
        .assert()
        .code(2)
        .get_output()
        .stdout
        .clone();
    assert_reef_sixth_rejection(&check_json(&output));
}

#[test]
fn build_reef_rejects_unexported_producer_reference() {
    // RT-1 F1: the same bare-call attack must also be rejected by
    // `chelis build` (RT-1 confirmed it emitted C in W1).
    let (_dir, pkg) = reef_package(&[
        ("types.ch", REEF_SIXTH_TYPES_CH),
        (
            "attack.ch",
            "module Demo.Attack
import Demo.Types (prob_value)
def attack(x: f32) -> f32 = prob_value(raw_make(x))
",
        ),
    ]);
    let out_dir = pkg.join("out");
    let assert = chelis()
        .current_dir(&pkg)
        .args([
            "build",
            pkg.join("src/attack.ch").to_str().unwrap(),
            "-o",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .failure();
    let stderr = String::from_utf8(assert.get_output().stderr.clone()).expect("utf8 stderr");
    assert!(
        stderr.contains("OpaqueTypeViolation"),
        "build must fail on the unexported-producer reference, stderr: {stderr}"
    );
}

#[test]
fn check_reef_exported_producer_reference_stays_clean() {
    // Negative parity: the exported producer/reader pair is callable
    // from another module in the same package without any violation.
    let (_dir, pkg) = reef_package(&[
        ("types.ch", REEF_SIXTH_TYPES_CH),
        (
            "user.ch",
            "module Demo.User
import Demo.Types (probability, prob_value)
def fine(x: f32) -> f32 = prob_value(probability(x))
",
        ),
    ]);
    let output = chelis()
        .current_dir(&pkg)
        .args(["check", pkg.join("src/user.ch").to_str().unwrap()])
        .assert()
        .code(0)
        .get_output()
        .stdout
        .clone();
    let json = check_json(&output);
    assert!(
        errors_of(&json).is_empty(),
        "exported producer/reader references must stay clean: {json}"
    );
}

// ── `chelis build` rejects the violation ─────────────────────────

#[test]
fn build_dp_rejects_out_of_module_construction() {
    let dir = tempdir().expect("tempdir");
    let dp = dir.path().join("violation.dp");
    write_file(&dp, VIOLATION_DP);
    let out_dir = dir.path().join("out");

    let assert = chelis()
        .args([
            "build",
            dp.to_str().unwrap(),
            "-o",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .failure();
    let stderr = String::from_utf8(assert.get_output().stderr.clone()).expect("utf8 stderr");
    assert!(
        stderr.contains("OpaqueTypeViolation"),
        "build must fail on the opacity violation, stderr: {stderr}"
    );
}

// ── Declaration rule through the Surf CLI surface ────────────────

#[test]
fn check_ch_opaque_outside_named_module_is_declaration_error() {
    // D-CHECK / RT-0 M6: `@opaque` requires a named enclosing module.
    let dir = tempdir().expect("tempdir");
    let ch = dir.path().join("bare.ch");
    write_file(
        &ch,
        "@opaque
type Probability =
  | Probability { value: f32 }
",
    );
    let output = chelis()
        .args(["check", ch.to_str().unwrap()])
        .assert()
        .code(2)
        .get_output()
        .stdout
        .clone();
    let json = check_json(&output);
    let errors = errors_of(&json);
    let violations = opaque_violations(&errors);
    assert_eq!(
        violations.len(),
        1,
        "expected exactly one declaration error, got: {errors:?}"
    );
    assert_eq!(
        violations[0].get("message").and_then(Value::as_str),
        Some("@opaque type `Probability` requires a named enclosing module"),
        "pinned declaration-error message mismatch: {errors:?}"
    );
}

// ── Formatter round-trip (baseline, passes before and after) ─────

#[test]
fn fmt_round_trips_opaque_module_fixture() {
    let dir = tempdir().expect("tempdir");
    let ch = dir.path().join("prob.ch");
    write_file(&ch, PROB_MODULE_CH);

    // Canonical input is fmt-stable...
    chelis()
        .args(["fmt", "--check", ch.to_str().unwrap()])
        .assert()
        .code(0);
    // ...and the formatted output is byte-identical to the input.
    let output = chelis()
        .args(["fmt", ch.to_str().unwrap()])
        .assert()
        .code(0)
        .get_output()
        .stdout
        .clone();
    let formatted = String::from_utf8(output).expect("utf8 fmt output");
    assert_eq!(
        formatted, PROB_MODULE_CH,
        "fmt must round-trip the @opaque module fixture byte-identically"
    );
}
