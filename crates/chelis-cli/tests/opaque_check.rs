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
const VIOLATION_DP: &str = r"(module {}
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
";

/// Defining module only: everything stays in-module and clean.
const CLEAN_DP: &str = r"(module {}
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
";

const EXPECTED_DP_VIOLATION_MSG: &str = "in def `bad`: record construction of opaque type \
     `Probability` outside its defining module `stats.prob`; exported producers of \
     `stats.prob`: probability: (f32) -> Probability";

/// RT-1 F2: the opaque defining module is RE-OPENED by a second
/// `(module ...)` wrapper that forges + accesses the type.
const MODULE_REOPEN_DP: &str = r"(module {}
  stats.prob
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
  stats.prob
  (def {}
    forge
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (record {} Probability (kv {} value (var {} x))))))
";

const EXPECTED_REOPEN_MSG: &str = "module `stats.prob` is opened by more than one module wrapper in this check unit; \
     a named module may be opened at most once";

/// RT-1 F2 bypass (RFC v5): pure-flat program of reef-internal mangled
/// names, NO wrappers. The mangled deftype + def stem-key to module
/// `foo` and forge + inspect the opaque type as in-module.
const STEM_ONLY_DP: &str = r"(deftype {opaque: true}
  Pkg__foo__Secret
  ()
  (variant {} Pkg__foo__Secret (field {} value (t-prim {} f32))))
(defsig {} pkg__foo__forge (t-fn {} (t-prim {} f32) (t-adt {} Pkg__foo__Secret)))
(def {}
  pkg__foo__forge
  (fn {}
    (params {} (x {type: (t-prim {} f32)}))
    (record {} Pkg__foo__Secret (kv {} value (var {} x)))))
(defsig {} pkg__foo__peek (t-fn {} (t-adt {} Pkg__foo__Secret) (t-prim {} f32)))
(def {}
  pkg__foo__peek
  (fn {}
    (params {} (p {type: (t-adt {} Pkg__foo__Secret)}))
    (access {} (var {} p) value)))
";

/// RT-1 F2 bypass: stem deftype + a lexical `(module {} foo ...)`
/// wrapper re-opening it.
const STEM_DOTFREE_DP: &str = r"(deftype {opaque: true}
  Pkg__foo__Secret
  ()
  (variant {} Pkg__foo__Secret (field {} value (t-prim {} f32))))
(module {}
  foo
  (def {}
    forge
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (record {} Pkg__foo__Secret (kv {} value (var {} x))))))
";

/// RT-1 F2 bypass: dotted-key variant of the stem-plus-wrapper forge.
const STEM_COLLIDE_DP: &str = r"(deftype {opaque: true}
  Pkg__foo__bar__Secret
  ()
  (variant {} Pkg__foo__bar__Secret (field {} value (t-prim {} f32))))
(module {}
  foo.bar
  (def {}
    forge
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (record {} Pkg__foo__bar__Secret (kv {} value (var {} x))))))
";

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
    assert_eq!(
        msg, EXPECTED_REEF_SIXTH_MSG,
        "reef sixth-rejection message must be de-mangled and byte-exact"
    );
    assert!(
        !msg.contains("pkg__") && !msg.contains("Pkg__"),
        "message must not leak any mangled reef name: {msg}"
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

/// RT-1 F3: the reef-surface message renders fully de-mangled names
/// (def `sneak`, type `Probability`, module `Demo.Types`, producer
/// `probability: (f32) -> Probability`), NOT the internal
/// `pkg__opq__Demo__Types__...` forms. Exact-message assertion.
const EXPECTED_REEF_CONSTRUCTION_MSG: &str = "in def `sneak`: record construction of opaque type \
     `Probability` outside its defining module `Demo.Types`; exported producers of \
     `Demo.Types`: probability: (f32) -> Probability";

/// RT-1 F1 + F3: de-mangled sixth-rejection message for `raw_make`.
/// `prob_value` is exported but returns f32 (its RESULT does not
/// mention the opaque type), so it is not a producer.
const EXPECTED_REEF_SIXTH_MSG: &str = "in def `attack`: reference to unexported binding \
     `raw_make` of module `Demo.Types` whose signature mentions opaque type `Probability`; \
     exported producers of `Demo.Types`: probability: (f32) -> Probability";

fn assert_reef_violation(json: &Value) {
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
    assert_eq!(
        msg, EXPECTED_REEF_CONSTRUCTION_MSG,
        "reef construction message must be de-mangled and byte-exact"
    );
    assert!(
        !msg.contains("pkg__") && !msg.contains("Pkg__"),
        "message must not leak any mangled reef name: {msg}"
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

// ── Module re-open forge across the surfaces (RT-1 F2) ───────────

#[test]
fn check_dp_rejects_module_reopen() {
    let dir = tempdir().expect("tempdir");
    let dp = dir.path().join("reopen.dp");
    write_file(&dp, MODULE_REOPEN_DP);

    let output = chelis()
        .args(["check", dp.to_str().unwrap()])
        .assert()
        .code(2)
        .get_output()
        .stdout
        .clone();
    let json = check_json(&output);
    let errors = errors_of(&json);
    let dup: Vec<&Value> = errors
        .iter()
        .filter(|e| e.get("kind").and_then(Value::as_str) == Some("DuplicateModule"))
        .collect();
    assert_eq!(
        dup.len(),
        1,
        "expected exactly one DuplicateModule error, got: {errors:?}"
    );
    assert_eq!(
        dup[0].get("message").and_then(Value::as_str),
        Some(EXPECTED_REOPEN_MSG),
        "pinned re-open message mismatch: {errors:?}"
    );
}

#[test]
fn build_dp_rejects_module_reopen() {
    let dir = tempdir().expect("tempdir");
    let dp = dir.path().join("reopen.dp");
    write_file(&dp, MODULE_REOPEN_DP);
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
        stderr.contains("DuplicateModule"),
        "build must fail on the module re-open, stderr: {stderr}"
    );
}

#[test]
fn validate_deep_rejects_module_reopen() {
    let dir = tempdir().expect("tempdir");
    let dp = dir.path().join("reopen.dp");
    write_file(&dp, MODULE_REOPEN_DP);

    let assert = chelis()
        .args(["validate", "--deep", dp.to_str().unwrap()])
        .assert()
        .failure();
    let stderr = String::from_utf8(assert.get_output().stderr.clone()).expect("utf8 stderr");
    assert!(
        stderr.contains("stats.prob") && stderr.to_lowercase().contains("module"),
        "validate --deep must reject the module re-open, stderr: {stderr}"
    );
}

// ── Reef-stem mangled-name forge (RT-1 F2 bypass, RFC v5) ────────

/// `chelis check <dp>` -> the parsed JSON report (any exit code).
fn check_dp_report(dir: &Path, name: &str, contents: &str) -> Value {
    let dp = dir.join(name);
    write_file(&dp, contents);
    let output = chelis()
        .args(["check", dp.to_str().unwrap()])
        .assert()
        .get_output()
        .stdout
        .clone();
    check_json(&output)
}

fn has_kind(errors: &[Value], kind: &str) -> bool {
    errors
        .iter()
        .any(|e| e.get("kind").and_then(Value::as_str) == Some(kind))
}

#[test]
fn check_dp_rejects_stem_only_mangled_forge() {
    let dir = tempdir().expect("tempdir");
    let json = check_dp_report(dir.path(), "stem_only.dp", STEM_ONLY_DP);
    let errors = errors_of(&json);
    assert!(
        has_kind(&errors, "ReservedLinkerName"),
        "pure-flat mangled-name forge must be rejected as ReservedLinkerName: {errors:?}"
    );
    assert!(
        json.get("score").and_then(Value::as_f64) != Some(1.0),
        "the forge must not score 1: {json}"
    );
}

#[test]
fn check_dp_rejects_stem_dotfree_and_collide_forges() {
    let dir = tempdir().expect("tempdir");
    for (name, dp) in [
        ("stem_dotfree.dp", STEM_DOTFREE_DP),
        ("stem_collide.dp", STEM_COLLIDE_DP),
    ] {
        let json = check_dp_report(dir.path(), name, dp);
        let errors = errors_of(&json);
        assert!(
            has_kind(&errors, "ReservedLinkerName"),
            "{name}: mangled deftype must be ReservedLinkerName: {errors:?}"
        );
        assert!(
            has_kind(&errors, "DuplicateModule"),
            "{name}: stem-vs-wrapper collision must also be DuplicateModule: {errors:?}"
        );
    }
}

#[test]
fn build_dp_rejects_stem_mangled_forges() {
    let dir = tempdir().expect("tempdir");
    for (name, dp) in [
        ("stem_only.dp", STEM_ONLY_DP),
        ("stem_dotfree.dp", STEM_DOTFREE_DP),
        ("stem_collide.dp", STEM_COLLIDE_DP),
    ] {
        let path = dir.path().join(name);
        write_file(&path, dp);
        let out_dir = dir.path().join(format!("out_{name}"));
        let assert = chelis()
            .args([
                "build",
                path.to_str().unwrap(),
                "-o",
                out_dir.to_str().unwrap(),
            ])
            .assert()
            .failure();
        let stderr = String::from_utf8(assert.get_output().stderr.clone()).expect("utf8 stderr");
        assert!(
            stderr.contains("ReservedLinkerName") || stderr.contains("DuplicateModule"),
            "{name}: build must reject the forge, stderr: {stderr}"
        );
    }
}

#[test]
fn validate_deep_rejects_stem_mangled_forges() {
    let dir = tempdir().expect("tempdir");
    for (name, dp) in [
        ("stem_only.dp", STEM_ONLY_DP),
        ("stem_dotfree.dp", STEM_DOTFREE_DP),
        ("stem_collide.dp", STEM_COLLIDE_DP),
    ] {
        let path = dir.path().join(name);
        write_file(&path, dp);
        let assert = chelis()
            .args(["validate", "--deep", path.to_str().unwrap()])
            .assert()
            .failure();
        let stderr = String::from_utf8(assert.get_output().stderr.clone()).expect("utf8 stderr");
        assert!(
            stderr.contains("reserved internal-name format"),
            "{name}: validate --deep must reject the mangled name, stderr: {stderr}"
        );
    }
}

#[test]
fn check_reef_package_with_internal_names_stays_clean() {
    // CRITICAL no-regression (RFC v5): a genuine reef package -- whose
    // decls the linker rewrites to the SAME `Pkg__`/`pkg__` internal
    // format that the forge uses -- must still check clean, because the
    // linked-program provenance flag is TRUE on the reef path. If this
    // breaks, the fix would have rejected the linker's own output.
    let (_dir, pkg) = reef_package(&[
        ("types.ch", REEF_SIXTH_TYPES_CH),
        (
            "main.ch",
            "module Demo.Main
import Demo.Types (probability, prob_value)
def use(x: f32) -> f32 = prob_value(probability(x))
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
        "a genuine reef package's internal names must check clean: {json}"
    );
    assert_eq!(json.get("score").and_then(Value::as_f64), Some(1.0));
}

#[test]
fn check_reef_package_with_internal_names_stays_clean_cache_disabled() {
    // Same no-regression with the stdlib cache off, exercising the
    // monolithic fallback (which checks `prepared.decls` directly).
    let (_dir, pkg) = reef_package(&[
        ("types.ch", REEF_SIXTH_TYPES_CH),
        (
            "main.ch",
            "module Demo.Main
import Demo.Types (probability, prob_value)
def use(x: f32) -> f32 = prob_value(probability(x))
",
        ),
    ]);
    let output = chelis()
        .env("CHELIS_STDLIB_CACHE_DISABLE", "1")
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
        "reef internal names must check clean with cache disabled: {json}"
    );
}

#[test]
fn build_reef_package_with_internal_names_stays_clean() {
    let (_dir, pkg) = reef_package(&[
        ("types.ch", REEF_SIXTH_TYPES_CH),
        (
            "main.ch",
            "module Demo.Main
import Demo.Types (probability, prob_value)
def use(x: f32) -> f32 = prob_value(probability(x))
",
        ),
    ]);
    let out_dir = pkg.join("out");
    chelis()
        .current_dir(&pkg)
        .args([
            "build",
            pkg.join("src/main.ch").to_str().unwrap(),
            "-o",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
}

#[test]
fn check_dp_distinct_modules_still_pass_construction_gate() {
    // Negative parity: two DISTINCT module wrappers in one .dp (the
    // ordinary out-of-module setup) must not trip the re-open rule --
    // only the opacity construction violation fires.
    let dir = tempdir().expect("tempdir");
    let dp = dir.path().join("two_modules.dp");
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
    assert!(
        !errors
            .iter()
            .any(|e| e.get("kind").and_then(Value::as_str) == Some("DuplicateModule")),
        "distinct module names must not trip the re-open rule: {errors:?}"
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

// ── Decompiler round-trip on the opaque path (RT-1 F4) ───────────

/// `chelis surf <file.dp>` -> decompiled Surf string.
fn decompile_dp(dp: &Path) -> String {
    let output = chelis()
        .args(["surf", dp.to_str().unwrap()])
        .assert()
        .code(0)
        .get_output()
        .stdout
        .clone();
    String::from_utf8(output).expect("utf8 decompiled surf")
}

#[test]
fn decompile_opaque_module_round_trips_through_check() {
    // RT-1 F4: decompile an opaque module's Deep, then re-parse and
    // re-check the Surf. Both decompiler bugs (multi-segment module
    // path `Stats.prob`, positional record variant
    // `Probability(value: f32)`) and the record-construction
    // separator (`{ value = x }`) broke this path in W1.
    let dir = tempdir().expect("tempdir");
    let dp = dir.path().join("opaque.dp");
    // A single-module opaque fixture: deftype + smart constructor.
    write_file(
        &dp,
        r"(module {}
  stats.prob
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
",
    );

    let decompiled = decompile_dp(&dp);
    // Structural checks on the de-sugared Surf.
    assert!(
        decompiled.contains("module Stats.Prob"),
        "module path must PascalCase every segment: {decompiled}"
    );
    assert!(
        decompiled.contains("| Probability { value: f32 }"),
        "record variant must use braces: {decompiled}"
    );
    assert!(
        decompiled.contains("Probability { value: x }"),
        "record construction must use `:`: {decompiled}"
    );

    // Re-parse + re-check: the decompiled Surf must score 1 clean.
    let ch = dir.path().join("roundtrip.ch");
    write_file(&ch, &decompiled);
    let output = chelis()
        .args(["check", ch.to_str().unwrap()])
        .assert()
        .code(0)
        .get_output()
        .stdout
        .clone();
    let json = check_json(&output);
    assert_eq!(
        json.get("score").and_then(Value::as_f64),
        Some(1.0),
        "decompiled opaque module must round-trip to a clean check: {json}"
    );
    assert!(
        errors_of(&json).is_empty(),
        "decompiled opaque module must have no check errors: {json}"
    );
}

#[test]
fn decompile_non_opaque_multi_segment_module_round_trips() {
    // Negative parity: the multi-segment module-path fix is not
    // opaque-specific. A plain `module Geo.Units` with a positional
    // ADT must also round-trip cleanly.
    let dir = tempdir().expect("tempdir");
    let dp = dir.path().join("plain.dp");
    write_file(
        &dp,
        r"(module {}
  geo.units
  (deftype {} Meters () (variant {} Meters (t-prim {} f32)))
  (defsig {} meters (t-fn {} (t-prim {} f32) (t-adt {} Meters)))
  (def {}
    meters
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (app {} (var {} Meters) (var {} x)))))
",
    );

    let decompiled = decompile_dp(&dp);
    assert!(
        decompiled.contains("module Geo.Units"),
        "non-opaque module path must PascalCase every segment: {decompiled}"
    );
    assert!(
        decompiled.contains("| Meters(f32)"),
        "positional variant must keep parens: {decompiled}"
    );

    let ch = dir.path().join("plain_roundtrip.ch");
    write_file(&ch, &decompiled);
    let output = chelis()
        .args(["check", ch.to_str().unwrap()])
        .assert()
        .code(0)
        .get_output()
        .stdout
        .clone();
    let json = check_json(&output);
    assert_eq!(
        json.get("score").and_then(Value::as_f64),
        Some(1.0),
        "decompiled non-opaque module must round-trip clean: {json}"
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

/// RT3-F4: verify that an `@opaque`-outside-a-module declaration error is
/// (1) VISIBLE on `chelis check` (non-empty errors array, exit non-zero,
/// not a silent score-1 pass) and (2) GATES the build/eval surfaces
/// (exit non-zero). `chelis check` is the scorer-with-exit-code (0 iff the
/// errors array is empty, else non-zero per Issue #207); the
/// build/eval/validate front-ends gate on a non-empty error list. So the
/// declaration error is never silently admitted -- there is no gap.
#[test]
fn rt3_f4_declaration_error_is_visible_on_check_and_gates_build() {
    let dir = tempdir().expect("tempdir");
    let ch = dir.path().join("nomodule.ch");
    write_file(
        &ch,
        "@opaque\n@invariant(p) p.value >= 0.0\ntype T = | T { value: f32 }\n",
    );

    // check: the error is listed AND the exit code is non-zero (it mirrors
    // the non-empty errors array -- check is NOT a silent score-1 pass for
    // a declaration error).
    let output = chelis()
        .args(["check", ch.to_str().unwrap()])
        .assert()
        .code(2)
        .get_output()
        .stdout
        .clone();
    let json = check_json(&output);
    let errors = errors_of(&json);
    assert_eq!(
        opaque_violations(&errors).len(),
        1,
        "the declaration error is listed: {errors:?}"
    );

    // build gates on the declaration error (exit non-zero, does not emit
    // artifacts for a type-broken/declaration-error module).
    chelis()
        .args(["build", ch.to_str().unwrap()])
        .assert()
        .failure();

    // eval --file gates likewise.
    chelis()
        .args(["eval", "--file", ch.to_str().unwrap()])
        .assert()
        .failure();
}
