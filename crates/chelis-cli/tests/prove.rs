use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;
use tempfile::tempdir;

fn write_prop(source: &str) -> tempfile::TempDir {
    let dir = tempdir().expect("tempdir");
    std::fs::write(dir.path().join("prop.ch"), source).expect("write property");
    dir
}

#[cfg(feature = "smt")]
fn write_file(path: &std::path::Path, contents: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create parent");
    }
    std::fs::write(path, contents).expect("write file");
}

fn property_records(output: &[u8]) -> Vec<Value> {
    String::from_utf8_lossy(output)
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str::<Value>(line).expect("json line"))
        .filter(|record| record.get("kind").and_then(Value::as_str) == Some("property"))
        .collect()
}

#[test]
fn prove_passes_filtered_samples() {
    // Pin `--tier fuzz-only` so the test asserts exactly one behavior
    // across the default and `--features smt` builds: a trivial linear
    // property auto-proves at the SMT tier under `--tier auto` in the smt
    // build (which would print "proved (smt)", not "3/3 passed"), so the
    // tier must be pinned for the fuzz assertion to be deterministic.
    let dir = write_prop(
        r#"
@property non_negative forall(x: f32) where x >= 0.0:
  x >= 0.0
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--samples",
            "3",
            "--seed",
            "0",
            "--tier",
            "fuzz-only",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "property: non_negative -- 3/3 passed",
        ));
}

#[test]
fn prove_reports_counterexample_exit_one() {
    let dir = write_prop(
        r#"
@property always_non_negative forall(x: f32):
  x >= 0.0
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--samples",
            "10",
            "--seed",
            "0",
        ])
        .assert()
        .code(1)
        .stdout(predicate::str::contains(
            "property failure: always_non_negative",
        ));
}

#[test]
fn prove_fuzz_json_shrinks_counterexample() {
    let dir = write_prop(
        r#"
@property always_positive forall(x: f32):
  x > 0.0
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--json",
            "--tier",
            "fuzz-only",
            "--samples",
            "10",
            "--seed",
            "0",
        ])
        .output()
        .expect("run prove");
    assert_eq!(
        output.status.code(),
        Some(1),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let props = property_records(&output.stdout);
    assert_eq!(props.len(), 1, "records: {props:?}");
    assert_eq!(props[0]["status"], "failed");
    assert_eq!(props[0]["counterexample"]["x"], 0.0);
    assert_eq!(props[0]["shrink_steps"], 1);
}

#[test]
fn prove_json_schema_has_property_and_summary_records() {
    let dir = write_prop(
        r#"
@property truth forall(x: bool):
  x == x
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--samples",
            "2",
            "--json",
        ])
        .output()
        .expect("run prove");
    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let lines = String::from_utf8(output.stdout).expect("utf8");
    let records = lines
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("json line"))
        .collect::<Vec<_>>();
    assert_eq!(records[0]["kind"], "property");
    assert_eq!(records[0]["name"], "truth");
    assert_eq!(records[0]["status"], "passed");
    assert_eq!(records[1]["kind"], "summary");
    assert_eq!(records[1]["passed"], 1);
}

#[cfg(feature = "smt")]
#[test]
fn prove_surf_reef_input_lowers_against_linked_declarations() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join("myapp");
    write_file(
        &root.join("reef.toml"),
        &format!(
            r#"[package]
name = "myapp"
version = "0.1.0"
compiler = "={}"
module_prefix = "App"

[dependencies]
mylib = {{ path = "./mylib" }}
"#,
            env!("CARGO_PKG_VERSION")
        ),
    );
    write_file(
        &root.join("mylib/reef.toml"),
        &format!(
            r#"[package]
name = "mylib"
version = "0.1.0"
compiler = "={}"
module_prefix = "Mylib"
"#,
            env!("CARGO_PKG_VERSION")
        ),
    );
    write_file(
        &root.join("mylib/src/math.ch"),
        "module Mylib.Math\nexport (double)\ndef double(x: f32) -> f32 = x + x\n",
    );
    let entry = root.join("src/proofs.ch");
    write_file(
        &entry,
        r#"module App.Proofs
import Mylib.Math (double)
@property double_identity forall(x: f32):
  double(x) == x + x
"#,
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            entry.to_str().unwrap(),
            "--json",
            "--tier",
            "smt-only",
        ])
        .output()
        .expect("run prove");
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let props = property_records(&output.stdout);
    assert_eq!(props.len(), 1, "records: {props:?}");
    assert_eq!(props[0]["name"], "double_identity");
    assert_eq!(props[0]["status"], "passed");
    assert_eq!(props[0]["proof_tier"], "smt");
    assert_eq!(props[0]["arith_model"], "real");
}

// chelis#422: a top-level call-form comparison predicate
// (`gte(mul(x, x), 0.0)`) used to return `None` from the predicate-position
// lowering and silently drop to a Tier C fuzz pass, so a measure-zero-false
// call-form rendered a false green. The call-form must now lower to SMT
// exactly like the operator-form.
#[cfg(feature = "smt")]
#[test]
fn call_form_predicate_proves_at_smt_like_operator_form() {
    let dir = write_prop(
        r#"
@property callform_nonneg forall(x: f32):
  gte(mul(x, x), 0.0)
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--tier",
            "auto",
            "--json",
        ])
        .output()
        .expect("run prove");
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let props = property_records(&output.stdout);
    assert_eq!(props.len(), 1, "records: {props:?}");
    assert_eq!(props[0]["name"], "callform_nonneg");
    assert_eq!(props[0]["status"], "passed");
    assert_eq!(
        props[0]["proof_tier"], "smt",
        "call-form must lower to SMT, not drop to fuzz: {}",
        props[0]
    );
    // chelis#422: an SMT proof is over the reals, so the green discloses the
    // machine-arithmetic gap as `proven_modulo_real_arithmetic` with
    // `real_arithmetic` in qualifiers[]. The key invariant for this test is
    // that call-form matches operator-form exactly (same badge, same SMT tier).
    assert_eq!(
        props[0]["composite_verdict"],
        "proven_modulo_real_arithmetic"
    );
    assert_eq!(
        props[0]["qualifiers"],
        serde_json::json!(["real_arithmetic"])
    );
}

// chelis#422: the operator-form of the same property keeps lowering to SMT
// unchanged -- the call-form arm must not perturb the operator path.
#[cfg(feature = "smt")]
#[test]
fn operator_form_predicate_still_proves_at_smt() {
    let dir = write_prop(
        r#"
@property operator_nonneg forall(x: f32):
  (x * x) >= 0.0
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--tier",
            "auto",
            "--json",
        ])
        .output()
        .expect("run prove");
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let props = property_records(&output.stdout);
    assert_eq!(props.len(), 1, "records: {props:?}");
    assert_eq!(props[0]["name"], "operator_nonneg");
    assert_eq!(props[0]["status"], "passed");
    assert_eq!(props[0]["proof_tier"], "smt");
    // chelis#422: over-reals SMT proof discloses the machine-arith gap.
    assert_eq!(
        props[0]["composite_verdict"],
        "proven_modulo_real_arithmetic"
    );
    assert_eq!(
        props[0]["qualifiers"],
        serde_json::json!(["real_arithmetic"])
    );
}

// chelis#422 negative test: a measure-zero-false call-form predicate
// (`(x - 12345.0)^2 > 0.0` under `x > 0`, false at x = 12345.0) used to fuzz
// to a false green because fuzz never sampled the exact root. It must now be
// REFUTED at SMT with the exact counterexample, not fuzz-passed.
#[cfg(feature = "smt")]
#[test]
fn measure_zero_false_call_form_is_refuted_at_smt_not_fuzz_passed() {
    let dir = write_prop(
        r#"
@property callform_false forall(x: f32) where gt(x, 0.0):
  gt(mul(sub(x, 12345.0), sub(x, 12345.0)), 0.0)
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--tier",
            "auto",
            "--samples",
            "100",
            "--json",
        ])
        .output()
        .expect("run prove");
    assert_eq!(
        output.status.code(),
        Some(1),
        "a measure-zero-false call-form must be refuted (exit 1), not fuzz-passed: stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let props = property_records(&output.stdout);
    assert_eq!(props.len(), 1, "records: {props:?}");
    assert_eq!(props[0]["name"], "callform_false");
    assert_eq!(
        props[0]["status"], "failed",
        "must be refuted, not passed: {}",
        props[0]
    );
    assert_eq!(
        props[0]["proof_tier"], "smt",
        "refutation must come from SMT, not fuzz: {}",
        props[0]
    );
    // chelis#422 (symmetric Disproved hedge): the disproof is over the REALS
    // (the goal is Real f32 `sub`/`mul`), so the SMT counterexample may be a
    // false counterexample at machine arithmetic; the badge is the hedged
    // failure `disproved_modulo_real_arithmetic`, NOT a definite `failed`. The
    // coarse `status` is still `failed` and the run still exits 1; the
    // qualifiers[] array discloses `real_arithmetic` symmetric to the proof
    // side. (The predicate hedges every over-reals disproof; it does not
    // separately confirm the counterexample survives machine rounding.)
    assert_eq!(
        props[0]["composite_verdict"],
        "disproved_modulo_real_arithmetic"
    );
    let qualifiers = props[0]["qualifiers"].as_array().expect("qualifiers array");
    assert!(
        qualifiers.iter().any(|q| q == "real_arithmetic"),
        "a hedged disproof discloses real_arithmetic symmetric to the proof side: {}",
        props[0]
    );
    assert_eq!(
        props[0]["counterexample"]["x"], "12345.0",
        "SMT must report the exact root as counterexample: {}",
        props[0]
    );
}

#[cfg(feature = "smt")]
#[test]
fn user_smt_property_json_carries_real_arith_model_and_refutation_model() {
    let dir = write_prop(
        r#"
@property false_claim forall(x: f32):
  x > x
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--json",
            "--tier",
            "smt-only",
        ])
        .output()
        .expect("run prove");
    assert_eq!(
        output.status.code(),
        Some(1),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let props = property_records(&output.stdout);
    assert_eq!(props.len(), 1, "records: {props:?}");
    assert_eq!(props[0]["status"], "failed");
    assert_eq!(props[0]["proof_tier"], "smt");
    assert_eq!(props[0]["arith_model"], "real");
    assert!(
        props[0]["counterexample"].get("x").is_some(),
        "SMT counterexample should surface model bindings: {}",
        props[0]
    );
}

#[cfg(feature = "smt")]
#[test]
fn prove_k1_parity_consumes_bundled_normal_cdf_contract() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join("myapp");
    write_file(
        &root.join("reef.toml"),
        &format!(
            r#"[package]
name = "myapp"
version = "0.1.0"
compiler = "={}"
module_prefix = "App"

[dependencies]
chelis-std = {{ version = "0.4.0" }}
"#,
            env!("CARGO_PKG_VERSION")
        ),
    );
    let entry = root.join("src/proofs.ch");
    write_file(
        &entry,
        r#"module App.Proofs
import Std.Contracts (normal_cdf)
@property put_call_parity_with_cdf_contract forall(s: f32, k: f32, disc: f32, d1: f32, d2: f32)
  where s >= 0.0, k >= 0.0, disc >= 0.0, disc <= 1.0:
  (((s * normal_cdf(d1)) - (k * (disc * normal_cdf(d2)))) - ((k * (disc * normal_cdf(-d2))) - (s * normal_cdf(-d1)))) == (s - (k * disc))
  with contract = "std.normal_cdf.reflection"
"#,
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            entry.to_str().unwrap(),
            "--json",
            "--tier",
            "smt-only",
        ])
        .output()
        .expect("run prove");
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let props = property_records(&output.stdout);
    assert_eq!(props.len(), 1, "records: {props:?}");
    let prop = &props[0];
    assert_eq!(prop["name"], "put_call_parity_with_cdf_contract");
    assert_eq!(prop["status"], "passed");
    assert_eq!(prop["proof_tier"], "smt");
    // chelis#422: the LEGIT contract case is unchanged at the token level -- an
    // exact SMT base discharged modulo a fuzz-validated CONTRACT keeps the
    // weakest token `proven_modulo_fuzz_validated_contract`. Additively, the
    // qualifiers[] array now also discloses `real_arithmetic` (the SMT base is
    // over reals). This is the case the fuzz-only-base fix must NOT collapse.
    assert_eq!(
        prop["composite_verdict"],
        "proven_modulo_fuzz_validated_contract"
    );
    let qualifiers = prop["qualifiers"].as_array().expect("qualifiers array");
    assert!(
        qualifiers.iter().any(|q| q == "fuzz"),
        "the fuzz CONTRACT discharge is disclosed: {prop}"
    );
    assert!(
        qualifiers.iter().any(|q| q == "real_arithmetic"),
        "the over-reals SMT base is disclosed alongside the fuzz contract: {prop}"
    );
    assert!(
        !qualifiers.iter().any(|q| q == "fuzz_base"),
        "the BASE was SMT-proved (not fuzz): must not carry fuzz_base: {prop}"
    );
    let assumptions = prop["assumptions"].as_array().expect("assumptions array");
    assert!(
        assumptions
            .iter()
            .any(|a| a["name"] == "std.normal_cdf.reflection"
                && a["discharge"]["evidence"]["implementation"] == "Std.Contracts.normal_cdf"),
        "reflection assumption must name bundled implementation: {prop}"
    );
}

#[cfg(feature = "smt")]
#[test]
fn contract_rejects_local_linker_shaped_normal_cdf_spoof() {
    let dir = write_prop(
        r#"
def pkg__chelis__std__Std__Contracts__normal_cdf(x: f32) -> f32 = x
@property spoofed_reflection forall(x: f32):
  pkg__chelis__std__Std__Contracts__normal_cdf(-x) == 1.0 - pkg__chelis__std__Std__Contracts__normal_cdf(x)
  with contract = "std.normal_cdf.reflection"
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--json",
            "--tier",
            "smt-only",
        ])
        .output()
        .expect("run prove");
    assert_ne!(
        output.status.code(),
        Some(0),
        "local linker-shaped spoof must not be a clean green\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let props = property_records(&output.stdout);
    assert!(
        props.iter().all(|prop| prop["status"] != "passed"),
        "spoofed contract property must not pass: {props:?}"
    );
}

#[cfg(feature = "smt")]
#[test]
fn reef_rejects_path_dependency_named_chelis_std() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join("myapp");
    write_file(
        &root.join("reef.toml"),
        &format!(
            r#"[package]
name = "myapp"
version = "0.1.0"
compiler = "={}"
module_prefix = "App"

[dependencies]
chelis-std = {{ path = "./fake-std" }}
"#,
            env!("CARGO_PKG_VERSION")
        ),
    );
    write_file(
        &root.join("fake-std/reef.toml"),
        &format!(
            r#"[package]
name = "chelis-std"
version = "0.4.0"
compiler = "={}"
module_prefix = "Std"
"#,
            env!("CARGO_PKG_VERSION")
        ),
    );
    write_file(
        &root.join("fake-std/src/contracts.ch"),
        "module Std.Contracts\nexport (normal_cdf)\ndef normal_cdf(x: f32) -> f32 = x\n",
    );
    let entry = root.join("src/proofs.ch");
    write_file(
        &entry,
        r#"module App.Proofs
import Std.Contracts (normal_cdf)
@property spoofed_reflection forall(x: f32):
  normal_cdf(-x) == 1.0 - normal_cdf(x)
  with contract = "std.normal_cdf.reflection"
"#,
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            entry.to_str().unwrap(),
            "--json",
            "--tier",
            "smt-only",
        ])
        .output()
        .expect("run prove");
    assert_eq!(
        output.status.code(),
        Some(3),
        "path-backed chelis-std must be rejected\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("cannot be supplied as a path dependency"),
        "diagnostic should name chelis-std path rejection: stdout={}",
        String::from_utf8_lossy(&output.stdout)
    );
}

#[test]
fn property_binder_types_are_required() {
    let dir = write_prop(
        r#"
@property missing_type forall(x):
  true
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["prove", dir.path().join("prop.ch").to_str().unwrap()])
        .assert()
        .code(3)
        .stderr(predicate::str::contains("explicit property binder type"));
}

#[test]
fn symbolic_tensor_binder_is_unsupported_exit_two() {
    let dir = write_prop(
        r#"
@property symbolic_tensor forall(x: tensor[n, f32]):
  true
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["prove", dir.path().join("prop.ch").to_str().unwrap()])
        .assert()
        .code(2)
        .stdout(predicate::str::contains(
            "symbolic tensor dimensions are not supported in L2 v1",
        ));
}

#[test]
fn duplicate_property_name_is_parse_error() {
    let dir = write_prop(
        r#"
@property repeated forall(x: f32):
  true
@property repeated forall(y: f32):
  true
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["prove", dir.path().join("prop.ch").to_str().unwrap()])
        .assert()
        .code(3)
        .stderr(predicate::str::contains("unique property name"));
}

#[test]
fn cli_samples_and_seed_override_source_options() {
    // `--samples`/`--seed` overriding the source options is a FUZZ-tier
    // behavior (the SMT tier ignores sample count and reports samples:0).
    // Pin `--tier fuzz-only` so the assertion holds across the default and
    // `--features smt` builds (under `--tier auto` in the smt build, the
    // trivial `x == x` property auto-proves at the SMT tier => samples:0).
    let dir = write_prop(
        r#"
@property source_options forall(x: f32):
  x == x
  with samples = 5
  with seed = 7
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--samples",
            "2",
            "--seed",
            "42",
            "--tier",
            "fuzz-only",
            "--json",
        ])
        .output()
        .expect("run prove");
    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let records = String::from_utf8(output.stdout)
        .expect("utf8")
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("json line"))
        .collect::<Vec<_>>();
    assert_eq!(records[0]["samples"], 2);
    assert_eq!(records[0]["seed"], 42);
}

#[test]
fn f64_tensor_binder_is_supported() {
    let dir = write_prop(
        r#"
@property f64_tensor forall(x: tensor[2, 3, f64]):
  {
    _ = drop(x)
    true
  }
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--samples",
            "1",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "property: f64_tensor -- 1/1 passed",
        ));
}

#[test]
fn tensor_binder_can_be_used_by_shape_queries() {
    let dir = write_prop(
        r#"
@property tensor_shape forall(x: tensor[3, f32]):
  {
    n = shape(x, 0)
    _ = drop(x)
    n == 3
  }
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--samples",
            "1",
            "--seed",
            "0",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "property: tensor_shape -- 1/1 passed",
        ));
}

#[test]
fn desugared_deep_property_uses_source_samples_and_seed() {
    let dir = write_prop(
        r#"
@property source_options forall(x: f32):
  x == x
  with samples = 2
  with seed = 7
"#,
    );
    let deep_path = dir.path().join("prop.dp");
    let deep = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep", dir.path().join("prop.ch").to_str().unwrap()])
        .output()
        .expect("desugar");
    assert!(
        deep.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&deep.stderr)
    );
    std::fs::write(&deep_path, deep.stdout).expect("write deep");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["prove", deep_path.to_str().unwrap(), "--json"])
        .output()
        .expect("run prove");
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let records = String::from_utf8(output.stdout)
        .expect("utf8")
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("json line"))
        .collect::<Vec<_>>();
    assert_eq!(records[0]["samples"], 2);
    assert_eq!(records[0]["seed"], 7);
}

#[test]
fn deep_property_uses_metadata_quantifiers_and_cli_samples() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("prop.dp");
    std::fs::write(
        &path,
        r#"
(def {chelis_role: "property",
      property_source_kind: "user",
      property_quantifiers: (params {} (x {type: (t-prim {} f32)})),
      property_preconditions: (tuple {} (app {} (var {} gte) (var {} x) (lit {type: (t-prim {} f32)} 0.0)))}
  deep_non_negative
  (fn {}
    (params {} (x {type: (t-prim {} f32)}))
    (app {} (var {} gte) (var {} x) (lit {type: (t-prim {} f32)} 0.0))))
"#,
    )
    .expect("write deep");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            path.to_str().unwrap(),
            "--samples",
            "2",
            "--seed",
            "0",
            "--json",
        ])
        .output()
        .expect("run prove");
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let records = String::from_utf8(output.stdout)
        .expect("utf8")
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("json line"))
        .collect::<Vec<_>>();
    assert_eq!(records[0]["name"], "deep_non_negative");
    assert_eq!(records[0]["samples"], 2);
}

// --- WI-7 mandatory non-vacuity: close the CLI-local green bypass ---
//
// The CLI-local Tier-C fuzz path (non-capability user properties and bridge
// c-earchin properties) used to render a green with `"assumptions": []`,
// recording NO non-vacuity even when the property carried preconditions. That
// is the WI-7 bypass: a green should never be reported without recording the
// non-vacuity it established. These tests enumerate the green-rendering paths
// and pin that each now carries an established non-vacuity record, and that the
// vacuous (unsatisfiable-precondition) case cannot reach a green at all.

/// Helper: the single non-vacuity record a green precondition-bearing local
/// property must now carry.
fn assert_established_precondition_non_vacuity(record: &Value, property_name: &str) {
    let assumptions = record["assumptions"]
        .as_array()
        .unwrap_or_else(|| panic!("assumptions must be an array: {record}"));
    assert_eq!(
        assumptions.len(),
        1,
        "a green over a non-empty precondition set records one non-vacuity assumption: {record}"
    );
    let assumption = &assumptions[0];
    assert_eq!(assumption["name"], format!("preconditions:{property_name}"));
    assert_eq!(
        assumption["non_vacuity"]["status"], "established",
        "the precondition non-vacuity is established by the accepted samples: {record}"
    );
    assert_eq!(assumption["discharge"]["method"], "fuzz");
    assert_eq!(assumption["discharge"]["evidence"]["status"], "validated");
}

#[test]
fn wi7_deep_user_green_with_preconditions_carries_established_non_vacuity() {
    // A green user `.dp` property WITH a precondition must carry an established
    // non-vacuity record, not an empty assumption list. (In the default build
    // this is the CLI-local path; the smt build routes user properties through
    // the shared runner, which records non-vacuity too.)
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("prop.dp");
    std::fs::write(
        &path,
        r#"
(def {chelis_role: "property",
      property_source_kind: "user",
      property_quantifiers: (params {} (x {type: (t-prim {} f32)})),
      property_preconditions: (tuple {} (app {} (var {} gte) (var {} x) (lit {type: (t-prim {} f32)} 0.0)))}
  deep_pre_green
  (fn {}
    (params {} (x {type: (t-prim {} f32)}))
    (app {} (var {} gte) (var {} x) (lit {type: (t-prim {} f32)} 0.0))))
"#,
    )
    .expect("write deep");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            path.to_str().unwrap(),
            "--tier",
            "fuzz-only",
            "--samples",
            "4",
            "--seed",
            "0",
            "--json",
        ])
        .output()
        .expect("run prove");
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let records = property_records(&output.stdout);
    let record = &records[0];
    assert_eq!(record["status"], "passed");
    // chelis#422: a fuzz-only base (--tier fuzz-only, no SMT) is empirically
    // validated, not proven. The honest badge is `fuzz_validated`, NEVER a
    // `proven_*` badge -- the proven-modulo-fuzz-CONTRACT badge is reserved for
    // an exact SMT base discharged modulo a fuzz contract.
    assert_eq!(record["composite_verdict"], "fuzz_validated");
    assert!(
        !record["composite_verdict"]
            .as_str()
            .unwrap()
            .starts_with("proven"),
        "a fuzz-only base must never read as proven_*: {record}"
    );
    // The fuzz-only base contributes `fuzz_base` (which dominates the badge to
    // `fuzz_validated`); the precondition's fuzz discharge may additionally
    // disclose `fuzz`. The invariant under test: `fuzz_base` is present and the
    // badge is never proven_* -- a fuzz-only base is not a proof.
    let qualifiers = record["qualifiers"].as_array().expect("qualifiers array");
    assert!(
        qualifiers.iter().any(|q| q == "fuzz_base"),
        "a fuzz-only base discloses fuzz_base: {record}"
    );
    assert_established_precondition_non_vacuity(record, "deep_pre_green");
}

#[test]
fn wi7_deep_user_green_without_preconditions_carries_no_assumption() {
    // A green property with NO preconditions has nothing that could be vacuous,
    // so it carries an empty assumption list -- non-vacuity is only recorded
    // where there is an assumption set to be non-vacuous about.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("prop.dp");
    std::fs::write(
        &path,
        r#"
(def {chelis_role: "property",
      property_source_kind: "user",
      property_quantifiers: (params {} (x {type: (t-prim {} f32)})),
      property_preconditions: (tuple {})}
  deep_no_pre_green
  (fn {}
    (params {} (x {type: (t-prim {} f32)}))
    (app {} (var {} gte) (var {} x) (var {} x))))
"#,
    )
    .expect("write deep");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            path.to_str().unwrap(),
            "--tier",
            "fuzz-only",
            "--samples",
            "4",
            "--seed",
            "0",
            "--json",
        ])
        .output()
        .expect("run prove");
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let records = property_records(&output.stdout);
    assert_eq!(records[0]["status"], "passed");
    assert_eq!(
        records[0]["assumptions"]
            .as_array()
            .map(Vec::len)
            .unwrap_or(usize::MAX),
        0,
        "a no-precondition green records no non-vacuity assumption: {}",
        records[0]
    );
}

#[test]
fn wi7_vacuous_preconditions_cannot_reach_a_green_on_the_local_path() {
    // The negative test: a property whose preconditions are jointly
    // unsatisfiable cannot reach a green by the CLI-local fuzz path. Rejection
    // sampling never accepts a sample, so the run exhausts into a
    // generator-exhaustion error (exit 3), never a green -- there is no
    // green-rendering path for a vacuous precondition set.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("prop.dp");
    std::fs::write(
        &path,
        r#"
(def {chelis_role: "property",
      property_source_kind: "user",
      property_quantifiers: (params {} (x {type: (t-prim {} f32)})),
      property_preconditions: (tuple {}
        (app {} (var {} gt) (var {} x) (lit {type: (t-prim {} f32)} 0.0))
        (app {} (var {} lt) (var {} x) (lit {type: (t-prim {} f32)} 0.0)))}
  deep_vacuous
  (fn {}
    (params {} (x {type: (t-prim {} f32)}))
    (app {} (var {} gte) (var {} x) (var {} x))))
"#,
    )
    .expect("write deep");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            path.to_str().unwrap(),
            "--tier",
            "fuzz-only",
            "--samples",
            "4",
            "--seed",
            "0",
            "--json",
        ])
        .output()
        .expect("run prove");
    assert!(
        !output.status.success(),
        "a vacuous precondition set must not exit success: stdout={}",
        String::from_utf8_lossy(&output.stdout)
    );
    let records = property_records(&output.stdout);
    assert_ne!(
        records[0]["status"], "passed",
        "vacuous preconditions cannot render a green: {}",
        records[0]
    );
    assert_ne!(
        records[0]["composite_verdict"], "proven",
        "vacuous preconditions cannot render proven: {}",
        records[0]
    );
    assert_ne!(
        records[0]["composite_verdict"], "proven_modulo_fuzz_validated_contract",
        "vacuous preconditions cannot render a green badge: {}",
        records[0]
    );
    assert_ne!(
        records[0]["composite_verdict"], "fuzz_validated",
        "vacuous preconditions cannot render any green badge, fuzz_validated included: {}",
        records[0]
    );
}

#[test]
fn deep_property_quantifiers_must_match_fn_params() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("mismatch.dp");
    std::fs::write(
        &path,
        r#"
(def {chelis_role: "property",
      property_source_kind: "user",
      property_quantifiers: (params {} (x {type: (t-prim {} f32)})),
      property_preconditions: (tuple {})}
  mismatch
  (fn {}
    (params {} (y {type: (t-prim {} f32)}))
    (lit {type: (t-prim {} bool)} true)))
"#,
    )
    .expect("write deep");
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["prove", path.to_str().unwrap()])
        .assert()
        .code(3)
        .stderr(predicate::str::contains(
            "property_quantifiers must match fn parameters",
        ));
}

#[test]
fn deep_property_source_kind_must_be_valid_and_drives_json_source_kind() {
    let dir = tempdir().expect("tempdir");
    let invalid = dir.path().join("invalid.dp");
    std::fs::write(
        &invalid,
        r#"
(def {chelis_role: "property",
      property_source_kind: "bogus",
      property_quantifiers: (params {}),
      property_preconditions: (tuple {})}
  invalid_kind
  (fn {} (params {}) (lit {type: (t-prim {} bool)} true)))
"#,
    )
    .expect("write invalid deep");
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["prove", invalid.to_str().unwrap()])
        .assert()
        .code(3)
        .stderr(predicate::str::contains("property_source_kind"));

    let user = dir.path().join("user.dp");
    std::fs::write(
        &user,
        r#"
(def {chelis_role: "property",
      property_source_kind: "user",
      property_quantifiers: (params {}),
      property_preconditions: (tuple {})}
  user_kind
  (fn {} (params {}) (lit {type: (t-prim {} bool)} true)))
"#,
    )
    .expect("write user deep");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["prove", user.to_str().unwrap(), "--json"])
        .output()
        .expect("run prove");
    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let records = String::from_utf8(output.stdout)
        .expect("utf8")
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("json line"))
        .collect::<Vec<_>>();
    assert_eq!(records[0]["source"]["kind"], "user");
}

#[test]
fn spans_override_requires_single_explicit_deep_file() {
    let dir = tempfile::Builder::new()
        .prefix("chelis-prove-spans")
        .tempdir()
        .expect("tempdir");
    for name in ["one.dp", "two.dp"] {
        std::fs::write(
            dir.path().join(name),
            r#"
(def {chelis_role: "property",
      property_source_kind: "user",
      property_quantifiers: (params {}),
      property_preconditions: (tuple {})}
  trivial
  (fn {} (params {}) (lit {type: (t-prim {} bool)} true)))
"#,
        )
        .expect("write deep");
    }
    let spans = dir.path().join("override.spans.json");
    std::fs::write(&spans, "{}").expect("write spans");
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().to_str().unwrap(),
            "--spans",
            spans.to_str().unwrap(),
        ])
        .assert()
        .code(3)
        .stderr(predicate::str::contains(
            "--spans can only be used with a single explicit .dp input",
        ));
}

#[test]
fn bridge_failure_resolves_spans_manifest_to_ears_source() {
    let dir = tempdir().expect("tempdir");
    let deep = dir.path().join("options.dp");
    let spans = dir.path().join("options.spans.json");
    std::fs::write(
        &deep,
        r#"
(def {chelis_role: "property",
      property_source_kind: "bridge:c-earchin",
      property_source_id: "FIN-003",
      property_quantifiers: (params {}),
      property_preconditions: (tuple {})}
  req_FIN_003
  (fn {} (params {}) (lit {type: (t-prim {} bool)} false)))
"#,
    )
    .expect("write deep");
    std::fs::write(
        &spans,
        r#"
{
  "source": "references/finance_options/options_rules.ears",
  "source_hash": "sha256:test",
  "spans": [
    {
      "deep_node_id": "req_FIN_003",
      "deep_path": "module.def[2]",
      "ears_id": "FIN-003",
      "ears_file": "references/finance_options/options_rules.ears",
      "ears_text": "WHILE the exchange is open, the portfolio delta shall be at most the limit.",
      "ears": {
        "start_byte": 143,
        "end_byte": 224,
        "start_line": 3,
        "start_column": 1,
        "end_line": 3,
        "end_column": 81
      },
      "clauses": []
    }
  ]
}
"#,
    )
    .expect("write spans");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            deep.to_str().unwrap(),
            "--spans",
            spans.to_str().unwrap(),
        ])
        .assert()
        .code(1)
        .stdout(predicate::str::contains(
            "references/finance_options/options_rules.ears:3:1 FIN-003",
        ))
        .stdout(predicate::str::contains(
            "WHILE the exchange is open, the portfolio delta shall be at most the limit.",
        ));

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            deep.to_str().unwrap(),
            "--spans",
            spans.to_str().unwrap(),
            "--json",
        ])
        .output()
        .expect("run prove");
    assert_eq!(output.status.code(), Some(1));
    let records = String::from_utf8(output.stdout)
        .expect("utf8")
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("json line"))
        .collect::<Vec<_>>();
    assert_eq!(records[0]["source"]["requirement"]["id"], "FIN-003");
    assert_eq!(records[0]["source"]["requirement"]["line"], 3);
    assert_eq!(
        records[0]["source"]["requirement"]["text"],
        "WHILE the exchange is open, the portfolio delta shall be at most the limit."
    );
}

#[test]
fn wi8_bridge_green_assumption_discharge_tier_joins_to_c_earchin_source_id() {
    // WI-8: the artifact JSON shows the prover-stamped discharge_tier joined to
    // its c-earchin source id. A bridge (c-earchin) property carrying a
    // precondition, proved green through the CLI-local fuzz path, records the
    // WI-7 precondition non-vacuity assumption; that assumption now carries a
    // prover-stamped discharge_tier (engine + guarantee + the source identity
    // it is keyed to), and the same artifact record carries the c-earchin
    // source id (source.requirement.id), so the tier joins to the source. The
    // c-earchin emission (the def metadata + spans) carries only source
    // identity -- the property_source_id and the requirement -- and NO tier.
    let dir = tempdir().expect("tempdir");
    let deep = dir.path().join("req.dp");
    let spans = dir.path().join("req.spans.json");
    std::fs::write(
        &deep,
        r#"
(def {chelis_role: "property",
      property_source_kind: "bridge:c-earchin",
      property_source_id: "FIN-007",
      property_quantifiers: (params {} (x {type: (t-prim {} f32)})),
      property_preconditions: (tuple {} (app {} (var {} gte) (var {} x) (lit {type: (t-prim {} f32)} 0.0)))}
  req_FIN_007
  (fn {}
    (params {} (x {type: (t-prim {} f32)}))
    (app {} (var {} gte) (var {} x) (lit {type: (t-prim {} f32)} 0.0))))
"#,
    )
    .expect("write deep");
    std::fs::write(
        &spans,
        r#"
{
  "source": "references/finance_options/options_rules.ears",
  "source_hash": "sha256:test",
  "spans": [
    {
      "deep_node_id": "req_FIN_007",
      "deep_path": "module.def[2]",
      "ears_id": "FIN-007",
      "ears_file": "references/finance_options/options_rules.ears",
      "ears_text": "The premium shall be non-negative.",
      "ears": {
        "start_byte": 1,
        "end_byte": 34,
        "start_line": 7,
        "start_column": 1,
        "end_line": 7,
        "end_column": 34
      },
      "clauses": []
    }
  ]
}
"#,
    )
    .expect("write spans");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            deep.to_str().unwrap(),
            "--spans",
            spans.to_str().unwrap(),
            "--tier",
            "fuzz-only",
            "--samples",
            "4",
            "--seed",
            "0",
            "--json",
        ])
        .output()
        .expect("run prove");
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let records = property_records(&output.stdout);
    let record = &records[0];
    assert_eq!(record["status"], "passed");

    // The c-earchin source identity is in the artifact (prover-untouched).
    assert_eq!(
        record["source"]["requirement"]["id"], "FIN-007",
        "the artifact carries the c-earchin source id: {record}"
    );

    // The prover-stamped discharge tier is on the (precondition non-vacuity)
    // assumption, joined to its source identity.
    let assumption = &record["assumptions"][0];
    let tier = &assumption["discharge_tier"];
    assert_eq!(
        tier["engine"], "fuzz-sampler",
        "the tier names the discharging engine: {record}"
    );
    assert_eq!(
        tier["guarantee"], "fuzz",
        "the tier names the guarantee kind: {record}"
    );
    assert_eq!(
        tier["source"], "preconditions:req_FIN_007",
        "the tier is keyed to the assumption source identity: {record}"
    );
}

#[test]
fn prove_accepts_dotted_deep_symbols_for_bridge_references() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bridge.dp");
    std::fs::write(
        &path,
        r#"
(module {}
  cearchin.generated.vocabularymiss
  (def {span: "req:MISS-001", c_earchin_role: "property_witness"}
    req_MISS_001
    (fn {} (params {}) (lit {type: (t-prim {} bool)} true))))
"#,
    )
    .expect("write deep");
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["prove", path.to_str().unwrap(), "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"name\":\"req_MISS_001\""));
}

// The producer-obligation machinery is gated behind the `smt` feature
// (`chelis-prove` is an optional dep enabled only by `smt`). In a default
// build `chelis prove` cannot check obligations, so it warns on stderr when
// a module declares invariant-carrying opaque types -- without touching the
// stdout NDJSON stream or the exit code. These assertions only hold in the
// non-smt build; under `--features smt` the warning does not exist and the
// obligation oracles cover the behavior instead.
#[cfg(not(feature = "chelis-prove"))]
#[test]
fn non_smt_prove_warns_for_invariant_carrying_opaque_types() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("inv.ch");
    std::fs::write(
        &path,
        "module Stats.Prob\n\
         export (probability)\n\
         @opaque\n\
         @invariant(p) ((p.value >= 0.0) && (p.value <= 1.0))\n\
         type Probability =\n  | Probability { value: f32 }\n\
         def probability(x: f32) -> Option[Probability] = if ((x >= 0.0) && (x <= 1.0)) then Some(Probability { value: x }) else None\n",
    )
    .expect("write");
    let assert = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["prove", path.to_str().unwrap(), "--json"])
        .assert()
        .success();
    let out = assert.get_output();
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Warning is on stderr, names the count and the smt remedy.
    assert!(
        stderr.contains("obligation verification requires the smt-enabled build")
            && stderr.contains("Rebuild with --features smt"),
        "expected stderr warning; stderr={stderr}"
    );
    // Machine-facing (review residual-risk): under --json the skipped
    // obligation verification is ALSO surfaced as a stdout record, so a
    // consumer reading stdout alone does not mistake a clean summary for a
    // verified proof run.
    let records: Vec<Value> = stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .collect();
    let skipped = records
        .iter()
        .find(|v| v["kind"] == "warning" && v["stage"] == "obligations")
        .expect("a machine-facing skipped-obligations warning record on stdout");
    assert_eq!(
        skipped["skipped"], 1,
        "the record names the count of unverified invariant-carrying types: {skipped}"
    );
    let summary = records
        .iter()
        .find(|v| v["kind"] == "summary")
        .expect("a summary record on stdout");
    assert_eq!(
        summary["obligations"], 0,
        "non-smt build checks no obligations"
    );
}

// CR2-5: a `chelis-prove`-without-`smt` build compiles the obligation
// machinery and runs it via Tier C (fuzz), NOT cvc5. The warning must
// still fire (gated on the `smt` capability, not on the optional
// `chelis-prove` dependency), so a clean fuzz-only run is not mistaken for
// formal SMT verification. This config compiles the obligation path, so
// obligation records DO appear on stdout -- but the stderr warning is still
// present and the exit code is unchanged.
#[cfg(all(feature = "chelis-prove", not(feature = "smt")))]
#[test]
fn chelis_prove_without_smt_still_warns_obligations_not_smt_verified() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("inv.ch");
    std::fs::write(
        &path,
        "module Stats.Prob\n\
         export (probability)\n\
         @opaque\n\
         @invariant(p) ((p.value >= 0.0) && (p.value <= 1.0))\n\
         type Probability =\n  | Probability { value: f32 }\n\
         def probability(x: f32) -> Option[Probability] = if ((x >= 0.0) && (x <= 1.0)) then Some(Probability { value: x }) else None\n",
    )
    .expect("write");
    let assert = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["prove", path.to_str().unwrap(), "--json"])
        .assert()
        .success();
    let out = assert.get_output();
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    // The stderr warning fires even though obligations ran (via fuzz).
    assert!(
        stderr.contains("obligation verification requires the smt-enabled build")
            && stderr.contains("Rebuild with --features smt"),
        "chelis-prove-without-smt must still warn; stderr={stderr}"
    );
    // The warning never leaks into the stdout NDJSON stream.
    assert!(
        !stdout.contains("obligation verification requires"),
        "warning must not leak into stdout; stdout={stdout}"
    );
}

#[cfg(not(feature = "chelis-prove"))]
#[test]
fn non_smt_prove_does_not_warn_for_plain_property_file() {
    let dir = write_prop(
        r#"
@property nonneg forall(x: f32):
  (x * x) >= 0.0
"#,
    );
    let assert = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--json",
        ])
        .assert()
        .success();
    let stderr = String::from_utf8_lossy(&assert.get_output().stderr).into_owned();
    assert!(
        !stderr.contains("obligation verification requires"),
        "a module with no invariant-carrying opaque type must not warn; stderr={stderr}"
    );
}

// chelis#422 negative test (non-smt build): a measure-zero-false @property
// (`(x - 12345.0)^2 > 0.0` under `x > 0`, false at x = 12345.0) used to
// fuzz-pass with the proven-flavored `proven_modulo_fuzz_validated_contract`
// badge -- a false green, because fuzz never sampled the exact root and the
// non-smt build has no SMT to refute it. On a non-smt build the run must now
// be HONEST: the green carries `fuzz_validated` (never `proven_*`) with
// `qualifiers:["fuzz_base"]`. Exit stays success -- a fuzz pass is still a
// pass, it just is not proven.
#[cfg(not(feature = "smt"))]
#[test]
fn non_smt_measure_zero_false_property_is_not_a_proven_green() {
    let dir = write_prop(
        r#"
@property always_positive forall(x: f32) where x > 0.0:
  (x - 12345.0) * (x - 12345.0) > 0.0
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--samples",
            "100",
            "--seed",
            "0",
            "--json",
        ])
        .output()
        .expect("run prove");
    assert!(
        output.status.success(),
        "a clean fuzz pass still exits success: stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let records = property_records(&output.stdout);
    assert_eq!(records.len(), 1, "records: {records:?}");
    let record = &records[0];
    assert_eq!(record["status"], "passed");
    // The core of the fix: the badge is the honest empirical one, NEVER a
    // proven-flavored badge, for a fuzz-only base on a non-smt build.
    assert_eq!(record["composite_verdict"], "fuzz_validated");
    assert!(
        !record["composite_verdict"]
            .as_str()
            .unwrap()
            .starts_with("proven"),
        "a measure-zero-false fuzz pass must never read as proven_*: {record}"
    );
    let qualifiers = record["qualifiers"].as_array().expect("qualifiers array");
    assert!(
        qualifiers.iter().any(|q| q == "fuzz_base"),
        "a fuzz-only base discloses fuzz_base: {record}"
    );
}

// chelis#422 positive twin (non-smt build): an honest fuzz green that happens
// to be TRUE still carries `fuzz_validated`, not `proven_*` -- the distinction
// is about the verification METHOD (fuzz vs SMT), not the truth of the
// property. A non-smt build can never SMT-prove, so it never mints a proven
// badge even for a true property.
#[cfg(not(feature = "smt"))]
#[test]
fn non_smt_true_property_green_is_empirical_not_proven() {
    let dir = write_prop(
        r#"
@property nonneg forall(x: f32):
  (x * x) >= 0.0
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--samples",
            "8",
            "--seed",
            "0",
            "--json",
        ])
        .output()
        .expect("run prove");
    assert!(output.status.success());
    let records = property_records(&output.stdout);
    assert_eq!(records[0]["status"], "passed");
    assert_eq!(records[0]["composite_verdict"], "fuzz_validated");
}

// ===================================================================
// Review 5: the whitelist gate's abort-proof guarantee, made executable.
//
// A cvc5 process-abort cannot be caught in-process (it bare-exits the
// process), so the guarantee is asserted at the SUBPROCESS level: a
// diverse corpus of predicates -- including the exact shapes that aborted
// cvc5 under the old blacklist (a transcendental over an Int arg, a
// Bool-vs-Int comparison, a quantifier, a mixed-sort intrinsic, nested
// intrinsics) -- run through `chelis prove` must EACH return a clean
// status with non-empty stdout, never the abort signature (empty stdout +
// the cvc5 sort/type error on stderr).
//
// Gated on `smt` because Tier B (cvc5 lowering) only runs in the smt
// build; without it Tier B is a no-op and no abort is possible.
#[cfg(feature = "smt")]
#[test]
fn whitelist_corpus_never_aborts_cvc5() {
    // Each entry is a self-contained module exercising a distinct cvc5-risky
    // shape, via either a user @property (checker + Tier B/C) or an opaque
    // invariant whose DERIVED obligation reaches the Tier B whitelist gate
    // and cvc5 lowering directly. Auto tier so the lowering is exercised.
    let corpus: &[&str] = &[
        // --- user @property shapes ---
        // sqrt over an int param (transcendental-over-Int abort shape).
        "module M\n@property sqrt_int forall(n: int32):\n  sqrt(n) >= 0.0\n",
        // min/max mixing an int param and a float literal (mixed-sort min/max).
        "module M\n@property min_mixed forall(n: int32):\n  min(n, 0.5) <= 0.5\n",
        "module M\n@property max_mixed forall(n: int32):\n  max(n, 0.5) >= 0.5\n",
        // a bool param compared with an int param (Bool-vs-Int comparison).
        "module M\n@property bool_vs_int forall(b: bool, n: int32):\n  b == (n >= 0)\n",
        // nested intrinsics over reals (lowerable -- must NOT abort, may prove).
        "module M\n@property nested_real forall(x: f32):\n  sqrt(abs(x)) >= 0.0\n",
        // exp/sin/cos over an int param (more transcendental-over-Int shapes).
        "module M\n@property exp_int forall(n: int32):\n  exp(n) >= 0.0\n",
        "module M\n@property sin_int forall(n: int32):\n  sin(n) <= 1.0\n",
        // a clean f32 linear property (the flagship -- must still run clean).
        "module M\n@property linear_real forall(x: f32):\n  (x - 1.0) <= x\n",
        // int comparison against an int literal (all-int -- clean).
        "module M\n@property int_cmp forall(n: int32) where n >= 0:\n  n + 1 >= 1\n",
        // mixed int/real arithmetic in one comparison (Int-vs-Real abort shape).
        "module M\n@property mixed_arith forall(n: int32, x: f32):\n  n + x >= 0.0\n",
        // --- opaque invariants whose DERIVED obligation reaches Tier B/cvc5 ---
        // The f32 flagship: the obligation proves at proof_tier:smt (the
        // whitelist must admit it).
        "module M\nexport (probability)\n@opaque\n@invariant(p) p.value >= 0.0 && p.value <= 1.0\ntype Probability =\n  | Probability { value: f32 }\ndef probability(x: f32) -> Option[Probability] =\n  if x >= 0.0 && x <= 1.0 then Some(Probability { value: x }) else None\n",
        // An int32-field opaque: the obligation lowers an all-Int comparison
        // (the whitelist admits it; a mistaken Real-const lowering would have
        // aborted).
        "module M\nexport (mk)\n@opaque\n@invariant(c) c.n >= 0\ntype Counter =\n  | Counter { n: int32 }\ndef mk(x: int32) -> Option[Counter] =\n  if x >= 0 then Some(Counter { n: x }) else None\n",
        // An int8-field opaque: the obligation lowers an all-int8 comparison
        // (single-source int width; the obligation must not abort).
        "module M\nexport (mk8)\n@opaque\n@invariant(c) c.n >= (0 : int8)\ntype Counter8 =\n  | Counter8 { n: int8 }\ndef mk8(x: int8) -> Option[Counter8] =\n  if x >= (0 : int8) then Some(Counter8 { n: x }) else None\n",
        // A tensor-field opaque with a sqrt invariant over a Real sum (the
        // intrinsic-over-Real path; must not abort).
        "module M\nexport (mk_simplex)\n@opaque\n@invariant(p) sqrt(sum(p.weights)) >= 0.0\ntype Simplex =\n  | Simplex { weights: tensor[3, f32] }\ndef mk_simplex(w: tensor[3, f32]) -> Simplex = Simplex { weights: w }\n",
    ];

    // The cvc5 sort/type abort messages a malformed term would emit.
    let abort_markers = [
        "Subexpressions must have the same type",
        "Expecting a real term",
        "argument of bound var list",
        "Branches of the ITE must have comparable type",
        "doesn't include THEORY_QUANTIFIERS",
    ];

    for (i, module) in corpus.iter().enumerate() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join(format!("corpus_{i}.ch"));
        std::fs::write(&path, module).expect("write");
        let output = Command::cargo_bin("chelis")
            .expect("binary")
            // The opaque corpus entries synthesize ad-hoc opaque construction
            // that the opaque-domain-construction lint blocks; disable the
            // style gate so the prove pipeline runs (the lint discipline is
            // covered elsewhere). Harmless for the @property entries.
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args([
                "prove",
                path.to_str().unwrap(),
                "--json",
                "--tier",
                "auto",
                "--seed",
                "0",
            ])
            .output()
            .expect("run prove");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        // The abort signature: the process did not exit cleanly (signal /
        // None code) OR cvc5 printed a sort/type abort error. Either means a
        // term reached cvc5 that should have routed to Tier C.
        assert!(
            output.status.code().is_some(),
            "module {i} aborted the prove process (no exit code); stderr={stderr}\nmodule={module}"
        );
        for marker in abort_markers {
            assert!(
                !stderr.contains(marker),
                "module {i} hit the cvc5 abort marker `{marker}`; the whitelist must route it to Tier C\nstderr={stderr}\nmodule={module}"
            );
        }
        // A clean run emits at least a property / obligation / summary NDJSON
        // record on stdout (never the empty-stdout abort signature).
        assert!(
            stdout.lines().any(|l| {
                serde_json::from_str::<Value>(l)
                    .ok()
                    .and_then(|v| v.get("kind").and_then(|k| k.as_str()).map(String::from))
                    .is_some_and(|k| k == "summary" || k == "property" || k == "obligation")
            }),
            "module {i} produced no record (possible abort); stdout={stdout}\nstderr={stderr}\nmodule={module}"
        );
    }
}

// chelis#426 SOUNDNESS: a goal comparing two calls of a def whose body CALLS
// an ITE-bodied helper (an `fmax`-based max-over-actions), with DIFFERENT
// arguments, used to false-prove. The nested-call inlining lost the outer
// call-site's argument bindings, so both calls collapsed to identical SMT
// terms (the all-args-equal corner) and a mathematically FALSE goal reported
// `passed / smt / proven`. It must now be REFUTED with a counterexample, and
// that counterexample must bind the swapped `w*` variables (proof the two
// call-sites lower to DISTINCT terms, not the collapsed corner).
#[cfg(feature = "smt")]
#[test]
fn issue_426_two_calls_of_ite_bodied_def_refute_not_false_prove() {
    let dir = write_prop(
        r#"
def fmax(a: f32, b: f32) -> f32 = if (a >= b) then a else b
def bs(v0: f32, v1: f32, r0: f32, p00: f32, p01: f32, r1: f32, p10: f32, p11: f32, g: f32) -> f32 =
  fmax((r0 + (g * ((p00 * v0) + (p01 * v1)))), (r1 + (g * ((p10 * v0) + (p11 * v1)))))

@property cmp_unguarded forall(v0: f32, v1: f32, w0: f32, w1: f32, r0: f32, p00: f32, p01: f32, r1: f32, p10: f32, p11: f32, g: f32):
  (bs(v0, v1, r0, p00, p01, r1, p10, p11, g) <= bs(w0, w1, r0, p00, p01, r1, p10, p11, g))

@property sub_unguarded forall(v0: f32, v1: f32, w0: f32, w1: f32, r0: f32, p00: f32, p01: f32, r1: f32, p10: f32, p11: f32, g: f32):
  ((bs(v0, v1, r0, p00, p01, r1, p10, p11, g) - bs(w0, w1, r0, p00, p01, r1, p10, p11, g)) <= 0.0)
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--tier",
            "smt-only",
            "--json",
        ])
        .output()
        .expect("run prove");
    assert_eq!(
        output.status.code(),
        Some(1),
        "a FALSE two-call goal must be refuted (exit 1), not false-proven: stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let props = property_records(&output.stdout);
    assert_eq!(props.len(), 2, "records: {props:?}");
    for prop in &props {
        let name = prop["name"].as_str().unwrap();
        assert_eq!(
            prop["status"], "failed",
            "{name} is FALSE in general and must be refuted, not proven: {prop}"
        );
        assert_eq!(
            prop["proof_tier"], "smt",
            "{name} refutation must come from SMT (the false-prove tier): {prop}"
        );
        // The over-reals disproof discloses the machine-arith gap symmetrically
        // (chelis#422 hedge); the coarse status stays `failed` and exits 1.
        assert_eq!(
            prop["composite_verdict"], "disproved_modulo_real_arithmetic",
            "{name}: {prop}"
        );
        // The counterexample must exist AND bind the swapped `w*` variables:
        // their presence proves the two call-sites lowered to DISTINCT terms
        // (the collapse dropped `w*` entirely, so a green carried no model).
        let cex = prop
            .get("counterexample")
            .and_then(Value::as_object)
            .unwrap_or_else(|| panic!("{name} has no counterexample object: {prop}"));
        assert!(
            cex.contains_key("w0") && cex.contains_key("w1"),
            "{name} counterexample must bind the swapped w0/w1 (proof the calls did not collapse): {prop}"
        );
    }
}

// chelis#426 positive twin: the fix must not over-correct into false-refuting
// TRUE goals over the same nested-call ITE-bodied shape. A reflexive goal
// (same args both sides) and the genuine `max >= each-branch` facts of the
// `fmax`-bodied operator still prove at SMT.
#[cfg(feature = "smt")]
#[test]
fn issue_426_true_goals_over_ite_bodied_def_still_prove() {
    let dir = write_prop(
        r#"
def fmax(a: f32, b: f32) -> f32 = if (a >= b) then a else b
def bs(v0: f32, v1: f32, r0: f32, p00: f32, p01: f32, r1: f32, p10: f32, p11: f32, g: f32) -> f32 =
  fmax((r0 + (g * ((p00 * v0) + (p01 * v1)))), (r1 + (g * ((p10 * v0) + (p11 * v1)))))

@property reflexive forall(v0: f32, v1: f32, r0: f32, p00: f32, p01: f32, r1: f32, p10: f32, p11: f32, g: f32):
  (bs(v0, v1, r0, p00, p01, r1, p10, p11, g) <= bs(v0, v1, r0, p00, p01, r1, p10, p11, g))

@property max_ge_first forall(v0: f32, v1: f32, r0: f32, p00: f32, p01: f32, r1: f32, p10: f32, p11: f32, g: f32):
  (bs(v0, v1, r0, p00, p01, r1, p10, p11, g) >= (r0 + (g * ((p00 * v0) + (p01 * v1)))))

@property max_ge_second forall(v0: f32, v1: f32, r0: f32, p00: f32, p01: f32, r1: f32, p10: f32, p11: f32, g: f32):
  (bs(v0, v1, r0, p00, p01, r1, p10, p11, g) >= (r1 + (g * ((p10 * v0) + (p11 * v1)))))
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--tier",
            "smt-only",
            "--json",
        ])
        .output()
        .expect("run prove");
    assert!(
        output.status.success(),
        "all three goals are TRUE and must prove (exit 0): stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let props = property_records(&output.stdout);
    assert_eq!(props.len(), 3, "records: {props:?}");
    for prop in &props {
        let name = prop["name"].as_str().unwrap();
        assert_eq!(prop["status"], "passed", "{name} is TRUE: {prop}");
        assert_eq!(
            prop["proof_tier"], "smt",
            "{name} must prove at SMT, not drop to fuzz: {prop}"
        );
        assert_eq!(
            prop["composite_verdict"], "proven_modulo_real_arithmetic",
            "{name}: {prop}"
        );
    }
}

// ===================================================================
// chelis#436: the discharged proposition (goal) travels with the record.
//
// A verification UI must bind the verdict to the exact claim it certifies. The
// prove-JSON property and obligation records now carry a `goal` field with the
// canonical text of the proposition the prover discharged, so a consumer never
// reconstructs the claim from source (which can drift). These tests pin the
// goal's PRESENCE and exact VALUE across the green/refuted/obligation surfaces,
// with negative parity for the bodiless discovery-error record.
// ===================================================================

// Obligation records only exist under `--features smt` (the obligation engine
// is gated on `chelis-prove`); the only consumer is the smt-gated obligation
// goal test below, so the helper is gated to match.
#[cfg(feature = "smt")]
fn obligation_records(output: &[u8]) -> Vec<Value> {
    String::from_utf8_lossy(output)
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str::<Value>(line).expect("json line"))
        .filter(|record| record.get("kind").and_then(Value::as_str) == Some("obligation"))
        .collect()
}

// A fuzz-tier property pass carries its body as the goal, in canonical text.
// Runs in BOTH lanes: the smt build routes a fuzz-fallback through the shared
// runner, the default build through the CLI-local fuzz path; both must emit the
// same goal.
#[test]
fn goal_field_carries_the_property_body_for_a_fuzz_pass() {
    let dir = write_prop(
        r#"
@property log_below_self forall(x: f32) where x > 0.0:
  log(x) < x
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--tier",
            "fuzz-only",
            "--samples",
            "16",
            "--seed",
            "0",
            "--json",
        ])
        .output()
        .expect("run prove");
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let records = property_records(&output.stdout);
    assert_eq!(records.len(), 1, "records: {records:?}");
    assert_eq!(records[0]["status"], "passed");
    assert_eq!(
        records[0]["goal"], "(log(x) < x)",
        "the goal is the canonical body text: {}",
        records[0]
    );
}

// An SMT-proved property carries its body as the goal too (the proof tier does
// not change what proposition was discharged).
#[cfg(feature = "smt")]
#[test]
fn goal_field_carries_the_property_body_for_an_smt_proof() {
    let dir = write_prop(
        r#"
@property linear_le forall(x: f32):
  (x - 1.0) <= x
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--tier",
            "smt-only",
            "--seed",
            "0",
            "--json",
        ])
        .output()
        .expect("run prove");
    assert!(output.status.success());
    let records = property_records(&output.stdout);
    assert_eq!(records.len(), 1, "records: {records:?}");
    assert_eq!(records[0]["proof_tier"], "smt");
    assert_eq!(records[0]["status"], "passed");
    assert_eq!(
        records[0]["goal"], "((x - 1.0) <= x)",
        "an smt proof still carries the discharged proposition: {}",
        records[0]
    );
}

// A refuted property still carries its goal: the consumer must bind a FAILED
// verdict to the exact claim that failed, not just a pass.
#[cfg(feature = "smt")]
#[test]
fn goal_field_carries_the_property_body_for_a_refuted_property() {
    let dir = write_prop(
        r#"
@property false_pos forall(x: f32) where x > 0.0:
  (x - 12345.0) * (x - 12345.0) > 0.0
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--tier",
            "smt-only",
            "--seed",
            "0",
            "--json",
        ])
        .output()
        .expect("run prove");
    let records = property_records(&output.stdout);
    assert_eq!(records.len(), 1, "records: {records:?}");
    assert_eq!(records[0]["status"], "failed");
    assert_eq!(
        records[0]["goal"], "(((x - 12345.0) * (x - 12345.0)) > 0.0)",
        "a refuted property binds its FAILED verdict to the exact claim: {}",
        records[0]
    );
}

// An obligation record carries its discharged proposition (the invariant
// predicate) as the goal, in canonical Deep text with lowering/producer
// metadata stripped, so a consumer sees the bare proposition.
#[cfg(feature = "smt")]
#[test]
fn goal_field_carries_the_invariant_predicate_for_an_obligation() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("inv.ch");
    std::fs::write(
        &path,
        "module Stats.Prob\n\
         export (probability)\n\
         @opaque\n\
         @invariant(p) p.value >= 0.0 && p.value <= 1.0\n\
         type Probability =\n\
         \x20 | Probability { value: f32 }\n\
         def probability(x: f32) -> Option[Probability] =\n\
         \x20 if x >= 0.0 && x <= 1.0 then Some(Probability { value: x }) else None\n",
    )
    .expect("write inv");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "prove",
            path.to_str().unwrap(),
            "--tier",
            "smt-only",
            "--seed",
            "0",
            "--json",
        ])
        .output()
        .expect("run prove");
    let records = obligation_records(&output.stdout);
    assert_eq!(records.len(), 1, "obligation records: {records:?}");
    let goal = records[0]["goal"]
        .as_str()
        .unwrap_or_else(|| panic!("obligation record must carry a goal: {}", records[0]));
    // The bare predicate `p.value >= 0.0 && p.value <= 1.0`, lowered to Deep and
    // metadata-stripped: an `and` of two comparisons over the binder's `value`.
    assert_eq!(
        goal,
        "(app {} (var {} and) (app {} (var {} gte) (access {} (var {} p) value) (lit {} 0.0)) (app {} (var {} lte) (access {} (var {} p) value) (lit {} 1.0)))",
        "the obligation goal is the discharged invariant predicate, metadata-stripped"
    );
    // Negative: no internal lowering spans leak into the displayed proposition.
    assert!(
        !goal.contains("surf:") && !goal.contains("span"),
        "the obligation goal must not carry lowering-internal span metadata: {goal}"
    );
}

// Negative parity (chelis#436 + schema §3.4 "representable as absent"): a
// bodiless discovery-error record carries NO goal field rather than a defaulted
// or empty one, so a consumer renders absent as absent.
#[cfg(feature = "smt")]
#[test]
fn malformed_property_discovery_error_record_has_no_goal() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bad.dp");
    // Classified `user` (so the shared runner owns it) but missing
    // `property_quantifiers`: a discovery error that never reaches a property
    // body, so its error record has no proposition to carry. (Same fixture
    // shape as prove_deep_malformed_property.rs.)
    std::fs::write(
        &path,
        r#"
(def {c_earchin_role: "property_witness",
      property_source_kind: "user"}
  malformed_missing_quantifiers
  (fn {} (params {}) (lit {type: (t-prim {} bool)} true)))
"#,
    )
    .expect("write deep");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["prove", path.to_str().unwrap(), "--json"])
        .output()
        .expect("run prove");
    let error_record = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|record| record.get("kind").and_then(Value::as_str) == Some("error"))
        .expect("a malformed property emits a discovery-error record");
    assert!(
        error_record.get("goal").is_none(),
        "a bodiless discovery-error record carries no goal (representable-as-absent): {error_record}"
    );
}

// ===================================================================
// chelis#435 honesty contract, locked in the smt lane. The two cases the issue
// distinguishes must not collapse into each other:
//   - a PURE-FUZZ base (no SMT-discharged contract) -> `fuzz_validated`,
//   - an SMT base discharged modulo a FUZZ-VALIDATED CONTRACT
//     -> `proven_modulo_fuzz_validated_contract`.
// A pure-fuzz result must never read as a `proven_*` badge. (The fix shipped in
// chelis#445/#447; this test locks the contract against regression.)
// ===================================================================
#[cfg(feature = "smt")]
#[test]
fn issue_435_pure_fuzz_base_reads_fuzz_validated_not_proven_modulo_contract() {
    // A transcendental property with a precondition: under --tier auto on the
    // smt build it does not lower to SMT and falls through to the fuzz tier, and
    // its precondition is discharged by fuzz -- the exact pure-fuzz shape the
    // issue reports. The honest verdict is `fuzz_validated`.
    let dir = write_prop(
        r#"
@property bs_call_positive forall(x: f32) where x > 0.0:
  log(x) < x
"#,
    );
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--tier",
            "auto",
            "--samples",
            "100",
            "--seed",
            "0",
            "--json",
        ])
        .output()
        .expect("run prove");
    assert!(output.status.success());
    let records = property_records(&output.stdout);
    assert_eq!(records.len(), 1, "records: {records:?}");
    let record = &records[0];
    assert_eq!(record["proof_tier"], "fuzz", "the base is a fuzz fallback");
    assert_eq!(record["status"], "passed");
    // The only assumption is discharged by fuzz -- nothing SMT-discharged.
    assert_eq!(record["assumptions"][0]["discharge"]["method"], "fuzz");
    // The honesty contract: a pure-fuzz base is `fuzz_validated`, NEVER a
    // `proven_*` badge.
    assert_eq!(record["composite_verdict"], "fuzz_validated");
    assert!(
        !record["composite_verdict"]
            .as_str()
            .unwrap()
            .starts_with("proven"),
        "a pure-fuzz base must never read as proven_*: {record}"
    );
}
