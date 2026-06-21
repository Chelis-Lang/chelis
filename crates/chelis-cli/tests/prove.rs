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
    assert_eq!(
        prop["composite_verdict"],
        "proven_modulo_fuzz_validated_contract"
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
    assert_eq!(
        record["composite_verdict"],
        "proven_modulo_fuzz_validated_contract"
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
