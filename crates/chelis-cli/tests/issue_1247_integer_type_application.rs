//! chelis#1247 / chelis#1258: nominal arguments are kinded.
//!
//! Integer arguments are dimension syntax, never inference type variables.
//! The checker must reject them at type parameters, enforce them at dimension
//! parameters, and keep check/test/Surf/migration/build signals honest.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;
use std::fs;
use std::process::Command as StdCommand;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

const COLUMN_DECL: &str = "\
type Column[n] =
  | FloatCol(tensor[n, f32])
";

fn check_json(source: &str) -> (std::process::ExitStatus, Value) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("case.ch");
    write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    let json = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "chelis check must emit JSON ({error}); stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output.status, json)
}

fn assert_check_rejects(source: &str, expected_kind: &str) {
    let (status, json) = check_json(source);
    assert!(
        !status.success(),
        "ill-kinded program must exit nonzero: {json}"
    );
    assert!(
        json["score"].as_f64().is_some_and(|score| score < 1.0),
        "ill-kinded program must score below one: {json}"
    );
    let errors = json["errors"].as_array().expect("errors array");
    assert!(
        !errors.is_empty(),
        "ill-kinded program must report errors: {json}"
    );
    assert!(
        errors.iter().any(|error| error["kind"] == expected_kind),
        "expected {expected_kind}, got {errors:#?}"
    );
}

#[test]
fn integer_arguments_reject_at_builtin_and_user_type_parameters() {
    assert_check_rejects(
        "def unwrap(value: Option[732]) -> f32 = match value with { | Some(item) => item | None => cast(0.0, f32) }\n",
        "TypeMismatch",
    );
    assert_check_rejects(
        "type Box[a] = | Full { value: a }\ndef unwrap(value: Box[732]) -> f32 = match value with { | Full { value } => value }\n",
        "TypeMismatch",
    );
}

#[test]
fn type_arguments_reject_at_dimension_parameters() {
    assert_check_rejects(
        &format!("{COLUMN_DECL}def wrong(value: Column[f32]) -> Column[f32] = value\n"),
        "TypeMismatch",
    );
}

#[test]
fn concrete_dimension_arguments_enforce_extent() {
    assert_check_rejects(
        &format!(
            "{COLUMN_DECL}def wrong() -> Column[3] = FloatCol(to_tensor([cast(1.0, f32), cast(2.0, f32)]))\n"
        ),
        "DimensionMismatch",
    );
}

#[test]
fn chelis_test_rejects_ill_kinded_test_file() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    fs::create_dir_all(root.join("src")).expect("create src");
    fs::create_dir_all(root.join("tests")).expect("create tests");
    write_file(
        &root.join("reef.toml"),
        &format!(
            "[package]\nname = \"kinded-test\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"Kinded\"\n",
            chelis_compiler_api::COMPILER_VERSION
        ),
    );
    write_file(
        &root.join("src/main.ch"),
        "module Kinded.Main\ndef noop() -> unit = test_assert(true, \"noop\")\n",
    );
    let path = root.join("tests/bad_test.ch");
    write_file(
        &path,
        &format!(
            "module Kinded.Bad\n{COLUMN_DECL}def wrong() -> Column[3] = FloatCol(to_tensor([cast(1.0, f32), cast(2.0, f32)]))\ndef test_wrong() -> unit = test_assert(true, \"must not run\")\n"
        ),
    );
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(root)
        .args(["test", "tests/bad_test.ch"])
        .assert()
        .failure()
        .stdout(predicate::str::contains("Column 2"))
        .stdout(predicate::str::contains("Column 3"));
}

#[test]
fn dimension_application_round_trips_through_deep_surf_and_migration() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("roundtrip.ch");
    let source = "\
module Repro.M
export (Box, make, pass)
type Box[n] =
  | Box { items: tensor[n, f32] }
def make(t: tensor[3, f32]) -> Box[3] = Box { items: t }
def pass(b: Box[3]) -> Box[3] = b
";
    write_file(&path, source);

    let deep = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["deep", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("(d-lit {} 3)"))
        .stdout(predicate::str::contains("(t-var {} 3)").not())
        .get_output()
        .stdout
        .clone();
    let deep_path = dir.path().join("roundtrip.dp");
    fs::write(&deep_path, deep).expect("write Deep fixture");
    let surfaced = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["surf", deep_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("Box[3]"))
        .get_output()
        .stdout
        .clone();
    let surfaced_path = dir.path().join("surfaced.ch");
    fs::write(&surfaced_path, surfaced).expect("write surfaced fixture");
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", surfaced_path.to_str().unwrap()])
        .assert()
        .success();
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["migrate", "surf", "--from", "0.18", "--check"])
        .arg(&path)
        .assert()
        .success();
}

#[test]
fn concrete_dimension_survives_in_structured_inferred_json() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("inferred.ch");
    write_file(
        &path,
        &format!("{COLUMN_DECL}def identity(value: Column[2]) -> Column[2] = value\n"),
    );
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", "--show-inferred", path.to_str().unwrap()])
        .output()
        .expect("check inferred JSON");
    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json: Value = serde_json::from_slice(&output.stdout).expect("valid JSON");
    let signatures = json["inferred_signatures"]
        .as_array()
        .expect("inferred signatures");
    let identity = signatures
        .iter()
        .find(|signature| signature["function"] == "identity")
        .expect("identity signature");
    let argument = &identity["checked_signature_structured"]["args"][0]["args"][0];
    assert_eq!(
        argument,
        &serde_json::json!({
            "kind": "dimension",
            "dim": {"kind": "lit", "size": 2}
        }),
        "structured inferred type must preserve the nominal dimension: {json}"
    );
}

#[test]
fn aliases_and_forward_nominal_headers_preserve_dimension_kind() {
    let source = "\
type Frame[n] =
  | Frame { column: Column[n] }
type Column[n] =
  | Column(tensor[n, f32])
type Pair[n] = (Frame[n], Column[n])
def good(value: Pair[2]) -> Pair[2] = value
";
    let (status, json) = check_json(source);
    assert!(
        status.success(),
        "forward/alias dimension kinds must resolve: {json}"
    );
    assert_eq!(json["score"], 1.0);
    assert_eq!(json["errors"], serde_json::json!([]));
}

#[test]
fn dimension_kinded_aliases_are_transparent_and_preserve_extent() {
    let matching = format!(
        "{COLUMN_DECL}type Pair[n] = Column[n]\ndef good() -> Pair[2] = FloatCol(to_tensor([cast(1.0, f32), cast(2.0, f32)]))\n"
    );
    let (status, json) = check_json(&matching);
    assert!(
        status.success(),
        "matching alias application must check: {json}"
    );
    assert_eq!(json["score"], 1.0);
    assert_eq!(json["errors"], serde_json::json!([]));

    assert_check_rejects(
        &format!(
            "{COLUMN_DECL}type Pair[n] = Column[n]\ndef wrong() -> Pair[3] = FloatCol(to_tensor([cast(1.0, f32), cast(2.0, f32)]))\n"
        ),
        "DimensionMismatch",
    );
}

#[test]
fn mixed_nominal_headers_keep_type_and_dimension_slots_distinct() {
    let source = "\
type Packet[a, n] =
  | Packet { item: a, payload: tensor[n, f32] }
def good() -> Packet[int32, 2] = Packet { item: 7i32, payload: to_tensor([cast(1.0, f32), cast(2.0, f32)]) }
";
    let (status, json) = check_json(source);
    assert!(status.success(), "mixed nominal kinds must check: {json}");
    assert_eq!(json["score"], 1.0);
    assert_eq!(json["errors"], serde_json::json!([]));

    assert_check_rejects(
        "type Packet[a, n] = | Packet { item: a, payload: tensor[n, f32] }\ndef wrong() -> Packet[int32, 3] = Packet { item: 7i32, payload: to_tensor([cast(1.0, f32), cast(2.0, f32)]) }\n",
        "DimensionMismatch",
    );
}

#[test]
fn kinded_nominals_preserve_record_access_and_exhaustiveness_checks() {
    let access = "\
type Frame[n] =
  | Frame { items: tensor[n, f32] }
def first(frame: Frame[2]) -> f32 = index(to_list(frame.items), 0)
";
    let (status, json) = check_json(access);
    assert!(status.success(), "kinded record access must check: {json}");
    assert_eq!(json["errors"], serde_json::json!([]));

    assert_check_rejects(
        "type Choice[n] = | Left(tensor[n, f32]) | Right(tensor[n, f32])\ndef incomplete(value: Choice[2]) -> f32 = match value with { | Left(items) => index(to_list(items), 0) }\n",
        "NonExhaustiveMatch",
    );
}

#[test]
fn matching_dimension_application_evaluates_and_c_backend_runs() {
    let source = format!(
        "{COLUMN_DECL}def total(value: Column[2]) -> f32 = match value with {{ | FloatCol(items) => add(index(to_list(items), 0), index(to_list(items), 1)) }}\ndef main() -> f32 = total(FloatCol(to_tensor([cast(1.0, f32), cast(2.0, f32)])))\nout = print(main())\n"
    );
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("positive.ch");
    let out_dir = dir.path().join("out");
    write_file(&path, &source);
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::starts_with("3.0"));
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    let linked = common::link_generated(&out_dir, "positive.c", "run");
    assert!(linked.success(), "generated C must link");
    let output = StdCommand::new(out_dir.join("run"))
        .output()
        .expect("run C binary");
    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).starts_with("3.0"));
}
