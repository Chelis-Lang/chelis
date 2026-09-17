//! chelis#1125 PP7: tensor-precision diagnostic ownership is carrier-independent.

use chelis_deep::{Expr, parse_and_stamp_file};
use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{check_ir_program, check_typed_program};
use serde_json::Value;
use std::fs;
use std::process::{Command, Output};
use tempfile::tempdir;

const TWO_DEFINITIONS: &str = "\
(def {} first (fn {} (params {} \
  (x {type: (t-tensor {} (d-lit {} 2) (t-prim {} madeup))})) \
  (var {} x)))
(def {} second (fn {} (params {} \
  (y {type: (t-tensor {} (d-lit {} 3) (t-prim {} madeup))})) \
  (var {} y)))
";

const ONE_DEFINITION_TWO_SITES: &str = "\
(def {} only (fn {} (params {} \
  (x {type: (t-tensor {} (d-lit {} 2) (t-prim {} madeup))}) \
  (y {type: (t-tensor {} (d-lit {} 3) (t-prim {} madeup))})) \
  (var {} x)))
";

const SAME_SPELLING_CROSS_NAMESPACE: &str = "\
(typealias {} shared () \
  (t-tensor {} (d-lit {} 2) (t-prim {} madeup)))
(def {} shared (fn {} (params {} \
  (x {type: (t-tensor {} (d-lit {} 3) (t-prim {} madeup))})) \
  (var {} x)))
";

const DUPLICATE_VALUE_DECLARATIONS: &str = "\
(def {} shared (fn {} (params {} \
  (x {type: (t-tensor {} (d-lit {} 2) (t-prim {} madeup))})) \
  (var {} x)))
(def {} shared (fn {} (params {} \
  (y {type: (t-tensor {} (d-lit {} 3) (t-prim {} madeup))})) \
  (var {} y)))
";

const MATCHED_SIGNATURE_AND_DEFINITION: &str = "\
(defsig {} shared \
  (t-fn {} \
    (t-tensor {} (d-lit {} 2) (t-prim {} madeup)) \
    (t-tensor {} (d-lit {} 2) (t-prim {} madeup))))
(def {} shared (fn {} (params {} \
  (x {type: (t-tensor {} (d-lit {} 2) (t-prim {} madeup))})) \
  (var {} x)))
";

fn diagnostics(errors: &[CheckError]) -> Vec<String> {
    let mut diagnostics = errors
        .iter()
        .map(|error| format!("[{:?}] {}", error.kind, error.message))
        .collect::<Vec<_>>();
    diagnostics.sort();
    diagnostics
}

fn unsupported_count(errors: &[CheckError]) -> usize {
    errors
        .iter()
        .filter(|error| matches!(error.kind, CheckErrorKind::UnsupportedTensorPrecision))
        .count()
}

fn checker_errors(program: &[Expr]) -> (Vec<CheckError>, Vec<CheckError>) {
    let ir = match check_ir_program(program) {
        Ok(_) => panic!("normalizing checker must reject the invalid precision"),
        Err(result) => result.errors,
    };
    let typed = match check_typed_program(program) {
        Ok(_) => panic!("stamped checker must reject the invalid precision"),
        Err(result) => result.errors,
    };
    (ir, typed)
}

fn run_cli(source: &str, command: &str) -> Output {
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join("precision.dp");
    fs::write(&path, source).expect("write Deep fixture");
    let mut process = Command::new(env!("CARGO_BIN_EXE_chelis"));
    process
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg(command)
        .arg(&path);
    if command == "prove" {
        process.arg("--json");
    }
    process.output().expect("run chelis")
}

fn check_unsupported_count(source: &str) -> usize {
    let output = run_cli(source, "check");
    assert_eq!(
        output.status.code(),
        Some(2),
        "check must reject during type checking: stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "check must emit JSON: {error}; stdout={}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    report["errors"]
        .as_array()
        .expect("check errors")
        .iter()
        .filter(|error| error["kind"] == "UnsupportedTensorPrecision")
        .count()
}

fn prove_unsupported_count(source: &str) -> usize {
    let output = run_cli(source, "prove");
    assert_eq!(
        output.status.code(),
        Some(3),
        "prove must stop at the check gate: stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| {
            serde_json::from_str::<Value>(line)
                .unwrap_or_else(|error| panic!("prove must emit NDJSON: {error}; line={line:?}"))
        })
        .filter(|record| record["kind"] == "error" && record["stage"] == "check")
        .flat_map(|record| {
            record["diagnostics"]
                .as_array()
                .expect("prove check diagnostics")
                .clone()
        })
        .filter(|diagnostic| {
            diagnostic.as_str().is_some_and(|text| {
                text.contains("tensor element precision `madeup` is not a recognized primitive")
            })
        })
        .count()
}

fn assert_multiplicity(source: &str, expected: usize, label: &str) {
    let program = parse_and_stamp_file(source).expect("fixture must stamp");
    let (ir, typed) = checker_errors(&program);
    assert_eq!(
        diagnostics(&typed),
        diagnostics(&ir),
        "{label}: check_ir_program and check_typed_program must report the same defects"
    );
    assert_eq!(unsupported_count(&ir), expected, "{label}: IR diagnostics");
    assert_eq!(
        unsupported_count(&typed),
        expected,
        "{label}: typed diagnostics"
    );
    assert_eq!(
        check_unsupported_count(source),
        expected,
        "{label}: chelis check diagnostics"
    );
    assert_eq!(
        prove_unsupported_count(source),
        expected,
        "{label}: chelis prove diagnostics"
    );
}

#[test]
fn repeated_invalid_precision_in_separate_definitions_keeps_both_diagnostics() {
    assert_multiplicity(
        TWO_DEFINITIONS,
        2,
        "same invalid precision in separate definitions",
    );
}

#[test]
fn repeated_invalid_precision_within_one_definition_remains_deduplicated() {
    assert_multiplicity(
        ONE_DEFINITION_TWO_SITES,
        1,
        "same invalid precision repeated within one definition",
    );
}

#[test]
fn same_spelling_in_type_and_value_namespaces_keeps_both_diagnostics() {
    assert_multiplicity(
        SAME_SPELLING_CROSS_NAMESPACE,
        2,
        "same spelling in distinct declaration namespaces",
    );
}

#[test]
fn repeated_value_declaration_occurrences_keep_both_diagnostics() {
    assert_multiplicity(
        DUPLICATE_VALUE_DECLARATIONS,
        2,
        "repeated value declaration occurrences",
    );
}

#[test]
fn matching_signature_and_definition_share_one_value_owner() {
    assert_multiplicity(
        MATCHED_SIGNATURE_AND_DEFINITION,
        1,
        "matching signature and definition",
    );
}
