//! chelis#1527 / chelis#1606: one diagnostic belongs to each rejected source site.

use assert_cmd::Command;
use serde::Deserialize;
use std::fs;
use tempfile::tempdir;

#[derive(Debug, Deserialize)]
struct CheckOutput {
    score: f64,
    errors: Vec<ReservedDiagnostic>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
enum DiagnosticKind {
    UnsupportedTensorPrecision,
    DimensionMismatch,
    TypeMismatch,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "span", rename_all = "snake_case")]
enum PointLocation {
    Point { offset: usize },
}

#[derive(Debug, Deserialize)]
struct ReservedDiagnostic {
    #[serde(rename = "kind")]
    kind: DiagnosticKind,
    message: String,
    span: Option<PointLocation>,
    span_id: Option<String>,
}

fn check(source: &str) -> (bool, CheckOutput, String) {
    let temp = tempdir().expect("create fixture directory");
    let path = temp.path().join("main.ch");
    fs::write(&path, source).expect("write fixture");
    Command::cargo_bin("chelis")
        .expect("find CLI")
        .arg("fmt")
        .arg("--inplace")
        .arg(&path)
        .assert()
        .success();
    let canonical = fs::read_to_string(&path).expect("read canonical fixture");
    let output = Command::cargo_bin("chelis")
        .expect("find CLI")
        .arg("check")
        .arg(&path)
        .output()
        .expect("run checker");
    let report = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "parse check report: {error}\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output.status.success(), report, canonical)
}

fn check_deep(source: &str) -> (bool, CheckOutput) {
    let temp = tempdir().expect("create fixture directory");
    let path = temp.path().join("main.dp");
    fs::write(&path, source).expect("write fixture");
    let output = Command::cargo_bin("chelis")
        .expect("find CLI")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("check")
        .arg(&path)
        .output()
        .expect("run checker");
    let report = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "parse check report: {error}\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output.status.success(), report)
}

fn assert_rejected_sites(source: &str, names: &[&str]) {
    let (success, report, canonical) = check(source);
    assert!(!success, "reserved types must fail: {report:?}");
    assert!(
        report.score < 1.0,
        "a rejected program cannot score 1: {report:?}"
    );
    let mut expected: Vec<_> = names
        .iter()
        .flat_map(|name| {
            canonical
                .match_indices(name)
                .map(move |(offset, _)| (*name, offset))
        })
        .collect();
    expected.sort_by_key(|(_, offset)| *offset);
    assert_eq!(report.errors.len(), expected.len(), "{report:?}");
    for (error, (name, offset)) in report.errors.iter().zip(expected) {
        let PointLocation::Point { offset: reported } =
            error.span.as_ref().expect("reserved dtype location");
        assert_eq!(*reported, offset, "{error:?}");
        let expected_span = format!("source:{offset}..{}", offset + name.len());
        assert_eq!(error.span_id.as_deref(), Some(expected_span.as_str()));
        assert!(error.message.contains(name), "{error:?}");
    }
}

fn prove_file(source: &str, extension: &str) -> (i32, Vec<serde_json::Value>) {
    let temp = tempdir().expect("create fixture directory");
    let path = temp.path().join(format!("main.{extension}"));
    fs::write(&path, source).expect("write fixture");
    let output = Command::cargo_bin("chelis")
        .expect("find CLI")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("prove")
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run prove");
    let records = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| {
            serde_json::from_str(line).unwrap_or_else(|error| {
                panic!(
                    "parse prove record: {error}; line={line:?}; stderr={}",
                    String::from_utf8_lossy(&output.stderr)
                )
            })
        })
        .collect();
    (output.status.code().unwrap_or(-1), records)
}

fn prove(source: &str) -> (i32, Vec<serde_json::Value>) {
    prove_file(source, "ch")
}

fn hand_authored_deep_property(signature_type: &str, parameter_type: &str) -> String {
    hand_authored_deep_property_types(
        &format!("(t-prim {{}} {signature_type})"),
        &format!("(t-prim {{}} {parameter_type})"),
    )
}

fn hand_authored_deep_property_types(signature_type: &str, parameter_type: &str) -> String {
    format!(
        "(defsig {{}} classify \
           (t-fn {{}} {signature_type} (t-prim {{}} bool)))\n\
         (def {{chelis_role: \"property\", property_source_kind: \"user\", \
                property_quantifiers: \
                  (params {{}} (x {{type: {parameter_type}}})), \
                property_preconditions: (tuple {{}})}} \
           classify \
           (fn {{}} (params {{}} (x {{type: {parameter_type}}})) \
             (lit {{}} true)))\n"
    )
}

#[test]
fn one_reserved_parameter_site_produces_one_cli_error() {
    for name in ["f8e4m3", "f8e5m2"] {
        assert_rejected_sites(
            &format!("def classify(x: {name}) -> int32 = 0i32\n"),
            &[name],
        );
    }
}

#[test]
fn no_clause_inline_precision_accepts_and_explicit_clauses_remain_authoritative() {
    for source in [
        "def inspect(x: tensor[3, p]) -> tensor[3, p] = x\n",
        "def inspect[p](x: tensor[3, p]) -> tensor[3, p] = x\n",
    ] {
        let (success, report, _) = check(source);
        assert!(success, "{source}: {report:?}");
        assert_eq!(report.score, 1.0, "{source}: {report:?}");
        assert!(report.errors.is_empty(), "{source}: {report:?}");
    }

    let (success, report, _) = check("def inspect[q](x: tensor[3, p]) -> tensor[3, p] = x\n");
    assert!(!success, "{report:?}");
    assert!(report.score < 1.0, "{report:?}");
    assert!(
        report
            .errors
            .iter()
            .any(|error| error.message.contains("`p`")),
        "{report:?}"
    );
}

#[test]
fn property_quantifier_reserved_types_produce_one_check_error() {
    for name in ["f8e4m3", "f8e5m2"] {
        for ty in [name.to_string(), format!("tensor[3, {name}]")] {
            assert_rejected_sites(
                &format!("@property classify forall(x: {ty}):\n  true\n"),
                &[name],
            );
        }
    }
}

#[test]
fn property_quantifier_reserved_types_produce_one_prove_check_diagnostic() {
    for name in ["f8e4m3", "f8e5m2"] {
        for ty in [name.to_string(), format!("tensor[3, {name}]")] {
            let source = format!("@property classify forall(x: {ty}):\n  true\n");
            let (code, records) = prove(&source);
            assert_eq!(code, 3, "{source}: {records:#?}");
            let check_errors = records
                .iter()
                .filter(|record| record["kind"] == "error" && record["stage"] == "check")
                .collect::<Vec<_>>();
            assert_eq!(check_errors.len(), 1, "{source}: {records:#?}");
            let diagnostics = check_errors[0]["diagnostics"]
                .as_array()
                .expect("prove check error carries diagnostics");
            assert_eq!(diagnostics.len(), 1, "{source}: {records:#?}");
            assert!(
                diagnostics[0]
                    .as_str()
                    .is_some_and(|text| text.contains(name)),
                "{source}: {records:#?}"
            );
        }
    }
}

#[test]
fn hand_authored_deep_property_conflict_rejects_check_and_prove() {
    let source = hand_authored_deep_property("f64", "f32");
    let (success, report) = check_deep(&source);
    assert!(!success, "{report:?}");
    assert!(report.score < 1.0, "{report:?}");
    assert_eq!(report.errors.len(), 1, "{report:?}");
    assert_eq!(report.errors[0].kind, DiagnosticKind::TypeMismatch);
    assert!(
        report.errors[0].message.contains("f32") && report.errors[0].message.contains("f64"),
        "{report:?}"
    );

    let (code, records) = prove_file(&source, "dp");
    assert_eq!(code, 3, "{records:#?}");
    let check_errors = records
        .iter()
        .filter(|record| record["kind"] == "error" && record["stage"] == "check")
        .collect::<Vec<_>>();
    assert_eq!(check_errors.len(), 1, "{records:#?}");
    let diagnostics = check_errors[0]["diagnostics"]
        .as_array()
        .expect("prove check error carries diagnostics");
    assert_eq!(diagnostics.len(), 1, "{records:#?}");
    assert!(
        diagnostics[0]
            .as_str()
            .is_some_and(|text| text.contains("f32") && text.contains("f64")),
        "{records:#?}"
    );
}

#[test]
fn matching_hand_authored_deep_property_checks_cleanly() {
    let source = hand_authored_deep_property("f32", "f32");
    let (success, report) = check_deep(&source);
    assert!(success, "{report:?}");
    assert_eq!(report.score, 1.0, "{report:?}");
    assert!(report.errors.is_empty(), "{report:?}");

    let (code, records) = prove_file(&source, "dp");
    assert_ne!(code, 3, "{records:#?}");
    assert!(
        !records
            .iter()
            .any(|record| record["kind"] == "error" && record["stage"] == "check"),
        "{records:#?}"
    );
}

fn assert_deep_property_has_reserved_owner(
    signature_type: &str,
    parameter_type: &str,
    name: &str,
    expected: usize,
) {
    let source = hand_authored_deep_property_types(signature_type, parameter_type);
    let (success, report) = check_deep(&source);
    assert!(!success, "{source}: {report:?}");
    assert!(report.score < 1.0, "{source}: {report:?}");
    assert_eq!(
        report
            .errors
            .iter()
            .filter(|error| error.kind == DiagnosticKind::UnsupportedTensorPrecision)
            .count(),
        expected,
        "{source}: {report:?}"
    );

    let (code, records) = prove_file(&source, "dp");
    assert_eq!(code, 3, "{source}: {records:#?}");
    let diagnostics = records
        .iter()
        .filter(|record| record["kind"] == "error" && record["stage"] == "check")
        .flat_map(|record| {
            record["diagnostics"]
                .as_array()
                .expect("prove check error carries diagnostics")
        })
        .filter(|diagnostic| diagnostic.as_str().is_some_and(|text| text.contains(name)))
        .count();
    assert_eq!(diagnostics, expected, "{source}: {records:#?}");
}

#[test]
fn property_copy_cli_ownership_uses_spanless_semantic_type_syntax() {
    for name in ["f8e4m3", "f8e5m2"] {
        for parameter in [
            format!("(t-prim {{doc: \"same\"}} {name})"),
            format!(
                "(t-tensor {{doc: \"same\"}} \
                   (d-lit {{doc: \"dimension\"}} 3) \
                   (t-prim {{doc: \"precision\"}} {name}))"
            ),
            format!(
                "(t-prim {{doc: \"same\", span: \"parameter\", \
                   loc: (loc \"parameter.dp\" 9 8), source: (parameter copy), \
                   tool_data: {{nested: (payload \"same\")}}}} {name})"
            ),
            format!(
                "(t-prim {{doc: \"same\", span_start: 7, span_end: 13, \
                   span_file: \"parameter.dp\"}} {name})"
            ),
        ] {
            let signature = parameter
                .replace("span: \"parameter\"", "span: \"signature\"")
                .replace("parameter.dp", "signature.dp")
                .replace("(parameter copy)", "(signature copy)")
                .replace("span_start: 7", "span_start: 70")
                .replace("span_end: 13", "span_end: 130");
            assert_deep_property_has_reserved_owner(&signature, &parameter, name, 1);
        }
    }
}

#[test]
fn property_copy_cli_ownership_preserves_semantic_metadata_differences() {
    for name in ["f8e4m3", "f8e5m2"] {
        for (signature, parameter) in [
            (
                format!("(t-prim {{doc: \"signature\"}} {name})"),
                format!("(t-prim {{doc: \"parameter\"}} {name})"),
            ),
            (
                format!(
                    "(t-tensor {{doc: \"signature\"}} \
                       (d-lit {{}} 3) (t-prim {{}} {name}))"
                ),
                format!(
                    "(t-tensor {{doc: \"parameter\"}} \
                       (d-lit {{}} 3) (t-prim {{}} {name}))"
                ),
            ),
        ] {
            assert_deep_property_has_reserved_owner(&signature, &parameter, name, 2);
        }
    }
}

#[test]
fn one_reserved_tensor_element_site_produces_one_located_cli_error() {
    for name in ["f8e4m3", "f8e5m2"] {
        assert_rejected_sites(
            &format!("def classify(x: tensor[3, {name}]) -> int32 = 0i32\n"),
            &[name],
        );
    }
}

#[test]
fn a_rejected_tensor_precision_keeps_the_independent_shape_error() {
    for name in ["f8e4m3", "f8e5m2"] {
        let (success, report, source) = check(&format!(
            "def inspect(x: tensor[3, {name}]) -> int64 = shape(x, 1i32)\n"
        ));
        assert!(!success, "{name}: {report:?}");
        assert!(report.score < 1.0, "{name}: {report:?}");
        assert_eq!(report.errors.len(), 2, "{name}: {report:?}");

        let reserved: Vec<_> = report
            .errors
            .iter()
            .filter(|error| error.kind == DiagnosticKind::UnsupportedTensorPrecision)
            .collect();
        assert_eq!(reserved.len(), 1, "{name}: {report:?}");
        let offset = source.find(name).expect("reserved precision site");
        let PointLocation::Point { offset: reported } =
            reserved[0].span.as_ref().expect("reserved dtype location");
        assert_eq!(*reported, offset, "{name}: {reserved:?}");
        let expected_span = format!("source:{offset}..{}", offset + name.len());
        assert_eq!(reserved[0].span_id.as_deref(), Some(expected_span.as_str()));
        assert!(reserved[0].message.contains(name), "{name}: {reserved:?}");

        let dimensions: Vec<_> = report
            .errors
            .iter()
            .filter(|error| error.kind == DiagnosticKind::DimensionMismatch)
            .collect();
        assert_eq!(dimensions.len(), 1, "{name}: {report:?}");
        assert!(
            dimensions[0]
                .message
                .contains("shape axis 1 is out of bounds for rank 1 tensor"),
            "{name}: {dimensions:?}"
        );
    }
}

#[test]
fn separate_declarations_keep_their_own_cli_errors() {
    assert_rejected_sites(
        "def left(x: f8e4m3) -> int32 = 0i32\ndef right(x: f8e5m2) -> int32 = 0i32\n",
        &["f8e4m3", "f8e5m2"],
    );
}

#[test]
fn repeated_reserved_spelling_keeps_distinct_cli_locations() {
    assert_rejected_sites(
        "def left(x: f8e4m3) -> int32 = 0i32\ndef right(x: f8e4m3) -> int32 = 0i32\n",
        &["f8e4m3"],
    );
}

#[test]
fn a_call_does_not_report_its_failed_signature_again() {
    assert_rejected_sites(
        "def classify(x: f8e4m3) -> int32 = 0i32\nresult = classify(1i32)\n",
        &["f8e4m3"],
    );
}

#[test]
fn distinct_parameter_sites_in_one_signature_keep_separate_cli_errors() {
    assert_rejected_sites(
        "def classify(x: f8e4m3, y: f8e5m2) -> int32 = 0i32\n",
        &["f8e4m3", "f8e5m2"],
    );
    assert_rejected_sites(
        "def classify(x: f8e4m3, y: f8e4m3) -> int32 = 0i32\n",
        &["f8e4m3"],
    );
}

#[test]
fn parameter_and_return_sites_keep_separate_cli_errors() {
    assert_rejected_sites(
        "def classify(x: f8e4m3) -> f8e5m2 = x\n",
        &["f8e4m3", "f8e5m2"],
    );
}

#[test]
fn nested_type_components_keep_separate_cli_errors() {
    for source in [
        "def classify(x: (f8e4m3, f8e5m2)) -> int32 = 0i32\n",
        "def classify(x: Dict[f8e4m3, f8e5m2]) -> int32 = 0i32\n",
    ] {
        assert_rejected_sites(source, &["f8e4m3", "f8e5m2"]);
    }
}

#[test]
fn a_failed_signature_does_not_hide_a_cli_body_error() {
    let (success, report, source) =
        check("def classify(x: f8e4m3) -> int32 = cast(0i32, f8e5m2)\n");
    assert!(!success, "{report:?}");
    assert!(report.score < 1.0, "{report:?}");
    assert_eq!(report.errors.len(), 2, "{report:?}");
    assert!(report.errors[0].message.contains("f8e4m3"), "{report:?}");
    assert!(report.errors[1].message.contains("f8e5m2"), "{report:?}");
    assert_eq!(report.errors[0].span_id.as_deref(), Some("source:16..22"));
    let start = source.find("cast(").expect("cast site");
    let PointLocation::Point { offset } = report.errors[1]
        .span
        .as_ref()
        .expect("cast diagnostic location");
    assert_eq!(*offset, start);
    let expected_span = format!("surf:{start}..{}", source.trim_end().len());
    assert_eq!(
        report.errors[1].span_id.as_deref(),
        Some(expected_span.as_str())
    );
}

#[test]
fn active_float_parameters_keep_a_successful_empty_report() {
    for name in ["f32", "f64", "f16", "bf16"] {
        let (success, report, _) = check(&format!("def classify(x: {name}) -> int32 = 0i32\n"));
        assert!(success, "{report:?}");
        assert_eq!(report.score, 1.0, "{report:?}");
        assert!(report.errors.is_empty(), "{report:?}");
    }
}
