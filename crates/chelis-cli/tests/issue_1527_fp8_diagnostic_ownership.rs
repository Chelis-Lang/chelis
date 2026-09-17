//! chelis#1527 / chelis#1606: declaration-owned type positions share one
//! diagnostic per rejected spelling, while independent use sites remain distinct.

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
    UnknownForm,
    MalformedForm,
    Other,
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
    let mut search_start = 0;
    let expected = names
        .iter()
        .map(|name| {
            let relative = canonical[search_start..].find(name).unwrap_or_else(|| {
                panic!("expected `{name}` after byte {search_start}: {canonical}")
            });
            let offset = search_start + relative;
            search_start = offset + name.len();
            (*name, offset)
        })
        .collect::<Vec<_>>();
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

fn prove_file_detailed(source: &str, extension: &str) -> (i32, Vec<serde_json::Value>, String) {
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
    (
        output.status.code().unwrap_or(-1),
        records,
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn prove_file(source: &str, extension: &str) -> (i32, Vec<serde_json::Value>) {
    let (code, records, _) = prove_file_detailed(source, extension);
    (code, records)
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
    hand_authored_deep_property_slots(
        &[signature_type.to_string()],
        &[parameter_type.to_string()],
        "(lit {} true)",
    )
}

fn hand_authored_deep_property_slots(
    signature_types: &[String],
    parameter_types: &[String],
    body: &str,
) -> String {
    let parameter_names = ["x", "y", "z", "w", "v", "u"];
    assert!(
        signature_types.len() <= parameter_names.len()
            && parameter_types.len() <= parameter_names.len()
    );
    let signature = signature_types.join(" ");
    let quantifiers = parameter_types
        .iter()
        .enumerate()
        .map(|(index, ty)| format!("({} {{type: {ty}}})", parameter_names[index]))
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "(defsig {{}} classify \
           (t-fn {{}} {signature} (t-prim {{}} bool)))\n\
         (def {{chelis_role: \"property\", property_source_kind: \"user\", \
                property_quantifiers: \
                  (params {{}} {quantifiers}), \
                property_preconditions: (tuple {{}})}} \
           classify \
           (fn {{}} (params {{}} {quantifiers}) {body}))\n"
    )
}

fn hand_authored_nominal_property(application: &str) -> String {
    format!(
        "(deftype {{}} Pair (a b) \
           (variant {{}} Pair \
             (field {{}} left (t-var {{}} a)) \
             (field {{}} right (t-var {{}} b))))\n\
         (defsig {{}} inspect (t-fn {{}} {application} (t-prim {{}} bool)))\n\
         (def {{}} inspect (fn {{}} (params {{}} x) (lit {{}} true)))\n"
    )
}

fn assert_cli_diagnostic_counts(source: &str, reserved: usize, mismatches: usize, label: &str) {
    let (success, report) = check_deep(source);
    assert!(!success, "{label}: {report:?}");
    assert!(report.score < 1.0, "{label}: {report:?}");
    assert_eq!(
        report
            .errors
            .iter()
            .filter(|error| error.kind == DiagnosticKind::UnsupportedTensorPrecision)
            .count(),
        reserved,
        "{label}: {report:?}"
    );
    assert_eq!(
        report
            .errors
            .iter()
            .filter(|error| error.kind == DiagnosticKind::TypeMismatch)
            .count(),
        mismatches,
        "{label}: {report:?}"
    );

    let (code, records, stderr) = prove_file_detailed(source, "dp");
    assert_eq!(code, 3, "{label}: {records:#?}\nstderr: {stderr}");
    let diagnostics = records
        .iter()
        .filter(|record| record["kind"] == "error" && record["stage"] == "check")
        .flat_map(|record| {
            record["diagnostics"]
                .as_array()
                .expect("prove check error carries diagnostics")
        })
        .filter_map(|diagnostic| diagnostic.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        diagnostics
            .iter()
            .filter(|text| {
                text.contains("cannot use") && (text.contains("f8e4m3") || text.contains("f8e5m2"))
            })
            .count(),
        reserved,
        "{label}: {records:#?}\nstderr: {stderr}"
    );
    assert_eq!(
        diagnostics.len(),
        reserved + mismatches,
        "{label}: {records:#?}\nstderr: {stderr}"
    );
}

#[test]
fn one_reserved_parameter_site_produces_one_cli_error() {
    for name in ["f8e4m3", "f8e5m2"] {
        assert_rejected_sites(&format!("def classify(x: {name}) -> i32 = 0i32\n"), &[name]);
    }
}

#[test]
fn no_clause_inline_precision_accepts_and_explicit_clauses_remain_authoritative() {
    for source in [
        "def inspect[p](x: tensor[3, p]) -> tensor[3, p] = x\n",
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
fn property_copy_cli_ownership_ignores_nonsemantic_metadata_differences() {
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
            assert_deep_property_has_reserved_owner(&signature, &parameter, name, 1);
        }
    }
}

#[test]
fn property_copy_cli_ownership_is_classified_per_parameter_slot() {
    for name in ["f8e4m3", "f8e5m2"] {
        for reserved in [
            format!("(t-prim {{}} {name})"),
            format!("(t-tensor {{}} (d-lit {{}} 3) (t-prim {{}} {name}))"),
        ] {
            for (signature, parameter) in [
                (
                    vec![reserved.clone(), "(t-prim {} f64)".to_string()],
                    vec![reserved.clone(), "(t-prim {} f32)".to_string()],
                ),
                (
                    vec!["(t-prim {} f64)".to_string(), reserved.clone()],
                    vec!["(t-prim {} f32)".to_string(), reserved.clone()],
                ),
            ] {
                let source =
                    hand_authored_deep_property_slots(&signature, &parameter, "(lit {} true)");
                assert_cli_diagnostic_counts(
                    &source,
                    1,
                    1,
                    &format!("{name}/{signature:?}/{parameter:?}"),
                );
            }
        }
    }
}

#[test]
fn property_copy_cli_ownership_handles_multiple_and_arity_disagreements() {
    for name in ["f8e4m3", "f8e5m2"] {
        let scalar = format!("(t-prim {{}} {name})");
        let tensor = format!("(t-tensor {{}} (d-lit {{}} 3) (t-prim {{}} {name}))");
        for (signature, parameter, reserved, label) in [
            (
                vec![
                    scalar.clone(),
                    "(t-prim {} f64)".to_string(),
                    tensor.clone(),
                    "(t-prim {} i64)".to_string(),
                ],
                vec![
                    scalar.clone(),
                    "(t-prim {} f32)".to_string(),
                    tensor.clone(),
                    "(t-prim {} bool)".to_string(),
                ],
                1,
                "multiple matching and disagreeing slots",
            ),
            (
                vec![scalar.clone(), "(t-prim {} f32)".to_string()],
                vec![scalar.clone()],
                1,
                "missing parameter",
            ),
            (
                vec![scalar.clone()],
                vec![scalar.clone(), "(t-prim {} f32)".to_string()],
                1,
                "extra parameter",
            ),
        ] {
            let source = hand_authored_deep_property_slots(&signature, &parameter, "(lit {} true)");
            assert_cli_diagnostic_counts(&source, reserved, 1, label);
        }
    }
}

#[test]
fn property_copy_cli_ownership_fails_closed_per_invalid_slot() {
    for name in ["f8e4m3", "f8e5m2"] {
        let reserved = format!("(t-prim {{}} {name})");
        let source = hand_authored_deep_property_slots(
            &[reserved.clone(), "(t-prim {} f32)".to_string()],
            &[reserved, "(t-prim {} madeup)".to_string()],
            "(lit {} true)",
        );
        let (success, report) = check_deep(&source);
        assert!(!success, "{name}: {report:?}");
        assert_eq!(
            report
                .errors
                .iter()
                .filter(|error| error.kind == DiagnosticKind::UnsupportedTensorPrecision)
                .count(),
            1,
            "{name}: {report:?}"
        );
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.message.contains("madeup")),
            "{name}: {report:?}"
        );

        let (code, records) = prove_file(&source, "dp");
        assert_eq!(code, 3, "{name}: {records:#?}");
        let diagnostics = records
            .iter()
            .filter(|record| record["kind"] == "error" && record["stage"] == "check")
            .flat_map(|record| {
                record["diagnostics"]
                    .as_array()
                    .expect("prove check error carries diagnostics")
            })
            .filter_map(|diagnostic| diagnostic.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            diagnostics
                .iter()
                .filter(|text| text.contains(name))
                .count(),
            1,
            "{name}: {records:#?}"
        );
        assert!(
            diagnostics.iter().any(|text| text.contains("madeup")),
            "{name}: {records:#?}"
        );
    }
}

#[test]
fn malformed_property_parameter_carriers_fail_closed_at_cli_ingress() {
    let source = hand_authored_deep_property_slots(
        &[
            "(t-prim {} f8e4m3)".to_string(),
            "(t-prim {} f32)".to_string(),
        ],
        &[
            "(t-prim {} f8e4m3)".to_string(),
            "(unknown-type {})".to_string(),
        ],
        "(lit {} true)",
    );
    let (success, report) = check_deep(&source);
    assert!(!success, "{report:?}");
    assert_eq!(report.score, 0.0, "{report:?}");
    assert_eq!(report.errors.len(), 1, "{report:?}");
    assert_eq!(report.errors[0].kind, DiagnosticKind::Other);
    assert!(report.errors[0].message.contains("type-expression node"));

    let (code, records, stderr) = prove_file_detailed(&source, "dp");
    assert_ne!(code, 0, "{records:#?}\nstderr: {stderr}");
    assert!(
        records.is_empty() && stderr.contains("type-expression node"),
        "{records:#?}\nstderr: {stderr}"
    );
}

#[test]
fn nominal_arity_cli_recovery_keeps_nested_rejections_and_the_arity_witness() {
    for name in ["f8e4m3", "f8e5m2"] {
        let other = if name == "f8e4m3" { "f8e5m2" } else { "f8e4m3" };
        for (application, reserved, label) in [
            (
                format!("(t-adt {{}} Pair (t-tuple {{}} (t-prim {{}} {name})))"),
                1,
                "too few with one nested rejection",
            ),
            (
                format!(
                    "(t-adt {{}} Pair \
                       (t-prim {{}} {name}) \
                       (t-tensor {{}} (d-lit {{}} 3) (t-prim {{}} {other})) \
                       (t-prim {{}} f32))"
                ),
                2,
                "too many with two nested rejections",
            ),
            (
                format!(
                    "(t-adt {{}} Pair \
                       (t-tuple {{}} (t-prim {{}} {name}) (t-prim {{}} {other})) \
                       (t-ref {{}} (t-prim {{}} {name})) \
                       (t-prim {{}} f32))"
                ),
                2,
                "nested composites preserve one rejection per spelling",
            ),
            (
                "(t-adt {} Pair (t-prim {} f32))".to_string(),
                0,
                "valid too-few control",
            ),
            (
                "(t-adt {} Pair (t-prim {} f32) (t-prim {} f64) (t-prim {} bool))".to_string(),
                0,
                "valid too-many control",
            ),
        ] {
            let source = hand_authored_nominal_property(&application);
            assert_cli_diagnostic_counts(&source, reserved, 1, label);
        }
    }
}

#[test]
fn one_reserved_tensor_element_site_produces_one_located_cli_error() {
    for name in ["f8e4m3", "f8e5m2"] {
        assert_rejected_sites(
            &format!("def classify(x: tensor[3, {name}]) -> i32 = 0i32\n"),
            &[name],
        );
    }
}

#[test]
fn a_rejected_tensor_precision_keeps_the_independent_shape_error() {
    for name in ["f8e4m3", "f8e5m2"] {
        let (success, report, source) = check(&format!(
            "def inspect(x: tensor[3, {name}]) -> i64 = shape(x, 1i32)\n"
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
        "def left(x: f8e4m3) -> i32 = 0i32\ndef right(x: f8e5m2) -> i32 = 0i32\n",
        &["f8e4m3", "f8e5m2"],
    );
}

#[test]
fn repeated_reserved_spelling_keeps_distinct_declaration_locations() {
    assert_rejected_sites(
        "def left(x: f8e4m3) -> i32 = 0i32\ndef right(x: f8e4m3) -> i32 = 0i32\n",
        &["f8e4m3", "f8e4m3"],
    );
}

#[test]
fn a_call_does_not_report_its_failed_signature_again() {
    assert_rejected_sites(
        "def classify(x: f8e4m3) -> i32 = 0i32\nresult = classify(1i32)\n",
        &["f8e4m3"],
    );
}

#[test]
fn one_signature_keys_reserved_cli_errors_by_spelling() {
    assert_rejected_sites(
        "def classify(x: f8e4m3, y: f8e5m2) -> i32 = 0i32\n",
        &["f8e4m3", "f8e5m2"],
    );
    assert_rejected_sites(
        "def classify(x: f8e4m3, y: f8e4m3) -> i32 = 0i32\n",
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
        "def classify(x: (f8e4m3, f8e5m2)) -> i32 = 0i32\n",
        "def classify(x: Dict[f8e4m3, f8e5m2]) -> i32 = 0i32\n",
    ] {
        assert_rejected_sites(source, &["f8e4m3", "f8e5m2"]);
    }
}

#[test]
fn a_failed_signature_does_not_hide_a_cli_body_error() {
    let (success, report, source) = check("def classify(x: f8e4m3) -> i32 = cast(0i32, f8e5m2)\n");
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
fn declaration_ownership_does_not_absorb_a_same_spelling_cli_cast_error() {
    let (success, report, source) = check("def classify(x: f8e4m3) -> i32 = cast(0i32, f8e4m3)\n");
    assert!(!success, "{report:?}");
    assert_eq!(report.errors.len(), 2, "{report:?}");
    assert!(
        report
            .errors
            .iter()
            .all(|error| error.message.contains("f8e4m3")),
        "{report:?}"
    );
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
        let (success, report, _) = check(&format!("def classify(x: {name}) -> i32 = 0i32\n"));
        assert!(success, "{report:?}");
        assert_eq!(report.score, 1.0, "{report:?}");
        assert!(report.errors.is_empty(), "{report:?}");
    }
}
