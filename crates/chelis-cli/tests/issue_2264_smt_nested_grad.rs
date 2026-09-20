//! chelis#2264: CLI precheck and property filtering must preserve the typed
//! SMT boundary for valid nested gradients without hiding malformed siblings.

#![cfg(feature = "smt")]

use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

fn prove_json_with_extension(
    source: &str,
    extension: &str,
    extra: &[&str],
) -> (i32, Vec<Value>, String) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("nested_grad.{extension}"));
    std::fs::write(&path, source).expect("write nested-gradient fixture");
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .arg("prove")
        .arg(&path)
        .arg("--json")
        .args(extra)
        .output()
        .expect("run chelis prove");
    let records = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("stdout must be NDJSON"))
        .collect();
    (
        output.status.code().unwrap_or(-1),
        records,
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn prove_json(source: &str, extra: &[&str]) -> (i32, Vec<Value>, String) {
    prove_json_with_extension(source, "ch", extra)
}

fn generated_deep(source: &str) -> String {
    let declarations = chelis_surf::parser::parse_str(source).expect("Surf fixture parses");
    let deep = chelis_surf::desugar::desugar_program(&declarations).expect("Surf fixture desugars");
    chelis_deep::printer::print_canonical(&deep)
}

fn nested_grad_pair() -> &'static str {
    "module M
@property nested_grad_left forall(x: f32):
  (grad(grad(fn (xx: f32) -> xx * xx, wrt=xx), wrt=xx)(x) >= 0.0)
@property nested_grad_right forall(x: f32):
  (grad(grad(fn (xx: f32) -> xx + xx, wrt=xx), wrt=xx)(x) >= 0.0)
"
}

#[test]
fn surf_and_generated_deep_preserve_nested_grad_reason_with_and_without_only() {
    let surf = nested_grad_pair();
    let deep = generated_deep(surf);
    let cases = [
        (
            "surf-all",
            surf,
            "ch",
            Vec::new(),
            vec!["nested_grad_left", "nested_grad_right"],
        ),
        (
            "surf-only",
            surf,
            "ch",
            vec!["--only", "nested_grad_left"],
            vec!["nested_grad_left"],
        ),
        (
            "deep-all",
            deep.as_str(),
            "dp",
            Vec::new(),
            vec!["nested_grad_left", "nested_grad_right"],
        ),
        (
            "deep-only",
            deep.as_str(),
            "dp",
            vec!["--only", "nested_grad_left"],
            vec!["nested_grad_left"],
        ),
    ];

    for (case, source, extension, filters, expected_names) in cases {
        let mut extra = vec!["--tier", "smt-only"];
        extra.extend(filters);
        let (code, records, stderr) = prove_json_with_extension(source, extension, &extra);

        assert_eq!(code, 2, "{case}: records={records:#?}\nstderr={stderr}");
        assert!(
            records
                .iter()
                .all(|record| record["kind"] != "error" || record["stage"] != "check"),
            "{case}: valid nested gradients must survive the CLI precheck: {records:#?}"
        );
        let properties = records
            .iter()
            .filter(|record| record["kind"] == "property")
            .collect::<Vec<_>>();
        let names = properties
            .iter()
            .map(|property| property["name"].as_str().expect("property name"))
            .collect::<Vec<_>>();
        assert_eq!(names, expected_names, "{case}: {records:#?}");
        for property in &properties {
            assert_eq!(property["status"], "unsupported", "{case}: {property}");
            assert_eq!(property["proof_tier"], "smt", "{case}: {property}");
            assert_eq!(
                property["reason"], "scalar grad SMT lowering does not support nested gradients",
                "{case}: {property}"
            );
        }
        let summary = records
            .iter()
            .find(|record| record["kind"] == "summary")
            .expect("summary record");
        assert_eq!(
            summary["unsupported"],
            properties.len(),
            "{case}: {summary}"
        );
        assert_eq!(summary["passed"], 0, "{case}: {summary}");
        assert_eq!(summary["failed"], 0, "{case}: {summary}");
        assert_eq!(summary["errors"], 0, "{case}: {summary}");
    }
}

#[test]
fn malformed_nested_grad_sibling_still_fails_closed_before_filtering() {
    let source = "module M
@property malformed_sibling forall(x: f32):
  (grad(grad(fn (xx: f32) -> xx * xx, wrt=xx), wrt=missing)(x) >= 0.0)
@property nested_grad_left forall(x: f32):
  (grad(grad(fn (xx: f32) -> xx * xx, wrt=xx), wrt=xx)(x) >= 0.0)
";
    let (code, records, stderr) = prove_json(
        source,
        &["--tier", "smt-only", "--only", "nested_grad_left"],
    );

    assert_eq!(code, 3, "records={records:#?}\nstderr={stderr}");
    let check_errors = records
        .iter()
        .filter(|record| record["kind"] == "error" && record["stage"] == "check")
        .collect::<Vec<_>>();
    assert_eq!(check_errors.len(), 1, "{records:#?}");
    assert!(
        check_errors[0].to_string().contains("missing"),
        "{check_errors:#?}"
    );
    assert!(
        records.iter().all(|record| record["kind"] != "property"),
        "a malformed sibling must prevent property verdicts: {records:#?}"
    );
}
