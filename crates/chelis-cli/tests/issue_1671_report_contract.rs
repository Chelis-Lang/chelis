//! [04-FIT-13]: reports carry fitness/diagnostics and requested signatures.
//! Checked Deep is a compiler product, outside this report document.
use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use tempfile::tempdir;

fn assert_report(report: &Value, requested: bool, accepted: bool) {
    let mut expected = vec![
        "score",
        "components",
        "typed_nodes",
        "untyped_nodes",
        "total_nodes",
        "unresolved_names",
        "errors",
    ];
    if requested {
        expected.push("inferred_signatures");
    }
    expected.sort_unstable();
    let mut actual = report
        .as_object()
        .expect("file report object")
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>();
    actual.sort_unstable();
    assert_eq!(
        actual, expected,
        "report must retain its decided document fields"
    );
    assert_eq!(report["errors"].as_array().unwrap().is_empty(), accepted);
    assert!(!report.as_object().unwrap().contains_key("typed_ast"));
    let _: chelis_compiler_api::schema::WireCheckResult =
        serde_json::from_value(report.clone()).expect("one typed report shape");
}

#[test]
fn surf_and_deep_success_and_failure_keep_the_report_contract() {
    let dir = tempdir().unwrap();
    let fixtures = [
        ("clean.ch", "def id(x: f32) -> f32 = x\n", true),
        ("bad_type.ch", "def id(x: f32) -> f32 = missing\n", false),
        ("bad_parse.ch", "def =\n", false),
        (
            "clean.dp",
            "(def {} id (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x)))\n",
            true,
        ),
        (
            "bad_type.dp",
            "(def {} id (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} missing)))\n",
            false,
        ),
        ("bad_parse.dp", "(def\n", false),
    ];
    for (name, source, accepted) in fixtures {
        let path = dir.path().join(name);
        fs::write(&path, source).unwrap();
        for requested in [false, true] {
            if requested && name.contains("bad_parse") {
                continue;
            }
            let mut cmd = Command::cargo_bin("chelis").unwrap();
            cmd.env("CHELIS_STYLE_GATE_DISABLE", "1")
                .arg("check")
                .arg(&path);
            if requested {
                cmd.arg("--show-inferred");
            }
            let out = cmd.output().unwrap();
            assert_eq!(
                out.status.code(),
                Some(if accepted { 0 } else { 2 }),
                "{name}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            let report: Value = serde_json::from_slice(&out.stdout).unwrap();
            assert_report(&report, requested, accepted);
        }
    }
}
