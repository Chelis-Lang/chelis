//! Ascription mismatches must be rejected before either execution lane.
use assert_cmd::Command;
use tempfile::tempdir;
mod common;

#[test]
fn scalar_literal_ascription_mismatches_reject_at_check_eval_and_c_build() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("ascription.ch");
    let output = dir.path().join("native");
    for expression in [
        "(1.5f64 : f32)",
        "(16777217.0f64 : f32)",
        "(1.1f32 : f64)",
        "(1.1 : f64)",
        "(1 : f64)",
        "((1.1f64 : f32) : f64)",
        "{\n  y: f32 = 1.5f64\n  y\n}",
        "{\n  y: f64 = 1.1\n  y\n}",
    ] {
        common::write_file(&path, &format!("out = {expression}\n"));
        for args in [
            vec!["check", path.to_str().unwrap()],
            vec!["eval", "--file", path.to_str().unwrap()],
            vec![
                "build",
                "--emit-c",
                path.to_str().unwrap(),
                "--target",
                "c",
                "--output",
                output.to_str().unwrap(),
            ],
        ] {
            let result = Command::cargo_bin("chelis")
                .unwrap()
                .env("CHELIS_STYLE_GATE_DISABLE", "1")
                .args(&args)
                .output()
                .unwrap();
            let text = format!(
                "{}{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
            assert!(!result.status.success(), "{args:?}: {expression}: {text}");
            assert!(
                text.contains("mismatch") || text.contains("does not match"),
                "{args:?}: {expression}: {text}"
            );
            assert!(!text.contains("integer atom cannot carry"), "{text}");
        }
    }
}
