//! Executable boundary witnesses for newly numbered builtin contracts.
use assert_cmd::Command;

fn evaluate(source: &str) -> std::process::Output {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("semantic_boundary.ch");
    std::fs::write(&file, source).unwrap();
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(file)
        .output()
        .unwrap()
}

#[test]
fn shifts_at_and_above_each_signed_width_follow_num_13() {
    for (dtype, width) in [("int8", 8), ("int16", 16), ("int32", 32), ("int64", 64)] {
        for count in [width, width + 1] {
            let source = format!(
                "left = shl(cast(1, {dtype}), cast({count}, {dtype}))\npositive = shr(cast(1, {dtype}), cast({count}, {dtype}))\nnegative = shr(cast(-1, {dtype}), cast({count}, {dtype}))\n"
            );
            let output = evaluate(&source);
            assert!(
                output.status.success(),
                "{source}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert_eq!(
                String::from_utf8(output.stdout).unwrap().trim(),
                "left = 0\npositive = 0\nnegative = -1",
                "{source}"
            );
        }
    }
}

#[test]
fn negative_shift_counts_fail_with_the_num_13_diagnostic() {
    for operation in ["shl", "shr"] {
        let output = evaluate(&format!("result = {operation}(1, -1)\n"));
        assert!(!output.status.success());
        let diagnostic = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            diagnostic.contains("shift amount must be non-negative, got -1"),
            "{diagnostic}"
        );
    }
}
