//! CLI execution of [05-OP-37], without relabeling compiled dropout.
use assert_cmd::Command;
use serde_json::Value;

fn cli(args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .unwrap()
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn executable_example_survives_format_check_and_exact_eval() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("dropout.ch");
    std::fs::write(
        &file,
        include_str!("../../../examples/dropout_fixed_stream.ch"),
    )
    .unwrap();
    let path = file.to_str().unwrap();
    let formatted = cli(&["fmt", "--inplace", path]);
    assert!(
        formatted.status.success(),
        "{}",
        String::from_utf8_lossy(&formatted.stderr)
    );
    let checked = cli(&["check", path]);
    assert!(
        checked.status.success(),
        "{}",
        String::from_utf8_lossy(&checked.stderr)
    );
    let checked: Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert_eq!(checked["score"], 1.0, "{checked}");
    assert_eq!(checked["errors"], serde_json::json!([]), "{checked}");
    for command in ["fmt", "lint"] {
        assert!(cli(&[command, "--check", path]).status.success());
    }
    assert!(cli(&["validate", "--surf", path]).status.success());
    let deep = cli(&["deep", path]);
    assert!(deep.status.success());
    let deep_path = dir.path().join("dropout.dp");
    std::fs::write(&deep_path, deep.stdout).unwrap();
    let surf = cli(&["surf", deep_path.to_str().unwrap()]);
    assert!(surf.status.success());
    let roundtrip_path = dir.path().join("roundtrip.ch");
    std::fs::write(&roundtrip_path, surf.stdout).unwrap();
    let roundtrip_path = roundtrip_path.to_str().unwrap();
    assert!(cli(&["fmt", "--check", roundtrip_path]).status.success());
    let roundtrip_check = cli(&["check", roundtrip_path]);
    assert!(roundtrip_check.status.success());
    let checked: Value = serde_json::from_slice(&roundtrip_check.stdout).unwrap();
    assert_eq!(checked["score"], 1.0, "{checked}");
    assert_eq!(checked["errors"], serde_json::json!([]), "{checked}");
    for source_path in [path, path, roundtrip_path] {
        let output = cli(&["eval", "--file", source_path, "--json"]);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let output: Value = serde_json::from_slice(&output.stdout).unwrap();
        let result: chelis_compiler_api::schema::EvalResult =
            serde_json::from_value(output).unwrap();
        let roots = result
            .roots
            .iter()
            .filter_map(|root| match &root.value {
                chelis_compiler_api::schema::ExecutionValue::Tensor { value } => {
                    Some((root.name.as_deref().unwrap(), value.data.to_f64_lossy_vec()))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        // Independently computed seed 42 ordinals 0 and 1; the second
        // differs, so a fresh-mask backward or omitted-forward mutant fails.
        assert_eq!(
            roots,
            [
                ("main.0", vec![0.0, 2.0, 0.0, 0.0]),
                ("main.1", vec![2.0, 0.0, 0.0, 0.0])
            ]
        );
    }
}

#[test]
fn invalid_empty_rate_is_a_real_cli_error_and_c_build_stays_unsupported() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("dropout.ch");
    std::fs::write(&file, "def empty() -> tensor[0, f32] = to_tensor([])\ndef invalid(x: tensor[0, f32]) -> tensor[0, f32] = dropout(x, 1.0f32)\ndef main() = with seed(42i64) { invalid(empty()) }\n").unwrap();
    let path = file.to_str().unwrap();
    assert!(cli(&["fmt", "--inplace", path]).status.success());
    let output = cli(&["eval", "--file", path, "--json"]);
    assert!(!output.status.success());
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        text.contains("numeric trap: domain in dropout at f32"),
        "{text}"
    );
    std::fs::write(&file, "def sample(x: tensor[4, f32]) -> tensor[4, f32] = with seed(42i64) { dropout(x, 0.5f32) }\n").unwrap();
    assert!(cli(&["fmt", "--inplace", path]).status.success());
    let output = cli(&[
        "build",
        path,
        "--target",
        "c",
        "--output",
        dir.path().join("out").to_str().unwrap(),
    ]);
    assert!(!output.status.success());
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        text.contains("dropout") || text.contains("Dropout"),
        "{text}"
    );
}
