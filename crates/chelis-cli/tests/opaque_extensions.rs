use assert_cmd::Command;
use std::{fs, path::Path, process::Output};

fn run(root: &Path, args: &[&str]) -> Output {
    Command::cargo_bin("chelis")
        .unwrap()
        .current_dir(root)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(args)
        .output()
        .unwrap()
}
fn succeeds(output: &Output) {
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn opaque_data_survives_tools_and_keeps_the_same_execution_result() {
    let dir = tempfile::tempdir().unwrap();
    let payload = r#"{type: "ablation", span: 1, surf_future: true, expr: (undefined_macro unbound), expr: (var {}), number: 1e-3f32}"#;
    let clean = "(def {} main (lit {type: (t-prim {} int32)} 7))";
    fs::write(dir.path().join("clean.dp"), clean).unwrap();
    fs::write(
        dir.path().join("data.dp"),
        clean.replacen("{}", &format!("{{tool_data: {payload}}}"), 1),
    )
    .unwrap();
    let formatted = run(dir.path(), &["fmt", "data.dp"]);
    succeeds(&formatted);
    assert!(String::from_utf8_lossy(&formatted.stdout).contains(payload));
    fs::write(dir.path().join("data.dp"), &formatted.stdout).unwrap();
    let again = run(dir.path(), &["fmt", "data.dp"]);
    succeeds(&again);
    assert_eq!(formatted.stdout, again.stdout);
    for args in [
        ["check", "data.dp"].as_slice(),
        &["validate", "--deep", "data.dp"],
        &["build", "data.dp", "--target", "c", "--output", "out.c"],
    ] {
        succeeds(&run(dir.path(), args));
    }
    assert!(dir.path().join("out.c").exists());
    let clean_eval = run(dir.path(), &["eval", "--file", "clean.dp", "--json"]);
    let data_eval = run(dir.path(), &["eval", "--file", "data.dp", "--json"]);
    succeeds(&clean_eval);
    succeeds(&data_eval);
    assert_eq!(data_eval.stdout, clean_eval.stdout);
    let result: serde_json::Value = serde_json::from_slice(&data_eval.stdout).unwrap();
    assert!(!result["roots"].as_array().unwrap().is_empty());
    let surf = run(dir.path(), &["surf", "data.dp"]);
    assert!(!surf.status.success());
    assert!(surf.stdout.is_empty(), "no partial Surf output");
    assert!(String::from_utf8_lossy(&surf.stderr).contains("tool_data"));
    succeeds(&run(dir.path(), &["surf", "clean.dp"]));
    fs::write(
        dir.path().join("bad.dp"),
        clean.replace("(t-prim {} int32)", "false"),
    )
    .unwrap();
    for args in [
        ["check", "bad.dp"].as_slice(),
        &["validate", "--deep", "bad.dp"],
        &["eval", "--file", "bad.dp"],
        &["surf", "bad.dp"],
        &["build", "bad.dp", "--target", "c", "--output", "bad.c"],
    ] {
        let output = run(dir.path(), args);
        assert!(!output.status.success(), "{args:?}");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(text.contains("type"), "{text}");
    }
    assert!(!dir.path().join("bad.c").exists());
}
