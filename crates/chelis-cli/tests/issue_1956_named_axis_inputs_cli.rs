//! #1956: actual Reef-context CLI admission and independent tagged/effect oracles.
use serde_json::{Value, json};
use std::path::Path;
use std::process::{Command, Output};

fn run(root: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_chelis"))
        .current_dir(root)
        .env_remove("CHELIS_STYLE_GATE_DISABLE")
        .args(args)
        .output()
        .unwrap()
}

fn success(root: &Path, args: &[&str]) -> Output {
    let output = run(root, args);
    assert!(
        output.status.success(),
        "{} failed:\n{}\n{}",
        args.join(" "),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn package(formal: bool, body: &str, failing: bool) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("src")).unwrap();
    std::fs::write(dir.path().join("reef.toml"), format!("[package]\nname = \"named_axis_inputs\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"Probe\"\n", env!("CARGO_PKG_VERSION"))).unwrap();
    let initializer = if failing {
        "_ = to_tensor([0.0f32, 0.0f32]) |> uniform_like(0.0f32, 1.0f32)\n to_tensor([floor_div(1i32, 0i32) |> cast(f32), 0.0f32])"
    } else {
        "to_tensor([3.0f32, 5.0f32])"
    };
    let (param, result) = if formal {
        ("weights", "tensor[..pre, ..post, f32]")
    } else {
        ("x", "(tensor[..pre, ..post, f32], tensor[f32])")
    };
    std::fs::write(dir.path().join("src/captures.ch"), format!("module Probe.Captures\nexport (total)\nbaseline = to_tensor([7.0f32, 11.0f32])\nweights = with seed(17i64) {{ _ = print(\"initialize\")\n {initializer} }}\ndef total[pre, post]({param}: &tensor[..pre, seq, ..post, f32]) -> {result} = {body}\n")).unwrap();
    let draws = "_ = uniform_like(copy(x), 0.0f32, 1.0f32)\n".repeat(5);
    std::fs::write(dir.path().join("src/client.ch"), format!("module Probe.Client\nimport Probe.Captures (total)\ndef entry(x: tensor[seq, f32]) = with seed(42i64) {{ _ = print(\"entry\")\n {draws} result = total(x)\n (result, uniform_like(x, 0.0f32, 1.0f32)) }}\nalias = entry\ndef main() = alias(to_tensor([7.0f32, 11.0f32]))\nout = main()\n")).unwrap();
    for file in ["src/captures.ch", "src/client.ch"] {
        success(dir.path(), &["fmt", "--inplace", file]);
        success(dir.path(), &["fmt", "--check", file]);
        success(dir.path(), &["lint", "--check", file]);
        success(dir.path(), &["validate", "--surf", file]);
        let checked = success(dir.path(), &["check", file]);
        let check: Value = serde_json::from_slice(&checked.stdout).unwrap();
        assert_eq!(check["score"], 1);
        assert_eq!(check["errors"], json!([]));
        assert_eq!(check["unresolved_names"], json!([]));
    }
    dir
}

fn tensor(shape: &[usize], bits: &[&str]) -> Value {
    json!({"type":"tensor","value":{"shape":shape,"data":{"dtype":"f32","bits":bits}}})
}

fn assert_values(formal: bool, body: &str, captured_bits: &str, transcript: &[&str]) {
    let dir = package(formal, body, false);
    let output = success(
        dir.path(),
        &[
            "eval",
            "--json",
            "--timeout",
            "10",
            "--file",
            "src/client.ch",
        ],
    );
    assert!(output.stderr.is_empty());
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["schema_version"], 3);
    let mut expected = Vec::new();
    // Existing manifest observes main and out independently; both calls matter.
    for entry in ["main", "out"] {
        if formal {
            expected.push((format!("{entry}.0"), tensor(&[], &["41900000"])));
        } else {
            expected.push((format!("{entry}.0.0"), tensor(&[], &["41900000"])));
            expected.push((format!("{entry}.0.1"), tensor(&[], &[captured_bits])));
        }
        expected.push((
            format!("{entry}.1"),
            tensor(&[2], &["3e68de41", "3f38fdad"]),
        ));
    }
    let roots = result["roots"].as_array().unwrap();
    assert_eq!(roots.len(), expected.len());
    for (index, (root, (name, value))) in roots.iter().zip(&expected).enumerate() {
        assert_eq!(root["node_id"], index);
        assert_eq!(root["name"], name.as_str());
        assert_eq!(&root["value"], value);
    }
    assert_eq!(result["manifest"]["target"], "Eval");
    assert_eq!(result["manifest"]["requires_main"], true);
    assert_eq!(
        result["manifest"]["entries"],
        Value::Array(
            expected
                .iter()
                .map(|(name, _)| json!({"name":name,"lane":"Host","required_inputs":[]}))
                .collect()
        )
    );
    assert_eq!(result["transcript"], json!(transcript));
}

#[test]
fn formal_shadow_keeps_default_style_admission_and_no_initializer() {
    assert_values(true, "sum(weights, seq)", "", &["entry", "entry"]);
}

#[test]
fn dead_capture_keeps_default_style_admission_and_no_initializer() {
    assert_values(
        false,
        "(sum(x, seq), if true then sum(baseline, 0i32) else sum(weights, 0i32))",
        "41900000",
        &["entry", "entry"],
    );
}

#[test]
fn selected_capture_preserves_values_and_initializer_events() {
    assert_values(
        false,
        "(sum(x, seq), sum(weights, 0i32))",
        "41000000",
        &["entry", "initialize", "entry"],
    );
}

#[test]
fn initializer_failure_preserves_original_error_and_channels() {
    let dir = package(false, "(sum(x, seq), sum(weights, 0i32))", true);
    for json in [true, false] {
        let mut args = vec!["eval", "--timeout", "10", "--file", "src/client.ch"];
        if json {
            args.push("--json");
        }
        let output = run(dir.path(), &args);
        assert!(!output.status.success());
        let error = "error: numeric trap: division by zero in floor_div at i32\n";
        if json {
            assert!(output.stdout.is_empty());
            assert_eq!(
                String::from_utf8(output.stderr).unwrap(),
                format!("entry\ninitialize\n{error}")
            );
        } else {
            assert_eq!(output.stdout, b"entry\ninitialize\n");
            assert_eq!(String::from_utf8(output.stderr).unwrap(), error);
        }
    }
}
