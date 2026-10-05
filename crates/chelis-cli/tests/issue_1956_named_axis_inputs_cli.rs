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
    std::fs::write(dir.path().join("reef.toml"), format!("schema = \"1\"\n[package]\nname = \"named-axis-inputs\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"Probe\"\n", env!("CARGO_PKG_VERSION"))).unwrap();
    let initializer = if failing {
        "uniform_like(key_from_seed(17i64), to_tensor([0.0f32, 0.0f32]), 0.0f32, 1.0f32); to_tensor([floor_div(1i32, 0i32) |> cast(f32), 0.0f32])"
    } else {
        "to_tensor([3.0f32, 5.0f32])"
    };
    let (param, result) = if formal {
        ("weights", "tensor[..pre, ..post, f32]")
    } else {
        ("x", "(tensor[..pre, ..post, f32], tensor[f32])")
    };
    std::fs::write(dir.path().join("src/captures.ch"), format!("module Probe.Captures\nexport (total)\nbaseline = to_tensor([7.0f32, 11.0f32])\nweights = do {{ print(\"initialize\"); {initializer} }}\ndef total[pre, post]({param}: &tensor[..pre, seq, ..post, f32]) -> {result} = {body}\n")).unwrap();
    // Five discarded draws along a `split_key` chain, then the kept draw
    // from the chain's last remainder.
    let draws = (0..5)
        .map(|j| {
            let parent = if j == 0 {
                "key_from_seed(42i64)".to_string()
            } else {
                format!("r{}", j - 1)
            };
            format!("(k{j}, r{j}) = split_key({parent})\n _ = uniform_like(k{j}, copy(x), 0.0f32, 1.0f32)\n")
        })
        .collect::<String>();
    std::fs::write(dir.path().join("src/client.ch"), format!("module Probe.Client\nimport Probe.Captures (total)\ndef entry(x: tensor[seq, f32]) = {{ _ = print(\"entry\")\n {draws} result = total(x)\n (result, uniform_like(r4, x, 0.0f32, 1.0f32)) }}\nalias = entry\ndef main() = alias(to_tensor([7.0f32, 11.0f32]))\nout = main()\n")).unwrap();
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
    assert_eq!(result["schema_version"], 4);
    let mut expected = Vec::new();
    // `entry` prints, so `alias = entry` carries its IO ([04-INF-9]: an alias
    // performs what it names, as a lambda would) and `main` is effectful. An
    // effectful nullary declaration is a callable, not a value root
    // (`chelis_effects::realizability::collect_manifest_entries`): observing
    // it would run an effect that an unselected declaration runs zero times.
    // Only `out` is a root, so `entry` runs once. Before chelis#3149 the alias
    // hid the IO, `main` was auto-applied as a root, and `entry` ran twice;
    // `pure_alias_chain_still_auto_applies_main_as_a_root` keeps that case
    // for a pure chain.
    let entry = "out";
    if formal {
        expected.push((format!("{entry}.0"), tensor(&[], &["41900000"])));
    } else {
        expected.push((format!("{entry}.0.0"), tensor(&[], &["41900000"])));
        expected.push((format!("{entry}.0.1"), tensor(&[], &[captured_bits])));
    }
    // `uniform_like` over 2 f32 elements keyed by the fifth remainder of
    // the `split_key` chain from `key_from_seed(42)` (0xb06b4e4bcc67415d),
    // from `key_ref.py`/`slice2_ref.py`.
    expected.push((
        format!("{entry}.1"),
        tensor(&[2], &["3edc4654", "3f027eac"]),
    ));
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
    assert_values(true, "sum(weights, seq)", "", &["entry"]);
}

#[test]
fn dead_capture_keeps_default_style_admission_and_no_initializer() {
    assert_values(
        false,
        "(sum(x, seq), if true then sum(baseline, 0i32) else sum(weights, 0i32))",
        "41900000",
        &["entry"],
    );
}

#[test]
fn selected_capture_preserves_values_and_initializer_events() {
    assert_values(
        false,
        "(sum(x, seq), sum(weights, 0i32))",
        "41000000",
        &["entry", "initialize"],
    );
}

/// The positive control for the expectations above: through the same
/// `alias = entry; def main() = alias(...); out = main()` shape, a pure
/// `entry` leaves `main` a value root that evaluation auto-applies, while an
/// `entry` that prints makes `main` a callable and leaves only `out`.
#[test]
fn pure_alias_chain_still_auto_applies_main_as_a_root() {
    for (entry_body, roots, transcript) in [
        (
            "add(x, x)",
            vec!["main", "out"],
            // A program that performs no effect reports no transcript.
            Value::Null,
        ),
        (
            "{\n  _ = print(\"entry\")\n  add(x, x)\n}",
            vec!["out"],
            json!(["entry"]),
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("alias_roots.ch");
        std::fs::write(
            &file,
            format!(
                "def entry(x: tensor[2, f32]) -> tensor[2, f32] = {entry_body}\n\
                 alias = entry\n\
                 def main() = alias(to_tensor([1.0f32, 2.0f32]))\n\
                 out = main()\n"
            ),
        )
        .unwrap();
        let path = file.to_str().unwrap();
        success(dir.path(), &["fmt", "--inplace", path]);
        let output = success(
            dir.path(),
            &["eval", "--json", "--timeout", "10", "--file", path],
        );
        let result: Value = serde_json::from_slice(&output.stdout).unwrap();
        let names: Vec<&str> = result["roots"]
            .as_array()
            .unwrap()
            .iter()
            .map(|root| root["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, roots, "{entry_body}: {result}");
        assert_eq!(result["transcript"], transcript, "{entry_body}: {result}");
    }
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
        let error = "numeric trap: division by zero in floor_div at i32\n";
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
