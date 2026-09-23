//! CLI execution of [05-OP-37], without relabeling compiled dropout.
mod common;

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
fn staged_size_example_checks_exact_mask_and_rejects_a_false_result_claim() {
    let directory = tempfile::tempdir().unwrap();
    let file = directory.path().join("staged.ch");
    let source = include_str!("../../../examples/dropout_staged_claim.ch");
    for valid in [true, false] {
        let source = if valid {
            source.to_owned()
        } else {
            source
                .replace(
                    "source = to_tensor([1.0f32, 1.0f32])",
                    "source = to_tensor([1.0f32, 1.0f32, 1.0f32])",
                )
                .replace(
                    "x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])",
                    "x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32])",
                )
        };
        std::fs::write(&file, source).unwrap();
        let path = file.to_str().unwrap();
        assert!(cli(&["fmt", "--inplace", path]).status.success());
        assert!(cli(&["lint", "--check", path]).status.success());
        let checked = cli(&["check", path]);
        assert!(
            checked.status.success(),
            "{}",
            String::from_utf8_lossy(&checked.stderr)
        );
        let checked: Value = serde_json::from_slice(&checked.stdout).unwrap();
        assert_eq!(checked["score"], 1.0);
        assert_eq!(checked["errors"], serde_json::json!([]));
        let output = cli(&["eval", "--file", path, "--json"]);
        if valid {
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let result: chelis_compiler_api::schema::EvalResult =
                serde_json::from_slice(&output.stdout).unwrap();
            let chelis_compiler_api::schema::ExecutionValue::Tensor { value } =
                &result.roots[0].value
            else {
                panic!("{result:?}")
            };
            assert_eq!(value.shape, vec![2, 2]);
            assert_eq!(value.data.to_f64_lossy_vec(), vec![2.0, 0.0, 0.0, 0.0]);
        } else {
            assert!(!output.status.success());
            let error = String::from_utf8_lossy(&output.stderr);
            assert!(
                error.contains("claimed = 2")
                    && error.contains("reshape axis 0 = 3")
                    && error.contains("numeric trap: domain in reshape at i64"),
                "{error}"
            );
        }
    }
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
fn invalid_empty_rate_traps_in_eval_and_c_and_a_runtime_rate_builds() {
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
    // Compiled C traps the same invalid rate at its draw, as eval does, even
    // over an empty tensor.
    let invalid_out = dir.path().join("invalid-out");
    let output = cli(&[
        "build",
        path,
        "--target",
        "c",
        "--output",
        invalid_out.to_str().unwrap(),
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(common::link_generated(&invalid_out, "dropout.c", "dropout").success());
    let run = std::process::Command::new(invalid_out.join("dropout"))
        .output()
        .unwrap();
    assert!(!run.status.success());
    let text = String::from_utf8_lossy(&run.stderr);
    assert!(
        text.contains("numeric trap: domain in dropout at f32"),
        "{text}"
    );
    // A runtime rate is an ordinary operand (chelis#2411).
    std::fs::write(&file, "def sample(x: tensor[4, f32], rate: f32) -> tensor[4, f32] = with seed(42i64) { dropout(x, rate) }\n").unwrap();
    assert!(cli(&["fmt", "--inplace", path]).status.success());
    let output = cli(&[
        "build",
        path,
        "--target",
        "c",
        "--output",
        dir.path().join("out").to_str().unwrap(),
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// [05-RNG-1] over f32 ones at rate 0.5, transcribed independently of every
/// evaluator: the mask for `ordinal` under `seed`, scaled by `1 / (1 - rate)`.
fn reference_mask(seed: u64, ordinal: u64, count: u64) -> Vec<f64> {
    fn mix(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
    (0..count)
        .map(|index| {
            let word = mix(seed ^ mix(ordinal).rotate_left(17) ^ mix(index).rotate_left(41));
            let unit = ((word >> 11) as f64 / 9007199254740992.0) as f32;
            if unit < 0.5 { 0.0 } else { 2.0 }
        })
        .collect()
}

/// [05-RNG-1]: a `dropout` in the unselected arm of a runtime `if` takes no
/// ordinal. Inlined into a caller's tensor segment the `if` would become a
/// kernel `where` that draws in both arms (chelis#2410), so IR lowering
/// refuses the draw and compiled C runs the branch as host control flow, as
/// eval does. The flags are computed from data, so no lane can fold them.
///
/// Evidentiary status: REGRESSION TEST. At 3b5f029d8 compiled C gave the
/// later draw the (7, 1) and (7, 2) masks, and the `not(flag)` call of the
/// last program an extra ordinal.
#[test]
fn a_dropout_in_an_unselected_runtime_arm_takes_no_ordinal_in_eval_or_c() {
    let layer = "def layer(x: tensor[8, f32], training: bool) -> tensor[8, f32] ! { Random } = if training then dropout(x, 0.5f32) else x\n";
    let ones = "x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32])";
    let not_training = "training = lt(tensor_to_scalar(sum(copy(x), 0i32)), 0.0f32)";
    let programs = [
        (
            "inline_flag",
            format!(
                "{layer}def main() =\n  with seed(7i64) {{\n    {ones}\n    {not_training}\n    y = layer(copy(x), training)\n    z = dropout(x, 0.5f32)\n    (y, z)\n  }}\n"
            ),
            vec![vec![1.0; 8], reference_mask(7, 0, 8)],
        ),
        (
            "two_layers",
            format!(
                "{layer}def model(x: tensor[8, f32], training: bool) -> tensor[8, f32] ! {{ Random }} = layer(layer(x, training), training)\ndef main() =\n  with seed(7i64) {{\n    {ones}\n    {not_training}\n    out = model(copy(x), training)\n    noise = dropout(x, 0.5f32)\n    (out, noise)\n  }}\n"
            ),
            vec![vec![1.0; 8], reference_mask(7, 0, 8)],
        ),
        (
            "taken_then_untaken",
            format!(
                "{layer}def main() =\n  with seed(7i64) {{\n    {ones}\n    flag = gt(tensor_to_scalar(sum(copy(x), 0i32)), 0.0f32)\n    a = layer(copy(x), flag)\n    b = dropout(copy(x), 0.5f32)\n    c = layer(copy(x), not(flag))\n    d = dropout(x, 0.5f32)\n    (a, b, c, d)\n  }}\n"
            ),
            vec![
                reference_mask(7, 0, 8),
                reference_mask(7, 1, 8),
                vec![1.0; 8],
                reference_mask(7, 2, 8),
            ],
        ),
    ];
    assert_ne!(reference_mask(7, 0, 8), reference_mask(7, 1, 8));
    for (name, source, expected) in programs {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join(format!("{name}.ch"));
        std::fs::write(&file, &source).unwrap();
        let path = file.to_str().unwrap();
        assert!(cli(&["fmt", "--inplace", path]).status.success(), "{name}");
        let eval = cli(&["eval", "--file", path]);
        assert!(
            eval.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&eval.stderr)
        );
        let eval = String::from_utf8(eval.stdout).unwrap();
        let compiled = common::build_and_run(&std::fs::read_to_string(&file).unwrap(), name);
        for (index, expected) in expected.iter().enumerate() {
            let root = format!("main.{index}");
            assert_eq!(
                &common::parse_tensor_data(&eval, &root),
                expected,
                "{name} eval {root}"
            );
            assert_eq!(
                &common::parse_tensor_data(&compiled, &root),
                expected,
                "{name} C {root}"
            );
        }
    }
}
