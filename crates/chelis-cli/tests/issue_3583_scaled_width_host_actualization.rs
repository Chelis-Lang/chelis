//! chelis#3583: a generic model block may call the same linear helper at
//! distinct widths. Each invocation must actualize its own input and output
//! axes before the C kernel's entry checks run (spec/04 §4.7).

#[path = "common/mod.rs"]
mod common;

use assert_cmd::Command;

const SOURCE: &str = r#"
type Linear[i, o] =
  | Linear { w: tensor[i, o, f32], b: tensor[o, f32] }
def linear[a, i, o](x: &tensor[a, i, f32], l: &Linear[i, o]) -> tensor[a, o, f32] = {
  n_a = shape(x, 0i32)
  match l with {
    | Linear { w, b } => add(matmul(x, w), insert(b, 0i32, n_a))
  }
}
def block[a, c, f](x: tensor[a, c, f32], first: &Linear[c, f], second: &Linear[f, c]) -> tensor[a, c, f32] = linear(linear(x, first), second)
def main() -> f32 = {
  x = reshape(insert(scalar_to_tensor(0.1f32), 0i32, 2i64), [1i64, 2i64])
  first: Linear[2, 8] = Linear { w: reshape(insert(scalar_to_tensor(0.1f32), 0i32, 16i64), [2i64, 8i64]), b: insert(scalar_to_tensor(0.1f32), 0i32, 8i64) }
  second: Linear[8, 2] = Linear { w: reshape(insert(scalar_to_tensor(0.1f32), 0i32, 16i64), [8i64, 2i64]), b: insert(scalar_to_tensor(0.1f32), 0i32, 2i64) }
  tensor_to_scalar(sum(reshape(block(x, first, second), [2i64]), 0i32))
}
"#;

fn numeric_result(output: &str) -> f64 {
    output
        .trim()
        .strip_prefix("main = ")
        .expect("main result")
        .parse()
        .expect("numeric result")
}

fn folded_source() -> String {
    SOURCE
        .replace(
            "def main() -> f32 = {",
            "type Block[c, f] = | Block { first: Linear[c, f], second: Linear[f, c] }\ndef stage[a, c, f](x: tensor[a, c, f32], blocks: &List[Block[c, f]]) -> tensor[a, c, f32] = fold(fn (acc: tensor[a, c, f32], blk: Block[c, f]) -> match blk with { | Block { first, second } => block(acc, first, second) }, x, copy(blocks))\ndef main() -> f32 = {",
        )
        .replace(
            "block(x, first, second)",
            "stage(x, [Block { first, second }])",
        )
}

fn activated_source() -> String {
    SOURCE.replace(
        "linear(linear(x, first), second)",
        "linear(gelu(linear(x, first)), second)",
    )
}

#[test]
fn two_widths_in_one_generic_block_match_eval_and_execute_in_c() {
    let dir = tempfile::tempdir().expect("source dir");
    let source = dir.path().join("scaled_width.ch");
    common::write_file(&source, SOURCE);
    let eval = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(&source)
        .output()
        .expect("eval");
    assert!(
        eval.status.success(),
        "{}",
        String::from_utf8_lossy(&eval.stderr)
    );
    let eval = numeric_result(&String::from_utf8(eval.stdout).expect("utf-8 eval"));
    let c = numeric_result(&common::build_and_run(SOURCE, "scaled_width"));
    assert!((eval - 0.392).abs() < 1e-6, "eval = {eval}");
    assert!((c - eval).abs() < 1e-6, "C = {c}, eval = {eval}");
}

#[test]
fn incompatible_second_layer_width_remains_a_type_error() {
    let source = SOURCE.replace("second: Linear[8, 2]", "second: Linear[7, 2]");
    let dir = tempfile::tempdir().expect("source dir");
    let path = dir.path().join("wrong_width.ch");
    common::write_file(&path, &source);
    let checked = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", "--allow-style-violations"])
        .arg(&path)
        .output()
        .expect("check");
    let report: serde_json::Value =
        serde_json::from_slice(&checked.stdout).expect("check JSON report");
    assert!(
        !checked.status.success() && report["errors"].as_array().is_some_and(|e| !e.is_empty()),
        "{report}"
    );
}

#[test]
fn fold_callback_keeps_each_scaled_width_call_separate() {
    let source = folded_source();
    let dir = tempfile::tempdir().expect("source dir");
    let path = dir.path().join("folded_width.ch");
    common::write_file(&path, &source);
    let eval = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(&path)
        .output()
        .expect("eval");
    assert!(
        eval.status.success(),
        "{}",
        String::from_utf8_lossy(&eval.stderr)
    );
    let eval = numeric_result(&String::from_utf8(eval.stdout).expect("utf-8 eval"));
    let c = numeric_result(&common::build_and_run(&source, "folded_width"));
    assert!((c - eval).abs() < 1e-6, "C = {c}, eval = {eval}");
}

#[test]
fn fold_callback_rejects_mismatched_scaled_width_block() {
    let source = folded_source().replace("second: Linear[8, 2]", "second: Linear[7, 2]");
    let dir = tempfile::tempdir().expect("source dir");
    let path = dir.path().join("wrong_folded_width.ch");
    common::write_file(&path, &source);
    let checked = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", "--allow-style-violations"])
        .arg(&path)
        .output()
        .expect("check");
    let report: serde_json::Value = serde_json::from_slice(&checked.stdout).expect("check JSON");
    assert!(
        !checked.status.success() && report["errors"].as_array().is_some_and(|e| !e.is_empty()),
        "{report}"
    );
}

#[test]
fn activation_between_widths_preserves_the_checked_middle_axis() {
    let source = activated_source();
    let dir = tempfile::tempdir().expect("source dir");
    let path = dir.path().join("activated_width.ch");
    common::write_file(&path, &source);
    let eval = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(&path)
        .output()
        .expect("eval");
    assert!(
        eval.status.success(),
        "{}",
        String::from_utf8_lossy(&eval.stderr)
    );
    let eval = numeric_result(&String::from_utf8(eval.stdout).expect("utf-8 eval"));
    let c = numeric_result(&common::build_and_run(&source, "activated_width"));
    assert!((c - eval).abs() < 1e-6, "C = {c}, eval = {eval}");
}

#[test]
fn activation_between_widths_rejects_wrong_second_layer() {
    let source = activated_source().replace("second: Linear[8, 2]", "second: Linear[7, 2]");
    let dir = tempfile::tempdir().expect("source dir");
    let path = dir.path().join("wrong_activated_width.ch");
    common::write_file(&path, &source);
    let checked = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", "--allow-style-violations"])
        .arg(&path)
        .output()
        .expect("check");
    let report: serde_json::Value = serde_json::from_slice(&checked.stdout).expect("check JSON");
    assert!(
        !checked.status.success() && report["errors"].as_array().is_some_and(|e| !e.is_empty()),
        "{report}"
    );
}
