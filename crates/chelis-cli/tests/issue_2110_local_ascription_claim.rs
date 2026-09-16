//! chelis#2110: Eval and generated C enforce authored local tensor ascriptions
//! at the initializer operation with [04-NUM-9]'s typed Domain trap.

mod common;

use assert_cmd::Command;
use common::{gcc_available, link_generated, write_file};
use std::process::Command as StdCommand;
use tempfile::TempDir;

#[derive(Debug)]
struct LaneResult {
    success: bool,
    text: String,
}

fn combined(output: &std::process::Output) -> String {
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    text
}

fn eval_result(dir: &TempDir, stem: &str, source: &str) -> LaneResult {
    let path = dir.path().join(format!("{stem}.ch"));
    write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "eval",
            "--allow-style-violations",
            "--file",
            path.to_str().expect("UTF-8 path"),
        ])
        .output()
        .expect("run eval");
    LaneResult {
        success: output.status.success(),
        text: combined(&output),
    }
}

fn c_result(dir: &TempDir, stem: &str, source: &str) -> LaneResult {
    assert!(
        gcc_available(),
        "the #2110 acceptance target requires an executed C lane"
    );
    let path = dir.path().join(format!("{stem}.ch"));
    let out_dir = dir.path().join(format!("{stem}-out"));
    write_file(&path, source);
    let built = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--allow-style-violations",
            path.to_str().expect("UTF-8 path"),
            "--target",
            "c",
            "-o",
            out_dir.to_str().expect("UTF-8 path"),
        ])
        .output()
        .expect("build generated C");
    assert!(
        built.status.success(),
        "{stem}: build must reach generated C: {}",
        combined(&built)
    );
    let linked = link_generated(&out_dir, &format!("{stem}.c"), stem);
    assert!(
        linked.success(),
        "{stem}: generated C must compile and link"
    );
    let output = StdCommand::new(out_dir.join(stem))
        .output()
        .expect("run generated binary");
    LaneResult {
        success: output.status.success(),
        text: combined(&output),
    }
}

fn both_lanes(dir: &TempDir, stem: &str, source: &str) -> [(&'static str, LaneResult); 2] {
    [
        ("eval", eval_result(dir, &format!("{stem}_eval"), source)),
        ("c", c_result(dir, &format!("{stem}_c"), source)),
    ]
}

fn assert_initializer_trap(dir: &TempDir, stem: &str, source: &str, claim: &str, operation: &str) {
    let context = format!("extent `{claim}`: claimed = 2, {operation} axis 0 = 3");
    let trap = format!("numeric trap: domain in {operation} at int64");
    for (lane, result) in both_lanes(dir, stem, source) {
        assert!(
            !result.success,
            "{lane}: local ascription must trap: {}",
            result.text
        );
        assert!(
            result.text.contains(&context),
            "{lane}: initializer owns `{context}`: {}",
            result.text
        );
        assert!(
            result.text.lines().any(|line| line == trap),
            "{lane}: exact typed trap line is missing: {}",
            result.text
        );
        assert!(
            !result.text.contains("assert") && !result.text.contains("Aborted"),
            "{lane}: this is a typed Domain trap, not a helper assertion/abort: {}",
            result.text
        );
    }
}

fn assert_pad_trap(dir: &TempDir, stem: &str, source: &str, claim: &str) {
    assert_initializer_trap(dir, stem, source, claim, "pad");
}

fn direct(body: &str) -> String {
    format!(
        "def f(x: tensor[*, f32]) -> tensor[*, f32] = {{\n  {body}\n}}\n\
         out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n"
    )
}

#[test]
fn direct_runtime_disagreement_traps_at_the_initializer_on_both_lanes() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_pad_trap(
        &dir,
        "direct",
        &direct("y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  y"),
        "2",
    );
}

#[test]
fn an_initializer_alias_keeps_the_local_claim_on_the_producing_operation() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_pad_trap(
        &dir,
        "initializer_alias",
        &direct(
            "raw = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
             y: tensor[2, f32] = raw\n  \
             y",
        ),
        "2",
    );
}

#[test]
fn a_later_return_alias_keeps_the_local_claim_on_the_initializer() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_pad_trap(
        &dir,
        "return_alias",
        &direct(
            "y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
             returned = y\n  \
             returned",
        ),
        "2",
    );
}

#[test]
fn a_value_dead_except_for_the_obligation_still_traps() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_pad_trap(
        &dir,
        "dead_value",
        &direct(
            "y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
             x",
        ),
        "2",
    );
}

#[test]
fn an_inlined_callee_retains_its_local_ascription_obligation() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def helper(x: tensor[*, f32]) -> tensor[*, f32] = {\n  \
                  y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
                  y\n\
                  }\n\
                  out = helper(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n";
    assert_pad_trap(&dir, "inlined", source, "2");
}

#[test]
fn a_named_local_claim_uses_its_declaring_runtime_extent() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(anchor: tensor[n, f32], x: tensor[*, f32]) -> tensor[*, f32] = {\n  \
                  y: tensor[n, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
                  y\n\
                  }\n\
                  out = f(\n  \
                  to_tensor([10.0f32, 20.0f32]),\n  \
                  to_tensor([1.0f32, 2.0f32, 3.0f32]),\n\
                  )\n";
    assert_pad_trap(&dir, "named", source, "n");
}

#[test]
fn a_staged_host_partition_traps_at_its_reshape_initializer_on_both_lanes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(source: tensor[*, f32], x: tensor[*, f32]) -> tensor[*, f32] = {\n  \
                  y: tensor[2, f32] = reshape(x, [numel(source)])\n  \
                  y\n\
                  }\n\
                  out = f(\n  \
                  to_tensor([10.0f32, 20.0f32, 30.0f32]),\n  \
                  to_tensor([1.0f32, 2.0f32, 3.0f32]),\n\
                  )\n";
    assert_initializer_trap(&dir, "staged_host", source, "2", "reshape");
}

#[test]
fn a_later_same_shape_consumer_does_not_take_the_initializer_claim() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_pad_trap(
        &dir,
        "same_shape_consumer",
        &direct(
            "y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
             add(y, y)",
        ),
        "2",
    );
}

#[test]
fn multiple_local_axis_claims_fail_in_authored_axis_order() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(x: tensor[*, *, f32]) -> tensor[*, *, f32] = {\n  \
                  y: tensor[2, 5, f32] = pad(\n  \
                  x,\n  \
                  [[0i64, 0i64], [0i64, 0i64]],\n  \
                  0.0f32,\n  \
                  )\n  \
                  y\n\
                  }\n\
                  out = f(to_tensor([\n  \
                  [1.0f32, 2.0f32, 3.0f32, 4.0f32],\n  \
                  [5.0f32, 6.0f32, 7.0f32, 8.0f32],\n  \
                  [9.0f32, 10.0f32, 11.0f32, 12.0f32],\n\
                  ]))\n";
    assert_pad_trap(&dir, "axis_order", source, "2");
}

#[test]
fn an_inferred_result_claim_does_not_replace_the_local_claim() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = "def f(x: tensor[*, f32]) = {\n  \
                  y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
                  y\n\
                  }\n\
                  out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n";
    assert_pad_trap(&dir, "inferred_result", source, "2");
}

#[test]
fn a_static_local_disagreement_is_a_checker_error_on_every_entry_lane() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("static_disagreement.ch");
    write_file(
        &path,
        "out = {\n  \
         y: tensor[9, f32] = pad(\n  \
         to_tensor([1.0f32]),\n  \
         [[1i64, 0i64]],\n  \
         0.0f32,\n  \
         )\n  \
         y\n\
         }\n",
    );
    let out_dir = dir.path().join("static-out");
    for (lane, args) in [
        (
            "check",
            vec!["check".to_string(), path.to_str().unwrap().to_string()],
        ),
        (
            "eval",
            vec![
                "eval".to_string(),
                "--allow-style-violations".to_string(),
                "--file".to_string(),
                path.to_str().unwrap().to_string(),
            ],
        ),
        (
            "build",
            vec![
                "build".to_string(),
                "--allow-style-violations".to_string(),
                path.to_str().unwrap().to_string(),
                "--target".to_string(),
                "c".to_string(),
                "-o".to_string(),
                out_dir.to_str().unwrap().to_string(),
            ],
        ),
    ] {
        let output = Command::cargo_bin("chelis")
            .expect("chelis")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(args)
            .output()
            .expect("run lane");
        let text = combined(&output);
        assert!(!output.status.success(), "{lane}: {text}");
        assert!(text.contains("DimensionMismatch"), "{lane}: {text}");
        assert!(!text.contains("numeric trap:"), "{lane}: {text}");
    }
}

fn assert_exact_on_both_lanes(stem: &str, source: &str) {
    let dir = tempfile::tempdir().expect("tempdir");
    let [(_, eval), (_, compiled)] = both_lanes(&dir, stem, source);
    assert!(eval.success, "eval: {}", eval.text);
    assert!(compiled.success, "c: {}", compiled.text);
    assert_eq!(eval.text, compiled.text);
    assert_eq!(eval.text, "out = tensor(shape=[3], data=[1.0, 2.0, 3.0])\n");
}

#[test]
fn an_agreeing_local_ascription_executes_exactly_on_both_lanes() {
    assert_exact_on_both_lanes(
        "agree",
        &direct("y: tensor[3, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  y"),
    );
}

#[test]
fn inferred_metadata_does_not_create_a_local_runtime_claim() {
    assert_exact_on_both_lanes(
        "inferred",
        &direct("y = pad(x, [[0i64, 0i64]], 0.0f32)\n  y"),
    );
}

#[test]
fn a_wildcard_local_ascription_creates_no_extent_obligation() {
    assert_exact_on_both_lanes(
        "wildcard_ascription",
        &direct("y: tensor[*, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  y"),
    );
}

#[test]
fn an_agreeing_literal_result_and_local_claim_execute_exactly() {
    assert_exact_on_both_lanes(
        "agreeing_literal_result",
        "def f(x: tensor[*, f32]) -> tensor[3, f32] = {\n  \
         y: tensor[3, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
         y\n\
         }\n\
         out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n",
    );
}
