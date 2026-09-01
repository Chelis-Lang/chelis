//! chelis#1137: `shrink(expand(...))` must retain its result rank when a
//! downstream tensor operation consumes the shrink result.

use std::fs;
use std::process::Command as StdCommand;

use assert_cmd::Command;
use tempfile::tempdir;

fn source(expand_size: &str, consume: bool) -> String {
    let tail = if consume { "relu(s)" } else { "s" };
    format!(
        "module Repro.ShrinkExpandRank\n\
         x = to_tensor([cast(11.0, f32), cast(22.0, f32)])\n\
         e = expand(x, cast(0, int32), {expand_size})\n\
         s = shrink(e, [[cast(0, int64), cast(1, int64)], [cast(0, int64), cast(2, int64)]])\n\
         out = {tail}\n"
    )
}

fn eval_stdout(source: &str, stem: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    fs::write(&path, source).expect("write source");
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().expect("UTF-8 source path")])
        .output()
        .expect("run eval");
    assert!(
        output.status.success(),
        "eval failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("UTF-8 eval stdout")
}

fn build_and_run(source: &str, stem: &str) -> (String, String) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    let out_dir = dir.path().join("build");
    fs::write(&path, source).expect("write source");
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().expect("UTF-8 source path"),
            "--target",
            "c",
            "--output",
            out_dir.to_str().expect("UTF-8 output path"),
        ])
        .output()
        .expect("build C");
    assert!(
        output.status.success(),
        "build failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let generated_path = out_dir.join(format!("{stem}.c"));
    let generated = fs::read_to_string(&generated_path).expect("read generated C");
    let binary = out_dir.join("program");
    let compile = StdCommand::new("gcc")
        .args(["-O0", "-std=c11", "-I"])
        .arg(&out_dir)
        .arg(&generated_path)
        .args(["-o"])
        .arg(&binary)
        .arg(out_dir.join("libchelis_runtime.a"))
        .args(["-lm", "-lpthread", "-ldl"])
        .output()
        .expect("invoke gcc");
    assert!(
        compile.status.success(),
        "gcc failed: {}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let run = StdCommand::new(&binary).output().expect("run generated C");
    assert!(
        run.status.success(),
        "generated C failed: stdout={} stderr={} generated C:\n{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr),
        generated
    );
    (
        generated,
        String::from_utf8(run.stdout).expect("UTF-8 generated stdout"),
    )
}

fn assert_consumed_case(expand_size: &str, stem: &str) {
    let source = source(expand_size, true);
    let eval = eval_stdout(&source, stem);
    let (generated, compiled) = build_and_run(&source, stem);
    assert_eq!(
        compiled, eval,
        "eval and generated C must agree byte-for-byte"
    );
    assert!(
        generated.contains("chelis_alloc(2, (int64_t[]){ 1, 2 }, CHELIS_DTYPE_F32)"),
        "consumed shrink must allocate its rank-2 result, generated C:\n{generated}"
    );
    assert!(
        !generated.contains("chelis_alloc(0, NULL, CHELIS_DTYPE_F32)"),
        "no rank-0 allocation may stand in for this rank-2 tensor, generated C:\n{generated}"
    );
}

#[test]
fn consumed_shrink_over_literal_expand_keeps_rank_two() {
    assert_consumed_case("cast(2, int64)", "shrink_expand_literal");
}

#[test]
fn consumed_shrink_over_shape_expand_keeps_rank_two() {
    assert_consumed_case("shape(x, cast(0, int32))", "shrink_expand_shape");
}

#[test]
fn consumed_shrink_over_cast_shape_expand_keeps_rank_two() {
    assert_consumed_case(
        "cast(shape(x, cast(0, int32)), int64)",
        "shrink_expand_cast_shape",
    );
}

#[test]
fn directly_returned_shrink_over_expand_remains_a_positive_control() {
    let source = source("cast(2, int64)", false);
    let eval = eval_stdout(&source, "shrink_expand_direct");
    let (_, compiled) = build_and_run(&source, "shrink_expand_direct");
    assert_eq!(compiled, eval, "direct-return control must stay green");
}

#[test]
fn consumed_shrink_without_expand_remains_a_positive_control() {
    let source = "module Repro.ShrinkDirectInput\n\
sig f: tensor[2, 2, f32] -> tensor[1, 2, f32]\n\
def f(x) = {\n\
  s = shrink(x, [[cast(0, int64), cast(1, int64)], [cast(0, int64), cast(2, int64)]])\n\
  relu(s)\n\
}\n\
out = f(to_tensor([[cast(11.0, f32), cast(22.0, f32)], [cast(33.0, f32), cast(44.0, f32)]]))\n";
    let eval = eval_stdout(source, "shrink_direct_input");
    let (_, compiled) = build_and_run(source, "shrink_direct_input");
    assert_eq!(compiled, eval, "non-expand shrink control must stay green");
}

#[test]
fn invalid_shrink_bounds_still_fail_loudly() {
    let source = "out = shrink(to_tensor([cast(1.0, f32), cast(2.0, f32)]), [[cast(1, int64), cast(1, int64)]])\n";
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("invalid_bounds.ch");
    fs::write(&path, source).expect("write source");
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().expect("UTF-8 source path")])
        .output()
        .expect("run eval");
    assert!(!output.status.success(), "empty shrink must not execute");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("empty or inverted"),
        "rejection must name the invalid bound: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
