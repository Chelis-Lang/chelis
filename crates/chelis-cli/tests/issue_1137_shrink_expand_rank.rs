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
         e = insert(x, cast(0, int32), {expand_size})\n\
         s = shrink(e, [[cast(0, int64), cast(1, int64)], [cast(0, int64), cast(2, int64)]])\n\
         out = {tail}\n"
    )
}

fn runtime_bound_source(start: &str, end: &str) -> String {
    format!(
        "module Repro.ShrinkExpandRuntimeBound\n\
         x = to_tensor([cast(11.0, f32), cast(22.0, f32)])\n\
         e = insert(x, cast(0, int32), cast(2, int64))\n\
         k = shape(x, cast(0, int32))\n\
         s = shrink(e, [[cast(0, int64), cast(1, int64)], [{start}, {end}]])\n\
         out = relu(s)\n"
    )
}

fn runtime_bound_movement_consumer_source(consumer: &str) -> String {
    format!(
        "module Repro.ShrinkExpandMovementConsumer\n\
         x = to_tensor([cast(11.0, f32), cast(22.0, f32)])\n\
         e = insert(x, cast(0, int32), cast(2, int64))\n\
         k = shape(x, cast(0, int32))\n\
         s = shrink(e, [[cast(0, int64), cast(1, int64)], [cast(k - k, int64), k]])\n\
         out = {consumer}\n"
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

/// These two programs compiled before chelis#1277's expand/insert split and
/// refuse now. That is a capability regression, and it is recorded as one
/// rather than adjusted away.
///
/// The program is unchanged in meaning: its consumer is a rank-2 `shrink`, so
/// the operand is rank 2 under either spelling. What changed is the route. The
/// deferred `expand` route reached C emission with a const node whose second
/// axis carried a source; the fixed `insert` route reaches it with an
/// anonymous dimension that `declared_shape_sources` yields no source for, and
/// the emitter refuses loudly rather than guessing.
///
/// That refusal is chelis#1482, an already tracked gap with its own row in the
/// runtime-extent oracle (`shrink.elementwise_const.build`, recorded at
/// `typed_unsupported(#1482)`). The split does not introduce it; it makes it
/// reachable from source that previously routed around it, so whoever flipped
/// `expand`'s meaning would have met it. These tests assert the receipt so the
/// day it is fixed they fail and come back as parity tests.
fn assert_typed_refusal(source: &str, stem: &str) {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join(format!("{stem}.ch"));
    fs::write(&path, source).expect("write source");
    let out_dir = dir.path().join("out");
    fs::create_dir_all(&out_dir).expect("create output dir");
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
        !output.status.success(),
        "chelis#1482 still refuses this shape; a success here means it was          fixed, so restore the C-parity assertion: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unimplemented chelis#1482")
            && stderr.contains("output-axis extent sources"),
        "the refusal must be chelis#1482's typed receipt, not another failure: \
         {stderr}"
    );
}

#[test]
fn consumed_shrink_with_runtime_end_refuses_c_emission_on_chelis_1482() {
    let source = runtime_bound_source("cast(0, int64)", "k");
    assert_typed_refusal(&source, "shrink_expand_runtime_end");
}

#[test]
fn consumed_shrink_with_runtime_start_refuses_c_emission_on_chelis_1482() {
    let source = runtime_bound_source("cast(k - k, int64)", "cast(2, int64)");
    assert_typed_refusal(&source, "shrink_expand_runtime_start");
}

#[test]
fn zero_pad_after_runtime_shrink_keeps_the_repaired_rank() {
    let source = runtime_bound_movement_consumer_source(
        "pad(s, [[cast(0, int64), cast(0, int64)], [cast(0, int64), cast(0, int64)]], cast(0.0, f32))",
    );
    let eval = eval_stdout(&source, "shrink_expand_zero_pad");
    let (_, compiled) = build_and_run(&source, "shrink_expand_zero_pad");
    assert_eq!(
        compiled, eval,
        "zero pad must preserve the actual rank repaired at Shrink"
    );
}

#[test]
fn identity_stride_after_runtime_shrink_keeps_the_repaired_rank() {
    let source =
        runtime_bound_movement_consumer_source("stride(s, cast(1, int64), cast(1, int64))");
    let eval = eval_stdout(&source, "shrink_expand_identity_stride");
    let (_, compiled) = build_and_run(&source, "shrink_expand_identity_stride");
    assert_eq!(
        compiled, eval,
        "identity stride must preserve the actual rank repaired at Shrink"
    );
}

#[test]
fn user_dimension_name_cannot_capture_runtime_shrink_extent() {
    // The generated candidate for this helper is `_rt_shrink_dim_8_0`.
    // A source dimension may legally spell that exact string, so the
    // runtime-extent allocator must detect the occupied identity and mint a
    // different one. Otherwise C incorrectly guards the three-element slice
    // against the four-element input dimension.
    let source = "module Repro.ShrinkGeneratedDimCollision\n\
sig crop: tensor[_rt_shrink_dim_8_0, f32] -> tensor[u, f32]\n\
def crop(x) = {\n\
  k = shape(x, cast(0, int32))\n\
  z = cast(k - k, int64)\n\
  stop = cast(k - cast(1, int64), int64)\n\
  shrink(x, [[z, stop]])\n\
}\n\
out = crop(to_tensor([\n\
  cast(1.0, f32),\n\
  cast(2.0, f32),\n\
  cast(3.0, f32),\n\
  cast(4.0, f32)\n\
]))\n";
    let eval = eval_stdout(source, "shrink_generated_dim_collision");
    let (_, compiled) = build_and_run(source, "shrink_generated_dim_collision");
    assert_eq!(
        compiled, eval,
        "a source-authored dimension must never capture a generated runtime extent"
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
