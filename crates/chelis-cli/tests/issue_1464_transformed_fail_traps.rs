//! chelis#1464: a taken `fail(...)` branch inside a transformed function must
//! abort with its authored message, in the evaluator and in the generated C
//! binary alike.
//!
//! Before [05-OP-68], `lower_if` selected between branch VALUES and the
//! `fail` branch was lowered to a zero placeholder, so a TAKEN abort returned
//! 0.0 with exit 0 in both lanes. The `#616` comment at the substitution site
//! justified that with "the placeholder's VALUE never matters on the taken
//! path", which is exactly backwards and had no test.
//!
//! Controlling rules: `spec/06-transformations.md` §2.10.1 (a scalar `if`
//! differentiates the branch the forward program executes) and §5.2 (a
//! potentially trapping node may be removed only when trap occurrence is
//! preserved), plus `spec/05-risc-primitives.md` [05-OP-68].
//!
//! Every taken case has an untaken twin. The untaken direction is the whole
//! reason this is a guarded abort rather than a rejection: a guarded `fail`
//! that does not fire must keep computing, and must keep differentiating, at
//! exactly the value it had before.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command as StdCommand, Output};

use assert_cmd::Command;
use tempfile::{TempDir, tempdir};

/// A shape guard: `fail` when the tensor is shorter than two elements. The
/// guard is the natural way to write this, and the extent is what decides
/// whether it fires.
fn shape_guarded(extent: usize, values: &str) -> String {
    format!(
        "module Repro.ShapeGuard\n\
         def loss(x: tensor[{extent}, f32]) -> tensor[f32] = \
         if gt(cast(2, i64), cast(shape(x, cast(0, i32)), i64)) \
         then fail(\"kernel exceeds input length\") \
         else sum(x, cast(0, i32))\n\
         out = grad(loss)(to_tensor([{values}]))\n"
    )
}

/// A data-dependent guard, so the condition cannot be folded at compile time
/// and the abort has to survive as a real runtime branch.
fn value_guarded(values: &str) -> String {
    format!(
        "module Repro.ValueGuard\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] = \
         if gt(tensor_to_scalar(sum(x, cast(0, i32))), cast(0.5, f32)) \
         then fail(\"value guard tripped\") \
         else sum(x, cast(0, i32))\n\
         out = grad(loss)(to_tensor([{values}]))\n"
    )
}

fn eval(source: &str, stem: &str) -> Output {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    fs::write(&path, source).expect("write source");
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().expect("UTF-8 source path")])
        .output()
        .expect("run chelis eval")
}

fn build_c(source: &str, stem: &str) -> (TempDir, PathBuf) {
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
    (dir, out_dir)
}

fn compile_and_run(build_dir: &Path, stem: &str) -> Output {
    let generated = build_dir.join(format!("{stem}.c"));
    let binary = build_dir.join("program");
    let compile = StdCommand::new("gcc")
        .args(["-O0", "-std=c11", "-I"])
        .arg(build_dir)
        .arg(&generated)
        .args(["-o"])
        .arg(&binary)
        .arg(build_dir.join("libchelis_runtime.a"))
        .args(["-lm", "-lpthread", "-ldl"])
        .output()
        .expect("invoke gcc");
    assert!(
        compile.status.success(),
        "gcc failed: {}",
        String::from_utf8_lossy(&compile.stderr)
    );
    StdCommand::new(binary).output().expect("run generated C")
}

/// Both lanes must abort nonzero AND carry the authored message. Checking the
/// exit status alone would pass for a crash, and checking the message alone
/// would pass for a program that printed it and returned 0.
fn assert_taken_in_both_lanes(source: &str, stem: &str, message: &str) {
    let evaluated = eval(source, stem);
    let eval_stderr = String::from_utf8_lossy(&evaluated.stderr).into_owned();
    assert!(
        !evaluated.status.success(),
        "eval must abort on the taken fail, not return a value: stdout={} stderr={eval_stderr}",
        String::from_utf8_lossy(&evaluated.stdout)
    );
    assert!(
        eval_stderr.contains(message),
        "eval must retain the authored message `{message}`; got: {eval_stderr}"
    );

    let (_dir, build_dir) = build_c(source, stem);
    let ran = compile_and_run(&build_dir, stem);
    let c_stdout = String::from_utf8_lossy(&ran.stdout).into_owned();
    let c_stderr = String::from_utf8_lossy(&ran.stderr).into_owned();
    assert!(
        !ran.status.success(),
        "the compiled binary must exit nonzero, not print a value: stdout={c_stdout} \
         stderr={c_stderr}"
    );
    assert!(
        c_stderr.contains(message),
        "the compiled abort must retain the authored message `{message}`; got: {c_stderr}"
    );
}

/// The untaken twin: both lanes compute, agree, and produce `expected`.
fn assert_untaken_in_both_lanes(source: &str, stem: &str, expected: &str) {
    let evaluated = eval(source, stem);
    let eval_stdout = String::from_utf8_lossy(&evaluated.stdout).into_owned();
    assert!(
        evaluated.status.success(),
        "an untaken guard must not abort: stderr={}",
        String::from_utf8_lossy(&evaluated.stderr)
    );
    assert!(
        eval_stdout.contains(expected),
        "eval must still compute `{expected}`; got: {eval_stdout}"
    );

    let (_dir, build_dir) = build_c(source, stem);
    let ran = compile_and_run(&build_dir, stem);
    let c_stdout = String::from_utf8_lossy(&ran.stdout).into_owned();
    assert!(
        ran.status.success(),
        "the compiled binary must not abort on an untaken guard: stderr={}",
        String::from_utf8_lossy(&ran.stderr)
    );
    assert!(
        c_stdout.contains(expected),
        "the compiled lane must agree with eval on `{expected}`; got: {c_stdout}"
    );
}

#[test]
fn taken_shape_guard_aborts_in_both_lanes() {
    // One element, so `2 > 1` holds and the guard fires. This is the issue's
    // own reproducer: it returned `tensor(shape=[1], data=[0.0])` with exit 0.
    assert_taken_in_both_lanes(
        &shape_guarded(1, "cast(1.0, f32)"),
        "taken_shape_guard",
        "kernel exceeds input length",
    );
}

#[test]
fn untaken_shape_guard_still_differentiates_in_both_lanes() {
    // Four elements, so the guard does not fire and `grad(sum)` is 1
    // everywhere. This is chelis#856's case and must stay green.
    assert_untaken_in_both_lanes(
        &shape_guarded(
            4,
            "cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32)",
        ),
        "untaken_shape_guard",
        "data=[1.0, 1.0, 1.0, 1.0]",
    );
}

#[test]
fn taken_value_guard_aborts_in_both_lanes() {
    // A runtime condition: the guard is decided by the data, not the shape.
    assert_taken_in_both_lanes(
        &value_guarded("cast(1.0, f32)"),
        "taken_value_guard",
        "value guard tripped",
    );
}

#[test]
fn untaken_value_guard_still_differentiates_in_both_lanes() {
    // The SAME program as the taken case, below the threshold. That the two
    // differ only by input is the reason a compile-time rejection would have
    // been the wrong shape of fix: it could not have told them apart.
    assert_untaken_in_both_lanes(
        &value_guarded("cast(0.25, f32)"),
        "untaken_value_guard",
        "data=[1.0]",
    );
}

#[test]
fn a_fail_outside_any_transform_still_aborts() {
    // The guard must not have captured the ordinary case: entry-level
    // `if`/`fail` belongs to the host lane and still traps there. The first
    // repair attempt for this issue broke exactly this.
    let source = "module Repro.OutsideTransform\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] = \
         if gt(cast(2, i64), cast(shape(x, cast(0, i32)), i64)) \
         then fail(\"outside boom\") \
         else sum(x, cast(0, i32))\n\
         out = loss(to_tensor([cast(1.0, f32)]))\n";
    let evaluated = eval(source, "outside_transform");
    let stderr = String::from_utf8_lossy(&evaluated.stderr).into_owned();
    assert!(
        !evaluated.status.success(),
        "a forward fail must still abort: stdout={}",
        String::from_utf8_lossy(&evaluated.stdout)
    );
    assert!(
        stderr.contains("outside boom"),
        "the forward abort must retain its message; got: {stderr}"
    );
}

/// A guard nested inside another runtime `if` branch. The inner guard's
/// condition is true, but the OUTER condition selects the sibling, so the
/// forward program never executes the inner branch and the guard must not
/// fire (spec/06 2.10.1: untaken branches are not evaluated).
fn nested_guard(outer: &str) -> String {
    format!(
        "module Repro.NestedGuard\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] = \
         if {outer} \
         then (if gt(tensor_to_scalar(sum(x, cast(0, i32))), cast(0.5, f32)) \
         then fail(\"SHOULD NOT FIRE\") else sum(x, cast(0, i32))) \
         else sum(mul(x, x), cast(0, i32))\n\
         out = grad(loss)(to_tensor([cast(1.0, f32)]))\n"
    )
}

#[test]
fn a_guard_under_an_untaken_outer_branch_does_not_fire() {
    // grad of x^2 at 1.0 is 2.0; the inner guard is on a path the forward
    // program does not take.
    assert_untaken_in_both_lanes(
        &nested_guard("gt(tensor_to_scalar(sum(x, cast(0, i32))), cast(100.0, f32))"),
        "nested_guard_runtime",
        "data=[2.0]",
    );
}

#[test]
fn a_nested_guard_agrees_whether_or_not_the_outer_condition_folds() {
    // The same program with a compile-time-resolvable outer condition. These
    // two must agree: the first version of this fix returned 2.0 when the
    // outer condition folded (chelis#620 pruning removed the guard) and
    // aborted when it did not, which is how the defect was caught.
    assert_untaken_in_both_lanes(
        &nested_guard("gt(cast(1, i64), cast(2, i64))"),
        "nested_guard_static",
        "data=[2.0]",
    );
}

#[test]
fn guards_fire_in_source_order_not_dag_order() {
    // Both conditions hold. The outer guard decides the branch, so it wins.
    // Lowering the inner guard as an INPUT to the outer one inverted this:
    // the inner was evaluated first and pre-empted the outer's message.
    let source = "module Repro.GuardPrecedence\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] = \
         if gt(tensor_to_scalar(sum(x, cast(0, i32))), cast(0.5, f32)) \
         then fail(\"OUTER GUARD FIRES FIRST\") \
         else (if gt(tensor_to_scalar(sum(x, cast(0, i32))), cast(0.2, f32)) \
         then fail(\"INNER GUARD\") else sum(x, cast(0, i32)))\n\
         out = grad(loss)(to_tensor([cast(1.0, f32)]))\n";
    assert_taken_in_both_lanes(source, "guard_precedence", "OUTER GUARD FIRES FIRST");
}

/// chelis#1464 names `vmap` and `vmap(grad(...))` in its acceptance surface,
/// so both get a taken and an untaken case.
fn vmapped_guard(rows: &str) -> String {
    format!(
        "module Repro.VmappedGuard\n\
         def row(t: tensor[1, f32]) -> tensor[f32] = \
         if gt(tensor_to_scalar(sum(t, cast(0, i32))), cast(50.0, f32)) \
         then fail(\"row too big\") else sum(mul(t, t), cast(0, i32))\n\
         def batched(b: tensor[2, 1, f32]) -> tensor[2, f32] = vmap(row)(b)\n\
         out = batched(to_tensor([{rows}]))\n"
    )
}

#[test]
fn a_vmapped_guard_aborts_when_any_row_fires() {
    // [05-OP-68]: a batched condition aborts when ANY mapped element is true.
    assert_taken_in_both_lanes(
        &vmapped_guard("[cast(1.0, f32)], [cast(99.0, f32)]"),
        "vmap_guard_taken",
        "row too big",
    );
}

#[test]
fn a_vmapped_guard_computes_when_no_row_fires() {
    assert_untaken_in_both_lanes(
        &vmapped_guard("[cast(1.0, f32)], [cast(2.0, f32)]"),
        "vmap_guard_untaken",
        "data=[1.0, 4.0]",
    );
}

#[test]
fn vmap_of_grad_differentiates_through_an_untaken_guard() {
    // grad of x^3 is 3x^2, so rows [1.0] and [2.0] give 3.0 and 12.0.
    let source = "module Repro.VmapGrad\n\
         def cube(t: tensor[1, f32]) -> tensor[f32] = \
         if gt(tensor_to_scalar(sum(t, cast(0, i32))), cast(50.0, f32)) \
         then fail(\"cube guard\") \
         else sum(mul(mul(t, t), t), cast(0, i32))\n\
         def batched(b: tensor[2, 1, f32]) -> tensor[2, 1, f32] = vmap(grad(cube))(b)\n\
         out = batched(to_tensor([[cast(1.0, f32)], [cast(2.0, f32)]]))\n";
    assert_untaken_in_both_lanes(source, "vmap_grad_untaken", "data=[3.0, 12.0]");
}

#[test]
fn an_indirect_fail_in_the_surviving_branch_is_also_rejected() {
    // Regression for a gap in this issue's own first fix. When one branch is
    // a direct `fail(...)`, the OTHER branch becomes the guard's fallback —
    // and it is still an `if` branch. Lowering it without branch depth let an
    // indirect `fail` there reach the placeholder arm, so
    // `if c then fail("a") else boom(x)` returned `data=[0.0]` with exit 0:
    // the very defect this issue fixes, surviving in the sibling of the
    // branch being fixed.
    let source = "module Repro.FallbackGap\n\
         def boom(x: tensor[1, f32]) -> tensor[f32] = fail(\"indirect in fallback\")\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] = \
         if gt(tensor_to_scalar(sum(x, cast(0, i32))), cast(9.0, f32)) \
         then fail(\"direct arm\") \
         else boom(x)\n\
         out = grad(loss)(to_tensor([cast(1.0, f32)]))\n";
    let evaluated = eval(source, "fallback_gap");
    let stdout = String::from_utf8_lossy(&evaluated.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&evaluated.stderr).into_owned();
    assert!(
        !evaluated.status.success(),
        "an indirect fail in the fallback must not silently produce a value: stdout={stdout}"
    );
    assert!(
        !stdout.contains("data=[0.0]"),
        "the placeholder zero must not reach the output: {stdout}"
    );
    assert!(
        stderr.contains("1464"),
        "the rejection must cite its owning issue; got: {stderr}"
    );
}

#[test]
fn an_indirect_branch_fail_is_rejected_not_substituted() {
    // `lower_if` recognizes a DIRECT `fail(...)` branch and guards it. A
    // `fail` reached through a helper has no guarded form, and the defect
    // this issue fixes was precisely a silent placeholder in that position.
    // It must be a loud rejection rather than a quiet zero.
    let source = "module Repro.IndirectFail\n\
         def boom(x: tensor[1, f32]) -> tensor[f32] = fail(\"indirect boom\")\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] = \
         if gt(tensor_to_scalar(sum(x, cast(0, i32))), cast(0.5, f32)) \
         then boom(x) \
         else sum(x, cast(0, i32))\n\
         out = grad(loss)(to_tensor([cast(1.0, f32)]))\n";
    let evaluated = eval(source, "indirect_fail");
    let stdout = String::from_utf8_lossy(&evaluated.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&evaluated.stderr).into_owned();
    assert!(
        !evaluated.status.success(),
        "an unrepresentable indirect fail must not silently produce a value: stdout={stdout}"
    );
    assert!(
        !stdout.contains("data=[0.0]"),
        "the placeholder zero must not reach the output: {stdout}"
    );
    assert!(
        stderr.contains("1464"),
        "the rejection must cite its owning issue; got: {stderr}"
    );
}
