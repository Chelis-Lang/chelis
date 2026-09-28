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

/// The taken twin of `vmap_of_grad_differentiates_through_an_untaken_guard`.
/// The file header promises every taken case has an untaken twin; this pair
/// was the one exception, and chelis#1464 names `vmap(grad(...))` in its
/// acceptance surface.
#[test]
fn vmap_of_grad_aborts_when_the_guard_fires() {
    let source = "module Repro.VmapGradTaken\n\
         def cube(t: tensor[1, f32]) -> tensor[f32] = \
         if gt(tensor_to_scalar(sum(t, cast(0, i32))), cast(1.5, f32)) \
         then fail(\"cube guard fired\") \
         else sum(mul(mul(t, t), t), cast(0, i32))\n\
         def batched(b: tensor[2, 1, f32]) -> tensor[2, 1, f32] = vmap(grad(cube))(b)\n\
         out = batched(to_tensor([[cast(1.0, f32)], [cast(2.0, f32)]]))\n";
    assert_taken_in_both_lanes(source, "vmap_grad_taken", "cube guard fired");
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

/// A guard whose condition is compile-time-resolvable and selects the
/// `fail`. chelis#620 prunes such an `if` to the selected branch alone,
/// which bypassed the guarded-abort lowering entirely: the `fail` reached
/// the placeholder arm and `grad` returned `data=[0.0]` with exit 0 — the
/// headline defect of this issue, on the one path no test covered.
fn statically_taken(guard: &str, body: &str) -> String {
    format!(
        "module Repro.StaticTaken\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] = {guard}\n\
         out = {body}\n"
    )
}

fn assert_rejects_without_a_placeholder(source: &str, stem: &str) {
    let evaluated = eval(source, stem);
    let stdout = String::from_utf8_lossy(&evaluated.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&evaluated.stderr).into_owned();
    assert!(
        !evaluated.status.success(),
        "an unconditional abort must not produce a value: stdout={stdout}"
    );
    assert!(
        !stdout.contains("data=[0.0]"),
        "the placeholder zero must not reach the output: {stdout}"
    );
    assert!(
        stderr.contains("always selects its `fail"),
        "the diagnostic must name the static selection, not the indirect-fail case; \
         got: {stderr}"
    );
}

#[test]
fn a_statically_taken_fail_does_not_become_a_placeholder_under_grad() {
    assert_rejects_without_a_placeholder(
        &statically_taken(
            "if gt(cast(2, i64), cast(1, i64)) then fail(\"STATIC TAKEN BOOM\") \
             else sum(mul(x, x), cast(0, i32))",
            "grad(loss)(to_tensor([cast(3.0, f32)]))",
        ),
        "static_taken_then",
    );
}

#[test]
fn a_statically_taken_else_fail_does_not_become_a_placeholder() {
    assert_rejects_without_a_placeholder(
        &statically_taken(
            "if gt(cast(1, i64), cast(2, i64)) then sum(mul(x, x), cast(0, i32)) \
             else fail(\"STATIC ELSE BOOM\")",
            "grad(loss)(to_tensor([cast(3.0, f32)]))",
        ),
        "static_taken_else",
    );
}

#[test]
fn a_statically_taken_fail_nested_in_a_runtime_branch_is_named_accurately() {
    // Also pins the diagnostic: this case used to report the indirect-`fail`
    // message, telling the user to write `fail(...)` directly as the branch
    // when that is exactly what they had written.
    assert_rejects_without_a_placeholder(
        &statically_taken(
            "if gt(tensor_to_scalar(sum(x, cast(0, i32))), cast(1.0, f32)) \
             then (if gt(cast(2, i64), cast(1, i64)) then fail(\"NESTED STATIC BOOM\") \
             else sum(x, cast(0, i32))) else sum(mul(x, x), cast(0, i32))",
            "grad(loss)(to_tensor([cast(3.0, f32)]))",
        ),
        "static_taken_nested",
    );
}

#[test]
fn a_statically_taken_fail_does_not_become_a_placeholder_under_vmap() {
    let source = "module Repro.StaticVmap\n\
         def row(t: tensor[1, f32]) -> tensor[f32] = \
         if gt(cast(2, i64), cast(1, i64)) then fail(\"STATIC VMAP BOOM\") \
         else sum(t, cast(0, i32))\n\
         def batched(b: tensor[2, 1, f32]) -> tensor[2, f32] = vmap(row)(b)\n\
         out = batched(to_tensor([[cast(1.0, f32)], [cast(2.0, f32)]]))\n";
    assert_rejects_without_a_placeholder(source, "static_taken_vmap");
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

/// chelis#2371: a `fail` with NO enclosing `if` inside a transform. The
/// depth counter guards `if` branches, so nothing guarded this route: the
/// placeholder became part of the answer rather than replacing it.
///
/// `grad` over `sum(add(x, fail("m")), 0i32)` returned `1.0` with exit 0 and
/// no `chelis_fail` anywhere in the generated C. It is unconditional here, so
/// it lowers to an [05-OP-68] guard whose condition is a constant true.
#[test]
fn a_fail_as_a_plain_operand_aborts_in_both_lanes() {
    let source = "module Repro.OperandFail\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] = \
         sum(add(x, fail(\"operand boom\")), cast(0, i32))\n\
         out = grad(loss)(to_tensor([cast(3.0, f32)]))\n";
    assert_taken_in_both_lanes(source, "operand_fail", "operand boom");
}

#[test]
fn a_fail_as_a_whole_transformed_body_aborts_in_both_lanes() {
    let source = "module Repro.BodyFail\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] = fail(\"body boom\")\n\
         out = grad(loss)(to_tensor([cast(3.0, f32)]))\n";
    assert_taken_in_both_lanes(source, "body_fail", "body boom");
}

#[test]
fn a_fail_reached_through_a_helper_with_no_if_aborts_in_both_lanes() {
    let source = "module Repro.HelperFail\n\
         def boom(x: tensor[1, f32]) -> tensor[f32] = fail(\"helper boom\")\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] = boom(x)\n\
         out = grad(loss)(to_tensor([cast(3.0, f32)]))\n";
    assert_taken_in_both_lanes(source, "helper_fail", "helper boom");
}

#[test]
fn a_fail_with_no_if_aborts_under_vmap_too() {
    let source = "module Repro.VmapBodyFail\n\
         def row(t: tensor[1, f32]) -> tensor[f32] = fail(\"vmap body boom\")\n\
         def batched(b: tensor[2, 1, f32]) -> tensor[2, f32] = vmap(row)(b)\n\
         out = batched(to_tensor([[cast(1.0, f32)], [cast(2.0, f32)]]))\n";
    assert_taken_in_both_lanes(source, "vmap_body_fail", "vmap body boom");
}

#[test]
fn a_fail_inside_a_where_arm_aborts_in_both_lanes() {
    // `where` is the tensor analogue of `if` and the natural spelling for an
    // elementwise guard. Arguments are call-by-value, so the helper traps
    // regardless of the condition -- this returned a value for BOTH inputs.
    let source = "module Repro.WhereFail\n\
         def boom(x: tensor[1, f32]) -> tensor[1, f32] = fail(\"where boom\")\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] = \
         sum(where(gt(&x, to_tensor([cast(0.5, f32)])), boom(copy(x)), mul(&x, &x)), \
         cast(0, i32))\n\
         out = grad(loss)(to_tensor([cast(0.9, f32)]))\n";
    assert_taken_in_both_lanes(source, "where_fail", "where boom");
}

/// chelis#2371 / chelis#2384 F1: `fail("")` IS a compile-time literal, so
/// it must not fall into the non-literal residue and become a placeholder.
/// [05-OP-68] makes an empty message a type error, and `lower_if` already
/// rejects the branch spelling by name — this pins the same rule and the
/// same diagnostic in the depth-0 position.
///
/// A mutation check found this branch had zero coverage: deleting the
/// emptiness handling left all tests green.
#[test]
fn an_empty_literal_message_is_rejected_not_placeholdered() {
    let source = "module Repro.EmptyOperand\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] = \
         sum(add(x, fail(\"\")), cast(0, i32))\n\
         out = grad(loss)(to_tensor([cast(3.0, f32)]))\n";
    let evaluated = eval(source, "empty_literal_operand");
    let stdout = String::from_utf8_lossy(&evaluated.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&evaluated.stderr).into_owned();
    assert!(
        !evaluated.status.success(),
        "an empty-message fail must not produce a value: stdout={stdout}"
    );
    assert!(
        !stdout.contains("data=[1.0]"),
        "the placeholder must not reach the output: {stdout}"
    );
    assert!(
        stderr.contains("has no message to report"),
        "the diagnostic must name the empty-message rule, not the non-literal \
         residue; got: {stderr}"
    );
}

#[test]
fn a_fail_inside_a_where_arm_aborts_on_the_false_direction_too() {
    // "Both directions" here means both INPUT values, not two lowering
    // paths: the arm is a call, evaluated call-by-value at depth 0, so both
    // inputs traverse the same path and no mutation separates this from its
    // twin (chelis#2384 review, FG). It is kept because the value-level
    // claim is the one a user reads.
    let source = "module Repro.WhereFailFalse\n\
         def boom(x: tensor[1, f32]) -> tensor[1, f32] = fail(\"where false boom\")\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] = \
         sum(where(gt(&x, to_tensor([cast(0.5, f32)])), boom(copy(x)), mul(&x, &x)), \
         cast(0, i32))\n\
         out = grad(loss)(to_tensor([cast(0.1, f32)]))\n";
    assert_taken_in_both_lanes(source, "where_fail_false", "where false boom");
}

/// chelis#2369: a `fail` in a statically-selected `match` arm. `lower_match`
/// never touches `if_branch_depth`, so the arm is lowered at depth 0 — which
/// is correct, because static arm selection lowers ONLY the selected arm, so
/// a `fail` reached there genuinely is unconditional.
///
/// This PR fixes it as a side effect of the depth-0 guard. chelis#2369's own
/// acceptance requires a negative control for a `fail` in an unselected arm,
/// and the claim had no test until this pair.
fn match_arm_fail(picked: &str, message: &str) -> String {
    format!(
        "module Repro.MatchArmFail\n\
         type Mode =\n\
           | ModeA\n\
           | ModeB\n\
         def pick() -> Mode = {picked}\n\
         def loss(x: tensor[2, f32]) -> f32 = match pick() with {{\n\
             | ModeA => fail(\"{message}\")\n\
             | ModeB => tensor_to_scalar(sum(mul(&x, &x), cast(0, i32)))\n\
           }}\n\
         out = grad(loss)(to_tensor([cast(1.0, f32), cast(2.0, f32)]))\n"
    )
}

#[test]
fn a_fail_in_a_statically_selected_match_arm_aborts_in_both_lanes() {
    assert_taken_in_both_lanes(
        &match_arm_fail("ModeA", "MATCH ARM BOOM"),
        "match_arm_selected",
        "MATCH ARM BOOM",
    );
}

#[test]
fn a_fail_in_an_unselected_match_arm_still_differentiates() {
    // grad of sum(x*x) is 2x. The `fail` arm is never lowered, so nothing
    // about it may reach the answer.
    assert_untaken_in_both_lanes(
        &match_arm_fail("ModeB", "UNSELECTED BOOM"),
        "match_arm_unselected",
        "data=[2.0, 4.0]",
    );
}

/// chelis#2368: a guarded abort whose result nothing consumes must still
/// fire. [05-OP-68] says it may not be removed, and an abort has no consumer
/// by design — so value reachability, which every pruner in the compiler
/// derives liveness from, is the wrong criterion for it.
///
/// Three independent liveness computations swept it before this: the DCE
/// pass, grad's own pruner, and the evaluator's root mask. A fix at any one
/// of them leaves the bug alive, which is why the predicate is shared.
#[test]
fn a_guard_whose_value_is_discarded_still_aborts() {
    let source = "module Repro.DiscardedGuard\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] = {\n\
           ignored = fail(\"discarded boom\")\n\
           sum(x, cast(0, i32))\n\
         }\n\
         out = grad(loss)(to_tensor([cast(3.0, f32)]))\n";
    let evaluated = eval(source, "discarded_guard");
    let stdout = String::from_utf8_lossy(&evaluated.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&evaluated.stderr).into_owned();
    assert!(
        !evaluated.status.success(),
        "a discarded guard must still abort: stdout={stdout}"
    );
    assert!(
        !stdout.contains("data=[1.0]"),
        "the surrounding value must not be returned: {stdout}"
    );
    assert!(
        stderr.contains("discarded boom"),
        "the authored message must survive; got: {stderr}"
    );
}

#[test]
fn a_guarded_if_whose_value_is_discarded_still_aborts() {
    // The issue's own reproducer: the guard is a whole `if` whose result is
    // bound and never used.
    //
    // Both lanes deliberately. The compiled lane was broken in the same way
    // and is fixed by the same change -- the generated C now emits the guard
    // loop and its `chelis_fail` -- so an eval-only assertion would leave a
    // real regression surface unwatched, against this file's own contract.
    let source = "module Repro.DiscardedGuardedIf\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] = {\n\
           unused = if gt(tensor_to_scalar(sum(&x, cast(0, i32))), cast(0.5, f32)) \
           then fail(\"discarded guard fired\") else sum(mul(&x, &x), cast(0, i32))\n\
           sum(x, cast(0, i32))\n\
         }\n\
         out = grad(loss)(to_tensor([cast(3.0, f32)]))\n";
    assert_taken_in_both_lanes(source, "discarded_guarded_if", "discarded guard fired");
}

#[test]
fn a_discarded_guard_that_does_not_fire_leaves_the_value_alone() {
    // The negative control: keeping the guard alive must not change the
    // answer when its condition is false.
    //
    // Note what this does and does not prove. Keeping an untaken guard alive
    // DOES change behaviour when its fallback can trap on its own, because
    // [05-OP-68] makes the fallback an ordinary operand evaluated under the
    // usual rules -- `a_discarded_untaken_guard_still_evaluates_its_fallback`
    // pins that. This control uses a fallback that cannot trap, so it
    // isolates the property it names: no change to a well-defined result.
    let source = "module Repro.DiscardedUntaken\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] = {\n\
           unused = if gt(tensor_to_scalar(sum(&x, cast(0, i32))), cast(99.0, f32)) \
           then fail(\"must not fire\") else sum(mul(&x, &x), cast(0, i32))\n\
           sum(x, cast(0, i32))\n\
         }\n\
         out = grad(loss)(to_tensor([cast(3.0, f32)]))\n";
    let evaluated = eval(source, "discarded_untaken");
    let stdout = String::from_utf8_lossy(&evaluated.stdout).into_owned();
    assert!(
        evaluated.status.success(),
        "an untaken discarded guard must not abort: stderr={}",
        String::from_utf8_lossy(&evaluated.stderr)
    );
    assert!(
        stdout.contains("data=[1.0]"),
        "grad of sum(x) is 1; got: {stdout}"
    );
}

/// chelis#2368, collateral and intended: an untaken discarded guard now
/// evaluates its fallback, so a fallback that traps on its own does trap.
///
/// [05-OP-68] is explicit that "the fallback is an ordinary operand and is
/// evaluated under the usual rules", so this follows from keeping the guard
/// alive. It is pinned because it is a real behaviour change for programs
/// with no FIRING guard at all, and the negative control above deliberately
/// cannot see it.
#[test]
fn a_discarded_untaken_guard_still_evaluates_its_fallback() {
    let source = "module Repro.DiscardedTrappingFallback\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] = {\n\
           unused = if gt(tensor_to_scalar(sum(&x, cast(0, i32))), cast(99.0, f32)) \
           then fail(\"never fires\") \
           else cast(to_tensor([cast(1.0e30, f32)]), i32)\n\
           sum(x, cast(0, i32))\n\
         }\n\
         out = grad(loss)(to_tensor([cast(3.0, f32)]))\n";
    let evaluated = eval(source, "discarded_trapping_fallback");
    let stdout = String::from_utf8_lossy(&evaluated.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&evaluated.stderr).into_owned();
    assert!(
        !evaluated.status.success(),
        "the fallback's own trap must fire: stdout={stdout}"
    );
    assert!(
        stderr.contains("numeric trap"),
        "the fallback traps on its own terms, not via the guard; got: {stderr}"
    );
    assert!(
        !stderr.contains("never fires"),
        "the guard itself must NOT fire -- its condition is false; got: {stderr}"
    );
}

/// chelis#2440: a discarded trapping node still traps.
///
/// `spec/06-transformations.md` §5.2: "Mark every effectful node, every
/// potentially trapping node ... as live", and "purity alone does not make a
/// possible trap dead". The cast overflows f32 into i32 and nothing consumes
/// it; before this it was swept and the trap simply did not occur.
#[test]
fn a_discarded_trapping_node_still_traps() {
    let source = "module Repro.DiscardedTrap\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] = {\n\
           dead = cast(mul(&x, to_tensor([cast(1.0e30, f32)])), i32)\n\
           sum(x, cast(0, i32))\n\
         }\n\
         out = grad(loss)(to_tensor([cast(1.0, f32)]))\n";
    let evaluated = eval(source, "discarded_trap");
    let stdout = String::from_utf8_lossy(&evaluated.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&evaluated.stderr).into_owned();
    assert!(
        !evaluated.status.success(),
        "a discarded trapping cast must still trap: stdout={stdout}"
    );
    assert!(
        stderr.contains("numeric trap"),
        "the trap must report on its own terms; got: {stderr}"
    );
}

/// chelis#2440, the reordering half of [05-OP-68]: two observable effects
/// report in source order regardless of which one is consumed.
///
/// This is why the atom carried a scoped parenthetical. A discarded trap
/// placed BEFORE a guard used to be swept, so the guard fired and the
/// earlier effect never reported — the guard was effectively reordered ahead
/// of it. The difference was purely the other effect's liveness.
#[test]
fn an_earlier_discarded_trap_reports_before_a_later_guard() {
    let source = "module Repro.TrapBeforeGuard\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] = {\n\
           dead = cast(mul(&x, to_tensor([cast(1.0e30, f32)])), i32)\n\
           guard = if gt(tensor_to_scalar(sum(&x, cast(0, i32))), cast(0.5, f32)) \
           then fail(\"GUARD AFTER DEAD TRAP\") else sum(&x, cast(0, i32))\n\
           sum(x, cast(0, i32))\n\
         }\n\
         out = grad(loss)(to_tensor([cast(1.0, f32)]))\n";
    let evaluated = eval(source, "trap_before_guard");
    let stderr = String::from_utf8_lossy(&evaluated.stderr).into_owned();
    assert!(
        !evaluated.status.success(),
        "one of the two effects must fire"
    );
    assert!(
        stderr.contains("numeric trap"),
        "the EARLIER effect reports; got: {stderr}"
    );
    assert!(
        !stderr.contains("GUARD AFTER DEAD TRAP"),
        "the later guard must not pre-empt the earlier trap; got: {stderr}"
    );
}

/// A `fail` with no enclosing `if` and NO transform is still the host lane's
/// whole-value abort.
///
/// Honest about its reach: it does NOT guard the depth-0 lowering this change
/// touches. Untransformed programs never reach that site, so reverting the
/// guard or rejecting every depth-0 `fail` both leave this test green
/// (measured — chelis#2384 review, FF). What it does guard is a *different*
/// implementation shape: one that routed the host lane through DAG lowering,
/// which is how an earlier attempt at this class went wrong.
#[test]
fn a_fail_with_no_if_and_no_transform_is_unchanged() {
    let source = "module Repro.PlainFail\n\
         def boom(x: tensor[1, f32]) -> tensor[f32] = fail(\"plain boom\")\n\
         def loss(x: tensor[1, f32]) -> tensor[f32] = boom(x)\n\
         out = loss(to_tensor([cast(3.0, f32)]))\n";
    // Both lanes: the claim this control protects is "outside a transform,
    // no behaviour change", and the compiled lane is where a regression
    // would actually reach a user. The census canary covers the C lane only
    // for a DYNAMIC message, so nothing else pins the literal path here.
    assert_taken_in_both_lanes(source, "plain_fail", "plain boom");
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
