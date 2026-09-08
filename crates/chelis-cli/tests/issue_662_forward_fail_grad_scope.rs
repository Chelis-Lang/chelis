//! chelis#662: a forward `fail` beside a grad/vmap expression must retain
//! real host control flow in generated C. Transformed-subtree behavior is
//! unchanged by this repair; chelis#1464 owns its taken-`fail` divergence.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command as StdCommand, Output};

use assert_cmd::Command;
use tempfile::{TempDir, tempdir};

const MIXED_GRAD: &str = "module Repro.MixedGradFail\n\
def mixed(x) = {\n\
  g = grad(sq_sum)(x)\n\
  s = tensor_to_scalar(sum(x, cast(0, int32)))\n\
  guarded = if gt(s, cast(0.0, f32)) then fail(\"forward boom\") else neg(x)\n\
  add(g, guarded)\n\
}\n\
def sq_sum(x: tensor[3, f32]) -> tensor[f32] = sum(mul(x, x), cast(0, int32))\n";

fn source_with_input(values: &str) -> String {
    format!("{MIXED_GRAD}out = mixed(to_tensor([{values}]))\n")
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

fn assert_failure_parity(source: &str, stem: &str, message: &str) {
    let evaluated = eval(source, stem);
    assert!(
        !evaluated.status.success(),
        "eval must take the forward fail: stdout={} stderr={}",
        String::from_utf8_lossy(&evaluated.stdout),
        String::from_utf8_lossy(&evaluated.stderr)
    );
    assert!(
        String::from_utf8_lossy(&evaluated.stderr).contains(message),
        "eval must retain the user's message: {}",
        String::from_utf8_lossy(&evaluated.stderr)
    );

    let (_dir, build_dir) = build_c(source, stem);
    let compiled = compile_and_run(&build_dir, stem);
    assert!(
        !compiled.status.success(),
        "compiled forward fail must abort: stdout={} stderr={}",
        String::from_utf8_lossy(&compiled.stdout),
        String::from_utf8_lossy(&compiled.stderr)
    );
    assert!(
        String::from_utf8_lossy(&compiled.stderr).contains(message),
        "compiled abort must retain the user's message: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
}

#[test]
fn forward_fail_beside_grad_aborts_in_both_lanes() {
    let source = source_with_input("cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)");
    let eval = eval(&source, "mixed_grad_fail_taken");
    assert!(!eval.status.success(), "eval must take the fail branch");
    assert!(
        String::from_utf8_lossy(&eval.stderr).contains("forward boom"),
        "eval must retain the user's message: {}",
        String::from_utf8_lossy(&eval.stderr)
    );

    let (_dir, build_dir) = build_c(&source, "mixed_grad_fail_taken");
    let compiled = compile_and_run(&build_dir, "mixed_grad_fail_taken");
    assert!(
        !compiled.status.success(),
        "compiled forward fail must abort, never return masked zeros: stdout={}",
        String::from_utf8_lossy(&compiled.stdout)
    );
    assert!(
        String::from_utf8_lossy(&compiled.stderr).contains("forward boom"),
        "compiled abort must retain the user's message: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
}

#[test]
fn untaken_forward_fail_beside_grad_has_exact_lane_parity() {
    let source = source_with_input("cast(-1.0, f32), cast(-2.0, f32), cast(-3.0, f32)");
    let eval = eval(&source, "mixed_grad_fail_untaken");
    assert!(
        eval.status.success(),
        "eval control failed: {}",
        String::from_utf8_lossy(&eval.stderr)
    );
    let (_dir, build_dir) = build_c(&source, "mixed_grad_fail_untaken");
    let compiled = compile_and_run(&build_dir, "mixed_grad_fail_untaken");
    assert!(
        compiled.status.success(),
        "compiled control failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    assert_eq!(
        compiled.stdout, eval.stdout,
        "the untaken branch must agree byte-for-byte"
    );
}

#[test]
fn untaken_fail_inside_grad_subtree_does_not_change_sibling_routing() {
    let source = "module Repro.GradInternalFail\n\
def loss(x: tensor[4, f32]) -> tensor[f32] =\n\
  if gt(cast(2, int64), cast(shape(x, cast(0, int32)), int64))\n\
  then fail(\"kernel exceeds input length\")\n\
  else sum(x, cast(0, int32))\n\
out = grad(loss)(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32)]))\n";
    let eval = eval(source, "grad_internal_fail");
    assert!(
        eval.status.success(),
        "grad-internal control failed in eval: {}",
        String::from_utf8_lossy(&eval.stderr)
    );
    let (_dir, build_dir) = build_c(source, "grad_internal_fail");
    let compiled = compile_and_run(&build_dir, "grad_internal_fail");
    assert!(
        compiled.status.success(),
        "grad-internal control failed in C: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    assert_eq!(compiled.stdout, eval.stdout);
}

#[test]
fn forward_fail_beside_vmap_aborts_in_both_lanes() {
    let source = "module Repro.MixedVmapFail\n\
def negate_row(x: tensor[2, f32]) -> tensor[2, f32] = neg(x)\n\
def mixed(xs: tensor[2, 2, f32]) -> tensor[2, 2, f32] = {\n\
  mapped = vmap(negate_row)(xs)\n\
  total = tensor_to_scalar(sum(sum(xs, cast(0, int32)), cast(0, int32)))\n\
  guarded = if gt(total, cast(0.0, f32)) then fail(\"vmap sibling boom\") else neg(xs)\n\
  add(mapped, guarded)\n\
}\n\
out = mixed(to_tensor([[cast(1.0, f32), cast(2.0, f32)], [cast(3.0, f32), cast(4.0, f32)]]))\n";
    let eval = eval(source, "mixed_vmap_fail_taken");
    assert!(!eval.status.success(), "eval must take the fail branch");
    assert!(
        String::from_utf8_lossy(&eval.stderr).contains("vmap sibling boom"),
        "eval must retain the user's message: {}",
        String::from_utf8_lossy(&eval.stderr)
    );
    let (_dir, build_dir) = build_c(source, "mixed_vmap_fail_taken");
    let compiled = compile_and_run(&build_dir, "mixed_vmap_fail_taken");
    assert!(
        !compiled.status.success(),
        "compiled forward fail beside vmap must abort: stdout={}",
        String::from_utf8_lossy(&compiled.stdout)
    );
    assert!(
        String::from_utf8_lossy(&compiled.stderr).contains("vmap sibling boom"),
        "compiled abort must retain the user's message: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
}

#[test]
fn transitive_forward_fail_through_defs_aborts_in_both_lanes() {
    let source = "module Repro.TransitiveForwardFail\n\
def fail_leaf(x: tensor[3, f32]) -> tensor[3, f32] = {\n\
  total = tensor_to_scalar(sum(x, cast(0, int32)))\n\
  if gt(total, cast(0.0, f32)) then fail(\"transitive boom\") else neg(x)\n\
}\n\
def fail_hop(x: tensor[3, f32]) -> tensor[3, f32] = fail_leaf(x)\n\
def sq_sum_transitive(x: tensor[3, f32]) -> tensor[f32] = sum(mul(x, x), cast(0, int32))\n\
def mixed_transitive(x: tensor[3, f32]) -> tensor[3, f32] = {\n\
  g = grad(sq_sum_transitive)(x)\n\
  add(g, fail_hop(x))\n\
}\n\
out = mixed_transitive(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]))\n";
    assert_failure_parity(source, "transitive_forward_fail", "transitive boom");
}

#[test]
fn shared_fail_def_is_visible_outside_transform_in_both_traversal_orders() {
    const PREFIX: &str = "module Repro.SharedFailDef\n\
def maybe_fail(x: tensor[3, f32]) -> tensor[f32] = {\n\
  total = sum(x, cast(0, int32))\n\
  scalar = tensor_to_scalar(total)\n\
  if gt(scalar, cast(0.0, f32)) then fail(\"shared def boom\") else total\n\
}\n";
    let inside_first = format!(
        "{PREFIX}def mixed_inside_first() -> tensor[f32] = {{\n\
           hidden = grad(maybe_fail)(to_tensor([cast(-1.0, f32), cast(-2.0, f32), cast(-3.0, f32)]))\n\
           maybe_fail(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]))\n\
         }}\n\
         out = mixed_inside_first()\n"
    );
    assert_failure_parity(&inside_first, "shared_fail_inside_first", "shared def boom");

    let outside_first = format!(
        "{PREFIX}def mixed_outside_first() -> tensor[f32] = {{\n\
           visible = maybe_fail(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]))\n\
           hidden = grad(maybe_fail)(to_tensor([cast(-1.0, f32), cast(-2.0, f32), cast(-3.0, f32)]))\n\
           visible\n\
         }}\n\
         out = mixed_outside_first()\n"
    );
    assert_failure_parity(
        &outside_first,
        "shared_fail_outside_first",
        "shared def boom",
    );
}
