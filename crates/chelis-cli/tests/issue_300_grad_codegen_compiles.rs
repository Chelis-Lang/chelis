//! Issue #300: the C backend EMITS C for a `grad` over the
//! constant-broadcast idiom `expand(scalar_to_tensor(c), axis, n)`, but the
//! emitted kernel did not gcc-compile:
//!
//! ```text
//! error: pointer value used where a floating-point was expected
//! error: pointer cannot be cast to type 'float'
//! ```
//!
//! Root cause (a C-backend host-lowering defect, NOT grad-specific): a
//! scalar-returning function body is lowered statement-by-statement in the
//! host lane. A `let k = expand(scalar_to_tensor(cast(2.5, f32)),
//! cast(0, i32), cast(2, i64))` binding lost its tensor type because:
//!
//!   * `expr_int_literal` did not see through `cast(0, i32)` /
//!     `cast(2, i32)`, so `infer_app_expr_host_type`'s `expand` shape
//!     handler bailed and the result type degraded to `Unknown`;
//!   * `infer_builtin_host_type_from_arg_tys` had no `expand` arm.
//!
//! With no tensor type, `expand` fell through to a host-lane `Builtin
//! "expand"` that the C emitter renders as `void* k = /* unsupported
//! builtin expand */ 0`. The consuming `sum(mul(x, k), 0)` tensor helper
//! then took `k` as a "scalar" input and emitted
//! `((float*)tmp->data)[0] = (float)(void* k);` -- the pointer-to-float
//! cast gcc rejects.
//!
//! The grad in the headline reproducer is incidental: the same defect
//! reproduces on the pure forward control (scalar return + constant
//! `expand`, no `grad`) -- see `issue_300_forward_scalar_const_expand_*`.
//!
//! This file is the closing oracle for the "emit but never compile"
//! coverage gap called out in #300 and in the #288 test's note: it EMITS
//! the C *and* invokes the native C compiler, asserting the emitted kernel
//! compiles cleanly and runs to the correct gradient `[2.5, 2.5]`.
//!
//! It also hosts the issue #308 closing tests (`issue_308_*`): the f64
//! sibling of the const-broadcast idiom must materialize the constant at
//! exact f64 precision in both the compiled C and `chelis eval`. The
//! former `#[ignore]`d manual gate
//! `issue_300_const_expand_f64_truncates_to_f32` (which pinned the bug
//! signature) was converted into the active
//! `issue_308_const_expand_f64_exact_precision` test when the fix landed.

use assert_cmd::Command;
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::tempdir;

/// Resolve the C compiler the way the existing C-backend exec tests do:
/// honor `$CC`, else `cc`. CI runners provide one.
fn c_compiler() -> String {
    std::env::var("CC").unwrap_or_else(|_| "cc".to_string())
}

/// Run `chelis build --target c` on `source`, returning the build dir
/// (containing the emitted `<stem>.c`, headers, and runtime static lib).
fn chelis_build_c(source: &str, stem: &str) -> tempfile::TempDir {
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("{stem}.ch"));
    fs::write(&src_path, source).expect("write .ch source");

    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            src_path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            dir.path().join(format!("{stem}.c")).to_str().unwrap(),
        ])
        .assert()
        .success();

    dir
}

/// Compile the emitted `<stem>.c` (which carries its own `main()` driving
/// the top-level `out = ...` binding) against the runtime static lib, run
/// it, and return stdout. Asserts both the compile and the run succeed --
/// the compile assertion is the #300 regression guard.
fn compile_and_run_emitted(build_dir: &Path, kernel_c: &Path) -> String {
    let runtime = build_dir.join("libchelis_runtime.a");
    assert!(
        runtime.is_file(),
        "`chelis build` did not stage the carried runtime at {}",
        runtime.display()
    );

    let bin = build_dir.join("issue_300_bin");
    let compile = StdCommand::new(c_compiler())
        .args([
            "-O0",
            "-std=c11",
            "-I",
            build_dir.to_str().unwrap(),
            kernel_c.to_str().unwrap(),
            "-o",
            bin.to_str().unwrap(),
            runtime.to_str().unwrap(),
            "-lm",
            "-lpthread",
            "-ldl",
        ])
        .output()
        .expect("invoke C compiler");
    assert!(
        compile.status.success(),
        "emitted grad/expand kernel must compile cleanly (issue #300); \
         compiler stderr=\n{}",
        String::from_utf8_lossy(&compile.stderr),
    );

    let run = StdCommand::new(&bin).output().expect("run emitted binary");
    assert!(
        run.status.success(),
        "emitted binary exited non-zero: stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr),
    );
    String::from_utf8_lossy(&run.stdout).into_owned()
}

/// The headline reproducer (verbatim from issue #300): `grad(f)(x)` over
/// the constant-broadcast idiom. The emitted C must compile cleanly and
/// run to `df(x) = [2.5, 2.5]` (the host evaluator's numeric oracle).
#[test]
fn issue_300_grad_const_expand_kernel_compiles_and_runs() {
    let source = "module Repro.GradExpandConst\n\
def f(x: tensor[2, f32]) -> f32 = {\n  \
  k = insert(scalar_to_tensor(cast(2.5, f32)), cast(0, i32), cast(2, i64))\n  \
  tensor_to_scalar(sum(mul(x, k), cast(0, i32)))\n\
}\n\
def df(x: tensor[2, f32]) -> tensor[2, f32] = grad(f)(x)\n\
out = df(to_tensor([3.0, 4.0]))\n";

    let build = chelis_build_c(source, "repro");
    let kernel_c = build.path().join("repro.c");
    let stdout = compile_and_run_emitted(build.path(), &kernel_c);
    assert!(
        stdout.contains("shape=[2]") && stdout.contains("data=[2.5, 2.5]"),
        "grad(f)(x) must print [2.5, 2.5]; got stdout={stdout:?}",
    );
}

/// Minimal non-grad control: the same defect reproduces on the pure
/// forward path. A scalar-returning function with a constant `expand`
/// binding consumed by a tensor reduction. `f(x) = sum(x * 2.5)` for
/// `x = [3, 4]` is `2.5 * 7 = 17.5`.
#[test]
fn issue_300_forward_scalar_const_expand_compiles_and_runs() {
    let source = "module Repro.ForwardConstExpand\n\
def h(x: tensor[2, f32]) -> f32 = {\n  \
  k = insert(scalar_to_tensor(cast(2.5, f32)), cast(0, i32), cast(2, i64))\n  \
  tensor_to_scalar(sum(mul(x, k), cast(0, i32)))\n\
}\n\
out = h(to_tensor([3.0, 4.0]))\n";

    let build = chelis_build_c(source, "fwd");
    let kernel_c = build.path().join("fwd.c");
    let stdout = compile_and_run_emitted(build.path(), &kernel_c);
    let trimmed = stdout.trim();
    assert!(
        trimmed.contains("17.5"),
        "h(x) = sum(x * 2.5) for x=[3,4] must be 17.5; got stdout={trimmed:?}",
    );
}

/// A tensor-returning function with a constant-`expand` binding consumed
/// by a tensor helper, driven through a host call (`out = scale(...)`).
/// This pins that the constant value is actually materialized into the
/// broadcast tensor: `scale(x) = x * 2.5` for `x = [3, 4]` is
/// `[7.5, 10.0]`. The pre-fix `scalar_to_tensor` f64-vs-f32 storage
/// mismatch zeroed the constant, so `mul(x, k)` collapsed to `x * 0`.
#[test]
fn issue_300_scale_const_expand_materializes_value() {
    let source = "module Repro.ScaleConstExpand\n\
def scale(x: tensor[2, f32]) -> tensor[2, f32] = {\n  \
  k = insert(scalar_to_tensor(cast(2.5, f32)), cast(0, i32), cast(2, i64))\n  \
  mul(x, k)\n\
}\n\
out = scale(to_tensor([3.0, 4.0]))\n";

    let build = chelis_build_c(source, "scale");
    let kernel_c = build.path().join("scale.c");
    let stdout = compile_and_run_emitted(build.path(), &kernel_c);
    assert!(
        stdout.contains("shape=[2]") && stdout.contains("data=[7.5, 10.0]"),
        "scale(x) = x * 2.5 for x=[3,4] must print [7.5, 10.0]; got stdout={stdout:?}",
    );
}

/// Run `chelis eval --file` on `source` and return trimmed stdout.
/// Asserts the eval call succeeds. (Mirror of
/// `cbackend_print_tensor_f64.rs::chelis_eval`.)
fn chelis_eval(source: &str, stem: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("{stem}.ch"));
    fs::write(&src_path, source).expect("write .ch source");
    let assert = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", src_path.to_str().unwrap()])
        .assert()
        .success();
    let out = assert.get_output();
    String::from_utf8_lossy(&out.stdout).trim_end().to_string()
}

/// Issue #308 (formerly the KNOWN-LIMITATION manual gate
/// `issue_300_const_expand_f64_truncates_to_f32`): an f64
/// const-broadcast `expand(scalar_to_tensor(cast(c, f64)), ...)` used to
/// materialize the constant at f32 precision, so the printed f64 result
/// carried the f32-truncation signature `1.100000023841858` instead of
/// `1.1`.
///
/// Root cause: the desugarer applied the spec §5.6 position-4 rule
/// (`cast(literal, p)` — the literal adopts `p`) only to tensor
/// literals; a bare scalar literal kept the §5.3 f32 default, so
/// `cast(1.1, f64)` lowered to an f32-typed IR `Const` plus a widening
/// `Cast`, and every value lane materialized f32(1.1) first. With the
/// literal bound at f64 the DAG `Const` is f64-typed and the C emit
/// fills exact f64 bits. The host-lane `scalar_to_tensor` inference
/// (`chelis_ir::host::infer_app_expr_host_type`) now also recovers the
/// operand's float precision from the Deep node so the
/// `chelis_scalar_tensor_from_f64` storage and the consuming helper's
/// f64 read stay paired even without a checker annotation.
#[test]
fn issue_308_const_expand_f64_exact_precision() {
    let source = "module Repro.ScaleConstExpandF64\n\
def scale64(x: tensor[2, f64]) -> tensor[2, f64] = {\n  \
  k = insert(scalar_to_tensor(cast(1.1, f64)), cast(0, i32), cast(2, i64))\n  \
  mul(x, k)\n\
}\n\
out = scale64(cast(to_tensor([1.0, 1.0]), f64))\n";

    let build = chelis_build_c(source, "scale64");
    let kernel_c = build.path().join("scale64.c");
    let stdout = compile_and_run_emitted(build.path(), &kernel_c);
    assert!(
        stdout.contains("shape=[2]") && stdout.contains("data=[1.1, 1.1]"),
        "f64 const-broadcast must materialize the exact f64 constant \
         (issue #308); got stdout={stdout:?}",
    );
    assert!(
        !stdout.contains("1.100000023841858"),
        "the f32-truncation signature must be gone (issue #308); got \
         stdout={stdout:?}",
    );

    // Evaluator/backend agreement: `chelis eval` must print the same
    // exact-f64 tensor the compiled C does.
    let eval_out = chelis_eval(source, "scale64");
    assert!(
        eval_out.contains("data=[1.1, 1.1]") && !eval_out.contains("1.100000023841858"),
        "`chelis eval` must agree with the C backend on the exact f64 \
         constant (issue #308); got eval stdout={eval_out:?}",
    );
}

/// Issue #308 headline reproducer (scalar-return host lane): the same
/// f64 const-broadcast consumed by a reduction in an f64-returning
/// function. `h(x) = sum(x * 1.1)` for `x = [1, 1]` at true f64
/// precision is exactly `2.2`; the pre-fix f32-truncated constant gave
/// `2.200000047683716` in the evaluator (and `2.2000000476...`-class
/// values wherever the DAG lane materialized the constant).
#[test]
fn issue_308_forward_scalar_const_expand_f64_exact() {
    let source = "module Repro.FwdConstExpandF64\n\
def h(x: tensor[2, f64]) -> f64 = {\n  \
  k = insert(scalar_to_tensor(cast(1.1, f64)), cast(0, i32), cast(2, i64))\n  \
  tensor_to_scalar(sum(mul(x, k), cast(0, i32)))\n\
}\n\
out = h(cast(to_tensor([1.0, 1.0]), f64))\n";

    let build = chelis_build_c(source, "fwd64");
    let kernel_c = build.path().join("fwd64.c");
    let stdout = compile_and_run_emitted(build.path(), &kernel_c);
    let trimmed = stdout.trim();
    assert!(
        trimmed.contains("2.2") && !trimmed.contains("2.200000047683716"),
        "h(x) = sum(x * 1.1) for f64 x=[1,1] must be exactly 2.2 \
         (issue #308); got stdout={trimmed:?}",
    );

    let eval_out = chelis_eval(source, "fwd64");
    assert!(
        eval_out.contains("2.2") && !eval_out.contains("2.200000047683716"),
        "`chelis eval` must agree with the C backend on the exact f64 \
         result (issue #308); got eval stdout={eval_out:?}",
    );
}
