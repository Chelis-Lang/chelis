//! Chelis-Lang/chelis#703 - unsupported cases silently substitute a value.
//!
//! ## The class
//!
//! When Chelis hits a case it does not support, it substitutes a plausible
//! value and compiles a working program instead of failing. The user gets a
//! binary that runs, returns numbers, and is wrong.
//!
//! This is a *sibling* of #695, not part of it. #695 is "integers have no
//! representation in the numeric layer"; this is "unimplemented evaluates to
//! zero". They keep meeting because integers are the most common unsupported
//! case, but the fixes are unrelated - and several instances here are wrong for
//! values like `100` or `3.5`, nothing to do with precision.
//!
//! Three substitution sites, all confirmed by compiling and running:
//!
//! | site | substitutes | issue |
//! |------|-------------|-------|
//! | `crates/chelis-ir/src/lower.rs:9838` | `RiscOp::Const { value: 0.0 }` for a non-float operand (operand DROPPED) | #699 |
//! | `crates/chelis-backend-c/src/host_emit.rs:2300` | `/* unsupported builtin */ 0` for an unimplemented builtin | #682, #704, #705 |
//! | `crates/chelis-backend-hip/src/emit.rs:3615` | `ElemKind::F32` for an unsupported dtype | #689 |
//!
//! ## Why the passing tests here matter as much as the failing ones
//!
//! Every `Broken` case below has a **paired control that passes**. That pairing
//! is not decoration:
//!
//!   * tensor `relu` is CORRECT; only the scalar overload is broken. An earlier
//!     draft of #704 claimed "a neural-network layer returns zero" - false, and
//!     caught only by running the tensor form. Idiomatic ML code is fine.
//!   * scalar `sqrt` and `exp` are CORRECT; only some scalar ops are missing.
//!     That localises #704 to per-op dispatch coverage, not "scalars are broken".
//!   * `sin` and `exp` on integer tensors are correctly REJECTED, while their
//!     siblings `cos`/`tan`/`atan` are not. That localises #699's checker gap.
//!   * f32 `abs` is CORRECT, which localises #699 to the `is_float()` branch.
//!
//! Without each control, the matching bug report overstates its blast radius.
//! These locks are what keep the issues honest.
//!
//! ## The fix mechanism already exists
//!
//! `lower_unsupported` (`crates/chelis-ir/src/lower.rs:10810-10817`) calls
//! `raise_lowering_error(...)` correctly, in the same file as #699's bug. The
//! codebase has the right tool and did not apply it next door - the same shape
//! as #387 in the #695 class.
//!
//! Tests that assert correct behavior and fail today are `#[ignore]`d with their
//! issue number, so CI stays green and each goes red-to-green when its fix lands.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

fn c_toolchain_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Build `program` to C. Returns `Ok((emitted_c, stdout_of_binary))` or the
/// build's stderr on failure.
///
/// A clean build failure is a legitimate outcome for an unsupported op - it is
/// the *silent success* that is the bug - so callers distinguish the two rather
/// than asserting success.
fn build_and_run_c(program: &str, name: &str) -> Result<(String, String), String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    let built = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("chelis build should run");
    if !built.status.success() {
        return Err(String::from_utf8_lossy(&built.stderr).into_owned());
    }
    let emitted = std::fs::read_to_string(out_dir.join(format!("{name}.c")))
        .map_err(|e| format!("read emitted C: {e}"))?;
    let status = common::link_generated(&out_dir, &format!("{name}.c"), name);
    if !status.success() {
        return Err(format!("link failed: {status}"));
    }
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    Ok((emitted, String::from_utf8_lossy(&run.stdout).into_owned()))
}

/// Run `program` through `chelis eval --file`; return the first printed line,
/// or the stderr on failure.
fn eval_first_line(program: &str) -> Result<String, String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string())
}

/// The marker the C emitter leaves when it substitutes zero for a builtin it
/// cannot emit (`host_emit.rs:2300`). Its presence in emitted C is the bug.
const STUB_MARKER: &str = "unsupported builtin";

// ===========================================================================
// CONTROLS THAT PASS. These bound the blast radius of every bug below.
// ===========================================================================

/// **Tensor `relu` is correct.** The control that proves #704 is scalar-only.
///
/// An earlier draft of #704 claimed a neural-network layer returns zero. That
/// was false: idiomatic ML code uses tensors, and the tensor path emits the real
/// helper with no stub. This lock exists so nobody re-derives the overclaim.
#[test]
fn tensor_relu_is_correct_and_emits_no_stub() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let (emitted, stdout) = build_and_run_c(
        "def f(x: tensor[4, f32]) -> tensor[4, f32] = relu(x)\n\
         out = f(to_tensor([-1.0, 2.0, -3.0, 4.0]))\n",
        "tensor_relu",
    )
    .expect("tensor relu should build and run");
    assert!(
        !emitted.contains(STUB_MARKER),
        "tensor relu must emit the real helper, not a stub"
    );
    assert!(
        stdout.contains("2.0") && stdout.contains("4.0"),
        "tensor relu must compute [0, 2, 0, 4]; got: {stdout}"
    );
}

/// **Scalar `sqrt` and `exp` are correct.** The control that localises #704 to
/// per-op dispatch coverage rather than "scalars are broken".
#[test]
fn scalar_sqrt_and_exp_are_correct() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let (emitted, stdout) = build_and_run_c(
        "def f(x: f32) -> f32 = sqrt(x)\nout = f(3.5)\n",
        "scalar_sqrt",
    )
    .expect("scalar sqrt should build and run");
    assert!(
        !emitted.contains(STUB_MARKER),
        "scalar sqrt must not emit a stub"
    );
    assert!(
        stdout.contains("1.87"),
        "scalar sqrt(3.5) must be ~1.8708; got: {stdout}"
    );

    let (emitted, stdout) = build_and_run_c(
        "def f(x: f32) -> f32 = exp(x)\nout = f(3.5)\n",
        "scalar_exp",
    )
    .expect("scalar exp should build and run");
    assert!(
        !emitted.contains(STUB_MARKER),
        "scalar exp must not emit a stub"
    );
    assert!(
        stdout.contains("33.1"),
        "scalar exp(3.5) must be ~33.115; got: {stdout}"
    );
}

/// **`sin` and `exp` on an integer tensor are correctly REJECTED.**
///
/// The control that localises #699's checker gap: these two are refused with a
/// `PrecisionMismatch`, while their siblings `cos`/`tan`/`atan` are not. Locks
/// that the checker CAN reject this shape, so the gap is per-op coverage rather
/// than a missing capability.
#[test]
fn sin_and_exp_on_integer_tensors_are_rejected() {
    for op in ["sin", "exp"] {
        let program = format!(
            "def run(x: tensor[4, int32]) -> tensor[4, int32] = {op}(x)\n\
             out = run(to_tensor([cast(1, int32), cast(2, int32), cast(3, int32), \
             cast(4, int32)]))\n"
        );
        let err = build_and_run_c(&program, &format!("reject_{op}"))
            .expect_err(&format!("`{op}` on an int32 tensor must be rejected"));
        assert!(
            err.contains("recision") || err.contains("recisionMismatch"),
            "`{op}` on an int32 tensor must be rejected with a precision \
             diagnostic; got: {err}"
        );
    }
}

// ===========================================================================
// #704: scalar ops that silently stub to 0.
// ===========================================================================

/// Assert a scalar op does not silently stub to zero.
fn assert_scalar_op_not_stubbed(op: &str, name: &str) {
    if !c_toolchain_available() {
        eprintln!("skipping {name}: no host C toolchain");
        return;
    }
    let program = format!("def f(x: f32) -> f32 = {op}(x)\nout = f(3.5)\n");
    match build_and_run_c(&program, name) {
        // A clean build error is an acceptable outcome: unimplemented should
        // fail loudly. Silent success returning 0 is the bug.
        Err(_) => {}
        Ok((emitted, stdout)) => {
            assert!(
                !emitted.contains(STUB_MARKER),
                "scalar `{op}` emitted `/* unsupported builtin {op} */ 0` and \
                 the build SUCCEEDED, so the compiled program silently returns \
                 zero. The scalar helper may already exist in host_emit.rs and \
                 simply not be wired into the dispatch. chelis#704"
            );
            assert!(
                !stdout.contains("out = 0"),
                "scalar `{op}(3.5)` returned 0. chelis#704. Got: {stdout}"
            );
        }
    }
}

macro_rules! scalar_stub_test {
    ($fn_name:ident, $op:literal) => {
        #[test]
        #[ignore = "chelis#704: this scalar op silently compiles to \
                    `/* unsupported builtin */ 0` and returns 0. The tensor form \
                    is CORRECT (see tensor_relu_is_correct_and_emits_no_stub). \
                    This test asserts the correct behavior and fails until the \
                    fix lands. Run with `cargo test -p chelis-cli --test \
                    issue_703_silent_placeholders -- --ignored`."]
        fn $fn_name() {
            assert_scalar_op_not_stubbed($op, stringify!($fn_name));
        }
    };
}

scalar_stub_test!(scalar_relu_is_not_stubbed_to_zero, "relu");
scalar_stub_test!(scalar_sigmoid_is_not_stubbed_to_zero, "sigmoid");
scalar_stub_test!(scalar_silu_is_not_stubbed_to_zero, "silu");
scalar_stub_test!(scalar_gelu_is_not_stubbed_to_zero, "gelu");
scalar_stub_test!(scalar_tan_is_not_stubbed_to_zero, "tan");
scalar_stub_test!(scalar_atan_is_not_stubbed_to_zero, "atan");
scalar_stub_test!(scalar_floor_is_not_stubbed_to_zero, "floor");
scalar_stub_test!(scalar_ceil_is_not_stubbed_to_zero, "ceil");
scalar_stub_test!(scalar_round_is_not_stubbed_to_zero, "round");

// ===========================================================================
// #699: cos/tan/atan on integer tensors (extends abs/floor/ceil/round).
// ===========================================================================

/// `cos`/`tan`/`atan` on an integer tensor BUILD and return zeros, while their
/// siblings `sin`/`exp` are correctly rejected.
///
/// The correct fix differs from `abs`/`floor`/`ceil`/`round`: those are
/// well-defined on integers and should be IMPLEMENTED, whereas `cos`/`tan`/
/// `atan` are undefined on integers and should be REJECTED exactly as `sin` is.
/// Either way the silent zero must go.
fn assert_int_tensor_transcendental_not_zeroed(op: &str, name: &str) {
    if !c_toolchain_available() {
        eprintln!("skipping {name}: no host C toolchain");
        return;
    }
    let program = format!(
        "def run(x: tensor[4, int32]) -> tensor[4, int32] = {op}(x)\n\
         out = run(to_tensor([cast(1, int32), cast(2, int32), cast(3, int32), \
         cast(4, int32)]))\n"
    );
    match build_and_run_c(&program, name) {
        // Rejection is the correct outcome for cos/tan/atan (matching sin/exp).
        Err(_) => {}
        Ok((_, stdout)) => {
            assert!(
                !stdout.contains("[0.0, 0.0, 0.0, 0.0]"),
                "`{op}` on an int32 tensor built successfully and returned all \
                 zeros. lower_transcendental's non-float branch emits \
                 `RiscOp::Const {{ value: 0.0 }}` with the operand dropped and \
                 raises no error. Its siblings `sin`/`exp` are correctly \
                 rejected. chelis#699. Got: {stdout}"
            );
        }
    }
}

#[test]
#[ignore = "chelis#699: cos on an integer tensor builds and returns zeros \
            (sin/exp are correctly rejected). Run with `cargo test -p chelis-cli \
            --test issue_703_silent_placeholders -- --ignored`."]
fn cos_on_integer_tensor_is_not_silently_zeroed() {
    assert_int_tensor_transcendental_not_zeroed("cos", "cos_i32");
}

#[test]
#[ignore = "chelis#699: tan on an integer tensor builds and returns zeros. \
            Run with `cargo test -p chelis-cli --test \
            issue_703_silent_placeholders -- --ignored`."]
fn tan_on_integer_tensor_is_not_silently_zeroed() {
    assert_int_tensor_transcendental_not_zeroed("tan", "tan_i32");
}

#[test]
#[ignore = "chelis#699: atan on an integer tensor builds and returns zeros. \
            Run with `cargo test -p chelis-cli --test \
            issue_703_silent_placeholders -- --ignored`."]
fn atan_on_integer_tensor_is_not_silently_zeroed() {
    assert_int_tensor_transcendental_not_zeroed("atan", "atan_i32");
}

// ===========================================================================
// #705: tensor_scan - a guard that exists, is documented, and never runs.
// ===========================================================================

/// `reject_host_only_builtins` (`crates/chelis-compiler-api/src/compiler.rs:2232`,
/// over `HOST_ONLY_BUILTINS = &["tensor_scan"]` at `:2230`) exists specifically
/// to stop this. It is only called from `compiler::compile()` at `compiler.rs:748`.
/// `chelis-cli` calls that function ZERO times - `cmd_build` runs its own
/// pipeline straight to `codegen_host_program`, screened only by its own
/// `EVAL_ONLY_HOST_BUILTINS = &["process_run"]` (`main.rs:8133`).
///
/// `spec/05-risc-primitives.md:849-855` asserts the guard means "the C/HIP
/// emitters never see a `tensor_scan` call". Verified false: the emitted C
/// contains `/* unsupported builtin tensor_scan */ 0` verbatim, which is the
/// exact string that paragraph says can no longer occur.
#[test]
#[ignore = "chelis#705: tensor_scan compiles to a silent C stub because \
            reject_host_only_builtins only guards compiler::compile, which \
            chelis build never calls. This test asserts the correct behavior \
            (build must fail loudly OR emit real code) and fails until the fix \
            lands. Run with `cargo test -p chelis-cli --test \
            issue_703_silent_placeholders -- --ignored`."]
fn tensor_scan_does_not_silently_compile_to_a_stub() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let program = "def gen() -> tensor[5, f32] = \
                   tensor_scan(0.0, fn (prev: f32, i: int64) -> add(prev, 1.0), \
                   cast(5, int64))\n\
                   out = gen()\n";
    match build_and_run_c(program, "tscan") {
        // A clean rejection is exactly what the guard was written to produce,
        // and what the spec claims already happens.
        Err(_) => {}
        Ok((emitted, stdout)) => {
            assert!(
                !emitted.contains("unsupported builtin tensor_scan"),
                "chelis build SUCCEEDED and emitted \
                 `/* unsupported builtin tensor_scan */ 0`. That is the exact \
                 string spec/05-risc-primitives.md:849-855 says the emitters \
                 `never see`, because reject_host_only_builtins guards \
                 compiler::compile, which the CLI does not call. eval returns \
                 [1, 2, 3, 4, 5] for this program. chelis#705. \
                 Binary stdout: {stdout}"
            );
        }
    }
}

/// The evaluator computes `tensor_scan` correctly, which is what makes #705 a
/// lane divergence rather than a missing feature. Passes today; locked so the
/// fix removes the divergence rather than breaking the working lane.
#[test]
fn tensor_scan_is_correct_in_the_eval_lane() {
    let got = eval_first_line(
        "module M.Main\n\
         def gen() -> tensor[5, f32] = \
         tensor_scan(0.0, fn (prev: f32, i: int64) -> add(prev, 1.0), cast(5, int64))\n\
         out = print(to_list(gen()))\n",
    )
    .expect("tensor_scan should evaluate");
    assert_eq!(
        got, "[1, 2, 3, 4, 5]",
        "the eval lane must compute tensor_scan correctly; it is the compiled \
         lane that stubs it out (chelis#705)"
    );
}

// ===========================================================================
// The class-level guard.
// ===========================================================================

/// No `chelis build` may ever emit the silent-stub marker.
///
/// This is the invariant that makes #703 a class rather than a list. Today, any
/// builtin without a C arm is a silent-wrong-answer bug BY DEFAULT: the fallback
/// at `host_emit.rs:2300` turns "unimplemented" into "returns 0 at runtime".
/// That default must invert - an unimplemented builtin should fail the build.
///
/// Checked across every op family known to hit the fallback: bitwise (#682),
/// scalar activations (#704), and host-only builtins (#705).
#[test]
#[ignore = "chelis#703: the `other => /* unsupported builtin */ 0` fallback at \
            crates/chelis-backend-c/src/host_emit.rs:2300 makes any builtin \
            without a C arm a silent-wrong-answer bug by default. Run with \
            `cargo test -p chelis-cli --test issue_703_silent_placeholders -- \
            --ignored`."]
fn no_build_ever_emits_a_silent_unsupported_builtin_stub() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let cases: &[(&str, &str)] = &[
        (
            "bitwise_682",
            "def f() -> int64 = bitand(cast(12, int64), cast(10, int64))\nout = f()\n",
        ),
        (
            "scalar_relu_704",
            "def f(x: f32) -> f32 = relu(x)\nout = f(3.5)\n",
        ),
        (
            "tensor_scan_705",
            "def gen() -> tensor[5, f32] = \
             tensor_scan(0.0, fn (prev: f32, i: int64) -> add(prev, 1.0), cast(5, int64))\n\
             out = gen()\n",
        ),
    ];
    let mut offenders = Vec::new();
    for (name, program) in cases {
        if let Ok((emitted, _)) = build_and_run_c(program, name)
            && emitted.contains(STUB_MARKER)
        {
            offenders.push(*name);
        }
    }
    assert!(
        offenders.is_empty(),
        "these builds SUCCEEDED while emitting a silent `{STUB_MARKER}` stub, \
         so each compiles to a program that returns garbage at runtime: {offenders:?}. \
         An unimplemented builtin must fail the build. chelis#703"
    );
}
