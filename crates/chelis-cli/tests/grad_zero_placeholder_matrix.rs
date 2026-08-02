//! chelis#722 - grad of a forward pass containing `abs`/`floor` on an
//! integer tensor returned ALL-ZERO gradients in BOTH lanes.
//!
//! This is #699's `Const { value: 0.0 }` placeholder
//! (`lower_transcendental`, lower.rs:9834-9845) reached through grad's
//! lowering: `grad` is built over the lowered DAG, so evaluating a
//! `grad(...)` routes the forward pass through the same lowering that
//! plants the zero - and eval's otherwise-correct `abs` does not survive
//! the trip. Both lanes then agree on the wrong answer, which no
//! cross-lane oracle (#687) can see; a zero gradient is the most
//! plausible wrong value in ML (training silently stalls).
//!
//! The controls are what make the diagnosis sharp:
//!   * the forward pass WITHOUT grad is correct in eval (prints 300.0);
//!   * the same gradient WITHOUT `abs` is correct in both lanes
//!     ([-100, 200, -300, 400]) - int64 weights, the cast, and the
//!     mul/sum adjoints are all fine. The trigger is exactly the
//!     placeholder.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

fn c_toolchain_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

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

fn c_first_line(program: &str, name: &str) -> Result<String, String> {
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
    let status = common::link_generated(&out_dir, &format!("{name}.c"), name);
    if !status.success() {
        return Err(format!("link failed: {status}"));
    }
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    String::from_utf8_lossy(&run.stdout)
        .lines()
        .find(|l| l.contains("tensor("))
        .map(|l| l.trim().to_string())
        .ok_or_else(|| "no tensor line".to_string())
}

/// The loss is sum(x * w) with w = OP(int64 weights) cast to f32; the
/// analytic gradient w.r.t. x is w itself. `print_form` toggles the
/// eval-shaped `print(...)` binding vs the compiled-shaped bare binding.
fn grad_program(weight_op: &str, print_form: bool) -> String {
    let out = if print_form {
        "out = print(compute_grad(to_tensor([0.1, 0.2, 0.3, 0.4])))"
    } else {
        "out = compute_grad(to_tensor([0.1, 0.2, 0.3, 0.4]))"
    };
    format!(
        "def g(x: tensor[4, f32]) -> tensor[f32] = {{\n\
           w = cast({weight_op}(to_tensor([cast(-100, int64), cast(200, int64), \
         cast(-300, int64), cast(400, int64)])), f32)\n\
           sum(mul(copy(x), w), 0)\n\
         }}\n\
         def compute_grad(x: tensor[4, f32]) -> tensor[4, f32] = grad(g, wrt=x)(x)\n\
         {out}\n"
    )
}

// ===========================================================================
// chelis#722 - evaluator support; the compiled half waits for Phase 3
// ===========================================================================

/// The correct gradient is `w = abs(weights) = [100, 200, 300, 400]`.
#[test]
fn eval_grad_through_int_abs_is_the_true_gradient() {
    let line = eval_first_line(&grad_program("abs", true)).expect("eval should run");
    assert!(
        line.contains("data=[100.0, 200.0, 300.0, 400.0]"),
        "grad of sum(x*w) wrt x must be w = abs(weights); got: {line}"
    );
}

/// Observed today: `[0.0, 0.0, 0.0, 0.0]` from the compiled binary too -
/// both lanes agree on the wrong answer, invisible to any cross-lane oracle.
#[test]
#[ignore = "chelis#722; since chelis#730 Phase 1 the placeholder is a LOUD lowering error \
            in both lanes (see grad_through_int_abs_fails_loudly_not_zero) - red for a \
            better reason until chelis#729 lands integer abs/floor. Original finding: grad through abs(int64 tensor) returns zeros in the compiled lane \
            as well; correct gradient is [100, 200, 300, 400]. Run with \
            `cargo test -p chelis-cli --test grad_zero_placeholder_matrix -- --ignored`."]
fn c_grad_through_int_abs_is_the_true_gradient() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let line = c_first_line(&grad_program("abs", false), "grad_abs_int").expect("C lane");
    assert!(
        line.contains("data=[100.0, 200.0, 300.0, 400.0]"),
        "grad of sum(x*w) wrt x must be w = abs(weights); got: {line}"
    );
}

/// `floor` on an already-integral int64 tensor is the identity, so the true
/// gradient is the raw weights.
#[test]
fn eval_grad_through_int_floor_is_the_true_gradient() {
    let line = eval_first_line(&grad_program("floor", true)).expect("eval should run");
    assert!(
        line.contains("data=[-100.0, 200.0, -300.0, 400.0]"),
        "grad of sum(x*w) wrt x must be w = floor(weights) = weights; got: {line}"
    );
}

/// `ceil` and `round` have the same exact identity semantics on integers as
/// `floor`; neither may route through a float kernel or fabricate a gradient.
#[test]
fn eval_grad_through_int_ceil_and_round_is_the_true_gradient() {
    for op in ["ceil", "round"] {
        let line = eval_first_line(&grad_program(op, true)).expect("eval should run");
        assert!(
            line.contains("data=[-100.0, 200.0, -300.0, 400.0]"),
            "grad of sum(x*{op}(w)) wrt x must be the exact integer weights; got: {line}"
        );
    }
}

/// Integer `abs` is not merely an identity rewrite: the minimum signed value
/// has no positive representative and must retain the [04-NUM-9] trap.
#[test]
fn eval_int64_abs_min_traps_instead_of_rounding_or_wrapping() {
    let program = "def int_abs(x: tensor[1, int64]) -> tensor[1, int64] = abs(x)\n\
                   out = print(int_abs(to_tensor([add(neg(cast(9223372036854775807, int64)), \
                   cast(-1, int64))])))\n";
    let err = eval_first_line(program).expect_err("abs(int64::MIN) must trap");
    assert!(
        err.contains("numeric trap: overflow in abs at int64"),
        "integer abs must preserve the exact overflow trap; got: {err}"
    );
}

// ===========================================================================
// chelis#722 compiled half: remains loud until Phase 3 + chelis#699
// ===========================================================================

/// Phase 2 intentionally enables the evaluator only. The compiled lane must
/// remain loud until Phase 3 supplies an exact integer kernel; emitting the
/// current float-only backend `fabsf` template would be a wrong answer.
#[test]
fn c_grad_through_int_abs_remains_loud_until_phase3() {
    if c_toolchain_available() {
        let err = c_first_line(&grad_program("abs", false), "grad_abs_int_loud")
            .expect_err("the compiled lane must reject integer abs");
        assert!(
            err.contains("unsupported:"),
            "the build-lane failure must carry the branded diagnostic; got: {err}"
        );
    }
}

// ===========================================================================
// CONTROLS (pass today; they isolate the trigger)
// ===========================================================================

/// The forward COMPUTATION without grad is correct in eval - re-authored
/// by chelis#730 Phase 1 (B2.5 adjudication). The original control used
/// the def-rooted program, whose eval output ALSO carried a fabricated
/// `g = tensor([0.0, ...])` trailing labeled root (the eval pipeline
/// lowers every tensor-signature def as a DAG root; the chelis#699
/// placeholder zeroed it silently). Two honest halves now:
/// the inline host-runtime forward computes 300.0, and the def-rooted
/// def-rooted program must now compute the same value instead of fabricating
/// a trailing zero root or failing during DAG lowering.
#[test]
fn forward_pass_without_grad_is_correct_in_eval() {
    let inline = "out = print(sum(mul(to_tensor([0.1, 0.2, 0.3, 0.4]), \
                  cast(abs(cast(to_tensor([-100.0, 200.0, -300.0, 400.0]), int64)), \
                  f32)), 0))\n";
    let line = eval_first_line(inline).expect("the host-runtime forward must evaluate");
    // chelis#732 P1 ([05-OBS-4]): the rank-0 result renders bare.
    assert_eq!(line, "300.0", "sum(x * abs(w)) must be 300.0; got: {line}");

    let def_rooted = "def g(x: tensor[4, f32]) -> tensor[f32] = {\n\
           w = cast(abs(to_tensor([cast(-100, int64), cast(200, int64), \
         cast(-300, int64), cast(400, int64)])), f32)\n\
           sum(mul(copy(x), w), 0)\n\
         }\n\
         out = print(g(to_tensor([0.1, 0.2, 0.3, 0.4])))\n";
    let line = eval_first_line(def_rooted).expect("the def-rooted program must evaluate");
    assert_eq!(
        line, "300.0",
        "the def-rooted evaluator path must preserve integer abs; got: {line}"
    );
}

/// The same gradient WITHOUT abs is correct in BOTH lanes: int64 weights,
/// the cast, grad itself, and the mul/sum adjoints are all fine. Any fix
/// for #722/#699 must keep this row green.
#[test]
fn grad_without_abs_is_correct_in_both_lanes() {
    let eval_program = "def g(x: tensor[4, f32]) -> tensor[f32] = {\n\
           w = cast(to_tensor([cast(-100, int64), cast(200, int64), \
         cast(-300, int64), cast(400, int64)]), f32)\n\
           sum(mul(copy(x), w), 0)\n\
         }\n\
         def compute_grad(x: tensor[4, f32]) -> tensor[4, f32] = grad(g, wrt=x)(x)\n\
         out = print(compute_grad(to_tensor([0.1, 0.2, 0.3, 0.4])))\n";
    let line = eval_first_line(eval_program).expect("eval should run");
    assert!(
        line.contains("data=[-100.0, 200.0, -300.0, 400.0]"),
        "grad of sum(x*w) wrt x = w; got: {line}"
    );
    if c_toolchain_available() {
        let c_program = eval_program.replace(
            "out = print(compute_grad(to_tensor([0.1, 0.2, 0.3, 0.4])))",
            "out = compute_grad(to_tensor([0.1, 0.2, 0.3, 0.4]))",
        );
        let line = c_first_line(&c_program, "grad_noabs").expect("C lane");
        assert!(
            line.contains("data=[-100.0, 200.0, -300.0, 400.0]"),
            "grad of sum(x*w) wrt x = w in the compiled lane; got: {line}"
        );
    }
}

// ===========================================================================
// chelis#856 - the `fail` placeholder's MESSAGE must not reach numeric IR
// ===========================================================================

/// The `fail` lowering arm (chelis#616) plants a masked zero `Const` at the
/// branch's rank and used to lower its own arguments for effect, discarding
/// the results. Those arguments are STRINGS, so the discarded work pushed a
/// `Prim::String` literal through `lower_lit`. That was harmless while a
/// constant was a bare `f64` (it smuggled `Const { value: 0.0 }` typed
/// `string` into the DAG); once chelis#856 sealed the payloads it reached
/// `finalize_scalar`'s `Prim::String` arm, which is an
/// unreachable-by-construction `panic!`, and every `grad`/`vmap` over a
/// `fail`-guarded function died with the bare panic string
/// "finalize_scalar: string is not a numeric dtype ...". The message is
/// host-lane data and is no longer lowered at all.
///
/// `sum` has gradient 1 everywhere; the 4-element input does not take the
/// guard.
#[test]
fn issue_856_grad_through_fail_guarded_branch_lowers() {
    let program = "def loss(x: tensor[4, f32]) -> f32 = \
         if gt(cast(2, int64), cast(shape(x, cast(0, int32)), int64)) \
         then fail(\"kernel exceeds input length\") \
         else tensor_to_scalar(sum(x, cast(0, int32)))\n\
         out = grad(loss)(to_tensor([cast(1.0, f32), cast(2.0, f32), \
         cast(3.0, f32), cast(4.0, f32)]))\n";
    let line =
        eval_first_line(program).expect("grad over a `fail`-guarded body must lower (chelis#856)");
    assert!(
        line.contains("data=[1.0, 1.0, 1.0, 1.0]"),
        "grad of sum(x) wrt x is 1 everywhere; got: {line}"
    );
}

/// Negative parity for the row above: dropping the message from the DAG
/// lane must not drop it from the program. When the guard IS taken the
/// host lane still owns `fail` and still reports the exact message.
#[test]
fn issue_856_fail_message_survives_when_the_guard_is_taken() {
    let program = "def loss(x: tensor[1, f32]) -> f32 = \
         if gt(cast(2, int64), cast(shape(x, cast(0, int32)), int64)) \
         then fail(\"kernel exceeds input length\") \
         else tensor_to_scalar(sum(x, cast(0, int32)))\n\
         out = loss(to_tensor([cast(1.0, f32)]))\n";
    let err = eval_first_line(program).expect_err("the taken `fail` branch must abort");
    assert!(
        err.contains("kernel exceeds input length"),
        "the taken `fail` must report its own message; got: {err}"
    );
}
