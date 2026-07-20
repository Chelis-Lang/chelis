//! chelis#722 - grad of a forward pass containing `abs`/`floor` on an
//! integer tensor returns ALL-ZERO gradients in BOTH lanes.
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
// chelis#722 - the zero gradients (both lanes)
// ===========================================================================

/// Observed today: `[0.0, 0.0, 0.0, 0.0]` from eval. The correct gradient is
/// `w = abs(weights) = [100, 200, 300, 400]`.
#[test]
#[ignore = "chelis#722; since chelis#730 Phase 1 the placeholder is a LOUD lowering error \
            in both lanes (see grad_through_int_abs_fails_loudly_not_zero) - red for a \
            better reason until chelis#729 lands integer abs/floor. Original finding: grad through abs(int64 tensor) returns zeros in eval (the #699 \
            placeholder poisons the grad-lowered forward pass); correct gradient is \
            [100, 200, 300, 400]. Run with \
            `cargo test -p chelis-cli --test grad_zero_placeholder_matrix -- --ignored`."]
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

/// floor on an already-integral int64 tensor is the identity, so the true
/// gradient is the raw weights. Observed today: zeros in eval.
#[test]
#[ignore = "chelis#722; since chelis#730 Phase 1 the placeholder is a LOUD lowering error \
            in both lanes (see grad_through_int_abs_fails_loudly_not_zero) - red for a \
            better reason until chelis#729 lands integer abs/floor. Original finding: grad through floor(int64 tensor) returns zeros in eval (same \
            placeholder as abs, per #699's op list); correct gradient is \
            [-100, 200, -300, 400]. Run with \
            `cargo test -p chelis-cli --test grad_zero_placeholder_matrix -- --ignored`."]
fn eval_grad_through_int_floor_is_the_true_gradient() {
    let line = eval_first_line(&grad_program("floor", true)).expect("eval should run");
    assert!(
        line.contains("data=[-100.0, 200.0, -300.0, 400.0]"),
        "grad of sum(x*w) wrt x must be w = floor(weights) = weights; got: {line}"
    );
}

// ===========================================================================
// chelis#730 Phase 1 (census row 1's grad half): loud, not zero
// ===========================================================================

/// The conversion's parity row: grad through `abs`/`floor` on an int64
/// tensor now fails LOUDLY in both lanes with the branded diagnostic -
/// never plausible zero gradients. Flips to the value tests above when
/// chelis#729 lands integer abs/floor support.
#[test]
fn grad_through_int_abs_fails_loudly_not_zero() {
    let err = eval_first_line(&grad_program("abs", true))
        .expect_err("chelis#722: grad through abs(int64) must fail loudly, not zero");
    assert!(
        err.contains("unsupported:"),
        "the eval-lane failure must carry the branded diagnostic; got: {err}"
    );
    if c_toolchain_available() {
        let err = c_first_line(&grad_program("abs", false), "grad_abs_int_loud")
            .expect_err("the compiled lane must reject the same program");
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
/// program fails LOUDLY with the branded diagnostic instead of printing
/// a correct first line above a fabricated zero root.
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
    let err = eval_first_line(def_rooted)
        .expect_err("the def-rooted program must fail loudly, never print a fabricated root");
    assert!(
        err.contains("unsupported:"),
        "the failure must carry the branded diagnostic; got: {err}"
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
