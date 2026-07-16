//! chelis#717 - the host evaluator's tensor lane narrows results to the
//! element dtype per-op inconsistently, and is wrong in three directions:
//!
//! 1. **f64 tensors are destroyed to f32** by every unary float op:
//!    `tensor_float_unop_f32` (`crates/chelis-compiler-api/src/runtime/
//!    host_ops.rs:518-544`) routes each element through `op(x as f32) as
//!    f64` regardless of the tensor's dtype. Its doc comment says it exists
//!    to stay byte-identical with the C backend's f32 helpers - the right
//!    goal for f32 tensors, silently destructive for f64 ones. The C lane
//!    emits genuine `double` math for f64 tensors, so the two lanes applied
//!    "match the other lane" in opposite directions.
//! 2. **f32 tensors skip narrowing on add/div/recip** (results are raw f64
//!    values not representable in f32), while tan/sqrt/sin do narrow.
//! 3. **f16/bf16 tensors are never rounded at all** (covered in
//!    narrow_dtype_matrix.rs; this file carries the f32/f64 rows).
//!
//! The compiled C lane is correct in every cell of this file, so each
//! broken row is also a cross-lane divergence in the direction
//! eval-wrong / C-right - the same direction as chelis#691, opposite to
//! chelis#680.
//!
//! Why nothing caught it: `eval_agreement.rs` compares lanes through f64
//! with a tolerance (chelis#687), and every f32/f64 row here is a relative
//! error of 1e-8 .. 1e-16 - exactly what a tolerance oracle ignores.
//!
//! Tests asserting correct behavior that fail today are `#[ignore]`d with
//! the issue number, the observed wrong value, and the run command. The
//! C-lane cells are locked as passing controls so a fix cannot regress the
//! correct lane.

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

/// Build + link + run through the C lane; first stdout line verbatim.
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
    Ok(String::from_utf8_lossy(&run.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string())
}

const F64_UNOP: &str = "module M.Main\n\
     def f(x: tensor[2, f64]) -> tensor[2, f64] = OP(x)\n\
     out = print(f(to_tensor([cast(A, f64), cast(B, f64)])))\n";

fn f64_unop_program(op: &str, a: &str, b: &str) -> String {
    F64_UNOP.replace("OP", op).replace('A', a).replace('B', b)
}

// ===========================================================================
// CONTROLS: the C lane is exact in every broken cell, and the parts of eval
// that are correct must stay correct.
// ===========================================================================

/// The C lane computes f64 tensor transcendentals at genuine f64 precision.
/// These exact strings are what eval must ALSO produce once #717 is fixed.
#[test]
fn c_f64_tensor_unary_ops_are_f64_precise() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let line = c_first_line(&f64_unop_program("tan", "1.5", "3.0"), "c_f64_tan")
        .expect("C lane should run");
    assert!(
        line.contains("14.10141994717172"),
        "C f64 tan(1.5) must be the true f64 value; got: {line}"
    );
    let line = c_first_line(&f64_unop_program("sqrt", "2.0", "3.0"), "c_f64_sqrt")
        .expect("C lane should run");
    assert!(
        line.contains("1.414213562373095"),
        "C f64 sqrt(2) must be f64-precise; got: {line}"
    );
}

/// f64 tensor binary ops are exact in BOTH lanes (the binary path does not
/// take the f32 wrapper). Bounds #717 to the unary path on f64.
#[test]
fn f64_tensor_div_is_exact_in_both_lanes() {
    let program = "module M.Main\n\
         def f(x: tensor[2, f64], y: tensor[2, f64]) -> tensor[2, f64] = div(x, y)\n\
         out = print(f(to_tensor([cast(1.0, f64), cast(2.0, f64)]), \
         to_tensor([cast(3.0, f64), cast(3.0, f64)])))\n";
    let eval_line = eval_first_line(program).expect("eval should run");
    assert!(
        eval_line.contains("0.3333333333333333"),
        "eval f64 div must be exact f64; got: {eval_line}"
    );
    if c_toolchain_available() {
        let c_line = c_first_line(program, "f64_div_both").expect("C lane should run");
        assert!(
            c_line.contains("0.3333333333333333"),
            "C f64 div must be exact f64; got: {c_line}"
        );
    }
}

/// The C lane rounds f32 tensor arithmetic through f32, correctly.
/// f32(0.1) + f32(0.2) = 0.30000001192092896 exactly.
#[test]
fn c_f32_tensor_add_rounds_to_f32() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let line = c_first_line(
        "module M.Main\n\
         def f(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[2, f32] = add(x, y)\n\
         out = print(f(to_tensor([0.1, 1.0]), to_tensor([0.2, 2.0])))\n",
        "c_f32_add",
    )
    .expect("C lane should run");
    assert!(
        line.contains("0.300000011920929"),
        "C f32 add(0.1, 0.2) must be the f32 sum; got: {line}"
    );
}

// ===========================================================================
// chelis#717 - f64 tensors destroyed to f32 by unary ops (eval)
// ===========================================================================

/// Observed today: eval prints 14.101419448852539 - the f32-precision tan -
/// for an f64 tensor. Eight significant digits of a declared-f64 value are
/// silently gone.
#[test]
#[ignore = "chelis#717: eval computes f64 tensor tan through f32 (tensor_float_unop_f32) \
            and prints 14.101419448852539; the f64 answer is 14.10141994717172 and the \
            compiled C lane produces it. Run with \
            `cargo test -p chelis-cli --test eval_tensor_narrowing_matrix -- --ignored`."]
fn eval_f64_tensor_tan_keeps_f64_precision() {
    let line = eval_first_line(&f64_unop_program("tan", "1.5", "3.0")).expect("eval should run");
    assert!(
        line.contains("14.10141994717172"),
        "f64 tensor tan must be f64-precise; got: {line}"
    );
}

/// Observed today: 1.4142135381698608 = f32(sqrt(2)).
#[test]
#[ignore = "chelis#717: eval computes f64 tensor sqrt through f32 and prints \
            1.4142135381698608 (= f32(sqrt 2)); the f64 answer is 1.4142135623730951. Run \
            with `cargo test -p chelis-cli --test eval_tensor_narrowing_matrix -- --ignored`."]
fn eval_f64_tensor_sqrt_keeps_f64_precision() {
    let line = eval_first_line(&f64_unop_program("sqrt", "2.0", "3.0")).expect("eval should run");
    assert!(
        line.contains("1.4142135623730951"),
        "f64 tensor sqrt must be f64-precise; got: {line}"
    );
}

/// Observed today: 7.389056205749512 = f32(exp(2)).
#[test]
#[ignore = "chelis#717: eval computes f64 tensor exp through f32 and prints \
            7.389056205749512; the f64 answer is 7.38905609893065. Run with \
            `cargo test -p chelis-cli --test eval_tensor_narrowing_matrix -- --ignored`."]
fn eval_f64_tensor_exp_keeps_f64_precision() {
    let line = eval_first_line(&f64_unop_program("exp", "2.0", "3.0")).expect("eval should run");
    assert!(
        line.contains("7.38905609893065"),
        "f64 tensor exp must be f64-precise; got: {line}"
    );
}

// ===========================================================================
// chelis#717 - f32 tensors: add/div/recip skip the f32 rounding (eval)
// ===========================================================================

/// Observed today: eval prints 0.30000000447034836, a value that does not
/// exist in f32 (it is the f64 sum of the two f32 inputs, never rounded).
/// Computing in f64 and rounding once IS correctly rounded for f32 - eval
/// just skips the rounding step for add.
#[test]
#[ignore = "chelis#717: eval f32 tensor add(0.1, 0.2) prints 0.30000000447034836, not an \
            f32 value; correct f32 sum is 0.30000001192092896 and the C lane produces it. \
            Run with `cargo test -p chelis-cli --test eval_tensor_narrowing_matrix -- --ignored`."]
fn eval_f32_tensor_add_rounds_to_f32() {
    let line = eval_first_line(
        "module M.Main\n\
         def f(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[2, f32] = add(x, y)\n\
         out = print(f(to_tensor([0.1, 1.0]), to_tensor([0.2, 2.0])))\n",
    )
    .expect("eval should run");
    assert!(
        line.contains("0.30000001192092896"),
        "f32 tensor add must round its result to f32; got: {line}"
    );
}

/// Observed today: 0.3333333333333333 (raw f64 quotient).
#[test]
#[ignore = "chelis#717: eval f32 tensor div(1, 3) prints 0.3333333333333333 (f64, not an \
            f32 value); correct f32 quotient is 0.3333333432674408. Run with \
            `cargo test -p chelis-cli --test eval_tensor_narrowing_matrix -- --ignored`."]
fn eval_f32_tensor_div_rounds_to_f32() {
    let line = eval_first_line(
        "module M.Main\n\
         def f(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[2, f32] = div(x, y)\n\
         out = print(f(to_tensor([1.0, 2.0]), to_tensor([3.0, 3.0])))\n",
    )
    .expect("eval should run");
    assert!(
        line.contains("0.3333333432674408"),
        "f32 tensor div must round its result to f32; got: {line}"
    );
}

/// Observed today: 0.6666666666666666 (raw f64). Note the irony: recip is on
/// the very op list #695 names as living on the lossy f64 helpers.
#[test]
#[ignore = "chelis#717: eval f32 tensor recip(1.5) prints 0.6666666666666666 (f64, not an \
            f32 value); correct f32 value is 0.6666666865348816. Run with \
            `cargo test -p chelis-cli --test eval_tensor_narrowing_matrix -- --ignored`."]
fn eval_f32_tensor_recip_rounds_to_f32() {
    let line = eval_first_line(
        "module M.Main\n\
         def f(x: tensor[2, f32]) -> tensor[2, f32] = recip(x)\n\
         out = print(f(to_tensor([1.5, 3.0])))\n",
    )
    .expect("eval should run");
    assert!(
        line.contains("0.6666666865348816"),
        "f32 tensor recip must round its result to f32; got: {line}"
    );
}
