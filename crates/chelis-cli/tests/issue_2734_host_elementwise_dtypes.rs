//! chelis#2734: an elementwise operation over a host-built tensor runs at
//! every dtype the spec admits for it.
//!
//! An operand such as `cast(to_tensor(range(..)), f16)` holds a run-time List
//! the tensor lane cannot represent, so the helper over the whole operation
//! failed and the operation fell back to the C host lane's own elementwise
//! loops. Those cover f32, f64, i32, i64 and bool for arithmetic and only f32
//! for unary functions: f16, bf16, i8 and i16 arithmetic and every non-f32
//! unary function built and then aborted at run time, and `cos`, `floor` and
//! their neighbours were refused at every dtype. Host lowering now keeps an
//! operation on the host loop only where that loop covers its dtype; any
//! other one evaluates the operand on the host and runs as a helper over its
//! value, so it uses the tensor lane's kernel for its dtype.
//!
//! The positive oracle: each dtype's program builds, runs, and prints what
//! `chelis eval` prints. The negative twins: an integer overflow traps with
//! the same line on both lanes, and integer `div` is refused before either
//! lane runs.
use assert_cmd::Command;
use std::process::Command as StdCommand;

#[path = "common/mod.rs"]
mod common;
use common::{build_and_run_app, make_app, write_file};

const LHS: &str = "to_tensor(range(1i64, 5i64))";
const RHS: &str = "to_tensor(range(2i64, 6i64))";

fn chelis(
    reef_home: &std::path::Path,
    app: &std::path::Path,
    args: &[&str],
) -> std::process::Output {
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app)
        .args(args)
        .output()
        .unwrap()
}

/// One root per operation, each over operands built inline from run-time
/// Lists at `dtype`.
fn program(dtype: &str, binary: &[&str], unary: &[&str]) -> String {
    let lhs = format!("cast({LHS}, {dtype})");
    let rhs = format!("cast({RHS}, {dtype})");
    let mut source = String::from("module Demo.Main\n");
    for op in binary {
        source.push_str(&format!("{op}_{dtype} = {op}({lhs}, {rhs})\n"));
    }
    for op in unary {
        source.push_str(&format!("{op}_{dtype} = {op}({lhs})\n"));
    }
    source
}

fn assert_compiled_matches_eval(name: &str, source: &str) {
    let (_dir, reef_home, app) = make_app(&format!("issue-2734-{name}"));
    let main = app.join("src/main.ch");
    write_file(&main, source);
    let evaluated = chelis(
        &reef_home,
        &app,
        &["eval", "--file", main.to_str().unwrap()],
    );
    assert!(
        evaluated.status.success(),
        "{name}: {}",
        String::from_utf8_lossy(&evaluated.stderr)
    );
    let expected = String::from_utf8(evaluated.stdout).unwrap();
    let compiled = build_and_run_app(&reef_home, &app, "main");
    assert_eq!(compiled, expected, "{name}:\n{source}");
}

const FLOAT_BINARY: &[&str] = &["add", "sub", "mul", "div", "max_elem", "min_elem"];
const FLOAT_UNARY: &[&str] = &[
    "exp", "log", "sin", "cos", "tan", "atan", "tanh", "sqrt", "relu", "floor", "ceil", "round",
    "recip", "abs", "neg",
];
/// Derived activations, which synthesize constants (section 3.3).
const ACTIVATIONS: &[&str] = &["sigmoid", "silu", "gelu"];
const INT_BINARY: &[&str] = &["add", "sub", "mul", "max_elem", "min_elem", "floor_div"];
const INT_UNARY: &[&str] = &["abs", "neg"];

// REGRESSION TEST. On `4bb166024` the f16, bf16, i8 and i16 arithmetic and
// the f64, f16 and bf16 unary functions aborted at run time with
// "unsupported dtype", and `cos`, `floor` and their neighbours were refused
// at build at every dtype.
#[test]
fn host_built_operands_run_at_every_float_dtype() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        assert_compiled_matches_eval(dtype, &program(dtype, FLOAT_BINARY, FLOAT_UNARY));
    }
}

#[test]
fn host_built_operands_run_at_every_integer_dtype() {
    for dtype in ["i8", "i16", "i32", "i64"] {
        assert_compiled_matches_eval(dtype, &program(dtype, INT_BINARY, INT_UNARY));
    }
}

/// The derived activations run on the C host lane's own f32 loop.
#[test]
fn host_built_operands_run_derived_activations_at_f32() {
    assert_compiled_matches_eval("activations-f32", &program("f32", &[], ACTIVATIONS));
}

/// At every other float dtype a derived activation runs in the tensor lane,
/// where its synthesized constants have no extent source over a run-time
/// extent yet: the build refuses before emission with chelis#1482's typed
/// receipt, never a program that aborts when run.
#[test]
fn a_derived_activation_off_the_host_loop_is_refused_before_emission() {
    for dtype in ["f16", "bf16", "f64"] {
        let (_dir, reef_home, app) = make_app(&format!("issue-2734-activation-{dtype}"));
        let main = app.join("src/main.ch");
        write_file(&main, &program(dtype, &[], &["sigmoid"]));
        let out = app.join("out");
        let built = chelis(
            &reef_home,
            &app,
            &[
                "build",
                main.to_str().unwrap(),
                "--target",
                "c",
                "--output",
                out.to_str().unwrap(),
            ],
        );
        let stderr = String::from_utf8_lossy(&built.stderr);
        assert_eq!(built.status.code(), Some(1), "{dtype}: {stderr}");
        assert!(
            stderr.contains("unimplemented chelis#1482") && stderr.contains("(codegen:c)"),
            "{dtype}: {stderr}"
        );
        assert!(
            !out.join("main").exists(),
            "{dtype}: no executable is emitted"
        );
    }
}

/// The issue's `def` witness: an f64 unary function over a parameter's List.
#[test]
fn an_f64_unary_function_over_a_list_parameter_runs() {
    assert_compiled_matches_eval(
        "def-exp",
        "module Demo.Main\ndef run(xs: List[f64]) -> List[f64] = to_list(exp(neg(to_tensor(xs))))\nresult = run([0.0f64, 1.0f64])\n",
    );
}

/// Negative twin: an integer result outside its width traps with the same
/// line on both lanes ([04-NUM-3]). i8 runs in the tensor lane's kernel; i32
/// and i64 stay on the C host lane's own loop, which wrapped silently before
/// its integer arms used checked arithmetic.
#[test]
fn an_integer_overflow_traps_on_both_lanes() {
    for (name, expression, trap) in [
        (
            "i8-add",
            "add(cast(to_tensor(range(100i64, 102i64)), i8), cast(to_tensor(range(100i64, 102i64)), i8))",
            "numeric trap: overflow in add at i8",
        ),
        (
            "i32-add",
            "add(cast(to_tensor(range(2147483646i64, 2147483648i64)), i32), cast(to_tensor(range(2147483646i64, 2147483648i64)), i32))",
            "numeric trap: overflow in add at i32",
        ),
        (
            "i64-mul",
            "mul(to_tensor(range(4611686018427387904i64, 4611686018427387906i64)), to_tensor(range(2i64, 4i64)))",
            "numeric trap: overflow in mul at i64",
        ),
        (
            "i32-neg",
            "neg(cast(to_tensor(range(-2147483648i64, -2147483646i64)), i32))",
            "numeric trap: overflow in neg at i32",
        ),
    ] {
        let (_dir, reef_home, app) = make_app(&format!("issue-2734-overflow-{name}"));
        let main = app.join("src/main.ch");
        write_file(&main, &format!("module Demo.Main\nx = {expression}\n"));
        let evaluated = chelis(
            &reef_home,
            &app,
            &["eval", "--file", main.to_str().unwrap()],
        );
        let stderr = String::from_utf8_lossy(&evaluated.stderr);
        assert!(!evaluated.status.success(), "{name}: eval accepted");
        assert!(stderr.contains(trap), "{name}: eval: {stderr}");
        let out = app.join("out");
        let built = chelis(
            &reef_home,
            &app,
            &[
                "build",
                main.to_str().unwrap(),
                "--target",
                "c",
                "--output",
                out.to_str().unwrap(),
            ],
        );
        assert!(
            built.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&built.stderr)
        );
        let ran = StdCommand::new(out.join("main")).output().unwrap();
        let stderr = String::from_utf8_lossy(&ran.stderr);
        assert!(!ran.status.success(), "{name}: the executable accepted");
        assert!(stderr.contains(trap), "{name}: executable: {stderr}");
    }
}

/// Negative twin: `div` is float-only, so an integer `div` over host-built
/// operands is refused by the checker on both lanes before anything runs.
#[test]
fn an_integer_div_is_refused_on_both_lanes() {
    let (_dir, reef_home, app) = make_app("issue-2734-int-div");
    let main = app.join("src/main.ch");
    write_file(&main, &program("i16", &["div"], &[]));
    let out = app.join("out");
    for args in [
        vec!["eval", "--file", main.to_str().unwrap()],
        vec![
            "build",
            main.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out.to_str().unwrap(),
        ],
    ] {
        let output = chelis(&reef_home, &app, &args);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{args:?} accepted integer div");
        assert!(stderr.contains("`div` is float-only"), "{args:?}: {stderr}");
    }
}
