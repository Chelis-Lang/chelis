//! #1741: checked scalar gradients must also execute in eval and property fuzzing.
use serde_json::Value;
use std::process::Command;

#[path = "common/mod.rs"]
mod common;

const PROPERTY: &str = "module M\n@property exp_grad_positive forall(x: f32)\nwhere x > 0.5, x < 9.5:\n  (grad(fn (xx: f32) -> exp(xx), wrt=xx)(x) > 0.0)\n";

#[test]
fn original_scalar_gradient_property_passes_auto_and_fuzz() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("property.ch");
    std::fs::write(&path, PROPERTY).unwrap();
    for tier in ["auto", "fuzz-only"] {
        let output = Command::new(assert_cmd::cargo_bin!("chelis"))
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["prove", path.to_str().unwrap(), "--tier", tier])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{tier}: {} {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout)
                .contains("1 passed, 0 failed, 0 unsupported, 0 errors"),
            "{output:?}"
        );
    }
}

#[test]
fn check_and_eval_agree_for_scalar_gradients_and_reject_real_mixed_surfaces() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gradient.ch");
    for dtype in ["f32", "f64"] {
        std::fs::write(&path, format!("def g(x: {dtype}) -> bool = gt(grad(fn (xx: {dtype}) -> exp(xx), wrt=xx)(x), 0.0{dtype})\nout = g(1.0{dtype})\n")).unwrap();
        for args in [
            vec!["check", path.to_str().unwrap()],
            vec!["eval", "--file", path.to_str().unwrap()],
        ] {
            let output = Command::new(assert_cmd::cargo_bin!("chelis"))
                .env("CHELIS_STYLE_GATE_DISABLE", "1")
                .args(&args)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{args:?}: {} {}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            if args[0] == "eval" {
                assert_eq!(
                    String::from_utf8(output.stdout).unwrap().trim(),
                    "out = true"
                );
            }
        }
    }
    for tensor in ["scalar_to_tensor(1.0f32)", "to_tensor([1.0f32])"] {
        std::fs::write(&path, format!("out = gt({tensor}, 0.0f32)\n")).unwrap();
        let output = Command::new(assert_cmd::cargo_bin!("chelis"))
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["check", path.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{output:?}");
    }
}

// chelis#2993 and chelis#3017: `chelis build` computes a top-level scalar
// `grad` at the operand dtype ([04-NUM-8]) through the same reverse-mode DAG
// `chelis eval` evaluates (spec/06 section 2.3), so the executable prints
// what eval prints. `eval --json` reports the operand dtype, and the emitted
// gradient code for an f32 program holds no `double` intermediate and no f64
// dtype tag.

/// The #2993 oracle bodies, each `f: T -> T`; `{t}` is the dtype.
const BODIES: [&str; 4] = ["mul(exp(x), x)", "div(1.0{t}, x)", "log(x)", "sqrt(x)"];

/// Bodies built only from correctly rounded operations, so a lane mismatch
/// can only come from the differentiation order (#3017's witnesses first).
/// The transcendentals are correctly rounded in both lanes ([05-OP-46],
/// chelis#2957), so bodies through them are bit-exact too.
const EXACT_BODIES: [&str; 13] = [
    "add(mul(x, x), x)",
    "mul(x, sqrt(x))",
    "sqrt(sqrt(x))",
    "mul(x, x)",
    "div(x, add(x, 1.0{t}))",
    "div(sqrt(x), add(x, 1.0{t}))",
    "sqrt(sqrt(sqrt(x)))",
    "div(1.0{t}, x)",
    "mul(exp(x), log(x))",
    "div(sin(x), add(cos(x), 2.0{t}))",
    "exp(neg(mul(x, x)))",
    "tanh(mul(x, sin(x)))",
    "log(add(x, 1.0{t}))",
];

fn width_program(dtype: &str, body: &str, root: &str) -> String {
    let body = body.replace("{t}", dtype);
    let root = root.replace("{t}", dtype);
    format!("module Probe.Case\ndef f(x: {dtype}) -> {dtype} = {body}\nout = {root}\n")
}

/// One program printing `grad(f)` at the 24 inputs 0.1, 0.237, ..., 3.251.
fn sweep_program(dtype: &str, body: &str) -> String {
    let body = body.replace("{t}", dtype);
    let mut source = format!("module Probe.Case\ndef f(x: {dtype}) -> {dtype} = {body}\n");
    for index in 0..24 {
        let input = 0.1 + 0.137 * f64::from(index);
        source.push_str(&format!("r{index} = print(grad(f)({input:.3}{dtype}))\n"));
    }
    source
}

fn chelis(args: &[&str]) -> std::process::Output {
    Command::new(assert_cmd::cargo_bin!("chelis"))
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(args)
        .output()
        .unwrap()
}

fn eval_stdout(source: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("p.ch");
    common::write_file(&path, source);
    let output = chelis(&["eval", "--file", path.to_str().unwrap()]);
    assert!(output.status.success(), "{source}\n{output:?}");
    String::from_utf8(output.stdout).unwrap()
}

fn eval_dtype(source: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("p.ch");
    common::write_file(&path, source);
    let output = chelis(&["eval", "--json", "--file", path.to_str().unwrap()]);
    assert!(output.status.success(), "{source}\n{output:?}");
    let json: Value = serde_json::from_slice(&output.stdout).expect("eval JSON");
    let root = json["roots"]
        .as_array()
        .expect("roots")
        .iter()
        .find(|root| root["name"] == "out")
        .unwrap_or_else(|| panic!("an `out` root: {json}"))
        .clone();
    root["value"]["value"]["dtype"]
        .as_str()
        .unwrap_or_else(|| panic!("a scalar dtype: {root}"))
        .to_string()
}

/// The emitted C translation unit for `source`.
fn emitted_c(source: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("p.ch");
    let out = dir.path().join("out");
    common::write_file(&path, source);
    let output = chelis(&[
        "build",
        path.to_str().unwrap(),
        "--target",
        "c",
        "--emit-c",
        "--output",
        out.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "{source}\n{output:?}");
    std::fs::read_to_string(out.join("p.c")).expect("emitted p.c")
}

/// One emitted C definition, from its signature to its closing brace.
fn definition<'a>(emitted: &'a str, name: &str) -> &'a str {
    let body = common::host_body_definition(emitted, name);
    let end = body.find("\n}\n").map_or(body.len(), |end| end + 2);
    &body[..end]
}

/// The code that computes and prints `out`: the emitted `main` and every
/// top-level tensor helper kernel (the reverse-DAG gradient).
fn gradient_code(emitted: &str) -> Vec<&str> {
    let mut code = vec![definition(emitted, "main")];
    let mut index = 0;
    while emitted.contains(&format!("p__global__tensor_{index}__private(")) {
        code.push(definition(
            emitted,
            &format!("p__global__tensor_{index}__private"),
        ));
        index += 1;
    }
    assert!(
        code.len() > 1,
        "a scalar gradient lowers to a tensor helper:\n{emitted}"
    );
    code
}

#[test]
fn executable_prints_what_eval_prints_over_the_exact_sweep() {
    for dtype in ["f32", "f64"] {
        for body in EXACT_BODIES {
            let source = sweep_program(dtype, body);
            let eval = eval_stdout(&source);
            assert_eq!(eval.lines().count(), 48, "{source}: 24 prints and roots");
            let built = common::build_and_run(&source, "p");
            assert_eq!(built, eval, "{source}");
        }
    }
}

#[test]
fn executable_prints_what_eval_prints_at_the_narrow_widths() {
    for dtype in ["f16", "bf16"] {
        for body in EXACT_BODIES {
            let source = width_program(dtype, body, "print(grad(f)(0.7{t}))");
            let eval = eval_stdout(&source);
            let built = common::build_and_run(&source, "p");
            assert_eq!(built, eval, "{source}");
        }
    }
}

#[test]
fn eval_reports_the_operand_dtype() {
    for (dtype, tag) in [("f32", "f32"), ("f64", "f64")] {
        for body in BODIES {
            let source = width_program(dtype, body, "grad(f)(0.7{t})");
            assert_eq!(eval_dtype(&source), tag, "{source}");
        }
    }
}

#[test]
fn emitted_f32_gradient_has_no_double_intermediate_or_f64_tag() {
    for body in BODIES {
        let source = width_program("f32", body, "print(grad(f)(0.7{t}))");
        let emitted = emitted_c(&source);
        for code in gradient_code(&emitted) {
            for forbidden in ["double", "CHELIS_DTYPE_F64", "chelis_f64_from_bits"] {
                assert!(
                    !code.contains(forbidden),
                    "{source}: `{forbidden}` in the emitted gradient code:\n{code}"
                );
            }
        }
        assert!(
            gradient_code(&emitted)
                .iter()
                .all(|code| code.contains("CHELIS_DTYPE_F32")),
            "{source}"
        );
    }
    // Control: the f64 program does compute in double.
    let source = width_program("f64", BODIES[0], "print(grad(f)(0.7{t}))");
    let emitted = emitted_c(&source);
    assert!(
        gradient_code(&emitted)
            .iter()
            .all(|code| code.contains("CHELIS_DTYPE_F64")),
        "{source}"
    );
}
