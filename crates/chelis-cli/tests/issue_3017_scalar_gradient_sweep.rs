//! chelis#3017 nightly sweep: every exact gradient body over #3017's 24-input
//! sweep at f32 and f64, and at 0.7 at f16 and bf16, one `chelis build` per
//! body and dtype. The standing canary `issue_1741_scalar_gradient_cli` keeps
//! #3017's named witnesses at every float width on every pull request; this
//! complete matrix is a target exclusion owned by the unfiltered nightly
//! workspace job (`.config/ci-test-targets.toml`).
use std::process::Command;

#[path = "common/mod.rs"]
mod common;

include!("support/scalar_gradient_lanes.rs");

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
