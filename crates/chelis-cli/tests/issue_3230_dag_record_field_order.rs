//! chelis#3230: a positional pattern binds a record's DECLARED fields
//! ([04-PAT-3]) in every lane, including the reverse-mode DAG `grad`
//! differentiates, whatever order the record literal wrote its fields in.
//!
//! The DAG lane stored record slots in written order and matched positional
//! sub-patterns against them by slot, so a literal written out of declared
//! order bound the wrong fields: the forward value (host lane) stayed right
//! while the gradient differentiated a different function. Every test runs
//! `eval --file` and `build --target c`, executes the built program, and
//! requires both lanes to print exactly the expected forward values and
//! gradients. Declared-order literals, positional constructor application,
//! and by-name record patterns are the controls.

#[path = "common/mod.rs"]
mod common;

use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

const PAIR: &str = "type P =\n  | P { w: tensor[2, f32], b: tensor[2, f32] }\n";
const TRIPLE: &str = "type Q =\n  | Q { a: tensor[2, f32], b: tensor[2, f32], c: tensor[2, f32] }\n\
     def combine(p: tensor[2, f32], q: tensor[2, f32], r: tensor[2, f32]) -> f32 = {\n  \
     pq = mul(&p, &q)\n  tensor_to_scalar(sum(sub(&pq, &r), 0i32))\n}\n";

fn eval(source: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("record_order.ch");
    fs::write(&path, source).expect("source");
    let output = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("eval runs");
    assert!(
        output.status.success(),
        "eval rejected: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("UTF-8")
}

/// Both lanes must print exactly `expected`, one `name = value` per line.
fn assert_lanes_print(source: &str, name: &str, expected: &[&str]) {
    let interpreted = eval(source);
    assert_eq!(
        interpreted.lines().collect::<Vec<_>>(),
        expected,
        "{name}: eval output"
    );
    let native = common::build_and_run(source, name);
    assert_eq!(
        native.lines().collect::<Vec<_>>(),
        expected,
        "{name}: compiled C output"
    );
}

/// `f(x) = sum(w - b)` over a `P` literal whose fields are `fields`, as
/// written, evaluated forward and differentiated at `[2, 3]`.
fn pair_program(fields: &str) -> String {
    format!(
        "{PAIR}def f(x: tensor[2, f32]) -> f32 = match P {{ {fields} }} with {{\n  \
         | P(w, b) => tensor_to_scalar(sum(sub(&w, &b), 0i32))\n}}\n\
         input = to_tensor([2.0f32, 3.0f32])\nforward = f(input)\n\
         out = grad(f)(to_tensor([2.0f32, 3.0f32]))\n"
    )
}

#[test]
fn the_issue_witness_differentiates_the_declared_fields() {
    // `b` is written before `w`; the declared order is `w, b`. With `w = x`
    // and `b = x * x`: forward `-8`, gradient `1 - 2x = [-3, -5]`. Binding by
    // written order differentiates `x * x - x`, giving `[3, 5]`.
    assert_lanes_print(
        &pair_program("b: mul(&x, &x), w: x"),
        "witness",
        &[
            "input = tensor(shape=[2], data=[2.0, 3.0])",
            "forward = -8.0",
            "out = tensor(shape=[2], data=[-3.0, -5.0])",
        ],
    );
}

#[test]
fn a_declared_order_literal_keeps_its_result() {
    // `w = 2x`, `b = x * x`, written in declared order: forward `-3`,
    // gradient `2 - 2x = [-2, -4]`.
    assert_lanes_print(
        &pair_program("w: add(&x, &x), b: mul(&x, &x)"),
        "declared_pair",
        &[
            "input = tensor(shape=[2], data=[2.0, 3.0])",
            "forward = -3.0",
            "out = tensor(shape=[2], data=[-2.0, -4.0])",
        ],
    );
}

#[test]
fn every_written_order_of_a_three_field_record_binds_declared_positions() {
    // `Q(p, q, r) => sum(p * q - r)` with `a = x`, `b = 2x`, `c = x * x` is
    // `sum(x * x)`: forward `13`, gradient `2x = [4, 6]`. `a: x` moves `x`,
    // so it is written last in each permutation.
    for (name, fields) in [
        ("triple_cba", "c: mul(&x, &x), b: add(&x, &x), a: x"),
        ("triple_bca", "b: add(&x, &x), c: mul(&x, &x), a: x"),
    ] {
        let source = format!(
            "{TRIPLE}def g(x: tensor[2, f32]) -> f32 = match Q {{ {fields} }} with {{\n  \
             | Q(p, q, r) => combine(p, q, r)\n}}\n\
             forward = g(to_tensor([2.0f32, 3.0f32]))\n\
             out = grad(g)(to_tensor([2.0f32, 3.0f32]))\n"
        );
        assert_lanes_print(
            &source,
            name,
            &["forward = 13.0", "out = tensor(shape=[2], data=[4.0, 6.0])"],
        );
    }
}

#[test]
fn a_scalar_record_written_out_of_order_binds_declared_positions() {
    // `h(x) = lo - hi` with `lo = x`, `hi = x * x`: at 3, forward `-6` and
    // gradient `1 - 2x = -5`. Binding by written order gives `6` and `5`.
    let source = "type S =\n  | S { lo: f32, hi: f32 }\n\
         def h(x: f32) -> f32 = match S { hi: mul(x, x), lo: x } with {\n  \
         | S(lo, hi) => sub(lo, hi)\n}\n\
         forward = h(3.0f32)\nout = grad(h)(3.0f32)\n";
    assert_lanes_print(source, "scalar_record", &["forward = -6.0", "out = -5.0"]);
}

#[test]
fn positional_construction_and_by_name_patterns_keep_their_results() {
    // `sum(w - b)` with `w = 2x`, `b = x * x` has gradient `2 - 2x`.
    let source = format!(
        "{PAIR}type Pos =\n  | Pos(tensor[2, f32], tensor[2, f32])\n\
         def positional(x: tensor[2, f32]) -> f32 = match Pos(add(&x, &x), mul(&x, &x)) with {{\n  \
         | Pos(w, b) => tensor_to_scalar(sum(sub(&w, &b), 0i32))\n}}\n\
         def by_name(x: tensor[2, f32]) -> f32 = match P {{ b: mul(&x, &x), w: add(&x, &x) }} with {{\n  \
         | P {{ w, b }} => tensor_to_scalar(sum(sub(&w, &b), 0i32))\n}}\n\
         p = grad(positional)(to_tensor([2.0f32, 3.0f32]))\n\
         n = grad(by_name)(to_tensor([2.0f32, 3.0f32]))\n"
    );
    assert_lanes_print(
        &source,
        "controls",
        &[
            "p = tensor(shape=[2], data=[-2.0, -4.0])",
            "n = tensor(shape=[2], data=[-2.0, -4.0])",
        ],
    );
}

#[test]
fn a_vmap_forward_over_an_out_of_order_record_binds_declared_positions() {
    // Per row, `sum(w - b)` with `w = x`, `b = x * x`: `[2, 1]` gives
    // `3 - 5 = -2` and `[1, 0]` gives `0`. Binding by written order makes the
    // vectorized FORWARD value `[2, 0]`.
    let source = format!(
        "{PAIR}def h(x: tensor[2, f32]) -> f32 = match P {{ b: mul(&x, &x), w: x }} with {{\n  \
         | P(w, b) => tensor_to_scalar(sum(sub(&w, &b), 0i32))\n}}\n\
         rows = to_tensor([[2.0f32, 1.0f32], [1.0f32, 0.0f32]])\nout = vmap(h)(rows)\n"
    );
    assert_lanes_print(
        &source,
        "vmap_forward",
        &[
            "rows = tensor(shape=[2, 2], data=[2.0, 1.0, 1.0, 0.0])",
            "out = tensor(shape=[2], data=[-2.0, 0.0])",
        ],
    );
}

#[test]
fn grad_with_respect_to_a_record_parameter_built_out_of_order() {
    // `loss(S(lo, hi)) = lo * lo - 3 * hi` at `lo = 2`, `hi = 1`: gradient
    // `S(2 * lo, -3) = S(4, -3)`. Compiled C bound written order and
    // printed `S(2.0, -3.0)`.
    let source = "type S =\n  | S { lo: f32, hi: f32 }\n\
         def loss(s: S) -> f32 = match s with {\n  \
         | S(lo, hi) => sub(mul(lo, lo), mul(3.0f32, hi))\n}\n\
         g = grad(loss)(S { hi: 1.0f32, lo: 2.0f32 })\n";
    assert_lanes_print(source, "record_param_grad", &["g = S(4.0, -3.0)"]);
}
