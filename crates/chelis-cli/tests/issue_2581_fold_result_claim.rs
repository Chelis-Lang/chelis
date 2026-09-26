//! chelis#2581, spec/04 section 4.7: a declared-result guard belongs to the
//! primitive that produced the returned value, whether the return expression
//! names it directly or reaches it through a block tail, a binding, a branch
//! or a callee. `fold` produces the tensor it returns, whichever iteration or
//! seed supplied it, so compiled C checks the declared extent there and traps
//! exactly as `chelis eval` does.
//!
//! On `d029224fd` compiled C emitted no guard for a `fold` result, so a
//! function declared `-> tensor[2, f32]` returned a three-element tensor.

mod common;
#[allow(dead_code)]
#[path = "common/result_claims.rs"]
mod result_claims;

use result_claims::run;

const ITEMS: &str = "[to_tensor([1.0, 2.0, 3.0])]";
const SEED: &str = "to_tensor([4.0, 5.0, 6.0])";

/// One producer shape. `{n}` is the declared extent; every shape returns a
/// three-element tensor, so `{n}` = 3 agrees and `{n}` = 2 traps.
struct Shape {
    name: &'static str,
    source: &'static str,
    /// The primitive the failing guard names.
    op: &'static str,
    /// The value the agreeing program prints.
    value: &'static str,
}

const SHAPES: &[Shape] = &[
    Shape {
        name: "fold_returns_its_seed",
        source: "def f(xs: List[tensor[*, f32]], t0: tensor[*, f32]) -> tensor[{n}, f32] =\n  \
                 fold(fn (acc: tensor[*, f32], x: tensor[*, f32]) -> acc, t0, xs)\n\
                 out = f({items}, {seed})\n",
        op: "fold",
        value: "tensor(shape=[3], data=[4.0, 5.0, 6.0])",
    },
    Shape {
        name: "fold_returns_an_iteration_result",
        source: "def f(xs: List[tensor[*, f32]], t0: tensor[*, f32]) -> tensor[{n}, f32] =\n  \
                 fold(fn (acc: tensor[*, f32], x: tensor[*, f32]) -> (acc + x), t0, xs)\n\
                 out = f({items}, {seed})\n",
        op: "fold",
        value: "tensor(shape=[3], data=[5.0, 7.0, 9.0])",
    },
    Shape {
        name: "fold_over_an_empty_list",
        source: "def f(xs: List[tensor[*, f32]], t0: tensor[*, f32]) -> tensor[{n}, f32] =\n  \
                 fold(fn (acc: tensor[*, f32], x: tensor[*, f32]) -> (acc + x), t0, take(xs, 0i64))\n\
                 out = f({items}, {seed})\n",
        op: "fold",
        value: "tensor(shape=[3], data=[4.0, 5.0, 6.0])",
    },
    Shape {
        name: "fold_selected_by_a_branch",
        source: "def f(flag: bool, xs: List[tensor[*, f32]], t0: tensor[*, f32]) -> tensor[{n}, f32] =\n  \
                 if flag then fold(fn (acc: tensor[*, f32], x: tensor[*, f32]) -> (acc + x), t0, xs) else t0\n\
                 out = f(true, {items}, {seed})\n",
        op: "fold",
        value: "tensor(shape=[3], data=[5.0, 7.0, 9.0])",
    },
    Shape {
        name: "fold_in_a_match_arm",
        source: "def f(o: Option[tensor[*, f32]], xs: List[tensor[*, f32]], t0: tensor[*, f32]) -> tensor[{n}, f32] =\n  \
                 match o with {\n    \
                 | Some(t) => fold(fn (acc: tensor[*, f32], x: tensor[*, f32]) -> (acc + x), t, xs)\n    \
                 | None => t0\n  }\n\
                 out = f(Some({seed}), {items}, {seed})\n",
        op: "fold",
        value: "tensor(shape=[3], data=[5.0, 7.0, 9.0])",
    },
    Shape {
        name: "fold_bound_before_a_later_effect",
        source: "def f(xs: List[tensor[*, f32]], t0: tensor[*, f32]) -> tensor[{n}, f32] ! { IO } = {\n  \
                 _ = print(\"before\")\n  \
                 r = fold(fn (acc: tensor[*, f32], x: tensor[*, f32]) -> (acc + x), t0, xs)\n  \
                 _ = print(\"after\")\n  \
                 r\n}\n\
                 out = f({items}, {seed})\n",
        op: "fold",
        value: "tensor(shape=[3], data=[5.0, 7.0, 9.0])",
    },
    Shape {
        name: "fold_as_a_block_tail_binding",
        source: "def f(xs: List[tensor[*, f32]], t0: tensor[*, f32]) -> tensor[{n}, f32] = {\n  \
                 r = fold(fn (acc: tensor[*, f32], x: tensor[*, f32]) -> (acc + x), t0, xs)\n  \
                 r\n}\n\
                 out = f({items}, {seed})\n",
        op: "fold",
        value: "tensor(shape=[3], data=[5.0, 7.0, 9.0])",
    },
    Shape {
        name: "fold_in_a_callee",
        source: "def g(xs: List[tensor[*, f32]], t0: tensor[*, f32]) -> tensor[*, f32] =\n  \
                 fold(fn (acc: tensor[*, f32], x: tensor[*, f32]) -> (acc + x), t0, xs)\n\
                 def f(xs: List[tensor[*, f32]], t0: tensor[*, f32]) -> tensor[{n}, f32] = g(xs, t0)\n\
                 out = f({items}, {seed})\n",
        op: "fold",
        value: "tensor(shape=[3], data=[5.0, 7.0, 9.0])",
    },
];

fn source(shape: &Shape, extent: usize) -> String {
    shape
        .source
        .replace("{n}", &extent.to_string())
        .replace("{items}", ITEMS)
        .replace("{seed}", SEED)
}

/// The two lines a failing result guard produces, in order.
fn trap_lines(output: &str) -> Vec<&str> {
    output
        .lines()
        .filter(|line| line.contains("extent `") || line.starts_with("numeric trap:"))
        .map(|line| line.trim_start_matches("error: "))
        .collect()
}

fn check_shape(shape: &Shape) -> Result<(), String> {
    let failing = source(shape, 2);
    let (ok, compiled) = run(&failing, true);
    let expected = [
        format!("extent `2`: claimed = 2, {} axis 0 = 3", shape.op),
        format!("numeric trap: domain in {} at i64", shape.op),
    ];
    if ok || trap_lines(&compiled) != expected {
        return Err(format!(
            "{}: compiled C must trap with {expected:?}\n{failing}\n{compiled}",
            shape.name
        ));
    }
    if compiled.contains("out =") || compiled.contains("after") {
        return Err(format!(
            "{}: an effect after the failing guard ran\n{compiled}",
            shape.name
        ));
    }
    let (eval_ok, evaluated) = run(&failing, false);
    if eval_ok || trap_lines(&evaluated) != trap_lines(&compiled) {
        return Err(format!(
            "{}: eval and compiled C traps differ\n{evaluated}\n---\n{compiled}",
            shape.name
        ));
    }
    // Negative parity: the true extent passes and returns the value.
    let agreeing = source(shape, 3);
    let (ok, compiled) = run(&agreeing, true);
    let line = format!("out = {}", shape.value);
    if !ok || !compiled.lines().any(|output| output == line) || compiled.contains("numeric trap:") {
        return Err(format!(
            "{}: the agreeing program must print `{line}`\n{agreeing}\n{compiled}",
            shape.name
        ));
    }
    let (eval_ok, evaluated) = run(&agreeing, false);
    if !eval_ok || evaluated != compiled {
        return Err(format!(
            "{}: eval and compiled C differ\n{evaluated}\n---\n{compiled}",
            shape.name
        ));
    }
    Ok(())
}

// REGRESSION TEST. On `d029224fd` every failing program ran to completion in
// compiled C and printed the three-element tensor; every agreeing program
// already passed there, so that half is a lock.
#[test]
fn a_fold_result_is_checked_against_the_declared_extent() {
    let failures: Vec<String> = SHAPES
        .iter()
        .filter_map(|shape| check_shape(shape).err())
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} shapes failed:\n\n{}",
        failures.len(),
        SHAPES.len(),
        failures.join("\n\n")
    );
}
