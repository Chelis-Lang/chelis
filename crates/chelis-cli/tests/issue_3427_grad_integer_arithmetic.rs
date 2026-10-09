//! chelis#3427: every signed-integer arithmetic operation has one `grad`
//! disposition.
//!
//! [05-OP-64]: "Integer operations ... structurally reject differentiation",
//! and spec/06 §2.1 gives an integer value the `unit` cotangent. An integer
//! operation is differentiated when its value derives from a selected
//! parameter and an active cotangent reaches it; then `grad` rejects it with
//! the signed-integer arithmetic reason, whichever arithmetic operation it is.
//! An integer value that does not derive from a selected parameter is a
//! discrete coefficient, retained exactly in the forward graph as spec/06
//! §2.1's `grad_bitwise.ch` shows, so `grad` accepts it for every arithmetic
//! operation. Each accepted gradient is checked on the evaluator and the C
//! lane. A value reached only through a comparison's zero cotangent keeps its
//! existing disposition (chelis#3426 owns that question).
use assert_cmd::Command;
use tempfile::tempdir;
#[path = "common/mod.rs"]
mod common;
use common::build_and_run;

fn write_file(path: &std::path::Path, source: &str) {
    common::write_file(
        path,
        &chelis_surf::format::format_source(source).expect("canonical Surf"),
    );
}

fn eval(source: &str, stem: &str) -> std::process::Output {
    let dir = tempdir().unwrap();
    let path = dir.path().join(format!("{stem}.ch"));
    write_file(&path, source);
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .unwrap()
}

/// The values printed for `out`, scalar or tensor.
fn printed_values(stdout: &str) -> Vec<f64> {
    let line = stdout
        .lines()
        .find_map(|line| line.strip_prefix("out = "))
        .unwrap_or_else(|| panic!("no `out` line:\n{stdout}"));
    let data = match line.find("data=[") {
        Some(start) => {
            let rest = &line[start + "data=[".len()..];
            &rest[..rest.find(']').expect("closing bracket")]
        }
        None => line,
    };
    data.split(',')
        .map(|value| value.trim().parse::<f64>().expect("numeric"))
        .collect()
}

fn assert_gradient(source: &str, expected: &[f64], stem: &str) {
    assert_eval_gradient(source, expected, stem);
    let c_values = printed_values(&build_and_run(source, stem));
    assert_eq!(c_values, expected, "{stem}: C");
}

fn assert_eval_gradient(source: &str, expected: &[f64], stem: &str) {
    let output = eval(source, stem);
    assert!(
        output.status.success(),
        "{stem}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let eval_values = printed_values(&String::from_utf8(output.stdout).unwrap());
    assert_eq!(eval_values, expected, "{stem}: eval");
}

fn assert_refused(source: &str, needle: &str, stem: &str) {
    let output = eval(source, stem);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "{stem} must refuse: {stderr}");
    assert!(stderr.contains(needle), "{stem}: {stderr}");
}

/// Elementwise integer operations over `v = [5, -3]` and `w = [2, 4]`, with
/// the integer result each yields.
const ELEMENTWISE: [(&str, &str, [f64; 2]); 7] = [
    ("add", "add(v, w)", [7.0, 1.0]),
    ("sub", "sub(v, w)", [3.0, -7.0]),
    ("mul", "mul(v, w)", [10.0, -12.0]),
    ("neg", "neg(v)", [-5.0, 3.0]),
    ("abs", "abs(v)", [5.0, 3.0]),
    ("max_elem", "max_elem(v, w)", [5.0, 4.0]),
    ("min_elem", "min_elem(v, w)", [2.0, -3.0]),
];

/// Integer reductions over `v = [5, -3]`, with their results and whether the
/// C lane emits the integer reduction (chelis#729 tracks the others).
const REDUCTIONS: [(&str, &str, f64, bool); 4] = [
    ("sum", "sum(v, 0i32)", 2.0, true),
    ("prod_reduce", "prod_reduce(v, 0i32)", -15.0, false),
    ("max_reduce", "max_reduce(v, 0i32)", 5.0, true),
    ("min_reduce", "min_reduce(v, 0i32)", -3.0, false),
];

#[test]
fn unselected_integer_coefficients_differentiate_for_every_arithmetic_operation() {
    for (name, expr, coefficient) in ELEMENTWISE {
        assert_gradient(
            &format!(
                r#"
def loss(x: tensor[2, f32], v: tensor[2, i32], w: tensor[2, i32]) -> tensor[f32] = sum(mul(x, cast({expr}, f32)), 0i32)
out = grad(loss, wrt=x)(to_tensor([1.5f32, 2.5f32]), to_tensor([5i32, -3i32]), to_tensor([2i32, 4i32]))
"#
            ),
            &coefficient,
            &format!("unselected_{name}"),
        );
    }
    for (name, expr, coefficient, c_lane) in REDUCTIONS {
        let check = if c_lane {
            assert_gradient
        } else {
            assert_eval_gradient
        };
        check(
            &format!(
                r#"
def loss(x: tensor[f32], v: tensor[2, i32]) -> tensor[f32] = mul(x, cast({expr}, f32))
out = grad(loss, wrt=x)(scalar_to_tensor(1.5f32), to_tensor([5i32, -3i32]))
"#
            ),
            &[coefficient],
            &format!("unselected_{name}"),
        );
    }
}

/// `s` derives from the selected `x` through a comparison and reaches the loss
/// through an active cotangent, so its integer arithmetic is differentiated.
#[test]
fn selected_integer_arithmetic_on_an_active_path_is_rejected_for_every_operation() {
    for (name, expr, _) in ELEMENTWISE {
        let expr = expr.replace('v', "s");
        assert_refused(
            &format!(
                r#"
def loss(x: tensor[2, f32], t: tensor[2, f32], w: tensor[2, i32]) -> tensor[f32] = {{
  s = cast(lt(&x, t), i32)
  sum(mul(x, cast({expr}, f32)), 0i32)
}}
out = grad(loss, wrt=x)(to_tensor([1.5f32, -2.5f32]), to_tensor([0.0f32, 0.0f32]), to_tensor([2i32, 4i32]))
"#
            ),
            &format!("grad: {name} is non-differentiable (signed-integer arithmetic output)"),
            &format!("selected_{name}"),
        );
    }
    for (name, expr, _, _) in REDUCTIONS {
        let expr = expr.replace('v', "s");
        assert_refused(
            &format!(
                r#"
def loss(x: tensor[2, f32], t: tensor[2, f32]) -> tensor[f32] = {{
  s = cast(lt(&x, t), i32)
  mul(sum(x, 0i32), cast({expr}, f32))
}}
out = grad(loss, wrt=x)(to_tensor([1.5f32, -2.5f32]), to_tensor([0.0f32, 0.0f32]))
"#
            ),
            &format!("grad: {name} is non-differentiable (signed-integer arithmetic output)"),
            &format!("selected_{name}"),
        );
    }
}

/// The piecewise-constant integer quotient keeps its own reason on an
/// unselected coefficient; chelis#3426 owns whether that should change.
#[test]
fn unselected_integer_floor_div_keeps_its_piecewise_constant_rejection() {
    assert_refused(
        r#"
def loss(x: tensor[2, f32], v: tensor[2, i32], w: tensor[2, i32]) -> tensor[f32] = sum(mul(x, cast(floor_div(v, w), f32)), 0i32)
out = grad(loss, wrt=x)(to_tensor([1.5f32, 2.5f32]), to_tensor([5i32, -3i32]), to_tensor([2i32, 4i32]))
"#,
        "grad: floor_div is non-differentiable (piecewise constant)",
        "unselected_floor_div",
    );
}

/// Selected integer arithmetic reached only through a comparison's exact zero
/// keeps its existing acceptance for every arithmetic operation.
#[test]
fn selected_integer_arithmetic_reached_only_through_a_predicate_is_accepted() {
    for (name, expr, _) in ELEMENTWISE {
        let expr = expr.replace('v', "s");
        assert_gradient(
            &format!(
                r#"
def loss(x: tensor[2, f32], t: tensor[2, f32], w: tensor[2, i32]) -> tensor[f32] = {{
  s = cast(lt(&x, t), i32)
  sum(where(lt({expr}, w), &x, mul(&x, x)), 0i32)
}}
out = grad(loss, wrt=x)(to_tensor([1.5f32, -2.5f32]), to_tensor([0.0f32, 0.0f32]), to_tensor([2i32, 4i32]))
"#
            ),
            &expected_predicate_gradient(name),
            &format!("predicate_{name}"),
        );
    }
}

/// `s = [0, 1]` for `x = [1.5, -2.5]`; where the predicate holds the term is
/// `x` (derivative 1), otherwise `x*x` (derivative `2x`).
fn expected_predicate_gradient(name: &str) -> [f64; 2] {
    let s = [0i64, 1];
    let w = [2i64, 4];
    let x = [1.5f64, -2.5];
    let mut out = [0.0; 2];
    for i in 0..2 {
        let value = match name {
            "add" => s[i] + w[i],
            "sub" => s[i] - w[i],
            "mul" => s[i] * w[i],
            "neg" => -s[i],
            "abs" => s[i].abs(),
            "max_elem" => s[i].max(w[i]),
            "min_elem" => s[i].min(w[i]),
            _ => unreachable!(),
        };
        out[i] = if value < w[i] { 1.0 } else { 2.0 * x[i] };
    }
    out
}
