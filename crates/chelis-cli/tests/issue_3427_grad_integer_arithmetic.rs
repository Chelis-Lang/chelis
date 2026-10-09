//! chelis#3427 and chelis#3426: integer computations under `grad`.
//!
//! [04-NUM-14]: "A bool or integer source is a discrete forward-only value and
//! carries no cotangent, irrespective of target", and spec/06 §2.1 gives an
//! integer value the `unit` cotangent. So a cast from an integer or bool value
//! is a zero-cotangent boundary, as a comparison is, and an integer
//! computation that reaches the loss only through such a boundary is never on
//! a cotangent path. Every integer arithmetic operation, `floor_div`,
//! `trunc_div`, `mod`, and `count` is then an exact forward value, whether or
//! not it depends on the differentiated parameter, and the gradient agrees
//! with a central finite difference on the evaluator and the C lane. The
//! conversions from differentiable data (a float-to-integer cast, a float
//! `floor_div`) still reject, as do the bitwise operations on a value that
//! depends on the differentiated parameter, which [05-OP-47] decides.
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

/// The values printed for `name`, scalar or tensor.
fn printed_values(stdout: &str, name: &str) -> Vec<f64> {
    let prefix = format!("{name} = ");
    let line = stdout
        .lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("no `{name}` line:\n{stdout}"));
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

fn eval_stdout(source: &str, stem: &str) -> String {
    let output = eval(source, stem);
    assert!(
        output.status.success(),
        "{stem}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

/// `out` matches the central differences `fd0` and `fd1` on each lane named.
fn assert_matches_finite_difference(source: &str, stem: &str, c_lane: bool) {
    let mut lanes = vec![("eval", eval_stdout(source, stem))];
    if c_lane {
        lanes.push(("C", build_and_run(source, stem)));
    }
    for (lane, stdout) in lanes {
        let gradient = printed_values(&stdout, "out");
        let differences = [
            printed_values(&stdout, "fd0")[0],
            printed_values(&stdout, "fd1")[0],
        ];
        assert_eq!(gradient.len(), 2, "{stem} {lane}: {gradient:?}");
        for (got, want) in gradient.iter().zip(differences) {
            assert!(
                (got - want).abs() <= 1e-6 * want.abs().max(1.0),
                "{stem} {lane}: gradient {gradient:?}, central differences {differences:?}"
            );
        }
    }
}

fn assert_refused(source: &str, needle: &str, stem: &str) {
    let output = eval(source, stem);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "{stem} must refuse: {stderr}");
    assert!(stderr.contains(needle), "{stem}: {stderr}");
}

/// Integer computations of `s` and `w`, and whether the C lane emits them
/// (it does not emit integer `prod_reduce` or `min_reduce`, chelis#729).
const COMPUTATIONS: [(&str, &str, bool); 15] = [
    ("add", "add(s, w)", true),
    ("sub", "sub(s, w)", true),
    ("mul", "mul(s, w)", true),
    ("neg", "neg(s)", true),
    ("abs", "abs(sub(s, w))", true),
    ("max_elem", "max_elem(s, w)", true),
    ("min_elem", "min_elem(s, w)", true),
    (
        "floor_div",
        "floor_div(sub(s, w), to_tensor([2i64, 2i64]))",
        true,
    ),
    (
        "trunc_div",
        "trunc_div(sub(s, w), to_tensor([2i64, 2i64]))",
        true,
    ),
    ("mod", "mod(sub(s, w), to_tensor([3i64, 3i64]))", true),
    ("sum", "insert(sum(sub(s, w), 0i32), 0i32, 2i64)", true),
    (
        "max_reduce",
        "insert(max_reduce(sub(s, w), 0i32), 0i32, 2i64)",
        true,
    ),
    (
        "min_reduce",
        "insert(min_reduce(sub(s, w), 0i32), 0i32, 2i64)",
        false,
    ),
    (
        "prod_reduce",
        "insert(prod_reduce(sub(s, w), 0i32), 0i32, 2i64)",
        false,
    ),
    ("count", "insert(count(lt(s, w), 0i32), 0i32, 2i64)", true),
];

/// `loss = sum(x * x * cast(c, f64))` with the integer coefficient `c`
/// computed from `s`, which `source` binds, and the central differences of
/// `loss` along each axis at `x0`.
fn coefficient_program(params: &str, source: &str, expr: &str, args: &str) -> String {
    format!(
        r#"
def loss(x: tensor[2, f64], {params}) -> f64 = {{
  s = {source}
  c = cast({expr}, f64)
  tensor_to_scalar(sum(mul(mul(&x, x), c), 0i32))
}}
x0 = to_tensor([1.5f64, -2.5f64])
e0 = to_tensor([0.0001f64, 0.0f64])
e1 = to_tensor([0.0f64, 0.0001f64])
fd0 = div(sub(loss(add(x0, e0), {args}), loss(sub(x0, e0), {args})), 0.0002f64)
fd1 = div(sub(loss(add(x0, e1), {args}), loss(sub(x0, e1), {args})), 0.0002f64)
out = grad(loss, wrt=x)(x0, {args})
"#
    )
}

/// The coefficient depends on no differentiated parameter.
#[test]
fn integer_coefficients_of_other_parameters_differentiate() {
    for (name, expr, c_lane) in COMPUTATIONS {
        assert_matches_finite_difference(
            &coefficient_program(
                "v: tensor[2, i64], w: tensor[2, i64]",
                "v",
                expr,
                "to_tensor([5i64, -3i64]), to_tensor([2i64, 4i64])",
            ),
            &format!("unselected_{name}"),
            c_lane,
        );
    }
}

/// The coefficient depends on `x` through a comparison and reaches the loss
/// through a cast; both are zero-cotangent boundaries.
#[test]
fn integer_coefficients_of_the_differentiated_parameter_differentiate() {
    for (name, expr, c_lane) in COMPUTATIONS {
        assert_matches_finite_difference(
            &coefficient_program(
                "t: tensor[2, f64], w: tensor[2, i64]",
                "cast(lt(&x, t), i64)",
                expr,
                "to_tensor([0.0f64, 0.0f64]), to_tensor([2i64, 4i64])",
            ),
            &format!("selected_{name}"),
            c_lane,
        );
    }
}

/// An integer that depends on `x` through a `gather` index and one that
/// depends on it through a `where` condition are both forward values, with the
/// same gradient.
#[test]
fn gather_and_where_dependence_agree() {
    let tail = r#"
x0 = to_tensor([1.5f32, -2.5f32])
t0 = to_tensor([0.0f32, 0.0f32])
"#;
    let gather = format!(
        r#"
def loss(x: tensor[2, f32], t: tensor[2, f32], v: tensor[2, i32], w: tensor[2, i32]) -> tensor[f32] = {{
  idx = cast(lt(&x, t), i64)
  s = gather(v, idx, 0i32)
  sum(mul(x, cast(add(s, w), f32)), 0i32)
}}{tail}out = grad(loss, wrt=x)(x0, t0, to_tensor([10i32, 20i32]), to_tensor([1i32, 1i32]))
"#
    );
    let selected = format!(
        r#"
def loss(x: tensor[2, f32], t: tensor[2, f32], a: tensor[2, i32], b: tensor[2, i32], w: tensor[2, i32]) -> tensor[f32] = {{
  s = where(lt(&x, t), a, b)
  sum(mul(x, cast(add(s, w), f32)), 0i32)
}}{tail}out = grad(loss, wrt=x)(x0, t0, to_tensor([20i32, 20i32]), to_tensor([10i32, 10i32]), to_tensor([1i32, 1i32]))
"#
    );
    for (stem, source) in [("gather_index", gather), ("where_condition", selected)] {
        let lanes = [
            ("eval", eval_stdout(&source, stem)),
            ("C", build_and_run(&source, stem)),
        ];
        for (lane, stdout) in lanes {
            assert_eq!(
                printed_values(&stdout, "out"),
                [11.0, 21.0],
                "{stem} {lane}"
            );
        }
    }
}

/// chelis#3426's block mask: an integer `floor_div` of `range` indices reaches
/// the loss only through a comparison under `where`.
#[test]
fn integer_floor_div_block_mask_differentiates() {
    let source = r#"
def loss(x: tensor[*, f32]) -> f32 = {
  idx = to_tensor(range(0i64, shape(&x, 0i32)))
  blk = floor_div(idx, insert(scalar_to_tensor(2i64), 0i32, shape(&x, 0i32)))
  m = eq(blk, insert(scalar_to_tensor(0i64), 0i32, shape(&x, 0i32)))
  tensor_to_scalar(sum(where(m, x, insert(scalar_to_tensor(0.0f32), 0i32, shape(&x, 0i32))), 0i32))
}
out = grad(loss)(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]))
"#;
    let lanes = [
        ("eval", eval_stdout(source, "block_mask")),
        ("C", build_and_run(source, "block_mask")),
    ];
    for (lane, stdout) in lanes {
        assert_eq!(
            printed_values(&stdout, "out"),
            [1.0, 1.0, 0.0, 0.0],
            "block mask {lane}"
        );
    }
}

/// Negative parity: a conversion from the differentiated float data still
/// rejects, and so does a bitwise operation on a value that depends on it.
#[test]
fn conversions_from_differentiable_data_and_selected_bitwise_still_reject() {
    assert_refused(
        r#"
def loss(x: tensor[2, f32]) -> tensor[f32] = sum(mul(&x, cast(cast(x, i32), f32)), 0i32)
out = grad(loss)(to_tensor([1.5f32, 2.5f32]))
"#,
        "grad: cast is non-differentiable (piecewise constant)",
        "float_to_int_cast",
    );
    assert_refused(
        r#"
def loss(x: tensor[2, f32], d: tensor[2, f32]) -> tensor[f32] = sum(mul(&x, floor_div(x, d)), 0i32)
out = grad(loss, wrt=x)(to_tensor([1.5f32, 2.5f32]), to_tensor([2.0f32, 2.0f32]))
"#,
        "grad: floor_div is non-differentiable (piecewise constant)",
        "float_floor_div",
    );
    assert_refused(
        r#"
def loss(x: tensor[2, f32], t: tensor[2, f32], one: tensor[2, i32]) -> tensor[f32] = {
  s = cast(lt(&x, t), i32)
  sum(mul(x, cast(bitand(s, one), f32)), 0i32)
}
out = grad(loss, wrt=x)(to_tensor([1.5f32, -2.5f32]), to_tensor([0.0f32, 0.0f32]), to_tensor([1i32, 1i32]))
"#,
        "grad: bitand is non-differentiable (signed-integer arithmetic output)",
        "selected_bitand",
    );
}
