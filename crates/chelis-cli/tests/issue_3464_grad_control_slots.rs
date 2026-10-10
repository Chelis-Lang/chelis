//! chelis#3464: spec/06 §7.5 control slots under `grad`.
//!
//! A comparison operand, a `where` or `if` condition, an index, and a
//! movement bound are control slots: their atoms assign exact zero cotangent
//! for every operand dtype. A structural rejection applies only to an
//! operation whose result has a data path to the differentiated output, one
//! that enters no control slot. A piecewise-constant conversion read only
//! through control slots executes forward and contributes nothing, so `grad`
//! returns the executed branch's derivative on the evaluator and the C lane.
//! A slot that is zero only because its operand is bool or integer
//! ([04-NUM-14]) is not a control slot, so a float-to-integer round trip used
//! as data still rejects.
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

fn c_build(source: &str, stem: &str) -> std::process::Output {
    let dir = tempdir().unwrap();
    let path = dir.path().join(format!("{stem}.ch"));
    let out_dir = dir.path().join(format!("{stem}-out"));
    write_file(&path, source);
    Command::cargo_bin("chelis")
        .unwrap()
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
        .unwrap()
}

/// The line printed for `out`, after `out = `.
fn printed_out(stdout: &str, stem: &str, lane: &str) -> String {
    stdout
        .lines()
        .find_map(|line| line.strip_prefix("out = "))
        .unwrap_or_else(|| panic!("{stem} {lane}: no `out` line:\n{stdout}"))
        .to_string()
}

/// `out` prints `expected` on the evaluator and on the C lane.
fn assert_out(source: &str, stem: &str, expected: &str) {
    let output = eval(source, stem);
    assert!(
        output.status.success(),
        "{stem} eval: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(printed_out(&stdout, stem, "eval"), expected, "{stem} eval");
    let stdout = build_and_run(source, stem);
    assert_eq!(printed_out(&stdout, stem, "C"), expected, "{stem} C");
}

/// Both lanes refuse `source` with a diagnostic containing `needle`, and
/// return the evaluator's diagnostic.
fn assert_refused(source: &str, stem: &str, needle: &str) -> String {
    let output = eval(source, stem);
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        !output.status.success(),
        "{stem} eval must refuse: {stderr}"
    );
    assert!(stderr.contains(needle), "{stem} eval: {stderr}");
    let output = c_build(source, stem);
    let c_stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "{stem} C must refuse: {c_stderr}");
    assert!(c_stderr.contains(needle), "{stem} C: {c_stderr}");
    stderr
}

/// `loss = sum(where(cmplt(guard, t), x, x * x))` at `x = [1, 2, 3]` and
/// `t = [2, 2, 2]`, whose gradient is `[1, 4, 6]` wherever `guard` agrees
/// with `x` on which side of 2 each element lies.
fn guarded(guard: &str) -> String {
    format!(
        r#"
def loss(x: tensor[3, f32], t: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(where(cmplt({guard}, t), x, mul(x, x)), 0i32))
out = grad(loss, wrt=x)(to_tensor([1.0f32, 2.0f32, 3.0f32]), to_tensor([2.0f32, 2.0f32, 2.0f32]))
"#
    )
}

/// The issue's witness: `x` cast to `i32` to compare with an integer `w`.
#[test]
fn integer_comparison_of_a_cast_parameter_differentiates() {
    let source = r#"
def loss(x: tensor[3, f32], w: tensor[3, i32]) -> f32 = tensor_to_scalar(sum(where(lt(cast(x, i32), w), x, mul(x, x)), 0i32))
out = grad(loss, wrt=x)(to_tensor([1.0f32, 2.0f32, 3.0f32]), to_tensor([2i32, 2i32, 2i32]))
"#;
    assert_out(
        source,
        "integer_comparison",
        "tensor(shape=[3], data=[1.0, 4.0, 6.0])",
    );
}

/// `lt(floor(x), 2)` is the same step function as the accepted `lt(x, 2)`.
#[test]
fn float_rounding_under_a_comparison_differentiates() {
    let source = r#"
def loss(x: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(where(lt(floor(x), to_tensor([2.0f32, 2.0f32, 2.0f32])), x, mul(x, x)), 0i32))
out = grad(loss, wrt=x)(to_tensor([1.5f32, 2.5f32, 3.5f32]))
"#;
    assert_out(
        source,
        "floor_comparison",
        "tensor(shape=[3], data=[1.0, 5.0, 7.0])",
    );
}

/// Every conversion rung, read only by `cmplt`.
#[test]
fn every_conversion_rung_under_cmplt_differentiates() {
    for (stem, conversion) in [
        ("cast", "cast(cast(x, i32), f32)"),
        ("cast_trunc", "cast(cast_trunc(x, i32), f32)"),
        ("cast_saturate", "cast(cast_saturate(x, i32), f32)"),
        ("cast_wrap", "cast(cast_wrap(cast(x, i64), i32), f32)"),
        ("floor", "floor(x)"),
        ("ceil", "ceil(x)"),
        ("round", "round(x)"),
    ] {
        assert_out(
            &guarded(conversion),
            &format!("rung_{stem}"),
            "tensor(shape=[3], data=[1.0, 4.0, 6.0])",
        );
    }
}

/// A float-to-bool cast read only as a `where` condition.
#[test]
fn float_to_bool_cast_condition_differentiates() {
    let source = r#"
def loss(x: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(where(cast(x, bool), x, mul(x, x)), 0i32))
out = grad(loss, wrt=x)(to_tensor([1.0f32, 0.0f32, 1.0f32]))
"#;
    assert_out(
        source,
        "bool_condition",
        "tensor(shape=[3], data=[1.0, 0.0, 1.0])",
    );
}

/// A scalar `if` condition is a control slot.
#[test]
fn scalar_if_condition_differentiates() {
    let source = r#"
def loss(x: f32) -> f32 = if lt(floor(x), 2.0f32) then x else mul(x, x)
out = grad(loss, wrt=x)(2.5f32)
"#;
    assert_out(source, "scalar_if", "5.0");
}

/// `and` of two comparisons, read only by a `where` condition.
#[test]
fn logical_guard_differentiates() {
    let source = r#"
def loss(x: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(where(and(lt(x, to_tensor([3.0f32, 3.0f32, 3.0f32])), gt(x, to_tensor([2.0f32, 2.0f32, 2.0f32]))), x, mul(x, x)), 0i32))
out = grad(loss, wrt=x)(to_tensor([1.5f32, 2.5f32, 3.5f32]))
"#;
    assert_out(
        source,
        "logical_guard",
        "tensor(shape=[3], data=[3.0, 1.0, 7.0])",
    );
}

/// Integer arithmetic on the conversion, under the comparison.
#[test]
fn integer_arithmetic_under_a_comparison_differentiates() {
    let source = r#"
def loss(x: tensor[3, f32], w: tensor[3, i32]) -> f32 = tensor_to_scalar(sum(where(lt(add(cast(x, i32), w), w), x, mul(x, x)), 0i32))
out = grad(loss, wrt=x)(to_tensor([-1.0f32, 0.0f32, 3.0f32]), to_tensor([2i32, 2i32, 2i32]))
"#;
    assert_out(
        source,
        "integer_arithmetic",
        "tensor(shape=[3], data=[1.0, 0.0, 6.0])",
    );
}

/// A parameter read only through control slots receives exact positive
/// zero on both lanes, even through a `neg`, whose adjoint would turn a
/// positive zero cotangent into a negative one.
#[test]
fn predicate_only_parameter_receives_positive_zero() {
    let source = r#"
def loss(x: tensor[3, f32], y: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(where(lt(floor(neg(y)), to_tensor([2.0f32, 2.0f32, 2.0f32])), x, mul(x, x)), 0i32))
out = grad(loss, wrt=y)(to_tensor([1.0f32, 2.0f32, 3.0f32]), to_tensor([1.5f32, 2.5f32, -3.5f32]))
"#;
    assert_out(
        source,
        "predicate_only",
        "tensor(shape=[3], data=[0.0, 0.0, 0.0])",
    );
}

/// A conversion used only as a gather index keeps working.
#[test]
fn gather_index_conversion_differentiates() {
    let source = r#"
def loss(x: tensor[2, f32], v: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(mul(x, gather(v, cast(floor(x), i64), 0i32)), 0i32))
out = grad(loss, wrt=x)(to_tensor([1.5f32, 2.5f32]), to_tensor([10.0f32, 20.0f32, 30.0f32]))
"#;
    assert_out(
        source,
        "gather_index",
        "tensor(shape=[2], data=[20.0, 30.0])",
    );
}

/// Second order through the same guard: `x^2` below 2, `x^3` from 2 on, so
/// the second derivative is `2` at 1.5 and `6x = 15` at 2.5.
#[test]
fn second_order_through_a_control_slot() {
    let source = r#"
def cubic(x: f32) -> f32 = if lt(floor(x), 2.0f32) then mul(x, x) else mul(mul(x, x), x)
def slope(x: f32) -> f32 = grad(cubic, wrt=x)(x)
lo = grad(slope, wrt=x)(1.5f32)
out = grad(slope, wrt=x)(2.5f32)
"#;
    assert_out(source, "second_order", "15.0");
    let output = eval(source, "second_order_lo");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("lo = 2.0"), "{stdout}");
}

/// `vmap` over the gradient of the same guard.
#[test]
fn vmap_of_grad_through_a_control_slot() {
    let source = r#"
def cubic(x: tensor[f32]) -> f32 = if lt(tensor_to_scalar(floor(x)), 2.0f32) then tensor_to_scalar(mul(x, x)) else tensor_to_scalar(mul(mul(x, x), x))
out = vmap(grad(cubic))(to_tensor([1.5f32, 2.5f32]))
"#;
    assert_out(source, "vmap_grad", "tensor(shape=[2], data=[3.0, 18.75])");
}

/// Negative parity: [04-NUM-14] zeroes a discrete value's cotangent by dtype,
/// not by slot, so a float-to-integer round trip used as data rejects.
#[test]
fn integer_round_trip_as_data_rejects() {
    let source = r#"
def loss(x: tensor[3, f32], y: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(add(mul(y, cast(cast(x, i32), f32)), x), 0i32))
out = grad(loss, wrt=x)(to_tensor([1.0f32, 2.0f32, 3.0f32]), to_tensor([5.0f32, 5.0f32, 5.0f32]))
"#;
    assert_refused(
        source,
        "round_trip",
        "grad: cast is non-differentiable (piecewise constant)",
    );
}

/// Negative parity: a conversion read by a comparison and also used as data
/// keeps its rejection through the data path.
#[test]
fn conversion_with_a_predicate_use_and_a_data_path_rejects() {
    let source = r#"
def loss(x: tensor[3, f32], t: tensor[3, f32]) -> f32 = {
  r = floor(x)
  tensor_to_scalar(sum(add(where(cmplt(r, t), x, mul(x, x)), r), 0i32))
}
out = grad(loss, wrt=x)(to_tensor([1.5f32, 2.5f32, 3.5f32]), to_tensor([2.0f32, 2.0f32, 2.0f32]))
"#;
    assert_refused(
        source,
        "predicate_and_data",
        "grad: floor is non-differentiable (piecewise constant)",
    );
}

/// Negative parity: `and` of comparisons of `x`, converted and used as data,
/// is a logical operation on a data path.
#[test]
fn logical_operation_as_data_rejects() {
    let source = r#"
def loss(x: tensor[3, f64]) -> f64 = tensor_to_scalar(sum(mul(x, cast(and(lt(x, to_tensor([3.0f64, 3.0f64, 3.0f64])), gt(x, to_tensor([2.0f64, 2.0f64, 2.0f64]))), f64)), 0i32))
out = grad(loss, wrt=x)(to_tensor([1.5f64, 2.5f64, 3.5f64]))
"#;
    assert_refused(
        source,
        "logical_as_data",
        "grad: and is non-differentiable (logical operation)",
    );
}

/// An operation reached only through control slots contributes exact zero
/// without evaluating its adjoint: `mul(x, y)` under the comparison would
/// otherwise send `0 * inf = NaN` to `x`.
#[test]
fn control_slot_operations_contribute_exact_zero() {
    let source = r#"
def loss(x: f32, y: f32) -> f32 = if lt(mul(x, y), 1.0f32) then x else mul(x, x)
out = grad(loss, wrt=x)(2.0f32, div(1.0f32, 0.0f32))
"#;
    assert_out(source, "control_zero", "4.0");
}

/// chelis#3487: a conversion of a parameter that is not differentiated is
/// inactive, a constant with respect to `x`, so the gradient of
/// `sum(x * conversion(m))` is `conversion(m)`.
#[test]
fn undifferentiated_conversion_is_a_constant() {
    for (stem, conversion, expected) in [
        (
            "floor",
            "floor(m)",
            "tensor(shape=[3], data=[1.0, 2.0, 3.0])",
        ),
        (
            "round",
            "round(m)",
            "tensor(shape=[3], data=[2.0, 2.0, 4.0])",
        ),
        (
            "cast_trunc",
            "cast(cast_trunc(m, i32), f32)",
            "tensor(shape=[3], data=[1.0, 2.0, 3.0])",
        ),
    ] {
        let source = format!(
            r#"
def loss(x: tensor[3, f32], m: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(mul(x, {conversion}), 0i32))
out = grad(loss, wrt=x)(to_tensor([1.5f32, 2.5f32, 3.5f32]), to_tensor([1.5f32, 2.5f32, 3.5f32]))
"#
        );
        assert_out(&source, &format!("inactive_{stem}"), expected);
    }
}

/// Negative parity: the same conversion of a value that depends on `x` is
/// active and rejects.
#[test]
fn conversion_of_a_differentiated_value_rejects() {
    let source = r#"
def loss(x: tensor[3, f32], m: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(mul(x, floor(add(m, x))), 0i32))
out = grad(loss, wrt=x)(to_tensor([1.5f32, 2.5f32, 3.5f32]), to_tensor([1.5f32, 2.5f32, 3.5f32]))
"#;
    assert_refused(
        source,
        "active_floor",
        "grad: floor is non-differentiable (piecewise constant)",
    );
}

/// Negative parity, spec/06 §7.5: a value gathered by an index computed from
/// `x` depends on `x`. The index is a control slot, so its cast never
/// rejects, but a conversion of the gathered value used as data is active and
/// rejects.
#[test]
fn conversion_of_an_index_dependent_value_rejects() {
    for (stem, conversion, needle) in [
        (
            "gathered_floor",
            "floor(gather(v, cast(lt(x, t), i64), 0i32))",
            "grad: floor is non-differentiable (piecewise constant)",
        ),
        (
            "gathered_bitand",
            "cast(bitand(gather(w, cast(lt(x, t), i64), 0i32), to_tensor([3i32, 3i32])), f32)",
            "grad: bitand is non-differentiable (signed-integer arithmetic output)",
        ),
    ] {
        let source = format!(
            r#"
def loss(x: tensor[2, f32], t: tensor[2, f32], v: tensor[2, f32], w: tensor[2, i32]) -> f32 = tensor_to_scalar(sum(mul(x, {conversion}), 0i32))
out = grad(loss, wrt=x)(to_tensor([1.5f32, -2.5f32]), to_tensor([0.0f32, 0.0f32]), to_tensor([10.5f32, 20.5f32]), to_tensor([5i32, 6i32]))
"#
        );
        assert_refused(&source, stem, needle);
    }
}

/// The rejection names the data path that makes it apply and how to keep the
/// conversion off one. It does not recommend `stop_gradient`, which no
/// program can call yet.
#[test]
fn active_conversion_diagnostic_names_the_data_path() {
    let source = r#"
def loss(x: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(mul(x, cast(cast(x, i32), f32)), 0i32))
out = grad(loss, wrt=x)(to_tensor([1.0f32, 2.0f32, 3.0f32]))
"#;
    let stderr = assert_refused(
        source,
        "data_path_diagnostic",
        "grad: cast is non-differentiable (piecewise constant) and its result has a data path \
         to the differentiated output",
    );
    assert!(
        stderr.contains("compute it from values that are not differentiated"),
        "{stderr}"
    );
    assert!(
        !stderr.contains("stop-gradient") && !stderr.contains("stop_gradient"),
        "{stderr}"
    );
}

/// spec/06 §7.5: a control slot receives no contribution, not even an exact
/// zero, so a predicate read of `x` cannot change the sign of `x`'s data
/// contribution `w = -0`. The accumulation then decides the sign alone: +0
/// while it starts from a +0 base leaf, -0 under chelis#3419's rule.
/// TODO(bf-zero, chelis#3419): once that rule lands, tighten this to
/// require exactly `-0.0`.
#[test]
fn a_predicate_read_adds_no_contribution_to_a_data_gradient() {
    let source = r#"
def loss(x: tensor[1, f32], w: tensor[1, f32], c: tensor[1, f32]) -> f32 = tensor_to_scalar(sum(add(mul(x, w), where(lt(x, to_tensor([0.0f32])), c, c)), 0i32))
out = grad(loss, wrt=x)(to_tensor([1.0f32]), neg(to_tensor([0.0f32])), to_tensor([2.0f32]))
"#;
    for (lane, stdout) in [
        ("eval", {
            let output = eval(source, "predicate_signed_zero");
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            String::from_utf8(output.stdout).unwrap()
        }),
        ("C", build_and_run(source, "predicate_signed_zero")),
    ] {
        let printed = printed_out(&stdout, "predicate_signed_zero", lane);
        assert!(
            printed == "tensor(shape=[1], data=[0.0])"
                || printed == "tensor(shape=[1], data=[-0.0])",
            "{lane}: {printed}"
        );
    }
}

/// A `shape` read is a control slot, so a parameter read only by it is
/// disconnected and its gradient is exact +0 on both lanes.
#[test]
fn shape_read_gradient_is_positive_zero() {
    let source = r#"
def loss(x: tensor[3, f32]) -> f32 = cast(shape(&x, 0i32), f32)
out = grad(loss, wrt=x)(to_tensor([1.0f32, 2.0f32, 3.0f32]))
"#;
    assert_out(
        source,
        "shape_read",
        "tensor(shape=[3], data=[0.0, 0.0, 0.0])",
    );
}

/// spec/06 §7.5 dependence, slot by slot. A value-read bound carries its
/// operand's dependence: a `floor` of a slice whose start comes from `x` is
/// active and rejects as data, while the same slice from an undifferentiated
/// `m` is a constant, `floor([2.5, 3.5, 4.5])` summed, 9.
#[test]
fn a_value_read_bound_carries_dependence() {
    let active = r#"
def loss(x: f32, v: tensor[4, f32]) -> f32 = mul(x, tensor_to_scalar(sum(floor(shrink(&v, [[cast_trunc(x, i64), 4i64]])), 0i32)))
out = grad(loss, wrt=x)(1.5f32, to_tensor([1.5f32, 2.5f32, 3.5f32, 4.5f32]))
"#;
    assert_refused(
        active,
        "bound_active",
        "grad: floor is non-differentiable (piecewise constant)",
    );
    let inactive = r#"
def loss(x: f32, m: f32, v: tensor[4, f32]) -> f32 = mul(x, tensor_to_scalar(sum(floor(shrink(&v, [[cast_trunc(m, i64), 4i64]])), 0i32)))
out = grad(loss, wrt=x)(1.5f32, 1.5f32, to_tensor([1.5f32, 2.5f32, 3.5f32, 4.5f32]))
"#;
    assert_out(inactive, "bound_inactive", "9.0");
}

/// spec/06 §7.5 dependence: a metadata read and a guard predicate carry no
/// element-value dependence, so a conversion read through one is a constant
/// with respect to `x`. Each row would reject if its slot carried dependence.
#[test]
fn metadata_reads_and_guard_predicates_carry_no_value_dependence() {
    for (stem, source, expected) in [
        (
            "shape_read",
            r#"
def loss(x: tensor[3, f32]) -> f32 = mul(tensor_to_scalar(sum(x, 0i32)), floor(cast(shape(&x, 0i32), f32)))
out = grad(loss, wrt=x)(to_tensor([1.5f32, 2.5f32, 3.5f32]))
"#,
            "tensor(shape=[3], data=[3.0, 3.0, 3.0])",
        ),
        (
            "uniform_template",
            r#"
def loss(x: tensor[3, f32]) -> f32 = {
  u = uniform_like(fold_in(key_from_seed(7i64), 1i64), x, 0.0f32, 4.0f32)
  tensor_to_scalar(sum(mul(x, floor(u)), 0i32))
}
out = grad(loss, wrt=x)(to_tensor([1.5f32, 2.5f32, 3.5f32]))
"#,
            "tensor(shape=[3], data=[3.0, 3.0, 0.0])",
        ),
        (
            "guard_predicate",
            r#"
def loss(x: f32, m: f32) -> f32 = {
  g = if gt(x, 10.0f32) then fail("big") else m
  mul(x, floor(g))
}
out = grad(loss, wrt=x)(2.5f32, 3.5f32)
"#,
            "3.0",
        ),
    ] {
        assert_out(source, stem, expected);
    }
}
