#![allow(clippy::disallowed_methods)]

use chelis_dag_core::dag::*;
use chelis_dag_core::grad::*;
use chelis_ir::eval::{TensorValue, eval_scalar};
use chelis_numeric_core::prim::Prim;
use chelis_unord::UnordMap;

// chelis#729 Phase 1: these fixtures verify the AUTODIFF machinery
// against f64-precision finite differences, so the DAG type is f64.
// (Pre-refactor they were typed f32 but evaluated at raw f64, the
// chelis#717 shape; per-dtype finalize now makes an f32-typed DAG
// genuinely round at f32, which the matrix suites cover.)
fn scalar_f32() -> TensorType {
    TensorType::scalar_f32()
}

fn scalar_f64() -> TensorType {
    TensorType {
        dims: vec![],
        precision: Prim::F64,
    }
}

#[test]
fn bitwise_coefficients_are_forward_values_and_selected_bitwise_paths_reject() {
    for kind in [
        chelis_numeric_core::bitwise::BitwiseKind::And,
        chelis_numeric_core::bitwise::BitwiseKind::Or,
        chelis_numeric_core::bitwise::BitwiseKind::Xor,
        chelis_numeric_core::bitwise::BitwiseKind::ShiftLeft,
        chelis_numeric_core::bitwise::BitwiseKind::ShiftRight,
    ] {
        let mut dag = Dag::new();
        let owner = dag.declare("test");
        let int_ty = TensorType {
            dims: vec![],
            precision: Prim::Int32,
        };
        let x = dag.add_node(
            owner,
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let n = dag.add_node(
            owner,
            RiscOp::Load { name: "n".into() },
            vec![],
            int_ty.clone(),
            None,
        );
        let count = dag.add_node(
            owner,
            RiscOp::Const {
                value: chelis_numeric_core::dtype_semantics::scalar_from_i64(
                    "test",
                    Prim::Int32,
                    2,
                )
                .unwrap(),
            },
            vec![],
            int_ty.clone(),
            None,
        );
        let discrete = dag.add_node(owner, RiscOp::Bitwise(kind), vec![n, count], int_ty, None);
        let coefficient = dag.add_node(
            owner,
            RiscOp::Cast {
                new_precision: Prim::F32,
            },
            vec![discrete],
            scalar_f32(),
            None,
        );
        let output = dag.add_node(owner, RiscOp::Mul, vec![x, coefficient], scalar_f32(), None);
        let result = grad_dag_checked(&dag, output, &[x]).expect("unselected runtime coefficient");
        assert!(grad_dag(&dag, output, &[x]).is_some());
        let inputs = UnordMap::from_iter([("x".to_string(), 1.25), ("n".to_string(), 6.0)]);
        let actual = eval_scalar(&result.dag, &inputs);
        let expected = match kind {
            chelis_numeric_core::bitwise::BitwiseKind::And => 2.0,
            chelis_numeric_core::bitwise::BitwiseKind::Or => 6.0,
            chelis_numeric_core::bitwise::BitwiseKind::Xor => 4.0,
            chelis_numeric_core::bitwise::BitwiseKind::ShiftLeft => 24.0,
            chelis_numeric_core::bitwise::BitwiseKind::ShiftRight => 1.0,
        };
        assert_eq!(actual[&result.grad_nodes[&x]], expected);
        assert!(
            matches!(grad_dag_checked(&dag, output, &[n]), Err(AdError::NotSupported { op, reason: AdRejectionReason::IntegerArithmeticOutput }) if op == kind.name())
        );
    }
}

/// chelis#3382: an `expand` whose size reads a differentiated
/// parameter's shape does not make its value selected data ([05-MOV-1]),
/// so a bitwise coefficient built from it stays a forward value. The
/// parameter that IS the expanded value still selects it.
#[test]
fn an_expand_extent_does_not_select_a_bitwise_coefficient() {
    let mut dag = Dag::new();
    let owner = dag.declare("test");
    let vector = |precision| TensorType {
        dims: vec![DimInfo::Lit(2)],
        precision,
    };
    let int_ty = TensorType {
        dims: vec![],
        precision: Prim::Int32,
    };
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        vector(Prim::F32),
        None,
    );
    let n = dag.add_node(
        owner,
        RiscOp::Load { name: "n".into() },
        vec![],
        int_ty,
        None,
    );
    let sized = dag.add_node(
        owner,
        RiscOp::Expand {
            axis: 0,
            size: RtDim::InputAxis {
                tensor: 1,
                axis: RtAxis::Lit(0),
            },
        },
        vec![n, x],
        vector(Prim::Int32),
        None,
    );
    let masked = dag.add_node(
        owner,
        RiscOp::Bitwise(chelis_numeric_core::bitwise::BitwiseKind::And),
        vec![sized, sized],
        vector(Prim::Int32),
        None,
    );
    let coefficient = dag.add_node(
        owner,
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![masked],
        vector(Prim::F32),
        None,
    );
    let weighted = dag.add_node(
        owner,
        RiscOp::Mul,
        vec![x, coefficient],
        vector(Prim::F32),
        None,
    );
    let output = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![weighted],
        scalar_f32(),
        None,
    );
    let result = grad_dag_checked(&dag, output, &[x]).expect("an extent selects nothing");
    let inputs = UnordMap::from([
        ("x".into(), TensorValue::from_vec(vec![2], vec![1.0, 2.0])),
        ("n".into(), TensorValue::from_vec(vec![], vec![3.0])),
    ]);
    let actual = chelis_ir::eval::eval_tensor(&result.dag, &inputs).unwrap();
    assert_eq!(
        actual[&result.grad_nodes[&x]].to_f64_lossy_vec(),
        vec![3.0, 3.0]
    );
    assert!(matches!(
        grad_dag_checked(&dag, output, &[n]),
        Err(AdError::NotSupported {
            reason: AdRejectionReason::IntegerArithmeticOutput,
            ..
        })
    ));
}

/// `bitwise(cast(cmplt(x, 1), i32), 1)` either compared under a `where`
/// condition, reaching the output only through control slots, or, with
/// `data_use`, cast back to f32 and multiplied by `x`, a data path.
fn bitwise_of_a_comparison(
    kind: chelis_numeric_core::bitwise::BitwiseKind,
    data_use: bool,
) -> (Dag, NodeId, NodeId) {
    let mut dag = Dag::new();
    let owner = dag.declare("test");
    let int_ty = TensorType {
        dims: vec![],
        precision: Prim::Int32,
    };
    let bool_ty = TensorType {
        dims: vec![],
        precision: Prim::Bool,
    };
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f32(),
        None,
    );
    let one_float = dag.add_node(
        owner,
        RiscOp::synth_const(Prim::F32, 1.0),
        vec![],
        scalar_f32(),
        None,
    );
    let first_control = dag.add_node(
        owner,
        RiscOp::Compare(ComparisonKind::CmpLt),
        vec![x, one_float],
        bool_ty.clone(),
        None,
    );
    let discrete = dag.add_node(
        owner,
        RiscOp::Cast {
            new_precision: Prim::Int32,
        },
        vec![first_control],
        int_ty.clone(),
        None,
    );
    let one_int = dag.add_node(
        owner,
        RiscOp::synth_const(Prim::Int32, 1.0),
        vec![],
        int_ty.clone(),
        None,
    );
    let selected_bitwise = dag.add_node(
        owner,
        RiscOp::Bitwise(kind),
        vec![discrete, one_int],
        int_ty,
        None,
    );
    let out = if data_use {
        let coefficient = dag.add_node(
            owner,
            RiscOp::Cast {
                new_precision: Prim::F32,
            },
            vec![selected_bitwise],
            scalar_f32(),
            None,
        );
        dag.add_node(owner, RiscOp::Mul, vec![x, coefficient], scalar_f32(), None)
    } else {
        let final_control = dag.add_node(
            owner,
            RiscOp::Compare(ComparisonKind::CmpLt),
            vec![selected_bitwise, one_int],
            bool_ty,
            None,
        );
        dag.add_node(
            owner,
            RiscOp::Where,
            vec![final_control, x, x],
            scalar_f32(),
            None,
        )
    };
    (dag, x, out)
}

/// chelis#3464: the same bitwise operation read only through a comparison
/// and a `where` condition, both control slots, is a forward value.
#[test]
fn bitwise_read_only_through_control_slots_is_a_forward_value() {
    for kind in [
        chelis_numeric_core::bitwise::BitwiseKind::And,
        chelis_numeric_core::bitwise::BitwiseKind::Or,
        chelis_numeric_core::bitwise::BitwiseKind::Xor,
        chelis_numeric_core::bitwise::BitwiseKind::ShiftLeft,
        chelis_numeric_core::bitwise::BitwiseKind::ShiftRight,
    ] {
        let (dag, x, out) = bitwise_of_a_comparison(kind, false);
        let result = match grad_dag_checked(&dag, out, &[x]) {
            Ok(result) => result,
            Err(error) => panic!("{} under a condition must not reject: {error}", kind.name()),
        };
        let values = eval_scalar(&result.dag, &UnordMap::from([("x".to_string(), 0.5)]));
        assert_eq!(values[&result.grad_nodes[&x]], 1.0, "{}", kind.name());
    }
}

/// Build a unary DAG: Load("x") -> op -> output.
fn build_unary_dag(
    op_fn: impl FnOnce(&mut Dag, Owner, NodeId, &TensorType) -> NodeId,
) -> (Dag, NodeId, NodeId) {
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let y = op_fn(&mut dag, owner, x, &scalar_f64());
    (dag, x, y)
}

/// Build a binary DAG: Load("x"), Load("y") -> op -> output.
fn build_binary_dag(
    op_fn: impl FnOnce(&mut Dag, Owner, NodeId, NodeId, &TensorType) -> NodeId,
) -> (Dag, NodeId, NodeId, NodeId) {
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let y = dag.add_node(
        owner,
        RiscOp::Load { name: "y".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let out = op_fn(&mut dag, owner, x, y, &scalar_f64());
    (dag, x, y, out)
}

/// Compute both analytical (via AD) and numerical (via finite differences) gradients.
fn finite_diff(
    dag: &Dag,
    output: NodeId,
    wrt: NodeId,
    input_name: &str,
    other_inputs: &[(&str, f64)],
    x0: f64,
    h: f64,
) -> (f64, f64) {
    let grad_result = grad_dag(dag, output, &[wrt]).unwrap();

    // Analytical gradient.
    let mut inputs: UnordMap<String, f64> = UnordMap::new();
    inputs.insert(input_name.to_string(), x0);
    for &(name, val) in other_inputs {
        inputs.insert(name.to_string(), val);
    }
    let vals = eval_scalar(&grad_result.dag, &inputs);
    let analytical = vals[&grad_result.grad_nodes[&wrt]];

    // Numerical gradient via central differences.
    inputs.insert(input_name.to_string(), x0 + h);
    let f_plus = eval_scalar(dag, &inputs)[&output];
    inputs.insert(input_name.to_string(), x0 - h);
    let f_minus = eval_scalar(dag, &inputs)[&output];
    let numerical = (f_plus - f_minus) / (2.0 * h);

    (analytical, numerical)
}

fn assert_grad_close(analytical: f64, numerical: f64) {
    assert!(
        (analytical - numerical).abs() < 1e-4,
        "gradient mismatch: analytical={analytical}, numerical={numerical}"
    );
}

// ---- Primitive adjoint tests ----

#[test]
fn grad_add() {
    let (dag, x, _y, out) = build_binary_dag(|dag, owner, a, b, ty| {
        dag.add_node(owner, RiscOp::Add, vec![a, b], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[("y", 3.0)], 2.0, 1e-5);
    assert_grad_close(a, n);
    assert!((a - 1.0).abs() < 1e-6); // d(x+y)/dx = 1
}

#[test]
fn grad_mul() {
    let (dag, x, _y, out) = build_binary_dag(|dag, owner, a, b, ty| {
        dag.add_node(owner, RiscOp::Mul, vec![a, b], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[("y", 3.0)], 2.0, 1e-5);
    assert_grad_close(a, n);
    assert!((a - 3.0).abs() < 1e-6); // d(x*y)/dx = y = 3
}

#[test]
fn grad_div_lhs() {
    let (dag, x, _y, out) = build_binary_dag(|dag, owner, a, b, ty| {
        dag.add_node(owner, RiscOp::Div, vec![a, b], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[("y", 4.0)], 2.0, 1e-5);
    assert_grad_close(a, n);
    assert!(
        (a - 0.25).abs() < 1e-6,
        "d(x/y)/dx at y=4 should be 0.25, got {a}"
    );
}

fn pow_dag() -> (Dag, NodeId, NodeId, NodeId) {
    build_binary_dag(|dag, owner, a, b, ty| {
        dag.add_node(owner, RiscOp::Pow, vec![a, b], ty.clone(), None)
    })
}

/// Both [05-OP-79] cotangents at one point, from the gradient DAG.
fn pow_grads(x0: f64, y0: f64) -> (f64, f64) {
    let (dag, x, y, out) = pow_dag();
    let grad = grad_dag(&dag, out, &[x, y]).unwrap();
    let inputs: UnordMap<String, f64> = [("x".to_string(), x0), ("y".to_string(), y0)]
        .into_iter()
        .collect();
    let vals = eval_scalar(&grad.dag, &inputs);
    (vals[&grad.grad_nodes[&x]], vals[&grad.grad_nodes[&y]])
}

/// [05-OP-79]: both cotangents agree with central differences, including a
/// negative base with an integer exponent and a negative exponent.
#[test]
fn grad_pow_matches_finite_differences() {
    let (dag, x, y, out) = pow_dag();
    for (x0, y0) in [
        (2.0, 3.0),
        (-1.5, 3.0),
        (-1.5, 2.0),
        (0.7, 2.5),
        (1.3, -0.5),
    ] {
        let (a, n) = finite_diff(&dag, out, x, "x", &[("y", y0)], x0, 1e-6);
        assert!(
            (a - n).abs() < 1e-4 * a.abs().max(1.0),
            "d pow/dx at ({x0}, {y0}): analytical {a}, numerical {n}"
        );
        let expected = y0 * x0.powf(y0 - 1.0);
        assert!(
            (a - expected).abs() < 1e-9 * expected.abs().max(1.0),
            "{a} vs {expected}"
        );
    }
    for (x0, y0) in [(2.0, 3.0), (0.7, 2.5), (1.3, -0.5)] {
        let (a, n) = finite_diff(&dag, out, y, "y", &[("x", x0)], y0, 1e-6);
        assert!(
            (a - n).abs() < 1e-4 * a.abs().max(1.0),
            "d pow/dy at ({x0}, {y0}): analytical {a}, numerical {n}"
        );
    }
}

/// [05-OP-79]: a zero exponent gives the base an exact zero cotangent and a
/// zero base gives the exponent one, where the unguarded products are
/// `0 * inf` and `inf * log(0)`.
#[test]
fn grad_pow_selects_exact_zeros_at_zero_exponent_and_zero_base() {
    let (dx, dy) = pow_grads(0.0, 0.0);
    assert_eq!(dx.to_bits(), 0.0_f64.to_bits(), "d pow/dx at (0, 0)");
    assert_eq!(dy.to_bits(), 0.0_f64.to_bits(), "d pow/dy at (0, 0)");
    let (dx, _) = pow_grads(3.0, 0.0);
    assert_eq!(dx, 0.0);
    let (dx, dy) = pow_grads(-0.0, 2.0);
    assert_eq!(dx, 0.0, "d pow/dx at (-0, 2) is 2 * (-0)^1");
    assert_eq!(dy.to_bits(), 0.0_f64.to_bits(), "d pow/dy at (-0, 2)");
    // The base cotangent at x = 0 is the real derivative where it exists.
    let (dx, _) = pow_grads(0.0, 1.0);
    assert_eq!(dx, 1.0);
    let (dx, _) = pow_grads(0.0, 0.5);
    assert_eq!(dx, f64::INFINITY);
}

/// The base cotangent of `pow(x, y)` at dtype `precision`, with `x` and `y`
/// loaded at f64 and cast to `precision` (each witness is exact there).
fn pow_base_grad_at(precision: Prim, x0: f64, y0: f64) -> f64 {
    pow_grads_at(precision, x0, y0).0
}

/// Both cotangents of `pow(x, y)` at dtype `precision`, as [`pow_base_grad_at`].
fn pow_grads_at(precision: Prim, x0: f64, y0: f64) -> (f64, f64) {
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let narrow = TensorType {
        dims: vec![],
        precision,
    };
    let load = |dag: &mut Dag, name: &str| {
        let wide = dag.add_node(
            owner,
            RiscOp::Load { name: name.into() },
            vec![],
            scalar_f64(),
            None,
        );
        let cast = dag.add_node(
            owner,
            RiscOp::Cast {
                new_precision: precision,
            },
            vec![wide],
            narrow.clone(),
            None,
        );
        (wide, cast)
    };
    let (x, xn) = load(&mut dag, "x");
    let (y, yn) = load(&mut dag, "y");
    let out = dag.add_node(owner, RiscOp::Pow, vec![xn, yn], narrow.clone(), None);
    let grad = grad_dag(&dag, out, &[x, y]).unwrap();
    let inputs: UnordMap<String, f64> = [("x".to_string(), x0), ("y".to_string(), y0)]
        .into_iter()
        .collect();
    let vals = eval_scalar(&grad.dag, &inputs);
    (vals[&grad.grad_nodes[&x]], vals[&grad.grad_nodes[&y]])
}

/// [05-OP-79] at an infinite base: the base cotangent is `y * x^(y-1)` on
/// both sides of `2^p`, its sign from the exact parity of `y - 1` (odd from
/// `2^p` on), never `inf / inf`; the exponent cotangent is an exact zero at
/// `+inf`, which is constant in `y` on each side of `y = 0`, and NaN at
/// `-inf` through `log`.
#[test]
fn grad_pow_cotangents_at_an_infinite_base() {
    for (precision, limit) in [
        (Prim::F16, 2048.0),
        (Prim::Bf16, 256.0),
        (Prim::F32, 16_777_216.0),
        (Prim::F64, 9_007_199_254_740_992.0),
    ] {
        let inf = f64::INFINITY;
        for y0 in [limit - 2.0, limit, 2.0 * limit] {
            let label = format!("{precision:?} y = {y0}");
            assert_eq!(
                pow_grads_at(precision, inf, y0).0,
                inf,
                "+inf base, {label}"
            );
            // y even: y - 1 odd, so (-inf)^(y-1) is -inf.
            assert_eq!(
                pow_grads_at(precision, -inf, y0).0,
                -inf,
                "-inf base, {label}"
            );
            // y even and negative: (+-inf)^(y-1) is a zero, so the
            // cotangent is a zero rather than the NaN of `inf / inf`.
            assert_eq!(
                pow_grads_at(precision, inf, -y0).0,
                0.0,
                "+inf base, -{label}"
            );
            assert_eq!(
                pow_grads_at(precision, -inf, -y0).0,
                0.0,
                "-inf base, -{label}"
            );
        }
        assert_eq!(
            pow_grads_at(precision, -inf, 3.0).0,
            inf,
            "{precision:?} (-inf)^2 * 3"
        );
        assert_eq!(
            pow_grads_at(precision, inf, 1.0).0,
            1.0,
            "{precision:?} y = 1"
        );
        for y0 in [-2.0, 0.0, 0.5, 2.0, limit] {
            let dy = pow_grads_at(precision, inf, y0).1;
            assert_eq!(
                dy.to_bits(),
                0.0f64.to_bits(),
                "{precision:?} d/dy at (+inf, {y0})"
            );
            assert!(
                pow_grads_at(precision, -inf, y0).1.is_nan(),
                "{precision:?} (-inf, {y0})"
            );
        }
    }
}

/// [05-OP-79]: from `2^p` on, `y - 1` rounds an even exponent to an even
/// one; the base cotangent still has the odd power's sign, through `r / x`,
/// and its magnitude `y * |x|^(y-1)` (the reviewer's witnesses).
#[test]
fn grad_pow_keeps_the_odd_power_sign_where_y_minus_one_rounds() {
    for (precision, x0, y0, tolerance) in [
        (Prim::F32, -1.000_000_119_209_289_6, 16_777_218.0, 1e-5),
        (
            Prim::F64,
            -(1.0 + f64::EPSILON),
            9_007_199_254_740_994.0,
            1e-12,
        ),
        (Prim::Bf16, -1.007_812_5, 258.0, 2e-2),
        (Prim::Bf16, -1.007_812_5, 512.0, 2e-2),
        (Prim::F16, -1.000_976_562_5, 2050.0, 4e-3),
        (Prim::F16, -1.000_976_562_5, 3000.0, 4e-3),
    ] {
        let dx = pow_base_grad_at(precision, x0, y0);
        let exact = -y0 * (-x0).powf(y0 - 1.0);
        assert!(
            dx < 0.0 && ((dx - exact) / exact).abs() < tolerance,
            "{precision:?} d pow/dx at ({x0}, {y0}): got {dx}, derivative {exact}"
        );
    }
}

/// Below `2^p` the base cotangent stays `y * pow(x, y - 1)`, which is finite
/// and nonzero where the forward result alone overflows or underflows (a
/// ratio `r / x` would give `inf` and `0` here).
#[test]
fn grad_pow_base_cotangent_survives_forward_overflow_and_underflow() {
    assert_eq!(pow_base_grad_at(Prim::F16, 300.0, 2.0), 600.0);
    assert_eq!(
        pow_base_grad_at(Prim::F32, 0.5, 150.0),
        150.0 * 2f64.powi(-149)
    );
    assert_eq!(
        pow_base_grad_at(Prim::F64, 0.5, 1075.0),
        1075.0 * f64::from_bits(1)
    );
    assert_eq!(pow_base_grad_at(Prim::F64, -2.0, 3.0), 12.0);
}

/// Central differences across the normal range, at f64: integer exponents
/// for either sign of base, non-integer ones for a positive base.
#[test]
fn grad_pow_base_matches_finite_differences_across_normal_ranges() {
    let (dag, x, _y, out) = pow_dag();
    let bases = [-3.0, -1.7, -0.9, -0.25, 0.25, 0.9, 1.7, 3.0];
    let exponents = [-4.0, -1.0, 1.0, 2.0, 5.0, 12.0, -2.5, 0.5, 3.75];
    let mut checked = 0;
    for x0 in bases {
        for y0 in exponents {
            if x0 < 0.0 && y0 != f64::round(y0) {
                continue;
            }
            let h = 1e-6 * f64::abs(x0);
            let (a, n) = finite_diff(&dag, out, x, "x", &[("y", y0)], x0, h);
            assert!(
                (a - n).abs() <= 1e-5 * a.abs().max(1.0),
                "d pow/dx at ({x0}, {y0}): analytical {a}, numerical {n}"
            );
            checked += 1;
        }
    }
    assert!(checked >= 60, "{checked}");
}

/// Negative partner: a negative base has no real exponent derivative, so its
/// exponent cotangent is NaN through `log`, while its base cotangent stays
/// the signed integer-power derivative.
#[test]
fn grad_pow_negative_base_exponent_cotangent_is_nan() {
    let (dx, dy) = pow_grads(-2.0, 3.0);
    assert_eq!(dx, 12.0);
    assert!(dy.is_nan(), "d pow/dy at (-2, 3) should be NaN, got {dy}");
}

#[test]
fn grad_div_rhs() {
    let (dag, _x, y, out) = build_binary_dag(|dag, owner, a, b, ty| {
        dag.add_node(owner, RiscOp::Div, vec![a, b], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, y, "y", &[("x", 2.0)], 4.0, 1e-5);
    assert_grad_close(a, n);
    assert!(
        (a - (-0.125)).abs() < 1e-6,
        "d(x/y)/dy at x=2,y=4 should be -0.125, got {a}"
    );
}

#[test]
fn grad_recip() {
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Recip, vec![a], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[], 2.0, 1e-5);
    assert_grad_close(a, n);
    assert!(
        (a - (-0.25)).abs() < 1e-6,
        "d(1/x)/dx at x=2 should be -0.25, got {a}"
    );
}

#[test]
fn grad_neg() {
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Neg, vec![a], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[], 2.0, 1e-5);
    assert_grad_close(a, n);
    assert!((a - (-1.0)).abs() < 1e-6); // d(-x)/dx = -1
}

#[test]
fn grad_exp() {
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Exp, vec![a], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[], 1.0, 1e-5);
    assert_grad_close(a, n);
    let expected = 1.0_f64.exp();
    assert!((a - expected).abs() < 1e-4); // d(exp(x))/dx = exp(x)
}

#[test]
fn grad_log() {
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Log, vec![a], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[], 2.0, 1e-5);
    assert_grad_close(a, n);
    assert!((a - 0.5).abs() < 1e-4); // d(log(x))/dx = 1/x = 0.5
}

#[test]
fn grad_sin() {
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Sin, vec![a], ty.clone(), None)
    });
    let x0 = 1.0;
    let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
    assert_grad_close(a, n);
    let expected = x0.cos();
    assert!((a - expected).abs() < 1e-4);
}

#[test]
fn grad_sqrt() {
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Sqrt, vec![a], ty.clone(), None)
    });
    let x0 = 4.0;
    let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
    assert_grad_close(a, n);
    let expected = 1.0 / (2.0 * x0.sqrt()); // = 0.25
    assert!((a - expected).abs() < 1e-4);
}

#[test]
fn grad_x_squared() {
    // f(x) = x * x, df/dx = 2x (tests accumulation: x used twice)
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let x_sq = dag.add_node(owner, RiscOp::Mul, vec![x, x], scalar_f64(), None);

    let (a, n) = finite_diff(&dag, x_sq, x, "x", &[], 3.0, 1e-5);
    assert_grad_close(a, n);
    assert!((a - 6.0).abs() < 1e-4); // 2 * 3 = 6
}

#[test]
fn grad_relu_positive() {
    // [05-OP-43]: d/dx relu(x) = 1 when x > 0.
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Relu, vec![a], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[], 2.0, 1e-5);
    assert_grad_close(a, n);
    assert!((a - 1.0).abs() < 1e-4);
}

#[test]
fn grad_relu_negative() {
    // [05-OP-43]: d/dx relu(x) = 0 when x < 0.
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Relu, vec![a], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[], -2.0, 1e-5);
    assert_grad_close(a, n);
    assert!(a.abs() < 1e-4);
}

#[test]
fn grad_chain_exp_neg() {
    // f(x) = exp(-x), df/dx = -exp(-x)
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        let neg = dag.add_node(owner, RiscOp::Neg, vec![a], ty.clone(), None);
        dag.add_node(owner, RiscOp::Exp, vec![neg], ty.clone(), None)
    });
    let x0 = 1.0;
    let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
    assert_grad_close(a, n);
    let expected = -(-x0).exp();
    assert!((a - expected).abs() < 1e-4);
}

#[test]
fn grad_multi_input() {
    // f(x, y, z) = x*y + z
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let y = dag.add_node(
        owner,
        RiscOp::Load { name: "y".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let z = dag.add_node(
        owner,
        RiscOp::Load { name: "z".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let xy = dag.add_node(owner, RiscOp::Mul, vec![x, y], scalar_f64(), None);
    let out = dag.add_node(owner, RiscOp::Add, vec![xy, z], scalar_f64(), None);

    let grad_result = grad_dag(&dag, out, &[x, y, z]).unwrap();
    let mut inputs = UnordMap::new();
    inputs.insert("x".to_string(), 2.0);
    inputs.insert("y".to_string(), 3.0);
    inputs.insert("z".to_string(), 5.0);
    let vals = eval_scalar(&grad_result.dag, &inputs);

    let dx = vals[&grad_result.grad_nodes[&x]];
    let dy = vals[&grad_result.grad_nodes[&y]];
    let dz = vals[&grad_result.grad_nodes[&z]];
    assert!((dx - 3.0).abs() < 1e-6); // d/dx(x*y + z) = y = 3
    assert!((dy - 2.0).abs() < 1e-6); // d/dy = x = 2
    assert!((dz - 1.0).abs() < 1e-6); // d/dz = 1
}

#[test]
fn grad_accumulation_3x() {
    // f(x) = x + x + x, df/dx = 3
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let sum1 = dag.add_node(owner, RiscOp::Add, vec![x, x], scalar_f64(), None);
    let out = dag.add_node(owner, RiscOp::Add, vec![sum1, x], scalar_f64(), None);

    let grad_result = grad_dag(&dag, out, &[x]).unwrap();
    let mut inputs = UnordMap::new();
    inputs.insert("x".to_string(), 7.0);
    let vals = eval_scalar(&grad_result.dag, &inputs);
    let dx = vals[&grad_result.grad_nodes[&x]];
    assert!((dx - 3.0).abs() < 1e-6);
}

#[test]
fn grad_accumulation_uses_forward_consumer_order_and_a_balanced_tree() {
    // The four uses of x have forward consumer order c1, c2, c3, c4.
    // With no base leaf, adjacent-pair balancing computes
    // (c1 + c2) + (c3 + c4) = 0 + 2 = 2 at f32. A positive-zero base leaf
    // would compute ((+0 + c1) + (c2 + c3)) + c4 = 1, and the historical
    // reverse-consumer left fold computes (((c4 + c3) + c2) + c1) = 0.
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let ty = scalar_f32();
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty.clone(),
        None,
    );
    let coefficients = [1.0e20, -1.0e20, 1.0, 1.0];
    let mut uses = Vec::new();
    for coefficient in coefficients {
        let constant = dag.add_node(
            owner,
            RiscOp::synth_const(Prim::F32, coefficient),
            vec![],
            ty.clone(),
            None,
        );
        uses.push(dag.add_node(owner, RiscOp::Mul, vec![x, constant], ty.clone(), None));
    }
    let left = dag.add_node(owner, RiscOp::Add, vec![uses[0], uses[1]], ty.clone(), None);
    let right = dag.add_node(owner, RiscOp::Add, vec![uses[2], uses[3]], ty.clone(), None);
    let output = dag.add_node(owner, RiscOp::Add, vec![left, right], ty, None);

    let grad_result = grad_dag(&dag, output, &[x]).expect("gradient");
    let values = eval_scalar(&grad_result.dag, &UnordMap::from([("x".to_string(), 0.0)]));

    assert_eq!(values[&grad_result.grad_nodes[&x]], 2.0);
}

#[test]
fn balanced_adjoint_sum_tracks_runtime_wildcard_shape() {
    use chelis_ir::eval::{TensorValue, eval_tensor};

    let concrete_ty = TensorType {
        dims: vec![DimInfo::Lit(3)],
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let source = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        concrete_ty.clone(),
        None,
    );
    let symbolic_ty = TensorType {
        dims: vec![DimInfo::Named("*".to_string(), None)],
        precision: Prim::F32,
    };
    let contribution_1 = dag.add_node(owner, RiscOp::Copy, vec![source], concrete_ty.clone(), None);
    let contribution_2 = dag.add_node(owner, RiscOp::Copy, vec![source], concrete_ty, None);
    let mut symbolic_forward = dag.get(source).expect("source exists").clone();
    symbolic_forward.output_type = symbolic_ty;
    let accumulated = balanced_adjoint_sum(
        &mut dag,
        &symbolic_forward,
        vec![contribution_1, contribution_2],
    );
    dag.add_root(accumulated);

    let values = eval_tensor(
        &dag,
        &UnordMap::from([(
            "x".to_string(),
            TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]),
        )]),
    )
    .expect("runtime-wildcard accumulation evaluates");
    let gradient = &values[&accumulated];

    assert_eq!(gradient.shape, vec![3]);
    assert_eq!(gradient.to_f64_lossy_vec(), vec![2.0, 4.0, 6.0]);
}

/// spec/06 §7.5: `where(cmplt(cast(conv(x), f32), 1), x * x, x)`, where
/// `conv` is the checked `cast` or a named cast rung to i32 (of
/// `cast(x, i64)` for `cast_wrap`, which reads only a signed integer),
/// optionally through integer `max_elem`. The conversion reaches the
/// output only through a comparison operand and a `where` condition, both
/// control slots. With `data_use`, the output also adds the restored
/// value, so the conversion has a data path as well.
fn comparison_with_discrete_cast(
    named: Option<chelis_numeric_core::named_cast::NamedCastMode>,
    through_integer_extrema: bool,
    data_use: bool,
) -> (Dag, NodeId, NodeId) {
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let scalar = |precision| TensorType {
        dims: vec![],
        precision,
    };
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f32(),
        None,
    );
    let source = if named == Some(chelis_numeric_core::named_cast::NamedCastMode::Wrap) {
        dag.add_node(
            owner,
            RiscOp::Cast {
                new_precision: Prim::Int64,
            },
            vec![x],
            scalar(Prim::Int64),
            None,
        )
    } else {
        x
    };
    let discrete = dag.add_node(
        owner,
        match named {
            Some(mode) => RiscOp::NamedCast {
                mode,
                new_precision: Prim::Int32,
            },
            None => RiscOp::Cast {
                new_precision: Prim::Int32,
            },
        },
        vec![source],
        scalar(Prim::Int32),
        None,
    );
    let discrete = if through_integer_extrema {
        let zero = dag.add_node(
            owner,
            RiscOp::synth_const(Prim::Int32, 0.0),
            vec![],
            scalar(Prim::Int32),
            None,
        );
        dag.add_node(
            owner,
            RiscOp::MaxElem,
            vec![discrete, zero],
            scalar(Prim::Int32),
            None,
        )
    } else {
        discrete
    };
    let restored = dag.add_node(
        owner,
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![discrete],
        scalar_f32(),
        None,
    );
    let one = dag.add_node(
        owner,
        RiscOp::synth_const(Prim::F32, 1.0),
        vec![],
        scalar_f32(),
        None,
    );
    let predicate = dag.add_node(
        owner,
        RiscOp::Compare(ComparisonKind::CmpLt),
        vec![restored, one],
        scalar(Prim::Bool),
        None,
    );
    let square = dag.add_node(owner, RiscOp::Mul, vec![x, x], scalar_f32(), None);
    let selected = dag.add_node(
        owner,
        RiscOp::Where,
        vec![predicate, square, x],
        scalar_f32(),
        None,
    );
    let out = if data_use {
        dag.add_node(
            owner,
            RiscOp::Add,
            vec![selected, restored],
            scalar_f32(),
            None,
        )
    } else {
        selected
    };
    (dag, x, out)
}

fn discrete_cast_variants()
-> impl Iterator<Item = (Option<chelis_numeric_core::named_cast::NamedCastMode>, bool)> {
    let rungs = chelis_numeric_core::named_cast::NamedCastMode::ALL
        .iter()
        .copied()
        .map(Some);
    std::iter::once(None)
        .chain(rungs)
        .flat_map(|named| [(named, false), (named, true)])
}

/// chelis#3464: a conversion read only through control slots executes
/// forward and contributes nothing, so `grad` differentiates the selected
/// branch, `2x` where the converted value is below one and `1` elsewhere.
/// This replaces the two units that pinned its PiecewiseConstant
/// rejection, which spec/06 §7.5 now confines to a data path.
#[test]
fn grad_conversion_read_only_through_control_slots_differentiates() {
    for (named, through_integer_extrema) in discrete_cast_variants() {
        let op = named.map_or("cast", |mode| mode.keyword());
        let (dag, x, out) = comparison_with_discrete_cast(named, through_integer_extrema, false);
        let result = match grad_dag_checked(&dag, out, &[x]) {
            Ok(result) => result,
            Err(error) => panic!("{op} read only by control slots must not reject: {error}"),
        };
        for (x0, expected) in [(-2.0, -4.0), (3.0, 1.0)] {
            let values = eval_scalar(&result.dag, &UnordMap::from([("x".to_string(), x0)]));
            assert_eq!(
                values[&result.grad_nodes[&x]], expected,
                "{op} (integer extrema: {through_integer_extrema}) at x = {x0}"
            );
        }
    }
}

/// A parameter read only through control slots receives no contribution
/// (spec/06 §7.5), so its gradient is the exact positive zero of a
/// parameter with none, built once for every consumer.
#[test]
fn grad_predicate_only_parameter_is_positive_zero() {
    let (mut dag, x, selected) = comparison_with_discrete_cast(None, false, false);
    let owner = dag.get(selected).unwrap().owner;
    // Read a second parameter only through a comparison and a `where`
    // condition: `where(cmplt(cast(cast(y, i32), f32), 1), out, -out)`.
    let y = dag.add_node(
        owner,
        RiscOp::Load { name: "y".into() },
        vec![],
        scalar_f32(),
        None,
    );
    let discrete = dag.add_node(
        owner,
        RiscOp::Cast {
            new_precision: Prim::Int32,
        },
        vec![y],
        TensorType {
            dims: vec![],
            precision: Prim::Int32,
        },
        None,
    );
    let restored = dag.add_node(
        owner,
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![discrete],
        scalar_f32(),
        None,
    );
    let one = dag.add_node(
        owner,
        RiscOp::synth_const(Prim::F32, 1.0),
        vec![],
        scalar_f32(),
        None,
    );
    let predicate = dag.add_node(
        owner,
        RiscOp::Compare(ComparisonKind::CmpLt),
        vec![restored, one],
        TensorType {
            dims: vec![],
            precision: Prim::Bool,
        },
        None,
    );
    let negated = dag.add_node(owner, RiscOp::Neg, vec![selected], scalar_f32(), None);
    let out = dag.add_node(
        owner,
        RiscOp::Where,
        vec![predicate, selected, negated],
        scalar_f32(),
        None,
    );
    let result = match grad_dag_checked(&dag, out, &[x, y]) {
        Ok(result) => result,
        Err(error) => panic!("a predicate-only parameter must not reject: {error}"),
    };
    for (y0, sign) in [(-5.0, 1.0), (4.0, -1.0)] {
        let values = eval_scalar(
            &result.dag,
            &UnordMap::from([("x".to_string(), -2.0), ("y".to_string(), y0)]),
        );
        assert_eq!(values[&result.grad_nodes[&x]], sign * -4.0, "y = {y0}");
        let dy = values[&result.grad_nodes[&y]];
        assert_eq!(dy.to_bits(), 0.0f64.to_bits(), "y = {y0}: {dy}");
    }
}

/// `mul(x, conversion(m))` over f32 scalars, where `conversion` is
/// `floor`, `ceil`, `round`, or the round trip `cast(cast(m, i32), f32)`.
fn product_with_converted(conversion: &str) -> (Dag, NodeId, NodeId, NodeId) {
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let load = |dag: &mut Dag, name: &str| {
        dag.add_node(
            owner,
            RiscOp::Load { name: name.into() },
            vec![],
            scalar_f32(),
            None,
        )
    };
    let x = load(&mut dag, "x");
    let m = load(&mut dag, "m");
    let converted = match conversion {
        "floor" => dag.add_node(owner, RiscOp::Floor, vec![m], scalar_f32(), None),
        "ceil" => dag.add_node(owner, RiscOp::Ceil, vec![m], scalar_f32(), None),
        "round" => dag.add_node(owner, RiscOp::Round, vec![m], scalar_f32(), None),
        "cast" => {
            let discrete = dag.add_node(
                owner,
                RiscOp::Cast {
                    new_precision: Prim::Int32,
                },
                vec![m],
                TensorType {
                    dims: vec![],
                    precision: Prim::Int32,
                },
                None,
            );
            dag.add_node(
                owner,
                RiscOp::Cast {
                    new_precision: Prim::F32,
                },
                vec![discrete],
                scalar_f32(),
                None,
            )
        }
        other => panic!("no conversion {other}"),
    };
    let out = dag.add_node(owner, RiscOp::Mul, vec![x, converted], scalar_f32(), None);
    (dag, x, m, out)
}

/// chelis#3487: a conversion of a parameter that is not differentiated is
/// inactive, a constant with respect to `wrt`, so `grad` returns the
/// converted value. Differentiating that parameter too makes the same
/// conversion active, and it rejects.
#[test]
fn grad_inactive_conversion_is_a_constant() {
    for (conversion, m0, expected) in [
        ("floor", 2.5, 2.0),
        ("ceil", 2.5, 3.0),
        ("round", 2.5, 2.0),
        ("cast", -3.0, -3.0),
    ] {
        let (dag, x, m, out) = product_with_converted(conversion);
        let result = match grad_dag_checked(&dag, out, &[x]) {
            Ok(result) => result,
            Err(error) => panic!("inactive {conversion} must not reject: {error}"),
        };
        let values = eval_scalar(
            &result.dag,
            &UnordMap::from([("x".to_string(), 1.5), ("m".to_string(), m0)]),
        );
        assert_eq!(values[&result.grad_nodes[&x]], expected, "{conversion}");
        let error = match grad_dag_checked(&dag, out, &[x, m]) {
            Ok(_) => panic!("active {conversion} must reject"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            AdError::NotSupported {
                op: conversion,
                reason: AdRejectionReason::PiecewiseConstant,
            }
        );
        assert!(grad_dag(&dag, out, &[x, m]).is_none());
    }
}

#[test]
fn grad_sum_expand() {
    // f(x) = sum(x, axis=0) for a 3-element vector, df/dx_i = 1
    use chelis_dag_core::dag::DimInfo;
    use chelis_ir::eval::{TensorValue, eval_tensor};

    let vec3_ty = TensorType {
        dims: vec![DimInfo::Lit(3)],
        precision: chelis_numeric_core::prim::Prim::F32,
    };
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec3_ty.clone(),
        None,
    );
    let out = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_numeric_core::prim::Prim::F32,
        },
        vec![x],
        scalar_f32(),
        None,
    );

    let grad_result = grad_dag(&dag, out, &[x]).unwrap();
    let mut inputs = UnordMap::new();
    inputs.insert(
        "x".to_string(),
        TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]),
    );
    let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
    let grad = &vals[&grad_result.grad_nodes[&x]];
    assert_eq!(grad.to_f64_lossy_vec(), vec![1.0, 1.0, 1.0]);
}

#[test]
fn grad_second_order() {
    // f(x) = x*x, f'(x) = 2x, f''(x) = 2
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let x_sq = dag.add_node(owner, RiscOp::Mul, vec![x, x], scalar_f64(), None);

    // First derivative
    let first = grad_dag(&dag, x_sq, &[x]).unwrap();
    let dx_node = first.grad_nodes[&x];

    // Second derivative: differentiate the first derivative DAG
    let second = grad_dag(&first.dag, dx_node, &[x]).unwrap();
    let ddx_node = second.grad_nodes[&x];

    let mut inputs = UnordMap::new();
    inputs.insert("x".to_string(), 5.0);
    let vals = eval_scalar(&second.dag, &inputs);
    let ddx = vals[&ddx_node];
    assert!((ddx - 2.0).abs() < 1e-4); // d^2(x^2)/dx^2 = 2
}

#[test]
fn grad_store_passthrough() {
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let exp_x = dag.add_node(owner, RiscOp::Exp, vec![x], scalar_f64(), None);
    let out = dag.add_node(
        owner,
        RiscOp::Store { name: "out".into() },
        vec![exp_x],
        scalar_f64(),
        None,
    );

    let (a, n) = finite_diff(&dag, out, x, "x", &[], 1.0, 1e-5);
    assert_grad_close(a, n);
}

#[test]
fn grad_cast_passthrough() {
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let f64_ty = TensorType {
        dims: vec![],
        precision: chelis_numeric_core::prim::Prim::F64,
    };
    let casted = dag.add_node(
        owner,
        RiscOp::Cast {
            new_precision: chelis_numeric_core::prim::Prim::F64,
        },
        vec![x],
        f64_ty,
        None,
    );
    let out = dag.add_node(owner, RiscOp::Exp, vec![casted], scalar_f64(), None);

    let grad_result = grad_dag(&dag, out, &[x]).unwrap();
    let mut inputs = UnordMap::new();
    inputs.insert("x".to_string(), 1.0);
    let vals = eval_scalar(&grad_result.dag, &inputs);
    let dx = vals[&grad_result.grad_nodes[&x]];
    let expected = 1.0_f64.exp();
    assert!((dx - expected).abs() < 1e-4);
}

/// chelis#2178 negative parity, dead-node half: the live-node scan
/// bounds the rejection to the gradient path. A float-to-integer
/// cast that the output does not reach is not on that path, so it
/// must not reject -- otherwise the fix would turn unrelated index
/// and metadata arithmetic into a grad failure.
#[test]
fn grad_cast_float_to_integer_off_the_gradient_path_is_accepted() {
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f32(),
        None,
    );
    // Dead: nothing downstream of `out` reaches this cast.
    let _dead = dag.add_node(
        owner,
        RiscOp::Cast {
            new_precision: Prim::Int32,
        },
        vec![x],
        TensorType {
            dims: vec![],
            precision: Prim::Int32,
        },
        None,
    );
    let out = dag.add_node(owner, RiscOp::Mul, vec![x, x], scalar_f32(), None);

    let grad_result =
        grad_dag_checked(&dag, out, &[x]).expect("a dead float-to-int cast must not reject grad");
    let mut inputs = UnordMap::new();
    inputs.insert("x".to_string(), 3.0);
    let vals = eval_scalar(&grad_result.dag, &inputs);
    let dx = vals[&grad_result.grad_nodes[&x]];
    assert!(
        (dx - 6.0).abs() < 1e-6,
        "d(x*x)/dx at x=3 must be 6.0; got {dx}"
    );
}

/// chelis#2178 negative parity, discrete-source half: [04-NUM-14]
/// says "A bool or integer source is a discrete forward-only value
/// and carries no cotangent, irrespective of target." That is not a
/// suppressed gradient, so an integer-to-float cast keeps its zero
/// seed and must NOT reject -- the behaviour
/// `grad_zero_placeholder_matrix::grad_without_abs_is_correct_in_both_lanes`
/// depends on.
#[test]
fn grad_cast_integer_source_is_accepted_and_contributes_zero() {
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let w = dag.add_node(
        owner,
        RiscOp::Load { name: "w".into() },
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::Int32,
        },
        None,
    );
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f32(),
        None,
    );
    let w_f32 = dag.add_node(
        owner,
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![w],
        scalar_f32(),
        None,
    );
    let out = dag.add_node(owner, RiscOp::Mul, vec![x, w_f32], scalar_f32(), None);

    let grad_result = grad_dag_checked(&dag, out, &[x, w])
        .expect("[04-NUM-14]: an integer source carries no cotangent, it does not reject");
    let mut inputs = UnordMap::new();
    inputs.insert("x".to_string(), 2.0);
    inputs.insert("w".to_string(), 5.0);
    let vals = eval_scalar(&grad_result.dag, &inputs);
    let dx = vals[&grad_result.grad_nodes[&x]];
    assert!(
        (dx - 5.0).abs() < 1e-6,
        "d(x*w)/dx must be w = 5.0; got {dx}"
    );
    let dw = vals[&grad_result.grad_nodes[&w]];
    assert!(
        dw.abs() < 1e-6,
        "an integer source carries no cotangent; got {dw}"
    );
}

#[test]
fn grad_reshape_roundtrip() {
    use chelis_ir::eval::{TensorValue, eval_tensor};

    for source_dim in [
        DimInfo::Lit(6),
        DimInfo::Named("n".into(), None),
        DimInfo::Named("*".into(), None),
        DimInfo::Named(String::new(), None),
    ] {
        let vec6_ty = TensorType {
            dims: vec![source_dim],
            precision: chelis_numeric_core::prim::Prim::F32,
        };
        let mat23_ty = TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            precision: chelis_numeric_core::prim::Prim::F32,
        };

        let mut dag = Dag::new();
        let owner = Owner::from(dag.declare("test"));
        let x = dag.add_node(
            owner,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec6_ty.clone(),
            None,
        );
        let reshaped = dag.add_node(
            owner,
            RiscOp::Reshape {
                new_shape: mat23_ty.dims.iter().map(RtDim::from_dim_info).collect(),
            },
            vec![x],
            mat23_ty.clone(),
            None,
        );
        // Sum all elements to get a scalar
        let sum0 = dag.add_node(
            owner,
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_numeric_core::prim::Prim::F32,
            },
            vec![reshaped],
            TensorType {
                dims: vec![DimInfo::Lit(3)],
                precision: chelis_numeric_core::prim::Prim::F32,
            },
            None,
        );
        let out = dag.add_node(
            owner,
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_numeric_core::prim::Prim::F32,
            },
            vec![sum0],
            scalar_f32(),
            None,
        );

        let grad_result = grad_dag(&dag, out, &[x]).unwrap();
        let mut inputs = UnordMap::new();
        inputs.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );
        let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
        let grad = &vals[&grad_result.grad_nodes[&x]];
        // d(sum(reshape(x)))/dx = ones
        assert_eq!(grad.to_f64_lossy_vec(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(grad.shape, vec![6]);
        inputs.insert("x".into(), TensorValue::from_vec(vec![5], vec![1.0; 5]));
        assert!(
            eval_tensor(&grad_result.dag, &inputs).is_err(),
            "a source with five elements cannot reshape to [2, 3]"
        );
    }
}

#[test]
fn grad_permute_transpose() {
    use chelis_ir::eval::{TensorValue, eval_tensor};

    let mat23_ty = TensorType {
        dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
        precision: chelis_numeric_core::prim::Prim::F32,
    };
    let mat32_ty = TensorType {
        dims: vec![DimInfo::Lit(3), DimInfo::Lit(2)],
        precision: chelis_numeric_core::prim::Prim::F32,
    };

    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        mat23_ty.clone(),
        None,
    );
    let transposed = dag.add_node(
        owner,
        RiscOp::Permute { axes: vec![1, 0] },
        vec![x],
        mat32_ty.clone(),
        None,
    );
    // Sum all elements for a scalar output
    let sum0 = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_numeric_core::prim::Prim::F32,
        },
        vec![transposed],
        TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: chelis_numeric_core::prim::Prim::F32,
        },
        None,
    );
    let out = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_numeric_core::prim::Prim::F32,
        },
        vec![sum0],
        scalar_f32(),
        None,
    );

    let grad_result = grad_dag(&dag, out, &[x]).unwrap();
    let mut inputs = UnordMap::new();
    inputs.insert(
        "x".to_string(),
        TensorValue::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    );
    let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
    let grad = &vals[&grad_result.grad_nodes[&x]];
    // d(sum(permute(x)))/dx = ones, shape should be 2x3
    assert_eq!(grad.to_f64_lossy_vec(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    assert_eq!(grad.shape, vec![2, 3]);
}

#[test]
fn scatter_add_adjoint_preserves_target_and_gathers_update_cotangents() {
    use chelis_ir::eval::{TensorValue, eval_tensor};

    let vec4_ty = TensorType {
        dims: vec![DimInfo::Lit(4)],
        precision: Prim::F32,
    };
    let vec2_ty = TensorType {
        dims: vec![DimInfo::Lit(2)],
        precision: Prim::F32,
    };
    let index_ty = TensorType {
        dims: vec![DimInfo::Lit(2)],
        precision: Prim::Int64,
    };
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let target = dag.add_node(
        owner,
        RiscOp::Load {
            name: "target".into(),
        },
        vec![],
        vec4_ty.clone(),
        None,
    );
    let indices = dag.add_node(
        owner,
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        index_ty,
        None,
    );
    let updates = dag.add_node(
        owner,
        RiscOp::Load {
            name: "updates".into(),
        },
        vec![],
        vec2_ty,
        None,
    );
    let scattered = dag.add_node(
        owner,
        RiscOp::ScatterAdd {
            axis: 0,
            batch_rank: 0,
        },
        vec![target, indices, updates],
        vec4_ty.clone(),
        None,
    );
    let coefficients = dag.add_node(
        owner,
        RiscOp::synth_const_tensor(Prim::F32, vec![2.0, 3.0, 5.0, 7.0]),
        vec![],
        vec4_ty.clone(),
        None,
    );
    let weighted = dag.add_node(
        owner,
        RiscOp::Mul,
        vec![scattered, coefficients],
        vec4_ty,
        None,
    );
    let output = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![weighted],
        scalar_f32(),
        None,
    );

    let gradient = grad_dag(&dag, output, &[target, updates]).expect("scatter-add gradient");
    let values = eval_tensor(
        &gradient.dag,
        &UnordMap::from([
            (
                "target".to_string(),
                TensorValue::from_vec(vec![4], vec![11.0, 13.0, 17.0, 19.0]),
            ),
            (
                "indices".to_string(),
                TensorValue::from_vec(vec![2], vec![1.0, 3.0]),
            ),
            (
                "updates".to_string(),
                TensorValue::from_vec(vec![2], vec![23.0, 29.0]),
            ),
        ]),
    )
    .expect("scatter-add adjoint evaluates");

    assert_eq!(
        values[&gradient.grad_nodes[&target]].to_f64_lossy_vec(),
        vec![2.0, 3.0, 5.0, 7.0],
    );
    assert_eq!(
        values[&gradient.grad_nodes[&updates]].to_f64_lossy_vec(),
        vec![3.0, 7.0],
    );
}

#[test]
fn grad_expand_sum_roundtrip() {
    use chelis_ir::eval::{TensorValue, eval_tensor};

    let vec3_ty = TensorType {
        dims: vec![DimInfo::Lit(3)],
        precision: chelis_numeric_core::prim::Prim::F32,
    };
    let mat23_ty = TensorType {
        dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
        precision: chelis_numeric_core::prim::Prim::F32,
    };

    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec3_ty.clone(),
        None,
    );
    let expanded = dag.add_node(
        owner,
        RiscOp::Expand {
            axis: 0,
            size: chelis_dag_core::dag::RtDim::Lit(2),
        },
        vec![x],
        mat23_ty.clone(),
        None,
    );
    // Sum back to scalar
    let sum0 = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_numeric_core::prim::Prim::F32,
        },
        vec![expanded],
        vec3_ty.clone(),
        None,
    );
    let out = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_numeric_core::prim::Prim::F32,
        },
        vec![sum0],
        scalar_f32(),
        None,
    );

    let grad_result = grad_dag(&dag, out, &[x]).unwrap();
    let mut inputs = UnordMap::new();
    inputs.insert(
        "x".to_string(),
        TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]),
    );
    let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
    let grad = &vals[&grad_result.grad_nodes[&x]];
    // expand by 2 then sum -> each element counted twice -> gradient = 2
    assert_eq!(grad.to_f64_lossy_vec(), vec![2.0, 2.0, 2.0]);
    assert_eq!(grad.shape, vec![3]);
}

/// The same-rank `Expand` adjoint, which a user program can reach for the
/// first time under `spec/04-type-system.md` section 4.7.2.
///
/// Both forward forms have always had an adjoint here, keyed on the node's
/// output rank against its operand's, but the same-rank branch was
/// reachable only from `tier2::align_matmul_operand`'s synthesized nodes:
/// lowering rewrote every surface `expand` into the rank-increasing form.
/// With one meaning per operation a surface `expand` lowers to a same-rank
/// node, so this branch now carries user gradients and owes a test.
///
/// Section 2.4 of `spec/05-risc-primitives.md` states the adjoint as
/// `insert(sum(g, axis), axis, 1i64)`: sum over the broadcast axis, then
/// restore the extent-1 slot so the cotangent keeps the operand's rank.
/// Each input element is read once per broadcast position, so a
/// `[2, 1] -> [2, 3]` broadcast summed to a scalar has gradient 3 at every
/// input element, at the operand's own shape.
///
/// This is a DISPOSITION LOCK, not a regression test, and the diff is the
/// evidence: S2b adds no line to this file outside this test, so the
/// branch it exercises is byte-identical to the base and the row passes
/// there too. What changed is who can reach the branch.
#[test]
fn grad_same_rank_expand_sums_over_the_broadcast_axis() {
    use chelis_ir::eval::{TensorValue, eval_tensor};

    let col_ty = TensorType {
        dims: vec![DimInfo::Lit(2), DimInfo::Lit(1)],
        precision: chelis_numeric_core::prim::Prim::F32,
    };
    let wide_ty = TensorType {
        dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
        precision: chelis_numeric_core::prim::Prim::F32,
    };
    let rows_ty = TensorType {
        dims: vec![DimInfo::Lit(2)],
        precision: chelis_numeric_core::prim::Prim::F32,
    };

    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        col_ty.clone(),
        None,
    );
    let broadcast = dag.add_node(
        owner,
        RiscOp::Expand {
            axis: 1,
            size: chelis_dag_core::dag::RtDim::Lit(3),
        },
        vec![x],
        wide_ty,
        None,
    );
    let per_row = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_numeric_core::prim::Prim::F32,
        },
        vec![broadcast],
        rows_ty,
        None,
    );
    let out = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_numeric_core::prim::Prim::F32,
        },
        vec![per_row],
        scalar_f32(),
        None,
    );

    let grad_result = grad_dag(&dag, out, &[x]).unwrap();
    let mut inputs = UnordMap::new();
    inputs.insert(
        "x".to_string(),
        TensorValue::from_vec(vec![2, 1], vec![5.0, 7.0]),
    );
    let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
    let grad = &vals[&grad_result.grad_nodes[&x]];
    assert_eq!(grad.to_f64_lossy_vec(), vec![3.0, 3.0]);
    assert_eq!(
        grad.shape,
        vec![2, 1],
        "the adjoint restores the extent-1 slot, so the cotangent keeps the \
             operand's rank"
    );
}

#[test]
fn grad_mul_by_const() {
    // f(x) = 5 * x, df/dx = 5
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let five = dag.add_node(
        owner,
        RiscOp::synth_const(scalar_f64().precision, 5.0),
        vec![],
        scalar_f64(),
        None,
    );
    let out = dag.add_node(owner, RiscOp::Mul, vec![five, x], scalar_f64(), None);

    let (a, n) = finite_diff(&dag, out, x, "x", &[], 3.0, 1e-5);
    assert_grad_close(a, n);
    assert!((a - 5.0).abs() < 1e-4);
}

#[test]
fn grad_log_chain() {
    // f(x) = log(x^2) = 2*log(x), df/dx = 2/x
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let x_sq = dag.add_node(owner, RiscOp::Mul, vec![x, x], scalar_f64(), None);
    let out = dag.add_node(owner, RiscOp::Log, vec![x_sq], scalar_f64(), None);

    let x0 = 3.0;
    let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
    assert_grad_close(a, n);
    let expected = 2.0 / x0;
    assert!((a - expected).abs() < 1e-4);
}

#[test]
fn grad_sin_chain() {
    // f(x) = sin(x^2), df/dx = 2x * cos(x^2)
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let x_sq = dag.add_node(owner, RiscOp::Mul, vec![x, x], scalar_f64(), None);
    let out = dag.add_node(owner, RiscOp::Sin, vec![x_sq], scalar_f64(), None);

    let x0 = 1.5;
    let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
    assert_grad_close(a, n);
}

#[test]
fn grad_max_elem_symmetric() {
    // f(x, y) = max(x, y), test that d/dy = 1 when y > x
    let (dag, _x, y, out) = build_binary_dag(|dag, owner, a, b, ty| {
        dag.add_node(owner, RiscOp::MaxElem, vec![a, b], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, y, "y", &[("x", 1.0)], 5.0, 1e-5);
    assert_grad_close(a, n);
    assert!((a - 1.0).abs() < 1e-4); // y > x, so d/dy = 1
}

/// `where(lt(5, x), x * log(x), x)` with the product and the `log` under
/// the arm's activation, as `lower_if` builds them.
fn arm_of_x_log_x() -> (Dag, NodeId, NodeId) {
    build_unary_dag(|dag, owner, x, ty| {
        let five = dag.add_node(
            owner,
            RiscOp::synth_const(Prim::F64, 5.0),
            vec![],
            ty.clone(),
            None,
        );
        let condition = dag.add_node(
            owner,
            RiscOp::Compare(ComparisonKind::CmpLt),
            vec![five, x],
            TensorType {
                dims: vec![],
                precision: Prim::Bool,
            },
            None,
        );
        let arm = Owner::new(owner.decl, Some(condition));
        let log = dag.add_node(arm, RiscOp::Log, vec![x], ty.clone(), None);
        let product = dag.add_node(arm, RiscOp::Mul, vec![x, log], ty.clone(), None);
        dag.add_node(
            owner,
            RiscOp::Where,
            vec![condition, product, x],
            ty.clone(),
            None,
        )
    })
}

/// spec/06 §2.10.1: at `x = 0` the arm is not taken and its product's
/// adjoint for `x` is `0 * log(0) = 0 * -inf`; the mask selects it away,
/// so the gradient is the else arm's 1. At `x = 7` the arm is taken and
/// the gradient is `log(7) + 1`.
///
/// Evidentiary status: REGRESSION TEST (NaN at `x = 0` without
/// [`mask_to_activation`]).
#[test]
fn an_untaken_arms_contribution_is_selected_away_by_its_activation() {
    let (dag, x, out) = arm_of_x_log_x();
    let grad_result = grad_dag(&dag, out, &[x]).unwrap();
    let gradient = |at: f64| {
        let inputs = UnordMap::from_iter([("x".to_string(), at)]);
        eval_scalar(&grad_result.dag, &inputs)[&grad_result.grad_nodes[&x]]
    };
    assert_eq!(gradient(0.0), 1.0);
    assert_grad_close(gradient(7.0), 7.0f64.ln() + 1.0);
}

#[test]
fn branch_broadcast_masks_rows_before_reducing_a_scalar_capture() {
    use chelis_ir::eval::{TensorValue, eval_tensor};

    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let ty = TensorType {
        dims: vec![DimInfo::Lit(3)],
        precision: Prim::F64,
    };
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let rows = dag.add_node(
        owner,
        RiscOp::Load {
            name: "rows".into(),
        },
        vec![],
        ty.clone(),
        None,
    );
    let zero = dag.add_node(
        owner,
        RiscOp::synth_const(Prim::F64, 0.0),
        vec![],
        ty.clone(),
        None,
    );
    let condition = dag.add_node(
        owner,
        RiscOp::Compare(ComparisonKind::CmpLt),
        vec![zero, rows],
        TensorType {
            dims: ty.dims.clone(),
            precision: Prim::Bool,
        },
        None,
    );
    let expanded = dag.add_node(
        Owner::new(owner.decl, Some(condition)),
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Lit(3),
        },
        vec![x],
        ty.clone(),
        None,
    );
    let chosen = dag.add_node(
        owner,
        RiscOp::Where,
        vec![condition, expanded, zero],
        ty,
        None,
    );
    let out = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F64,
        },
        vec![chosen],
        scalar_f64(),
        None,
    );
    let result = grad_dag_checked(&dag, out, &[x]).unwrap();
    for (rows, expected) in [(vec![1.0, -1.0, 1.0], 2.0), (vec![-1.0; 3], 0.0)] {
        let values = eval_tensor(
            &result.dag,
            &UnordMap::from([
                ("x".to_string(), TensorValue::from_vec(vec![], vec![7.0])),
                ("rows".to_string(), TensorValue::from_vec(vec![3], rows)),
            ]),
        )
        .unwrap();
        assert_eq!(
            values[&result.grad_nodes[&x]].to_f64_lossy_vec(),
            vec![expected]
        );
    }
}

#[test]
fn runtime_recurrence_callback_vjp_uses_explicit_tensor_seed() {
    use chelis_ir::eval::{TensorValue, eval_tensor};

    let ty = TensorType {
        dims: vec![DimInfo::Lit(2)],
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("recurrence_step"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty.clone(),
        None,
    );
    let y = dag.add_node(
        owner,
        RiscOp::Load { name: "y".into() },
        vec![],
        ty.clone(),
        None,
    );
    let seed = dag.add_node(
        owner,
        RiscOp::Load {
            name: "seed".into(),
        },
        vec![],
        ty.clone(),
        None,
    );
    let output = dag.add_node(owner, RiscOp::Mul, vec![x, y], ty, None);
    let vjp = vjp_dag_checked(&dag, output, seed, &[x, y]).expect("tensor VJP");
    let inputs = UnordMap::from([
        ("x".into(), TensorValue::from_vec(vec![2], vec![2.0, 3.0])),
        ("y".into(), TensorValue::from_vec(vec![2], vec![4.0, 5.0])),
        (
            "seed".into(),
            TensorValue::from_vec(vec![2], vec![7.0, 11.0]),
        ),
    ]);
    let values = eval_tensor(&vjp.dag, &inputs).expect("evaluate VJP");
    assert_eq!(
        values[&vjp.grad_nodes[&x]].to_f64_lossy_vec(),
        vec![28.0, 55.0]
    );
    assert_eq!(
        values[&vjp.grad_nodes[&y]].to_f64_lossy_vec(),
        vec![14.0, 33.0]
    );
}

#[test]
fn adv_wrt_non_leaf_node() {
    // Ask for gradient wrt a non-leaf node (e.g., an Add node)
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let y = dag.add_node(
        owner,
        RiscOp::Load { name: "y".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let add_node = dag.add_node(owner, RiscOp::Add, vec![x, y], scalar_f64(), None);
    let out = dag.add_node(owner, RiscOp::Exp, vec![add_node], scalar_f64(), None);

    // wrt the Add node, not a leaf
    let grad_result = grad_dag(&dag, out, &[add_node]).unwrap();
    // This should work -- it's asking for d(out)/d(add_node).
    // The answer should be exp(x+y), i.e., exp evaluated at the add node.
    let mut inputs = UnordMap::new();
    inputs.insert("x".to_string(), 1.0);
    inputs.insert("y".to_string(), 2.0);
    let vals = eval_scalar(&grad_result.dag, &inputs);
    let d_add = vals[&grad_result.grad_nodes[&add_node]];
    let expected = (1.0_f64 + 2.0).exp();
    assert!(
        (d_add - expected).abs() < 1e-4,
        "wrt non-leaf: expected {expected}, got {d_add}"
    );
}

// ---- Category B: Numerical correctness at specific values ----

#[test]
fn adv_exp_at_2() {
    // d(exp(x))/dx at x=2.0 should be exp(2) ≈ 7.389
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Exp, vec![a], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[], 2.0, 1e-5);
    assert_grad_close(a, n);
    let expected = 2.0_f64.exp();
    assert!(
        (a - expected).abs() < 1e-4,
        "d(exp(x))/dx at x=2: expected {expected}, got {a}"
    );
}

#[test]
fn adv_log_at_0_5() {
    // d(log(x))/dx at x=0.5 should be 2.0
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Log, vec![a], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[], 0.5, 1e-5);
    assert_grad_close(a, n);
    assert!(
        (a - 2.0).abs() < 1e-4,
        "d(log(x))/dx at x=0.5: expected 2.0, got {a}"
    );
}

#[test]
fn adv_sin_at_1() {
    // d(sin(x))/dx at x=1.0 should be cos(1) ≈ 0.5403
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Sin, vec![a], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[], 1.0, 1e-5);
    assert_grad_close(a, n);
    let expected = 1.0_f64.cos();
    assert!(
        (a - expected).abs() < 1e-4,
        "d(sin(x))/dx at x=1: expected {expected}, got {a}"
    );
}

#[test]
fn adv_sqrt_at_4() {
    // d(sqrt(x))/dx at x=4.0 should be 0.25
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Sqrt, vec![a], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[], 4.0, 1e-5);
    assert_grad_close(a, n);
    assert!(
        (a - 0.25).abs() < 1e-4,
        "d(sqrt(x))/dx at x=4: expected 0.25, got {a}"
    );
}

#[test]
fn adv_max_x_gt_y() {
    // d(max(x,y))/dx at x=3, y=1 should be 1.0
    let (dag, x, _y, out) = build_binary_dag(|dag, owner, a, b, ty| {
        dag.add_node(owner, RiscOp::MaxElem, vec![a, b], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[("y", 1.0)], 3.0, 1e-5);
    assert_grad_close(a, n);
    assert!(
        (a - 1.0).abs() < 1e-4,
        "d(max(x,y))/dx at x=3,y=1: expected 1.0, got {a}"
    );
}

#[test]
fn adv_max_x_lt_y() {
    // d(max(x,y))/dx at x=1, y=3 should be 0.0
    let (dag, x, _y, out) = build_binary_dag(|dag, owner, a, b, ty| {
        dag.add_node(owner, RiscOp::MaxElem, vec![a, b], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[("y", 3.0)], 1.0, 1e-5);
    assert_grad_close(a, n);
    assert!(
        a.abs() < 1e-4,
        "d(max(x,y))/dx at x=1,y=3: expected 0.0, got {a}"
    );
}

#[test]
fn adv_max_equal_inputs() {
    // Phase 0 tie-break convention routes the gradient to the left-hand side.
    let (dag, x, y, out) = build_binary_dag(|dag, owner, a, b, ty| {
        dag.add_node(owner, RiscOp::MaxElem, vec![a, b], ty.clone(), None)
    });
    let grad_result = grad_dag(&dag, out, &[x, y]).unwrap();
    let mut inputs = UnordMap::new();
    inputs.insert("x".to_string(), 2.0);
    inputs.insert("y".to_string(), 2.0);
    let vals = eval_scalar(&grad_result.dag, &inputs);
    let dx = vals[&grad_result.grad_nodes[&x]];
    let dy = vals[&grad_result.grad_nodes[&y]];
    assert!(
        (dx - 1.0).abs() < 1e-6,
        "expected left tie gradient 1.0, got {dx}"
    );
    assert!(dy.abs() < 1e-6, "expected right tie gradient 0.0, got {dy}");
}

// ---- Category C: Composed operations ----

#[test]
fn adv_exp_log_identity() {
    // d(exp(log(x)))/dx should be 1.0 (since exp(log(x)) = x)
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        let log_x = dag.add_node(owner, RiscOp::Log, vec![a], ty.clone(), None);
        dag.add_node(owner, RiscOp::Exp, vec![log_x], ty.clone(), None)
    });
    let x0 = 2.7;
    let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
    assert_grad_close(a, n);
    assert!(
        (a - 1.0).abs() < 1e-3,
        "d(exp(log(x)))/dx: expected 1.0, got {a}"
    );
}

#[test]
fn adv_x_times_exp_x() {
    // d(x * exp(x))/dx = (1+x) * exp(x)
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        let exp_x = dag.add_node(owner, RiscOp::Exp, vec![a], ty.clone(), None);
        dag.add_node(owner, RiscOp::Mul, vec![a, exp_x], ty.clone(), None)
    });
    let x0 = 1.5;
    let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
    assert_grad_close(a, n);
    let expected = (1.0 + x0) * x0.exp();
    assert!(
        (a - expected).abs() < 1e-3,
        "d(x*exp(x))/dx at x={x0}: expected {expected}, got {a}"
    );
}

#[test]
fn adv_sin_x_squared() {
    // d(sin(x^2))/dx = 2x * cos(x^2)
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        let x_sq = dag.add_node(owner, RiscOp::Mul, vec![a, a], ty.clone(), None);
        dag.add_node(owner, RiscOp::Sin, vec![x_sq], ty.clone(), None)
    });
    let x0 = 1.5;
    let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
    assert_grad_close(a, n);
    let expected = 2.0 * x0 * (x0 * x0).cos();
    assert!(
        (a - expected).abs() < 1e-3,
        "d(sin(x^2))/dx at x={x0}: expected {expected}, got {a}"
    );
}

#[test]
fn adv_sigmoid_gradient() {
    // sigmoid(x) = 1 / (1 + exp(-x))
    // Lowered as: div(1, add(1, exp(neg(x))))
    // d(sigmoid)/dx = sigmoid * (1 - sigmoid)
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        let neg_x = dag.add_node(owner, RiscOp::Neg, vec![a], ty.clone(), None);
        let exp_neg_x = dag.add_node(owner, RiscOp::Exp, vec![neg_x], ty.clone(), None);
        let one = dag.add_node(
            owner,
            RiscOp::synth_const(ty.precision, 1.0),
            vec![],
            ty.clone(),
            None,
        );
        let denom = dag.add_node(owner, RiscOp::Add, vec![one, exp_neg_x], ty.clone(), None);
        // div(1, denom) = 1 * recip(denom) = exp(-log(denom))
        chelis_dag_core::tier2::lower_div(owner, dag, one, denom, ty, None)
    });
    let x0 = -0.3;
    let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
    assert_grad_close(a, n);
    let sig = 1.0 / (1.0 + (-x0).exp());
    let expected = sig * (1.0 - sig);
    assert!(
        (a - expected).abs() < 1e-3,
        "sigmoid gradient at x={x0}: expected {expected}, got {a}"
    );
}

// ---- Category D: Edge cases ----

#[test]
fn adv_all_zero_grad_out() {
    // If grad_out is Const(0) everywhere (degenerate), gradient should still compute.
    // This happens when the output doesn't depend on x but we ask for dx anyway.
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let c = dag.add_node(
        owner,
        RiscOp::synth_const(scalar_f64().precision, 42.0),
        vec![],
        scalar_f64(),
        None,
    );
    // output is a constant, doesn't depend on x
    let grad_result = grad_dag(&dag, c, &[x]).unwrap();
    // x might not even be in grad_nodes since no path from c to x
    if let Some(&gx) = grad_result.grad_nodes.get(&x) {
        let mut inputs = UnordMap::new();
        inputs.insert("x".to_string(), 7.0);
        let vals = eval_scalar(&grad_result.dag, &inputs);
        let dx = vals[&gx];
        assert!(
            dx.abs() < 1e-10,
            "gradient of constant wrt x should be 0, got {dx}"
        );
    }
    // If x is not in grad_nodes, that's also fine (no path)
}

#[test]
fn adv_node_used_5_times() {
    // add(add(add(add(x, x), x), x), x) = 5x, gradient should be 5
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let s1 = dag.add_node(owner, RiscOp::Add, vec![x, x], scalar_f64(), None);
    let s2 = dag.add_node(owner, RiscOp::Add, vec![s1, x], scalar_f64(), None);
    let s3 = dag.add_node(owner, RiscOp::Add, vec![s2, x], scalar_f64(), None);
    let out = dag.add_node(owner, RiscOp::Add, vec![s3, x], scalar_f64(), None);

    let grad_result = grad_dag(&dag, out, &[x]).unwrap();
    let mut inputs = UnordMap::new();
    inputs.insert("x".to_string(), 2.7);
    let vals = eval_scalar(&grad_result.dag, &inputs);
    let dx = vals[&grad_result.grad_nodes[&x]];
    assert!(
        (dx - 5.0).abs() < 1e-6,
        "5x gradient: expected 5.0, got {dx}"
    );
}

#[test]
fn adv_deep_neg_chain_even() {
    // neg(neg(neg(neg(x)))) = x, gradient = 1.0
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        let n1 = dag.add_node(owner, RiscOp::Neg, vec![a], ty.clone(), None);
        let n2 = dag.add_node(owner, RiscOp::Neg, vec![n1], ty.clone(), None);
        let n3 = dag.add_node(owner, RiscOp::Neg, vec![n2], ty.clone(), None);
        dag.add_node(owner, RiscOp::Neg, vec![n3], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[], 2.7, 1e-5);
    assert_grad_close(a, n);
    assert!(
        (a - 1.0).abs() < 1e-6,
        "4x neg gradient: expected 1.0, got {a}"
    );
}

#[test]
fn adv_deep_neg_chain_odd() {
    // neg(neg(neg(x))) = -x, gradient = -1.0
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        let n1 = dag.add_node(owner, RiscOp::Neg, vec![a], ty.clone(), None);
        let n2 = dag.add_node(owner, RiscOp::Neg, vec![n1], ty.clone(), None);
        dag.add_node(owner, RiscOp::Neg, vec![n2], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[], 2.7, 1e-5);
    assert_grad_close(a, n);
    assert!(
        (a - (-1.0)).abs() < 1e-6,
        "3x neg gradient: expected -1.0, got {a}"
    );
}

// ---- Category E: MaxReduce adjoint (critical for MNIST) ----

#[test]
fn adv_max_reduce_gradient() {
    // Build a 2x3 tensor, apply max_reduce on axis 0, then sum to scalar.
    // Check gradient via finite differences on each element.
    use chelis_ir::eval::{TensorValue, eval_tensor};

    let mat23_ty = TensorType {
        dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
        precision: chelis_numeric_core::prim::Prim::F64,
    };
    let vec3_ty = TensorType {
        dims: vec![DimInfo::Lit(3)],
        precision: chelis_numeric_core::prim::Prim::F64,
    };

    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        mat23_ty.clone(),
        None,
    );
    let maxr = dag.add_node(
        owner,
        RiscOp::MaxReduce { axis: 0 },
        vec![x],
        vec3_ty.clone(),
        None,
    );
    let out = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_numeric_core::prim::Prim::F64,
        },
        vec![maxr],
        scalar_f64(),
        None,
    );

    let grad_result = grad_dag(&dag, out, &[x]).unwrap();

    // Input: [[1, 5, 3], [4, 2, 6]]
    // max_reduce(axis=0) = [4, 5, 6]
    // sum = 15
    // Gradient: 1 where element equals max along axis 0, 0 otherwise
    // Expected gradient: [[0, 1, 0], [1, 0, 1]]
    let input_data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let mut inputs = UnordMap::new();
    inputs.insert(
        "x".to_string(),
        TensorValue::from_vec(vec![2, 3], input_data.clone()),
    );
    let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
    let grad = &vals[&grad_result.grad_nodes[&x]];

    let expected_grad = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    for (i, (actual, expected)) in grad
        .to_f64_lossy_vec()
        .iter()
        .zip(expected_grad.iter())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() < 1e-6,
            "max_reduce gradient at index {i}: expected {expected}, got {actual}"
        );
    }

    // Also verify via finite differences on each element
    let h = 1e-5;
    for elem_idx in 0..6 {
        let mut data_plus = input_data.clone();
        data_plus[elem_idx] += h;
        let mut data_minus = input_data.clone();
        data_minus[elem_idx] -= h;

        let mut inputs_plus = UnordMap::new();
        inputs_plus.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![2, 3], data_plus),
        );
        let vals_plus = eval_tensor(&dag, &inputs_plus).unwrap();
        let f_plus = vals_plus[&out].to_f64_lossy_vec()[0];

        let mut inputs_minus = UnordMap::new();
        inputs_minus.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![2, 3], data_minus),
        );
        let vals_minus = eval_tensor(&dag, &inputs_minus).unwrap();
        let f_minus = vals_minus[&out].to_f64_lossy_vec()[0];

        let numerical = (f_plus - f_minus) / (2.0 * h);
        let analytical = grad.to_f64_lossy_vec()[elem_idx];
        assert!(
            (analytical - numerical).abs() < 1e-3,
            "max_reduce finite diff at elem {elem_idx}: analytical={analytical}, numerical={numerical}"
        );
    }
}

#[test]
fn adv_max_reduce_with_ties() {
    // What happens when multiple elements tie for max?
    // Input: [[3, 3], [3, 1]]
    // max_reduce(axis=0) = [3, 3]
    // The gradient should flow to ALL elements equal to max (eq mask).
    // So row 0 col 0 AND row 1 col 0 should both get gradient.
    use chelis_ir::eval::{TensorValue, eval_tensor};

    let mat22_ty = TensorType {
        dims: vec![DimInfo::Lit(2), DimInfo::Lit(2)],
        precision: chelis_numeric_core::prim::Prim::F32,
    };
    let vec2_ty = TensorType {
        dims: vec![DimInfo::Lit(2)],
        precision: chelis_numeric_core::prim::Prim::F32,
    };

    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        mat22_ty.clone(),
        None,
    );
    let maxr = dag.add_node(
        owner,
        RiscOp::MaxReduce { axis: 0 },
        vec![x],
        vec2_ty.clone(),
        None,
    );
    let out = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_numeric_core::prim::Prim::F32,
        },
        vec![maxr],
        scalar_f32(),
        None,
    );

    let grad_result = grad_dag(&dag, out, &[x]).unwrap();

    let mut inputs = UnordMap::new();
    inputs.insert(
        "x".to_string(),
        TensorValue::from_vec(vec![2, 2], vec![3.0, 3.0, 3.0, 1.0]),
    );
    let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
    let grad = &vals[&grad_result.grad_nodes[&x]];

    // Column 0 has two equal maxima, so each receives 1/2. Column 1
    // has one selected maximum, so it receives the full cotangent.
    assert_eq!(grad.to_f64_lossy_vec(), vec![0.5, 1.0, 0.5, 0.0]);
}

#[test]
fn adv_max_reduce_axis1() {
    // Test max_reduce along axis 1 (the last axis)
    use chelis_ir::eval::{TensorValue, eval_tensor};

    let mat23_ty = TensorType {
        dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
        precision: chelis_numeric_core::prim::Prim::F64,
    };
    let vec2_ty = TensorType {
        dims: vec![DimInfo::Lit(2)],
        precision: chelis_numeric_core::prim::Prim::F64,
    };

    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        mat23_ty.clone(),
        None,
    );
    let maxr = dag.add_node(
        owner,
        RiscOp::MaxReduce { axis: 1 },
        vec![x],
        vec2_ty.clone(),
        None,
    );
    let out = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_numeric_core::prim::Prim::F64,
        },
        vec![maxr],
        scalar_f64(),
        None,
    );

    let grad_result = grad_dag(&dag, out, &[x]).unwrap();

    // Input: [[1, 5, 3], [4, 2, 6]]
    // max_reduce(axis=1) = [5, 6]
    // sum = 11
    // Gradient: 1 at positions of max along axis 1
    // Row 0: max is 5 at col 1 -> [0, 1, 0]
    // Row 1: max is 6 at col 2 -> [0, 0, 1]
    // Expected: [[0, 1, 0], [0, 0, 1]]
    let input_data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let mut inputs = UnordMap::new();
    inputs.insert(
        "x".to_string(),
        TensorValue::from_vec(vec![2, 3], input_data.clone()),
    );
    let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
    let grad = &vals[&grad_result.grad_nodes[&x]];

    let expected_grad = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    for (i, (actual, expected)) in grad
        .to_f64_lossy_vec()
        .iter()
        .zip(expected_grad.iter())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() < 1e-6,
            "max_reduce axis=1 gradient at index {i}: expected {expected}, got {actual}"
        );
    }

    // Verify via finite differences
    let h = 1e-5;
    for elem_idx in 0..6 {
        let mut data_plus = input_data.clone();
        data_plus[elem_idx] += h;
        let mut data_minus = input_data.clone();
        data_minus[elem_idx] -= h;

        let mut inputs_plus = UnordMap::new();
        inputs_plus.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![2, 3], data_plus),
        );
        let vals_plus = eval_tensor(&dag, &inputs_plus).unwrap();
        let f_plus = vals_plus[&out].to_f64_lossy_vec()[0];

        let mut inputs_minus = UnordMap::new();
        inputs_minus.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![2, 3], data_minus),
        );
        let vals_minus = eval_tensor(&dag, &inputs_minus).unwrap();
        let f_minus = vals_minus[&out].to_f64_lossy_vec()[0];

        let numerical = (f_plus - f_minus) / (2.0 * h);
        let analytical = grad.to_f64_lossy_vec()[elem_idx];
        assert!(
            (analytical - numerical).abs() < 1e-3,
            "max_reduce axis=1 finite diff at elem {elem_idx}: analytical={analytical}, numerical={numerical}"
        );
    }
}

#[test]
fn adv_log_at_negative_input() {
    // log(x) at x=-1 is NaN. What does the gradient look like?
    // This shouldn't crash.
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Log, vec![a], ty.clone(), None)
    });
    let grad_result = grad_dag(&dag, out, &[x]).unwrap();
    let mut inputs = UnordMap::new();
    inputs.insert("x".to_string(), -1.0);
    let vals = eval_scalar(&grad_result.dag, &inputs);
    let dx = vals[&grad_result.grad_nodes[&x]];
    // Should be NaN or -1 (1/x = 1/-1 = -1, but log(-1) is NaN so the
    // chain involves NaN). Just verify no crash.
    eprintln!("log gradient at x=-1: {dx} (expected NaN or similar)");
}

#[test]
fn adv_sqrt_at_zero() {
    // d(sqrt(x))/dx = 1/(2*sqrt(x)), at x=0 this is infinity.
    // Just verify no crash.
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Sqrt, vec![a], ty.clone(), None)
    });
    let grad_result = grad_dag(&dag, out, &[x]).unwrap();
    let mut inputs = UnordMap::new();
    inputs.insert("x".to_string(), 0.0);
    let vals = eval_scalar(&grad_result.dag, &inputs);
    let dx = vals[&grad_result.grad_nodes[&x]];
    eprintln!("sqrt gradient at x=0: {dx} (expected inf)");
}

#[test]
fn adv_div_gradient() {
    // div(a, b) lowers to RiscOp::Div(a, b); the closed-form
    // adjoint pair is da = g / b, db = -g * y / b (where y =
    // a / b is the forward output). At a=6, b=3 the expected
    // gradients are d/da = 1/3, d/db = -6/9 = -2/3.
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let a = dag.add_node(
        owner,
        RiscOp::Load { name: "a".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let b = dag.add_node(
        owner,
        RiscOp::Load { name: "b".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let out = chelis_dag_core::tier2::lower_div(owner, &mut dag, a, b, &scalar_f64(), None);

    // Test d/da
    let (ana_a, num_a) = finite_diff(&dag, out, a, "a", &[("b", 3.0)], 6.0, 1e-5);
    assert_grad_close(ana_a, num_a);
    assert!(
        (ana_a - 1.0 / 3.0).abs() < 1e-3,
        "d(a/b)/da at a=6,b=3: expected {}, got {ana_a}",
        1.0 / 3.0
    );

    // Test d/db
    let (ana_b, num_b) = finite_diff(&dag, out, b, "b", &[("a", 6.0)], 3.0, 1e-5);
    assert_grad_close(ana_b, num_b);
    let expected_db = -6.0 / 9.0;
    assert!(
        (ana_b - expected_db).abs() < 1e-3,
        "d(a/b)/db at a=6,b=3: expected {expected_db}, got {ana_b}"
    );
}

#[test]
fn adv_pad_shrink_gradient() {
    // Test pad adjoint: pad then sum, check gradient
    use chelis_ir::eval::{TensorValue, eval_tensor};

    let vec3_ty = TensorType {
        dims: vec![DimInfo::Lit(3)],
        precision: chelis_numeric_core::prim::Prim::F32,
    };
    let vec5_ty = TensorType {
        dims: vec![DimInfo::Lit(5)],
        precision: chelis_numeric_core::prim::Prim::F32,
    };

    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec3_ty.clone(),
        None,
    );
    let padded = dag.add_node(
        owner,
        RiscOp::zero_pad(
            chelis_numeric_core::prim::Prim::F32,
            vec![(RtDim::Lit(1), RtDim::Lit(1))],
        ),
        vec![x],
        vec5_ty.clone(),
        None,
    );
    let out = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_numeric_core::prim::Prim::F32,
        },
        vec![padded],
        scalar_f32(),
        None,
    );

    let grad_result = grad_dag(&dag, out, &[x]).unwrap();
    let mut inputs = UnordMap::new();
    inputs.insert(
        "x".to_string(),
        TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]),
    );
    let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
    let grad = &vals[&grad_result.grad_nodes[&x]];
    // d(sum(pad(x)))/dx = ones (pad adds zeros, but gradient only flows to original elements)
    assert_eq!(grad.to_f64_lossy_vec(), vec![1.0, 1.0, 1.0]);
}

#[test]
fn adv_shrink_gradient() {
    // Test shrink adjoint: shrink then sum
    use chelis_ir::eval::{TensorValue, eval_tensor};

    let vec5_ty = TensorType {
        dims: vec![DimInfo::Lit(5)],
        precision: chelis_numeric_core::prim::Prim::F32,
    };
    let vec3_ty = TensorType {
        dims: vec![DimInfo::Lit(3)],
        precision: chelis_numeric_core::prim::Prim::F32,
    };

    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec5_ty.clone(),
        None,
    );
    let shrunk = dag.add_node(
        owner,
        RiscOp::Shrink {
            bounds: vec![(RtDim::Lit(1), RtDim::Lit(4))],
        },
        vec![x],
        vec3_ty.clone(),
        None,
    );
    let out = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_numeric_core::prim::Prim::F32,
        },
        vec![shrunk],
        scalar_f32(),
        None,
    );

    let grad_result = grad_dag(&dag, out, &[x]).unwrap();
    let mut inputs = UnordMap::new();
    inputs.insert(
        "x".to_string(),
        TensorValue::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]),
    );
    let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
    let grad = &vals[&grad_result.grad_nodes[&x]];
    // d(sum(shrink(x, [1,4])))/dx = [0, 1, 1, 1, 0]
    assert_eq!(
        grad.to_f64_lossy_vec(),
        vec![0.0, 1.0, 1.0, 1.0, 0.0],
        "shrink gradient mismatch: got {:?}",
        grad.to_f64_lossy_vec()
    );
}

#[test]
fn adv_max_elem_spec_compliance() {
    let (dag, x, _y, out) = build_binary_dag(|dag, owner, a, b, ty| {
        dag.add_node(owner, RiscOp::MaxElem, vec![a, b], ty.clone(), None)
    });
    let grad_result = grad_dag(&dag, out, &[x]).unwrap();
    let mut inputs = UnordMap::new();
    inputs.insert("x".to_string(), 5.0);
    inputs.insert("y".to_string(), 3.0);
    let vals = eval_scalar(&grad_result.dag, &inputs);
    let dx_gt = vals[&grad_result.grad_nodes[&x]];
    assert!(
        (dx_gt - 1.0).abs() < 1e-6,
        "max_elem x>y: dx should be 1.0, got {dx_gt}"
    );

    inputs.insert("x".to_string(), 3.0);
    inputs.insert("y".to_string(), 3.0);
    let vals2 = eval_scalar(&grad_result.dag, &inputs);
    let dx_eq = vals2[&grad_result.grad_nodes[&x]];
    assert!(
        (dx_eq - 1.0).abs() < 1e-6,
        "max_elem x==y should route gradient to lhs, got {dx_eq}"
    );
}

#[test]
fn adv_log_second_order() {
    // f(x) = log(x), f'(x) = 1/x, f''(x) = -1/x^2
    // At x=2: f''(2) = -0.25
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let out = dag.add_node(owner, RiscOp::Log, vec![x], scalar_f64(), None);

    let first = grad_dag(&dag, out, &[x]).unwrap();
    let dx_node = first.grad_nodes[&x];

    let second = grad_dag(&first.dag, dx_node, &[x]).unwrap();
    let ddx_node = second.grad_nodes[&x];

    let mut inputs = UnordMap::new();
    inputs.insert("x".to_string(), 2.0);
    let vals = eval_scalar(&second.dag, &inputs);
    let ddx = vals[&ddx_node];
    let expected = -1.0 / 4.0; // -1/x^2 = -0.25
    assert!(
        (ddx - expected).abs() < 1e-3,
        "d^2(log(x))/dx^2 at x=2: expected {expected}, got {ddx}"
    );
}

#[test]
fn adv_stride_gradient_scatters_into_strided_slots() {
    // Issue #291: the stride adjoint scatters the cotangent back into
    // the strided source slots (zeros elsewhere) using existing
    // movement primitives (reshape/pad/shrink). `f(x) = sum(stride(x,
    // 2)) = x0 + x2`, so `df/dx = [1, 0, 1, 0]`.
    use chelis_ir::eval::{TensorValue, eval_tensor};

    let vec4_ty = TensorType {
        dims: vec![DimInfo::Lit(4)],
        precision: chelis_numeric_core::prim::Prim::F32,
    };
    let vec2_ty = TensorType {
        dims: vec![DimInfo::Lit(2)],
        precision: chelis_numeric_core::prim::Prim::F32,
    };

    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec4_ty.clone(),
        None,
    );
    let strided = dag.add_node(
        owner,
        RiscOp::Stride {
            strides: vec![RtDim::Lit(2)],
        },
        vec![x],
        vec2_ty.clone(),
        None,
    );
    let out = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_numeric_core::prim::Prim::F32,
        },
        vec![strided],
        scalar_f32(),
        None,
    );
    let grad_result =
        grad_dag(&dag, out, &[x]).expect("stride gradients are supported (issue #291)");
    let mut inputs = UnordMap::new();
    inputs.insert(
        "x".to_string(),
        TensorValue::from_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]),
    );
    let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
    let grad = &vals[&grad_result.grad_nodes[&x]];
    assert_eq!(
        grad.to_f64_lossy_vec(),
        vec![1.0, 0.0, 1.0, 0.0],
        "stride gradient must scatter into strided slots: got {:?}",
        grad.to_f64_lossy_vec()
    );
}

#[test]
fn adv_gradient_at_nontrivial_values() {
    // Test all ops at non-trivial values (2.7, -0.3, 1.5)
    // to catch any issues with specific values.

    // exp at 2.7
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Exp, vec![a], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[], 2.7, 1e-5);
    assert_grad_close(a, n);

    // log at 1.5
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Log, vec![a], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[], 1.5, 1e-5);
    assert_grad_close(a, n);

    // sin at -0.3
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Sin, vec![a], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[], -0.3, 1e-5);
    assert_grad_close(a, n);

    // sqrt at 2.7
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Sqrt, vec![a], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[], 2.7, 1e-5);
    assert_grad_close(a, n);
}

// ---- Phase 3j-pre: adjoints for new reductions ----

#[test]
fn adv_min_reduce_gradient() {
    // Mirrors adv_max_reduce_gradient. Input 2x3:
    //   [[1, 5, 3], [4, 2, 6]]
    // min_reduce(axis=0) = [1, 2, 3]
    // sum(min) = 6
    // Gradient: 1 where element equals min along axis 0, else 0.
    // Expected: [[1, 0, 1], [0, 1, 0]]
    use chelis_ir::eval::{TensorValue, eval_tensor};

    let mat23 = TensorType {
        dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
        precision: Prim::F32,
    };
    let vec3 = TensorType {
        dims: vec![DimInfo::Lit(3)],
        precision: Prim::F32,
    };

    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        mat23.clone(),
        None,
    );
    let minr = dag.add_node(owner, RiscOp::MinReduce { axis: 0 }, vec![x], vec3, None);
    let out = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_numeric_core::prim::Prim::F32,
        },
        vec![minr],
        scalar_f32(),
        None,
    );

    let grad_result = grad_dag(&dag, out, &[x]).unwrap();
    let mut inputs = UnordMap::new();
    inputs.insert(
        "x".to_string(),
        TensorValue::from_vec(vec![2, 3], vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]),
    );
    let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
    let grad = &vals[&grad_result.grad_nodes[&x]];
    let expected = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    for (i, (got, want)) in grad
        .to_f64_lossy_vec()
        .iter()
        .zip(expected.iter())
        .enumerate()
    {
        assert!(
            (got - want).abs() < 1e-6,
            "min_reduce grad index {i}: expected {want}, got {got}"
        );
    }
}

#[test]
fn adv_prod_reduce_gradient_nonzero() {
    // Input: 1D length-4 vector [2, 3, 4, 5]
    // prod = 120
    // ∂prod/∂x_i = prod(x) / x_i = [60, 40, 30, 24]
    use chelis_ir::eval::{TensorValue, eval_tensor};

    let vec4 = TensorType {
        dims: vec![DimInfo::Lit(4)],
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(owner, RiscOp::Load { name: "x".into() }, vec![], vec4, None);
    let p = dag.add_node(
        owner,
        RiscOp::ProdReduce { axis: 0 },
        vec![x],
        scalar_f32(),
        None,
    );
    dag.add_root(p);

    let grad_result = grad_dag(&dag, p, &[x]).unwrap();
    let mut inputs = UnordMap::new();
    inputs.insert(
        "x".to_string(),
        TensorValue::from_vec(vec![4], vec![2.0, 3.0, 4.0, 5.0]),
    );
    let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
    let grad = &vals[&grad_result.grad_nodes[&x]];
    let expected = [60.0, 40.0, 30.0, 24.0];
    for (i, (got, want)) in grad
        .to_f64_lossy_vec()
        .iter()
        .zip(expected.iter())
        .enumerate()
    {
        assert!(
            (got - want).abs() < 1e-4,
            "prod_reduce grad index {i}: expected {want}, got {got}"
        );
    }
}

/// The critical safety test: one element of the reduced slice is zero.
/// The naive `g * prod(x) / x_i` rule divides by zero; the prefix*suffix
/// rule must produce finite, correct gradients.
///
/// Input: [2, 0, 4, 5]
/// prod = 0
/// ∂prod/∂x_0 = 0*4*5 = 0
/// ∂prod/∂x_1 = 2*4*5 = 40   (the interesting one — nonzero even though x_1=0)
/// ∂prod/∂x_2 = 2*0*5 = 0
/// ∂prod/∂x_3 = 2*0*4 = 0
#[test]
fn adv_prod_reduce_gradient_with_zero_element() {
    use chelis_ir::eval::{TensorValue, eval_tensor};

    let vec4 = TensorType {
        dims: vec![DimInfo::Lit(4)],
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(owner, RiscOp::Load { name: "x".into() }, vec![], vec4, None);
    let p = dag.add_node(
        owner,
        RiscOp::ProdReduce { axis: 0 },
        vec![x],
        scalar_f32(),
        None,
    );
    dag.add_root(p);

    let grad_result = grad_dag(&dag, p, &[x]).unwrap();
    let mut inputs = UnordMap::new();
    inputs.insert(
        "x".to_string(),
        TensorValue::from_vec(vec![4], vec![2.0, 0.0, 4.0, 5.0]),
    );
    let vals = eval_tensor(&grad_result.dag, &inputs).unwrap();
    let grad = &vals[&grad_result.grad_nodes[&x]];
    for v in &grad.to_f64_lossy_vec() {
        assert!(v.is_finite(), "prod_reduce grad must be finite; got {v}");
    }
    let expected = [0.0, 40.0, 0.0, 0.0];
    for (i, (got, want)) in grad
        .to_f64_lossy_vec()
        .iter()
        .zip(expected.iter())
        .enumerate()
    {
        assert!(
            (got - want).abs() < 1e-4,
            "prod_reduce zero-element grad index {i}: expected {want}, got {got}"
        );
    }
}

// ---- New scalar builtin AD tests ----

#[test]
fn grad_cos() {
    // d/dx cos(x) = -sin(x)
    // At x=0.5: -sin(0.5) ≈ -0.4794
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Cos, vec![a], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[], 0.5, 1e-5);
    assert_grad_close(a, n);
    let expected = -(0.5_f64).sin();
    assert!(
        (a - expected).abs() < 1e-4,
        "grad of cos at 0.5 should be ≈ {expected}, got {a}"
    );
}

#[test]
fn grad_tan() {
    // d/dx tan(x) = 1 / cos²(x)
    // At x=0.5: 1/cos²(0.5) ≈ 1.298
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Tan, vec![a], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[], 0.5, 1e-5);
    assert_grad_close(a, n);
}

#[test]
fn grad_atan() {
    // d/dx atan(x) = 1 / (1 + x²)
    // At x=1.0: 1 / (1+1) = 0.5
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Atan, vec![a], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[], 1.0, 1e-5);
    assert_grad_close(a, n);
    assert!(
        (a - 0.5).abs() < 1e-4,
        "grad of atan at 1.0 should be ≈ 0.5, got {a}"
    );
}

#[test]
fn grad_abs_positive() {
    // d/dx abs(x) = 1 for x > 0
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Abs, vec![a], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[], 2.0, 1e-5);
    assert_grad_close(a, n);
    assert!(
        (a - 1.0).abs() < 1e-4,
        "grad of abs at 2.0 should be 1.0, got {a}"
    );
}

#[test]
fn grad_abs_negative() {
    // d/dx abs(x) = -1 for x < 0
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Abs, vec![a], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[], -2.0, 1e-5);
    assert_grad_close(a, n);
    assert!(
        (a - (-1.0)).abs() < 1e-4,
        "grad of abs at -2.0 should be -1.0, got {a}"
    );
}

// ---- ADVERSARIAL TESTS: missing coverage from red-team spec ----

/// abs(x) at x=0 must give exactly 0.0 (sign convention: sign(0) = 0).
/// This is NOT covered by grad_abs_positive (x=2.0) or grad_abs_negative (x=-2.0).
#[test]
fn adv_grad_abs_at_zero_is_zero() {
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Abs, vec![a], ty.clone(), None)
    });
    let grad_result = grad_dag(&dag, out, &[x]).unwrap();
    let mut inputs = UnordMap::new();
    inputs.insert("x".to_string(), 0.0_f64);
    let vals = eval_scalar(&grad_result.dag, &inputs);
    let a = vals[&grad_result.grad_nodes[&x]];
    assert!(
        a.abs() < 1e-9,
        "abs gradient at x=0 must be 0.0 (sign(0)=0), got {a}"
    );
}

/// abs(x) at x=-2.7 must give exactly -1.0.
/// Spec says: abs(x) at x=-2.7 → -1.0.
#[test]
fn adv_grad_abs_at_neg2_7_is_neg1() {
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Abs, vec![a], ty.clone(), None)
    });
    let grad_result = grad_dag(&dag, out, &[x]).unwrap();
    let mut inputs = UnordMap::new();
    inputs.insert("x".to_string(), -2.7_f64);
    let vals = eval_scalar(&grad_result.dag, &inputs);
    let a = vals[&grad_result.grad_nodes[&x]];
    assert!(
        (a - (-1.0)).abs() < 1e-6,
        "abs gradient at x=-2.7 must be -1.0, got {a}"
    );
}

/// atan(x) at x=1.5: grad = 1/(1+1.5²) = 1/3.25 ≈ 0.3077.
/// Spec asks for this point; existing test only covers x=1.0.
#[test]
fn adv_grad_atan_at_1_5() {
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Atan, vec![a], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[], 1.5, 1e-5);
    assert_grad_close(a, n);
    let expected = 1.0 / (1.0 + 1.5_f64 * 1.5);
    assert!(
        (a - expected).abs() < 1e-4,
        "grad of atan at 1.5 should be ≈ {expected} (1/3.25), got {a}"
    );
}

/// [05-OP-46]: erf(x) at x=0.5 has gradient (2/sqrt(pi))*exp(-0.25)
/// ≈ 0.8788, and erfc(x) its negation; both agree with finite differences.
#[test]
fn grad_erf_and_erfc_at_0_5() {
    let expected = 0.878_782_578_935_444_8;
    for (op, sign) in [(RiscOp::Erf, 1.0), (RiscOp::Erfc, -1.0)] {
        let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
            dag.add_node(owner, op.clone(), vec![a], ty.clone(), None)
        });
        let (a, n) = finite_diff(&dag, out, x, "x", &[], 0.5, 1e-5);
        assert_grad_close(a, n);
        assert!(
            (a - sign * expected).abs() < 1e-4,
            "grad of {op:?} at 0.5 should be ≈ {}, got {a}",
            sign * expected
        );
    }
}

/// tan(x) at x=0.3: grad = 1/cos²(0.3) ≈ 1.047.
/// Spec asks for this exact point with exact value; existing test uses x=0.5 with no exact check.
#[test]
fn adv_grad_tan_at_0_3_exact() {
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Tan, vec![a], ty.clone(), None)
    });
    let (a, n) = finite_diff(&dag, out, x, "x", &[], 0.3, 1e-5);
    assert_grad_close(a, n);
    let expected = 1.0 / (0.3_f64.cos() * 0.3_f64.cos());
    assert!(
        (a - expected).abs() < 1e-4,
        "grad of tan at 0.3 should be ≈ {expected} (1/cos²(0.3)), got {a}"
    );
}

/// cos(add(x, const(0.5))) composition — grad should be -sin(x+0.5).
#[test]
fn adv_grad_cos_of_add_composition() {
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let half = dag.add_node(
        owner,
        RiscOp::synth_const(scalar_f64().precision, 0.5),
        vec![],
        scalar_f64(),
        None,
    );
    let xp = dag.add_node(owner, RiscOp::Add, vec![x, half], scalar_f64(), None);
    let out = dag.add_node(owner, RiscOp::Cos, vec![xp], scalar_f64(), None);

    let x0 = 0.7_f64;
    let (a, n) = finite_diff(&dag, out, x, "x", &[], x0, 1e-5);
    assert_grad_close(a, n);
    let expected = -(x0 + 0.5).sin();
    assert!(
        (a - expected).abs() < 1e-4,
        "grad of cos(x+0.5) at x=0.7 should be -sin(1.2) ≈ {expected}, got {a}"
    );
}

/// cos(x) at x=0.5: exact value check. Spec requires -sin(0.5) ≈ -0.4794.
#[test]
fn adv_grad_cos_at_0_5_exact_value() {
    let (dag, x, out) = build_unary_dag(|dag, owner, a, ty| {
        dag.add_node(owner, RiscOp::Cos, vec![a], ty.clone(), None)
    });
    let grad_result = grad_dag(&dag, out, &[x]).unwrap();
    let mut inputs = UnordMap::new();
    inputs.insert("x".to_string(), 0.5_f64);
    let vals = eval_scalar(&grad_result.dag, &inputs);
    let a = vals[&grad_result.grad_nodes[&x]];
    let expected = -(0.5_f64).sin(); // ≈ -0.4794
    assert!(
        (a - expected).abs() < 1e-5,
        "cos gradient at x=0.5 must be -sin(0.5) ≈ {expected:.6}, got {a:.6}"
    );
}

/// chelis#616: a stride along the SYMBOLIC axis itself now constructs a
/// RUNTIME adjoint cascade — the trim shrink's end bound is a
/// node-valued Shape read of the forward input's extent (`n_a`), and the
/// merge reshape's target extent is runtime `m_a * step` arithmetic —
/// instead of the pre-#616 loud panic.
#[test]
fn stride_adjoint_symbolic_strided_axis_builds_runtime_trim() {
    let mut dag = Dag::new();
    let owner = Owner::from(dag.declare("test"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: chelis_numeric_core::prim::Prim::F32,
        },
        None,
    );
    let strided = dag.add_node(
        owner,
        RiscOp::Stride {
            strides: vec![RtDim::Lit(2)],
        },
        vec![x],
        TensorType {
            dims: vec![DimInfo::Named("m".into(), None)],
            precision: chelis_numeric_core::prim::Prim::F32,
        },
        None,
    );
    let out = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_numeric_core::prim::Prim::F32,
        },
        vec![strided],
        scalar_f32(),
        None,
    );
    let grad_result =
        grad_dag(&dag, out, &[x]).expect("runtime strided-axis adjoint must construct");
    // The trim shrink's end bound is node-valued (a Shape read of the
    // forward input), never a guessed literal.
    let runtime_trim = grad_result.dag.nodes().iter().any(|node| {
        matches!(
            &node.op,
            RiscOp::Shrink { bounds }
                if bounds.first().is_some_and(|(s, e)| {
                    s.as_lit() == Some(0) && e.node_input().is_some()
                })
        )
    });
    assert!(
        runtime_trim,
        "runtime strided-axis adjoint must trim to a node-valued end bound"
    );
    // The gradient of sum(stride(x, 2)) at n = 5 is the upsample mask
    // [1, 0, 1, 0, 1].
    let mut inputs = UnordMap::new();
    inputs.insert(
        "x".to_string(),
        TensorValue::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]),
    );
    let grad_node = grad_result.grad_nodes[&x];
    let values =
        chelis_ir::eval::eval_tensor_roots_with_strict(&grad_result.dag, &[grad_node], |name| {
            inputs.get(name).cloned()
        })
        .expect("runtime strided-axis adjoint must evaluate");
    assert_eq!(values[&grad_node].shape, vec![5]);
    assert_eq!(
        values[&grad_node].to_f64_lossy_vec(),
        vec![1.0, 0.0, 1.0, 0.0, 1.0]
    );
}
