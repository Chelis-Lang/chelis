//! Issue #318: `grad` has no backward rule for the SHAPE-DERIVED
//! expand-of-scalar const-broadcast. Issue #288 (PR #296) fixed the
//! LITERAL-size form `expand(scalar_to_tensor(c), 0, cast(2, int32))`;
//! the canonical 0.7.x scalar-broadcast helper (`tensor_full_like` /
//! `tensor_full_1d`) instead writes the SHAPE-DERIVED form
//! `expand(scalar_to_tensor(c), 0, cast(shape(&x, 0), int32))`, whose
//! broadcast extent is a *runtime/symbolic* dimension rather than a
//! literal. That form still failed.
//!
//! Reproducer (Surf, the form the helper emits):
//! ```chelis
//! module Repro.GradExpandShape
//! def f(x: tensor[n, f32]) -> f32 = {
//!   k = expand(scalar_to_tensor(cast(3.0, f32)),
//!              cast(0, int32),
//!              cast(shape(&x, cast(0, int32)), int32))
//!   tensor_to_scalar(sum(mul(x, k), cast(0, int32)))
//! }
//! def df(x: tensor[n, f32]) -> tensor[n, f32] = grad(f)(x)
//! ```
//!
//! Before the fix the forward `chelis check`ed clean (the type checker
//! accepts the size-1 source broadcasting up to `tensor[n]`), but adding
//! `grad` failed:
//!
//! ```text
//! error: Lowering error: grad(...) lowering rejected: failed to
//! construct backward DAG (grad: constructed backward DAG failed
//! verification: binary op at node 4 has mismatched dimension at axis 0:
//! Lit(2) vs Lit(1); binary op at node 8 has mismatched dimension at
//! axis 0: Lit(2) vs Lit(1))
//! ```
//!
//! Root cause (lowering, not autodiff): the `expand` size argument is a
//! host-lane `shape(...)` application, which `extract_dim_expr_value`
//! cannot read. The lowering silently defaulted the size to `1`, so the
//! `expand` lowered to a `tensor[1]` instead of `tensor[n]`. The forward
//! `mul(x, k)` then mixed `tensor[n]` with `tensor[1]`; the type checker
//! accepts that via size-1 broadcasting, but the IR (no implicit
//! broadcasting) does not. The malformed `Mul` surfaced only when `grad`
//! verified the cloned forward inside the backward DAG. The fix recovers
//! the broadcast extent from the `shape(operand, axis)` argument's
//! operand dim during lowering (see the lowering-level siblings
//! `issue_318_expand_shape_arg_*` in `crates/chelis-ir/src/lower.rs`,
//! which drive the REAL `expand` lowering arm with the exact
//! `cast(shape(&x, ...), int32)` Deep the Surf idiom emits and are the
//! tests that detect the lowering failure).
//!
//! SCOPE OF THIS FILE: it does NOT exercise the lowering fix. It
//! hand-builds the forward DAG with the broadcast extent ALREADY supplied
//! (a `DimExpr::Sym` size and a `DimInfo::Named(_, None)` expand output),
//! so it cannot detect the lowering defect — the `lower.rs`
//! `issue_318_expand_shape_arg_*` units and the CLI end-to-end test own
//! that. What this file pins is that the `grad`
//! `RiscOp::Expand` adjoint is EXTENT-AGNOSTIC: given a well-formed
//! forward whose broadcast extent is symbolic (not a `Lit`),
//! `grad_dag_checked` still constructs the backward and `eval_tensor`
//! yields the correct numeric gradient (`[3, 3]`), for both const-source
//! ranks (`[]` and `[1]`). Both reviewers confirmed the adjoint is
//! correct and unchanged; this is the regression lock for that property.

use chelis_ir::dag::{Dag, DimExpr, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_ir::grad::{AdError, grad_dag_checked};
use chelis_types::types::Prim;
use std::collections::HashMap;

fn scalar_f32() -> TensorType {
    TensorType {
        dims: vec![],
        precision: Prim::F32,
    }
}

/// A rank-1 tensor whose single axis is the runtime/symbolic dimension
/// `name` (no statically-known size) — the shape of an `x` whose extent
/// is only known at call time, the case the `shape(&x, 0)`-derived
/// `expand` broadcasts to.
fn vec_sym_f32(name: &str) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Named(name.to_string(), None)],
        precision: Prim::F32,
    }
}

/// A rank-1 literal-sized tensor `tensor[n]`.
fn vec_lit_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

/// How the const-broadcast source `scalar_to_tensor(cast(c, f32))`
/// materializes. `scalar_to_tensor` is a no-op passthrough in lowering,
/// so the real idiom produces a rank-0 source (`[]`), but the static
/// build path can materialize the constant as a rank-1 size-1 source
/// (`[1]`). Both must back-prop identically, exactly as in the #288
/// sibling test.
const SOURCE_SHAPES: [&[usize]; 2] = [&[], &[1]];

/// The two broadcast-extent encodings whose backward must be identical.
/// `Literal` is the #288 form `cast(2, int32)`; `ShapeDerived` is the
/// #318 form `cast(shape(&x, 0), int32)`, where the extent is the
/// runtime/symbolic dimension carried by `x`.
#[derive(Clone, Copy)]
enum Extent {
    Literal,
    ShapeDerived,
}

/// Build the issue #318 forward DAG.
///
///   c   = Cast(Const c_val)        : f32  source_shape  [scalar_to_tensor(cast(c_val, f32))]
///   x   = Load("x")                : tensor[n] (literal) or tensor[<sym>] (shape-derived)
///   k   = Expand{axis:0, size}(c)  : same vector type as x  [expand(c, 0, size)]
///   m   = Mul(x, k)                : same vector type  [mul(x, k)]
///   out = Sum{axis:0}(m)           : f32 (rank 0)      [tensor_to_scalar(sum(m, 0))]
///
/// For `Literal`, `x` and the expand output are `tensor[2]` and the
/// expand size is `DimExpr::Concrete(2)`. For `ShapeDerived`, `x` and
/// the expand output carry the symbolic dimension `n` (`DimInfo::Named`)
/// and the expand size is `DimExpr::Sym("n")` — the runtime extent the
/// `shape(&x, 0)` argument lowers to. `f(x) = sum(x * c_val)`, so
/// `df/dx = [c_val; len(x)]` for either encoding and either source
/// shape.
fn build_forward(extent: Extent, source_shape: &[usize]) -> (Dag, NodeId, NodeId) {
    let mut dag = Dag::new();
    let (vec_ty, size) = match extent {
        Extent::Literal => (vec_lit_f32(2), DimExpr::Concrete(2)),
        Extent::ShapeDerived => (vec_sym_f32("n"), DimExpr::Sym("n".to_string())),
    };
    let source_ty = TensorType {
        dims: source_shape.iter().map(|&d| DimInfo::Lit(d)).collect(),
        precision: Prim::F32,
    };

    // scalar_to_tensor(cast(c_val, f32)) -> f32 constant of `source_ty`.
    let raw = dag.add_node(
        RiscOp::synth_const(source_ty.precision, 3.0),
            vec![],
            source_ty.clone(),
        None,
    );
    let c = dag.add_node(
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![raw],
        source_ty,
        None,
    );

    // expand(c, axis=0, size) -> vector type (symbolic for shape-derived).
    let k = dag.add_node(
        RiscOp::Expand { axis: 0, size },
        vec![c],
        vec_ty.clone(),
        None,
    );

    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_ty.clone(),
        None,
    );
    let m = dag.add_node(RiscOp::Mul, vec![x, k], vec_ty, None);
    let out = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default for f32"),
        vec![m],
        scalar_f32(),
        None,
    );
    (dag, x, out)
}

fn assert_close(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: length mismatch: got {} want {}",
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < 1e-5, "{label}: elem {i}: got {g}, want {w}",);
    }
}

/// Positive: the SHAPE-DERIVED forward evaluates to `c_val * sum(x)`.
/// Establishes the forward is well-formed (the symbolic extent binds
/// from `x`'s runtime shape) before any backward is requested.
#[test]
fn issue_318_forward_shape_derived_expand_evaluates() {
    for shape in SOURCE_SHAPES {
        let (dag, _x, out) = build_forward(Extent::ShapeDerived, shape);
        let mut inputs: HashMap<String, TensorValue> = HashMap::new();
        inputs.insert("x".into(), TensorValue::from_vec(vec![2], vec![5.0, 6.0]));
        let vals = eval_tensor(&dag, &inputs).expect("forward eval must succeed");
        // 3.0 * (5 + 6) = 33.0
        assert_close(
            &format!("forward shape-derived source={shape:?}"),
            &vals[&out].to_f64_lossy_vec(),
            &[33.0],
        );
    }
}

/// The #318 bug: `grad` must CONSTRUCT cleanly through the shape-derived
/// `expand(scalar_to_tensor(c), 0, shape(&x, 0))` idiom for BOTH source
/// shapes. Before the fix the lowered forward carried a `tensor[1]`
/// expand, so the cloned forward `Mul` failed backward verification with
/// `Lit(2) vs Lit(1)`; here the symbolic extent surfaces the same class
/// of mismatch the fix removes.
#[test]
fn issue_318_grad_through_shape_derived_expand_constructs() {
    for shape in SOURCE_SHAPES {
        let (dag, x, out) = build_forward(Extent::ShapeDerived, shape);
        match grad_dag_checked(&dag, out, &[x]) {
            Ok(_) => {}
            Err(AdError::NotSupported { op, reason }) => panic!(
                "grad through shape-derived expand(scalar_to_tensor(c), 0, \
                 shape(&x, 0)) with source shape {shape:?} must succeed \
                 (issue #318); got rejection op={op}, reason={reason:?}",
            ),
        }
    }
}

/// Numeric: `df/dx = [c_val; len(x)] = [3, 3]` for the shape-derived
/// form, both source shapes, exactly as the issue's Expected section
/// specifies. The grad DAG carries the symbolic extent; `eval_tensor`
/// binds it from the runtime shape of `x`.
#[test]
fn issue_318_grad_through_shape_derived_expand_is_correct() {
    for shape in SOURCE_SHAPES {
        let (dag, x, out) = build_forward(Extent::ShapeDerived, shape);
        let result = grad_dag_checked(&dag, out, &[x]).expect("grad must construct (issue #318)");
        let grad_x = result
            .grad_nodes
            .get(&x)
            .copied()
            .expect("gradient w.r.t. x must be present");
        let mut inputs: HashMap<String, TensorValue> = HashMap::new();
        inputs.insert("x".into(), TensorValue::from_vec(vec![2], vec![5.0, 6.0]));
        let vals = eval_tensor(&result.dag, &inputs).expect("grad DAG eval");
        assert_close(
            &format!("grad_x shape-derived source={shape:?}"),
            &vals[&grad_x].to_f64_lossy_vec(),
            &[3.0, 3.0],
        );
        assert_eq!(
            vals[&grad_x].shape,
            vec![2],
            "grad shape must be tensor[2] for source {shape:?}"
        );
    }
}

/// Negative parity: the LITERAL form (the #288 fix) must STILL construct
/// and produce the same gradient. The two encodings differ only in the
/// expand size argument (`Concrete(2)` vs `Sym("n")`); their backward
/// DAGs must agree. Running both side by side here pins that the #318
/// lowering fix did not regress the #288 literal path.
#[test]
fn issue_318_literal_and_shape_derived_agree() {
    for shape in SOURCE_SHAPES {
        let mut grads = Vec::new();
        for extent in [Extent::Literal, Extent::ShapeDerived] {
            let (dag, x, out) = build_forward(extent, shape);
            let result = grad_dag_checked(&dag, out, &[x])
                .expect("both literal and shape-derived forms must construct");
            let grad_x = result.grad_nodes[&x];
            let mut inputs = HashMap::new();
            inputs.insert("x".into(), TensorValue::from_vec(vec![2], vec![5.0, 6.0]));
            let vals = eval_tensor(&result.dag, &inputs).expect("grad eval");
            grads.push(vals[&grad_x].to_f64_lossy_vec().clone());
        }
        assert_close(
            &format!("literal-vs-shape-derived source={shape:?}"),
            &grads[1],
            &grads[0],
        );
        // Both must equal the analytic gradient [3, 3].
        assert_close(&format!("literal source={shape:?}"), &grads[0], &[3.0, 3.0]);
    }
}

/// Finite-difference cross-check for the shape-derived form (both source
/// shapes), per the backend-numerics discipline: the analytic gradient
/// must agree with a centered finite difference of the forward.
#[test]
fn issue_318_grad_matches_finite_difference() {
    for shape in SOURCE_SHAPES {
        let (dag, x, out) = build_forward(Extent::ShapeDerived, shape);
        let result = grad_dag_checked(&dag, out, &[x]).expect("grad must construct");
        let grad_x = result.grad_nodes[&x];
        let base = TensorValue::from_vec(vec![2], vec![0.7, -1.3]);
        let mut inputs: HashMap<String, TensorValue> = HashMap::new();
        inputs.insert("x".into(), base.clone());
        let analytic = eval_tensor(&result.dag, &inputs).expect("analytic eval")[&grad_x]
            .to_f64_lossy_vec()
            .clone();

        let h = 1e-3;
        let mut numerical = [0.0f64; 2];
        for (j, slot) in numerical.iter_mut().enumerate() {
            let mut plus_data = base.to_f64_lossy_vec();
            let mut minus_data = base.to_f64_lossy_vec();
            plus_data[j] += h;
            minus_data[j] -= h;
            let plus = TensorValue::from_vec(base.shape.clone(), plus_data);
            let minus = TensorValue::from_vec(base.shape.clone(), minus_data);
            let mut ip = HashMap::new();
            ip.insert("x".into(), plus);
            let mut im = HashMap::new();
            im.insert("x".into(), minus);
            let fp = eval_tensor(&dag, &ip).expect("plus eval")[&out].to_f64_lossy_vec()[0];
            let fm = eval_tensor(&dag, &im).expect("minus eval")[&out].to_f64_lossy_vec()[0];
            *slot = (fp - fm) / (2.0 * h);
        }
        for (i, (a, n)) in analytic.iter().zip(numerical.iter()).enumerate() {
            assert!(
                (a - n).abs() < 1e-3,
                "finite-diff mismatch at {i} (shape-derived source {shape:?}): \
                 analytic {a}, numerical {n}",
            );
        }
    }
}
