//! WS-A3 AD oracle: gradient flow through bf16 / f16 ops produces an
//! operand-precision adjoint per spec/04-type-system.md §5.7 / §5.7.1.
//!
//! The acceptance test in the WS-A3 brief is "bf16 → scalar bf16 loss
//! → grad → bf16 tensor (gradient precision = operand precision)".
//! This test pins the IR-level shape of that contract so a future
//! AD-rule refactor that accidentally widens the adjoint to f32 fails
//! loudly. The HIP backend integration test (tests/gpu_correctness.rs)
//! exercises the full host→device→host execution; this test exercises
//! the AD construction without requiring a HIP-capable GPU.

use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir::grad::grad_dag;
use chelis_types::types::Prim;

fn vec_at(precision: Prim, len: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(len)],
        precision,
    }
}

fn scalar_at(precision: Prim) -> TensorType {
    TensorType {
        dims: vec![],
        precision,
    }
}

/// Forward DAG: `loss = sum(x * x)` with bf16 throughout.
///
/// `Sum` carries the spec-default f32 accumulator for bf16 operands
/// (per §5.7.1) so the result of the reduction is `f32`, then we
/// downcast to bf16 to produce a `tensor[bf16]` scalar loss in line
/// with the rest of the operand-precision contract.
fn build_bf16_program() -> (Dag, NodeId, NodeId) {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_at(Prim::Bf16, 4),
        None,
    );
    let sq = dag.add_node(decl, RiscOp::Mul, vec![x, x], vec_at(Prim::Bf16, 4), None);
    let sum_op = RiscOp::sum_default(0, Prim::Bf16)
        .expect("bf16 reduce_sum constructs (default accumulator = f32)");
    // Reduction output dtype = accumulator dtype per spec §5.7.1; for
    // bf16 operand the default accumulator is f32, so the Sum node
    // output is f32. We then downcast back to bf16 for the loss
    // scalar so the AD entry-point sees a bf16 scalar (matching the
    // "scalar bf16 loss" wording in the WS-A3 brief).
    let summed = dag.add_node(decl, sum_op, vec![sq], scalar_at(Prim::F32), None);
    let loss = dag.add_node(
        decl,
        RiscOp::Cast {
            new_precision: Prim::Bf16,
        },
        vec![summed],
        scalar_at(Prim::Bf16),
        None,
    );
    (dag, x, loss)
}

#[test]
fn bf16_grad_returns_bf16_adjoint_tensor() {
    let (dag, x, loss) = build_bf16_program();
    let result = grad_dag(&dag, loss, &[x]).expect(
        "AD must produce a gradient for bf16 → scalar bf16 loss; the existing \
         is_scalar_float gate accepts any active float dtype per Prim::is_float",
    );
    let grad_x = result
        .grad_nodes
        .get(&x)
        .expect("bf16 input must have an adjoint node in the result");
    let grad_x_node = result
        .dag
        .get(*grad_x)
        .expect("adjoint NodeId must resolve in the gradient DAG");
    assert_eq!(
        grad_x_node.output_type.precision,
        Prim::Bf16,
        "WS-A3 contract: gradient precision matches operand precision \
         (operand-precision adjoint per spec §5.7); got {:?}",
        grad_x_node.output_type.precision,
    );
    assert_eq!(
        grad_x_node.output_type.dims,
        vec![DimInfo::Lit(4)],
        "adjoint tensor shape must match the operand tensor shape; got {:?}",
        grad_x_node.output_type.dims,
    );
}

/// Same shape for f16 — gradient precision = operand precision per
/// spec §5.7. Pinning both rows independently so a regression in one
/// dtype does not hide behind the other.
#[test]
fn f16_grad_returns_f16_adjoint_tensor() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_at(Prim::F16, 3),
        None,
    );
    let sq = dag.add_node(decl, RiscOp::Mul, vec![x, x], vec_at(Prim::F16, 3), None);
    let sum_op = RiscOp::sum_default(0, Prim::F16)
        .expect("f16 reduce_sum constructs (default accumulator = f32)");
    let summed = dag.add_node(decl, sum_op, vec![sq], scalar_at(Prim::F32), None);
    let loss = dag.add_node(
        decl,
        RiscOp::Cast {
            new_precision: Prim::F16,
        },
        vec![summed],
        scalar_at(Prim::F16),
        None,
    );

    let result =
        grad_dag(&dag, loss, &[x]).expect("AD must produce a gradient for f16 scalar loss");
    let grad_x = result.grad_nodes.get(&x).expect("f16 input adjoint exists");
    let grad_x_node = result.dag.get(*grad_x).expect("adjoint resolves");
    assert_eq!(
        grad_x_node.output_type.precision,
        Prim::F16,
        "f16 operand → f16 gradient per §5.7; got {:?}",
        grad_x_node.output_type.precision,
    );
}

/// Negative parity: AD on a bf16 scalar loss produces a non-empty
/// gradient DAG. Guards against a regression where the output-type
/// gate (`is_scalar_float`) silently rejects bf16/f16 because of a
/// future refactor that narrows it to `Prim::F32 | Prim::F64`.
#[test]
fn bf16_grad_does_not_silently_reject_loss_precision() {
    let (dag, x, loss) = build_bf16_program();
    let result = grad_dag(&dag, loss, &[x]);
    assert!(
        result.is_some(),
        "WS-A3 negative parity: bf16 scalar loss must NOT be silently rejected by \
         the AD entry-point's `is_scalar_float` gate; that gate accepts every \
         dtype in `Prim::is_float`, which includes bf16/f16"
    );
}
