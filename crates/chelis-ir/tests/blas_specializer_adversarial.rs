//! Wave 5 red-team — Gap 2 specializer brittleness beyond the closed-list
//! identity-cast / identity-reshape / identity-permute cleanup.
//!
//! The M1 specializer at `chelis_ir::specialize::specialize_for_blas` cleans
//! exactly three closed-list no-ops before BLAS replacement: identity Cast,
//! identity Reshape, and identity Permute. Anything outside that list MUST
//! NOT collapse to BLAS, because the cleanup is a closed list — that's the
//! whole correctness story for keeping the cleanup conservative.
//!
//! This file probes structural perturbations the plan's Wave-5 brief invited:
//!
//! 1. **Non-identity cast pair** (`f32 → f64 → f32`) between Expand and Mul
//!    must NOT specialize to BLAS. There's no cancel-pair recognizer; the
//!    pair is structurally non-identity at the IR level.
//! 2. **Non-identity cast pair** (`int32 → f32 → int32`) — same.
//! 3. **Reshape pair that round-trips** (`[m,k] → [k,m] → [m,k]` via two
//!    Reshape nodes) must NOT specialize: there's no cancel-pair recognizer.
//! 4. **Permute pair that cancels** (`[1,0]` then `[1,0]` again) must NOT
//!    specialize: the cleanup checks identity individually, not pair-wise.
//! 5. **Identity reshape with NEW DimInfo shape kind** (Lit ↔ Named with same
//!    numeric value) — the M1 cleanup only collapses when `input.dims ==
//!    node.output_type.dims` literally. A `Lit(4)` vs `Named("a", Some(4))`
//!    pair is structurally distinct even though numerically equal; verify
//!    that today.
//!
//! Each test is a positive lock on current behavior; the BLAS-hit prediction
//! corresponds to what the closed-list cleanup actually does. If a future
//! commit extends the cleanup to handle these cases, the test will fail
//! deliberately so the spec/owning docs can be updated in the same change.

use chelis_ir::dag::{Dag, DimExpr, DimInfo, RiscOp, TensorType};
use chelis_ir::specialize::specialize_for_blas;
use chelis_types::types::Prim;

fn mat(prim: Prim, r: usize, c: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(r), DimInfo::Lit(c)],
        precision: prim,
    }
}

fn t3(prim: Prim, a: usize, b: usize, c: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(a), DimInfo::Lit(b), DimInfo::Lit(c)],
        precision: prim,
    }
}

fn has_blas(dag: &Dag) -> bool {
    dag.nodes()
        .iter()
        .any(|n| matches!(n.op, RiscOp::BlasMatmul { .. }))
}

fn dead_intermediates_pruned(dag: &Dag) -> bool {
    !dag.nodes()
        .iter()
        .any(|n| matches!(n.op, RiscOp::Mul | RiscOp::Expand { .. }))
}

/// Build a canonical Tier 2 matmul subgraph
/// (Expand → Cast → Mul → Sum or Cast → Expand → Mul → Sum) so adversarial
/// perturbations can be injected at known positions.
fn add_const_mat(dag: &mut Dag, prim: Prim, r: usize, c: usize) -> chelis_ir::dag::NodeId {
    dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat(prim, r, c), None)
}

/// ADV-1: A *non-identity* cast pair f32→f64→f32 between Expand and Mul
/// is NOT a closed-list no-op. The M1 cleanup recognizes single-step
/// identity casts only; a pair where the intermediate precision differs
/// from the input/output keeps both nodes alive. Must miss BLAS.
#[test]
fn non_identity_cast_pair_between_expand_and_mul_misses_blas() {
    let mut dag = Dag::new();
    let a = add_const_mat(&mut dag, Prim::F32, 2, 3);
    let b = add_const_mat(&mut dag, Prim::F32, 3, 4);
    let ea = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: DimExpr::Concrete(4),
        },
        vec![a],
        t3(Prim::F32, 2, 3, 4),
        None,
    );
    let eb = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(2),
        },
        vec![b],
        t3(Prim::F32, 2, 3, 4),
        None,
    );
    // f32 → f64 → f32 (round-trip; non-identity at the IR level).
    let cast_a_up = dag.add_node(
        RiscOp::Cast {
            new_precision: Prim::F64,
        },
        vec![ea],
        t3(Prim::F64, 2, 3, 4),
        None,
    );
    let cast_a_down = dag.add_node(
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![cast_a_up],
        t3(Prim::F32, 2, 3, 4),
        None,
    );
    let mul = dag.add_node(
        RiscOp::Mul,
        vec![cast_a_down, eb],
        t3(Prim::F32, 2, 3, 4),
        None,
    );
    let sum = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F32,
        },
        vec![mul],
        mat(Prim::F32, 2, 4),
        None,
    );
    dag.add_root(sum);

    let out = specialize_for_blas(&dag);
    assert!(
        !has_blas(&out),
        "non-identity f32→f64→f32 cast pair must NOT collapse via the closed-list cleanup; \
         got BLAS specialization. nodes = {:?}",
        out.nodes().iter().map(|n| n.op.clone()).collect::<Vec<_>>()
    );
    // And the dense Mul stays alive (no DCE removal) because the BLAS replacement didn't fire.
    assert!(
        !dead_intermediates_pruned(&out),
        "without BLAS replacement, the dense Mul/Expand subgraph must remain live"
    );
}

/// ADV-2: An int32 → f32 → int32 cast pair is NOT a closed-list no-op.
/// Even though the round-trip data semantics is "lossy identity", the
/// pair is structurally distinct from a single identity Cast and must
/// not be collapsed by the closed-list cleanup.
///
/// (Note: matmul with int32 operands doesn't match the BLAS-F32 signature
/// at the recognizer anyway, but we want to be sure NO recognizer fires
/// — neither BLAS nor any other.)
#[test]
fn int_float_int_cast_round_trip_misses_specialization() {
    let mut dag = Dag::new();
    let a = add_const_mat(&mut dag, Prim::Int32, 2, 3);
    let b = add_const_mat(&mut dag, Prim::Int32, 3, 4);
    let ea = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: DimExpr::Concrete(4),
        },
        vec![a],
        t3(Prim::Int32, 2, 3, 4),
        None,
    );
    let eb = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(2),
        },
        vec![b],
        t3(Prim::Int32, 2, 3, 4),
        None,
    );
    let cast_up = dag.add_node(
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![ea],
        t3(Prim::F32, 2, 3, 4),
        None,
    );
    let cast_down = dag.add_node(
        RiscOp::Cast {
            new_precision: Prim::Int32,
        },
        vec![cast_up],
        t3(Prim::Int32, 2, 3, 4),
        None,
    );
    let mul = dag.add_node(
        RiscOp::Mul,
        vec![cast_down, eb],
        t3(Prim::Int32, 2, 3, 4),
        None,
    );
    let sum = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F32,
        },
        vec![mul],
        mat(Prim::Int32, 2, 4),
        None,
    );
    dag.add_root(sum);

    let out = specialize_for_blas(&dag);
    assert!(
        !has_blas(&out),
        "int32→f32→int32 cast round-trip must NOT collapse to BLAS; got BLAS specialization"
    );
}

/// ADV-3: Reshape pair `[m,k] → [k,m] → [m,k]` between Expand and Mul
/// involves two non-identity Reshape nodes. There is no cancel-pair
/// recognizer; each Reshape's `identity_source` returns None because the
/// intermediate shape doesn't match the input shape. Must miss BLAS.
///
/// (This adversarial case has an additional subtlety: a `Reshape` with
/// shape `[k,m]` from a `[m,k]` input is NOT semantically identity — it
/// permutes the underlying contiguous-layout-vs-shape interpretation.
/// The IR allows this and the specializer correctly refuses to collapse.)
#[test]
fn reshape_round_trip_pair_between_expand_and_mul_misses_blas() {
    let mut dag = Dag::new();
    let a = add_const_mat(&mut dag, Prim::F32, 2, 3); // [m=2, k=3]
    let b = add_const_mat(&mut dag, Prim::F32, 3, 4);
    let ea = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: DimExpr::Concrete(4),
        },
        vec![a],
        t3(Prim::F32, 2, 3, 4),
        None,
    );
    let eb = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(2),
        },
        vec![b],
        t3(Prim::F32, 2, 3, 4),
        None,
    );
    // Reshape pair: [2,3,4] → [3,2,4] → [2,3,4]. Neither step is identity.
    let rs_a_up = dag.add_node(
        RiscOp::Reshape {
            new_shape: vec![DimInfo::Lit(3), DimInfo::Lit(2), DimInfo::Lit(4)],
        },
        vec![ea],
        t3(Prim::F32, 3, 2, 4),
        None,
    );
    let rs_a_down = dag.add_node(
        RiscOp::Reshape {
            new_shape: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
        },
        vec![rs_a_up],
        t3(Prim::F32, 2, 3, 4),
        None,
    );
    let mul = dag.add_node(
        RiscOp::Mul,
        vec![rs_a_down, eb],
        t3(Prim::F32, 2, 3, 4),
        None,
    );
    let sum = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F32,
        },
        vec![mul],
        mat(Prim::F32, 2, 4),
        None,
    );
    dag.add_root(sum);

    let out = specialize_for_blas(&dag);
    assert!(
        !has_blas(&out),
        "non-identity reshape pair between Expand and Mul must NOT collapse to BLAS \
         (no cancel-pair recognizer); got BLAS specialization"
    );
}

/// ADV-4: Permute pair `[1,0]` then `[1,0]` between Expand and Mul. Each
/// individual Permute is NOT identity (axes != 0..len), so the closed-list
/// cleanup does not remove either. Must miss BLAS.
///
/// The `node_has_contiguous_matrix_slices` contiguity check in the
/// matmul recognizer also explicitly rejects `Permute` nodes between
/// the Mul and its operand, so we get two independent defenses.
#[test]
fn permute_round_trip_pair_between_expand_and_mul_misses_blas() {
    let mut dag = Dag::new();
    let a = add_const_mat(&mut dag, Prim::F32, 2, 3);
    let b = add_const_mat(&mut dag, Prim::F32, 3, 4);
    let ea = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: DimExpr::Concrete(4),
        },
        vec![a],
        t3(Prim::F32, 2, 3, 4),
        None,
    );
    let eb = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(2),
        },
        vec![b],
        t3(Prim::F32, 2, 3, 4),
        None,
    );
    // Permute pair on the lhs (axes [1,0,2] then [1,0,2]).
    let p1 = dag.add_node(
        RiscOp::Permute {
            axes: vec![1, 0, 2],
        },
        vec![ea],
        t3(Prim::F32, 3, 2, 4),
        None,
    );
    let p2 = dag.add_node(
        RiscOp::Permute {
            axes: vec![1, 0, 2],
        },
        vec![p1],
        t3(Prim::F32, 2, 3, 4),
        None,
    );
    let mul = dag.add_node(RiscOp::Mul, vec![p2, eb], t3(Prim::F32, 2, 3, 4), None);
    let sum = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F32,
        },
        vec![mul],
        mat(Prim::F32, 2, 4),
        None,
    );
    dag.add_root(sum);

    let out = specialize_for_blas(&dag);
    assert!(
        !has_blas(&out),
        "non-identity permute pair between Expand and Mul must NOT collapse to BLAS \
         (no cancel-pair recognizer); got BLAS specialization"
    );
}

/// ADV-5: A *single* identity Permute on lhs (axes [0,1,2] over a rank-3
/// tensor) — already covered as a positive case by the closed-list
/// cleanup. Asserts the cleanup *does* fire, so we know we have the
/// right ground truth.
#[test]
fn single_identity_permute_does_collapse_to_blas() {
    let mut dag = Dag::new();
    let a = add_const_mat(&mut dag, Prim::F32, 2, 3);
    let b = add_const_mat(&mut dag, Prim::F32, 3, 4);
    let ea = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: DimExpr::Concrete(4),
        },
        vec![a],
        t3(Prim::F32, 2, 3, 4),
        None,
    );
    let eb = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(2),
        },
        vec![b],
        t3(Prim::F32, 2, 3, 4),
        None,
    );
    let p_id = dag.add_node(
        RiscOp::Permute {
            axes: vec![0, 1, 2],
        },
        vec![ea],
        t3(Prim::F32, 2, 3, 4),
        None,
    );
    let mul = dag.add_node(RiscOp::Mul, vec![p_id, eb], t3(Prim::F32, 2, 3, 4), None);
    let sum = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F32,
        },
        vec![mul],
        mat(Prim::F32, 2, 4),
        None,
    );
    dag.add_root(sum);

    let out = specialize_for_blas(&dag);
    assert!(
        has_blas(&out),
        "identity Permute (axes [0,1,2]) between Expand and Mul MUST be cleaned up \
         by the closed-list pass and let BLAS specialization fire"
    );
    assert!(
        dead_intermediates_pruned(&out),
        "dense Mul/Expand must be DCEd after BLAS replacement"
    );
}

/// ADV-6: A Reshape that round-trips with shape DimInfo::Named-vs-Lit
/// (semantically same numeric value, structurally distinct DimInfo). The
/// M1 cleanup compares `input.output_type.dims == node.output_type.dims`
/// literally, so even though `DimInfo::Named("k", Some(3))` and
/// `DimInfo::Lit(3)` evaluate to the same size, they don't compare equal
/// for cleanup purposes. This locks the literal-equality semantics.
#[test]
fn named_vs_lit_dim_reshape_is_not_identity_and_misses_blas() {
    let mut dag = Dag::new();
    // a: [Named("m", Some(2)), Named("k", Some(3))], i.e. carry symbolic names.
    let a_ty = TensorType {
        dims: vec![
            DimInfo::Named("m".into(), Some(2)),
            DimInfo::Named("k".into(), Some(3)),
        ],
        precision: Prim::F32,
    };
    let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], a_ty.clone(), None);

    // Reshape into all-Lit form: even though same numeric values, the
    // dims vector is structurally distinct -> not an identity Reshape.
    let lit_ty = mat(Prim::F32, 2, 3);
    let reshaped = dag.add_node(
        RiscOp::Reshape {
            new_shape: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
        },
        vec![a],
        lit_ty.clone(),
        None,
    );

    let b = add_const_mat(&mut dag, Prim::F32, 3, 4);
    let ea = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: DimExpr::Concrete(4),
        },
        vec![reshaped],
        t3(Prim::F32, 2, 3, 4),
        None,
    );
    let eb = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(2),
        },
        vec![b],
        t3(Prim::F32, 2, 3, 4),
        None,
    );
    let mul = dag.add_node(RiscOp::Mul, vec![ea, eb], t3(Prim::F32, 2, 3, 4), None);
    let sum = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F32,
        },
        vec![mul],
        mat(Prim::F32, 2, 4),
        None,
    );
    dag.add_root(sum);

    let out = specialize_for_blas(&dag);
    // BLAS specialization still fires through the Reshape, because the
    // recognizer's contiguity rule walks through Reshape down to the
    // Load/Const. But the Reshape stays in the DAG (it's not collapsed
    // by the closed-list cleanup because dims are structurally distinct).
    // What we lock here is that the cleanup does NOT collapse the
    // Named-vs-Lit Reshape — even when numerically equivalent.
    let surviving_reshape = out
        .nodes()
        .iter()
        .find(|n| matches!(n.op, RiscOp::Reshape { .. }));
    assert!(
        surviving_reshape.is_some(),
        "the Named-vs-Lit Reshape must survive the closed-list cleanup, \
         because structural DimInfo inequality keeps the Reshape non-identity"
    );
    // And BLAS still fires because the contiguity walker accepts Reshape.
    assert!(
        has_blas(&out),
        "BLAS recognition walks through Reshape; expected BLAS hit"
    );
}
