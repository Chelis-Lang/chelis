//! C in-place fused-elementwise reuse consumes the shared planner's exact
//! shape, dtype, capacity, operation, and lifetime proof.

use chelis_ir::dag::{Dag, DimInfo, FusedInput, FusedStep, FusedStepOp, RiscOp, TensorType};
use chelis_types::types::Prim;

mod support;
use support::emit_dag;

fn vec_lit_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn vec_named_f32(name: &str, size: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Named(name.to_string(), Some(size))],
        precision: Prim::F32,
    }
}

fn vec_named_unsized_f32(name: &str) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Named(name.to_string(), None)],
        precision: Prim::F32,
    }
}

fn vec_lit_f64(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F64,
    }
}

/// Build a 3-input fan-in FusedElem chain: `((a + b) * c)` where `a` is
/// marked as the reusable input via `set_reusable_input`. Each external
/// input is intermediate (a `Copy` node so memory planning treats it
/// as SlotBacked, mirroring the copy-elision-probe fan-in shape).
///
/// `a_ty`, `b_ty`, `c_ty`, `out_ty` are independent so callers can probe
/// resolved exact equality, unresolved extents, and mismatches without
/// rebuilding the chain.
fn fan_in_dag(
    a_ty: TensorType,
    b_ty: TensorType,
    c_ty: TensorType,
    out_ty: TensorType,
) -> (Dag, chelis_ir::dag::NodeId, chelis_ir::dag::NodeId) {
    let mut dag = Dag::new();
    // Sources: Load + Copy stages so each fan-in input is SlotBacked
    // (intermediate), not BorrowedLoad. This mirrors how `copy(x)` arms
    // materialize into distinct backing slots in the real probe.
    let x_a = dag.add_node(
        RiscOp::Load { name: "x_a".into() },
        vec![],
        a_ty.clone(),
        None,
    );
    let a = dag.add_node(RiscOp::Copy, vec![x_a], a_ty.clone(), None);

    let x_b = dag.add_node(
        RiscOp::Load { name: "x_b".into() },
        vec![],
        b_ty.clone(),
        None,
    );
    let b = dag.add_node(RiscOp::Copy, vec![x_b], b_ty.clone(), None);

    let x_c = dag.add_node(
        RiscOp::Load { name: "x_c".into() },
        vec![],
        c_ty.clone(),
        None,
    );
    let c = dag.add_node(RiscOp::Copy, vec![x_c], c_ty.clone(), None);

    // Fused: `v0 = a + b; v1 = v0 * c;` over three external inputs.
    let ops = vec![
        FusedStep {
            op: FusedStepOp::Add,
            input_indices: vec![FusedInput::External(0), FusedInput::External(1)],
        },
        FusedStep {
            op: FusedStepOp::Mul,
            input_indices: vec![FusedInput::PreviousStep(0), FusedInput::External(2)],
        },
    ];
    let fused = dag.add_node(RiscOp::FusedElem { ops }, vec![a, b, c], out_ty, None);
    dag.set_reusable_input(fused, a);
    dag.add_root(fused);
    (dag, fused, a)
}

fn assert_reuse_proven(c: &str, fused: chelis_ir::dag::NodeId, reusable: chelis_ir::dag::NodeId) {
    let fused_id = fused.0;
    let reusable_id = reusable.0;
    assert!(
        !c.contains("chelis_alloc_view"),
        "the C backend must not restore the deleted unproved view allocator; got:\n{c}"
    );
    assert!(
        c.contains(&format!("chelis_tensor *t{fused_id} = t{reusable_id};")),
        "fused output must consume the shared reuse proof; got:\n{c}"
    );
    assert!(
        c.contains(&format!(
            "chelis_tensor_repurpose(t{fused_id}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, UINT64_C(1)), (chelis_scalar[]){{ chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)(int64_t)(4)) }});"
        )),
        "proven reuse must reset exact descriptor metadata; got:\n{c}"
    );
    assert!(
        c.contains(&format!(
            "float* __out_{fused_id} = (float*)t{fused_id}_data;"
        )),
        "aliased output must not claim restrict; got:\n{c}"
    );
}

fn assert_reuse_rejected(c: &str, fused: chelis_ir::dag::NodeId, reusable: chelis_ir::dag::NodeId) {
    assert!(
        !c.contains(&format!("chelis_tensor *t{} = t{};", fused.0, reusable.0)),
        "unproved reuse must not transfer the source descriptor; got:\n{c}"
    );
    assert!(
        c.contains(&format!("chelis_tensor *t{} = chelis_alloc(", fused.0)),
        "unproved reuse must allocate owned storage; got:\n{c}"
    );
}

/// Positive: a program-owned, terminal input with an exact literal shape
/// satisfies the shared proof and transfers its descriptor to the output.
#[test]
fn fan_in_literal_equal_shapes_consume_shared_proof() {
    let (dag, fused, a) = fan_in_dag(
        vec_lit_f32(4),
        vec_lit_f32(4),
        vec_lit_f32(4),
        vec_lit_f32(4),
    );
    let c = emit_dag(&dag, "test_fan_in_literal").unwrap();
    assert_reuse_proven(&c, fused, a);
}

/// Positive: `Lit(4)` and `Named(_, Some(4))` both carry the exact extent
/// value four. The shared `CapacityKey` proof ignores the non-semantic name.
#[test]
fn fan_in_resolved_lit_to_named_consumes_exact_proof() {
    let (dag, fused, a) = fan_in_dag(
        vec_lit_f32(4),
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
    );
    let c = emit_dag(&dag, "test_fan_in_binder_lit_to_named").unwrap();
    assert_reuse_proven(&c, fused, a);
}

/// Positive: resolved exact equality is symmetric when the source uses a
/// named extent and the consumer uses a literal extent.
#[test]
fn fan_in_resolved_named_to_lit_consumes_exact_proof() {
    let (dag, fused, a) = fan_in_dag(
        vec_named_f32("seq", 4),
        vec_lit_f32(4),
        vec_lit_f32(4),
        vec_lit_f32(4),
    );
    let c = emit_dag(&dag, "test_fan_in_binder_named_to_lit").unwrap();
    assert_reuse_proven(&c, fused, a);
}

/// Positive: matching resolved named extents satisfy the exact proof.
#[test]
fn fan_in_same_resolved_name_consumes_exact_proof() {
    let (dag, fused, a) = fan_in_dag(
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
    );
    let c = emit_dag(&dag, "test_fan_in_same_named_binder").unwrap();
    assert_reuse_proven(&c, fused, a);
}

/// Negative: an unresolved source extent does not become exact merely
/// because its spelling matches a resolved consumer extent.
#[test]
fn fan_in_named_binder_with_unknown_size_defers_reuse_without_shared_proof() {
    let (dag, fused, a) = fan_in_dag(
        vec_named_unsized_f32("seq"),
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
    );
    let c = emit_dag(&dag, "test_fan_in_named_unsized").unwrap();
    assert_reuse_rejected(&c, fused, a);
}

/// Positive: resolved extent values are the capacity authority; different
/// source-level names with the same exact value do not prevent reuse.
#[test]
fn fan_in_different_resolved_names_alias_exact_equal_shape() {
    let (dag, fused, a) = fan_in_dag(
        vec_named_f32("batch", 4),
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
    );
    let c = emit_dag(&dag, "test_fan_in_different_binders").unwrap();
    assert_reuse_proven(&c, fused, a);
}

/// Negative: a repeated name cannot equate conflicting exact extent values.
#[test]
fn fan_in_same_binder_different_known_size_does_not_alias() {
    let (dag, fused, a) = fan_in_dag(
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 8),
        vec_named_f32("seq", 8),
        vec_named_f32("seq", 8),
    );
    let c = emit_dag(&dag, "test_fan_in_size_mismatch").unwrap();
    assert_reuse_rejected(&c, fused, a);
}

/// Negative: exact representation, not equal extent or equal byte width,
/// is part of the shared proof.
#[test]
fn fan_in_different_precision_does_not_alias() {
    let (dag, fused, a) = fan_in_dag(
        vec_lit_f64(4),
        vec_lit_f32(4),
        vec_lit_f32(4),
        vec_lit_f32(4),
    );
    let c = emit_dag(&dag, "test_fan_in_different_precision").unwrap();
    assert_reuse_rejected(&c, fused, a);
}

/// Negative: even when capacities and representations are exact, an input
/// with more than one consumer must NOT be
/// aliased in-place. Mutating shared buffers breaks the second consumer.
#[test]
fn fan_in_multi_consumer_reusable_input_does_not_alias() {
    let mut dag = Dag::new();
    let x_a = dag.add_node(
        RiscOp::Load { name: "x_a".into() },
        vec![],
        vec_lit_f32(4),
        None,
    );
    let a = dag.add_node(RiscOp::Copy, vec![x_a], vec_named_f32("seq", 4), None);
    let x_b = dag.add_node(
        RiscOp::Load { name: "x_b".into() },
        vec![],
        vec_named_f32("seq", 4),
        None,
    );
    let b = dag.add_node(RiscOp::Copy, vec![x_b], vec_named_f32("seq", 4), None);

    let ops = vec![FusedStep {
        op: FusedStepOp::Add,
        input_indices: vec![FusedInput::External(0), FusedInput::External(1)],
    }];
    let fused = dag.add_node(
        RiscOp::FusedElem { ops },
        vec![a, b],
        vec_named_f32("seq", 4),
        None,
    );
    dag.set_reusable_input(fused, a);

    // The second consumer keeps `a` live and forces a fresh fused output.
    let other = dag.add_node(RiscOp::Neg, vec![a], vec_named_f32("seq", 4), None);
    dag.add_root(fused);
    dag.add_root(other);

    let c = emit_dag(&dag, "test_fan_in_multi_consumer").unwrap();
    assert_reuse_rejected(&c, fused, a);
}

/// Negative: equal total capacity does not satisfy the per-axis exact-shape
/// requirement when source and consumer ranks differ.
#[test]
fn fan_in_different_rank_does_not_alias() {
    let ty_r1 = TensorType {
        dims: vec![DimInfo::Lit(8)],
        precision: Prim::F32,
    };
    let ty_r2 = TensorType {
        dims: vec![DimInfo::Lit(2), DimInfo::Lit(4)],
        precision: Prim::F32,
    };
    let (dag, fused, a) = fan_in_dag(ty_r1.clone(), ty_r2.clone(), ty_r2.clone(), ty_r2);
    let c = emit_dag(&dag, "test_fan_in_different_rank").unwrap();
    assert_reuse_rejected(&c, fused, a);
}
