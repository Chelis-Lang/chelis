//! HIP in-place fused-elementwise aliasing consumes the shared planner's
//! exact shape, dtype, capacity, operation, and lifetime proof.
//!
//! Wire format expectations are pinned with exact full-statement
//! `contains` strings (not partial fragments) so a regression that
//! drops the in-place alias or reshapes the wrapper preamble is
//! immediately visible. Each `assert!` matches a complete line that
//! appears verbatim in the emitted HIP host code or the embedded
//! kernel-source string declaration.
//!
//! Boundary: this file exercises the shared proof as consumed by the HIP
//! kernel-signature and wrapper paths. It does not
//! touch the BLAS dispatch (W1-A) or the new `Scatter` emit branch
//! (W2-A); both are intentionally out of scope here.

mod support;
use chelis_ir::dag::{
    Dag, DimInfo, FusedInput, FusedStep, FusedStepOp, NodeId, RiscOp, TensorType,
};
use chelis_types::types::Prim;
use support::codegen_hip;

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

fn vec_lit_i32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::Int32,
    }
}

/// Build a 3-input fan-in FusedElem chain: `((a + b) * c)` where `a`
/// is the reusable input. Each external input is a `Copy` so memory
/// planning treats it as `SlotBacked` (intermediate), mirroring how
/// `copy(x)` fan-in shapes materialize after the linearity analyzer
/// attaches the reuse hint. This mirrors the C-side fan_in_dag helper
/// exactly (same node creation order, same `Copy` wrappers) so the
/// emitted node ids are predictable: `a = NodeId(1)`, `b = NodeId(3)`,
/// `c = NodeId(5)`, `fused = NodeId(6)`.
fn fan_in_dag(
    a_ty: TensorType,
    b_ty: TensorType,
    c_ty: TensorType,
    out_ty: TensorType,
) -> (Dag, NodeId, NodeId) {
    let mut dag = Dag::new();
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

/// Build the exact-string host-side branch that admits the in-place
/// alias view on HIP. The wrapper opens with the bare-declaration of
/// the FusedElem's GPU tensor, then a runtime contiguity guard. The
/// contiguous branch aliases the FusedElem's view onto the reusable
/// input's device data + storage_size.
fn expected_in_place_view_alias(fused_id: usize, reusable_id: usize, shape: &str) -> String {
    format!(
        "d_t{fused_id} = chelis_gpu_alloc_view(1, (int[]){{ {shape} }}, CHELIS_DTYPE_F32, d_t{reusable_id}->data, d_t{reusable_id}->storage_size);"
    )
}

fn expected_contiguity_guard(reusable_id: usize) -> String {
    format!("if (chelis_gpu_is_contiguous(d_t{reusable_id})) {{")
}

fn expected_bare_declaration(fused_id: usize) -> String {
    format!("chelis_gpu_tensor *d_t{fused_id};")
}

// ---------------------------------------------------------------------
// Positive cases
// ---------------------------------------------------------------------

/// Pinning regression: fan-in with literal-equal shapes aliases the
/// FusedElem's output view onto the reusable input's device buffer.
/// This is the W2-B shipped behavior on HIP. The wrapper must emit
/// the bare `chelis_gpu_tensor *d_t{fused};` declaration (not a slot
/// allocation), the runtime contiguity guard, and the
/// `chelis_gpu_alloc_view` aliased onto `d_t{reusable}->data`.
#[test]
fn fan_in_literal_equal_shapes_aliases_reusable_input() {
    let (dag, fused, a) = fan_in_dag(
        vec_lit_f32(4),
        vec_lit_f32(4),
        vec_lit_f32(4),
        vec_lit_f32(4),
    );
    let result = codegen_hip(&dag, "test_fan_in_literal").unwrap();
    let hip = &result.c_source;
    let fused_id = fused.0;
    let a_id = a.0;

    let bare_decl = expected_bare_declaration(fused_id);
    assert!(
        hip.contains(&bare_decl),
        "expected bare GPU tensor declaration `{bare_decl}` for in-place fused; got:\n{hip}"
    );
    let guard = expected_contiguity_guard(a_id);
    assert!(
        hip.contains(&guard),
        "expected contiguity guard `{guard}` on reusable input; got:\n{hip}"
    );
    let alias = expected_in_place_view_alias(fused_id, a_id, "4");
    assert!(
        hip.contains(&alias),
        "expected literal-shape fan-in to alias d_t{fused_id} onto d_t{a_id}->data; got:\n{hip}"
    );
    // Kernel must not put `__restrict__` on the aliased input (ext0) or out.
    assert!(
        !hip.contains("const float *__restrict__ ext0"),
        "aliased external ext0 must not carry __restrict__; got:\n{hip}"
    );
    assert!(
        !hip.contains("float *__restrict__ out"),
        "fused output must not carry __restrict__ when aliasing; got:\n{hip}"
    );
    // Non-aliased externals (ext1, ext2) MUST carry __restrict__.
    assert!(
        hip.contains("const float *__restrict__ ext1"),
        "non-aliased external ext1 must carry __restrict__; got:\n{hip}"
    );
    assert!(
        hip.contains("const float *__restrict__ ext2"),
        "non-aliased external ext2 must carry __restrict__; got:\n{hip}"
    );
}

/// Positive: a literal and a resolved named extent with value four have
/// exact-equal `CapacityKey`s despite their different source spelling.
#[test]
fn fan_in_resolved_lit_to_named_aliases_reusable_input() {
    let (dag, fused, a) = fan_in_dag(
        vec_lit_f32(4),
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
    );
    let result = codegen_hip(&dag, "test_fan_in_binder_lit_to_named").unwrap();
    let hip = &result.c_source;
    let fused_id = fused.0;
    let a_id = a.0;

    let alias = expected_in_place_view_alias(fused_id, a_id, "4");
    assert!(
        hip.contains(&alias),
        "exact Lit->resolved-Named fan-in must alias d_t{fused_id} onto d_t{a_id}->data; got:\n{hip}"
    );
    assert!(
        hip.contains(&expected_contiguity_guard(a_id)),
        "expected contiguity guard on reusable input; got:\n{hip}"
    );
    assert!(
        !hip.contains("const float *__restrict__ ext0"),
        "aliased external ext0 must not carry __restrict__; got:\n{hip}"
    );
}

/// Positive: resolved exact equality is symmetric when the source uses a
/// named extent and the consumer uses a literal extent.
#[test]
fn fan_in_resolved_named_to_lit_aliases_reusable_input() {
    let (dag, fused, a) = fan_in_dag(
        vec_named_f32("seq", 4),
        vec_lit_f32(4),
        vec_lit_f32(4),
        vec_lit_f32(4),
    );
    let result = codegen_hip(&dag, "test_fan_in_binder_named_to_lit").unwrap();
    let hip = &result.c_source;
    let fused_id = fused.0;
    let a_id = a.0;

    let alias = expected_in_place_view_alias(fused_id, a_id, "4");
    assert!(
        hip.contains(&alias),
        "exact resolved-Named->Lit fan-in must alias d_t{fused_id} onto d_t{a_id}->data; got:\n{hip}"
    );
}

/// Positive: matching resolved named extents satisfy the exact proof.
#[test]
fn fan_in_same_named_binder_aliases_reusable_input() {
    let (dag, fused, a) = fan_in_dag(
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
    );
    let result = codegen_hip(&dag, "test_fan_in_same_named_binder").unwrap();
    let hip = &result.c_source;
    let fused_id = fused.0;
    let a_id = a.0;

    let alias = expected_in_place_view_alias(fused_id, a_id, "4");
    assert!(
        hip.contains(&alias),
        "matching resolved Named fan-in must alias d_t{fused_id} onto d_t{a_id}->data; got:\n{hip}"
    );
}

/// Negative — an unresolved source axis is not an exact proof of literal
/// four, even if its display name matches the resolved output name.
#[test]
fn fan_in_unresolved_axis_does_not_alias_resolved_literal() {
    let (dag, fused, a) = fan_in_dag(
        vec_named_unsized_f32("seq"),
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
    );
    let result = codegen_hip(&dag, "test_fan_in_named_unsized").unwrap();
    let hip = &result.c_source;
    let fused_id = fused.0;
    let a_id = a.0;

    let forbidden_alias = expected_in_place_view_alias(fused_id, a_id, "4");
    assert!(
        !hip.contains(&forbidden_alias),
        "an unresolved axis must not authorize reuse as literal four; got:\n{hip}"
    );
    assert!(!hip.contains(&expected_contiguity_guard(a_id)), "{hip}");
}

// ---------------------------------------------------------------------
// Negative cases
// ---------------------------------------------------------------------

/// Positive — resolved named dimensions are exact literals. Different source
/// spellings do not defeat a proof that both axes are exactly four.
#[test]
fn fan_in_different_resolved_names_alias_exact_equal_shape() {
    let (dag, fused, a) = fan_in_dag(
        vec_named_f32("batch", 4),
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
    );
    let result = codegen_hip(&dag, "test_fan_in_different_binders").unwrap();
    let hip = &result.c_source;
    let fused_id = fused.0;
    let a_id = a.0;

    let alias = expected_in_place_view_alias(fused_id, a_id, "4");
    assert!(
        hip.contains(&alias),
        "resolved exact-equal shapes must alias d_t{fused_id} onto d_t{a_id}->data; got:\n{hip}"
    );
    assert!(
        hip.contains(&expected_contiguity_guard(a_id)),
        "exact reuse must emit the guard derived from the same source token; got:\n{hip}"
    );
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
    let result = codegen_hip(&dag, "test_fan_in_size_mismatch").unwrap();
    let hip = &result.c_source;
    let fused_id = fused.0;
    let a_id = a.0;

    // FusedElem output is `Named("seq", Some(8))` so the alias view
    // shape literal would be `(int[]){ 8 }` if the gate misfired.
    let forbidden_alias = expected_in_place_view_alias(fused_id, a_id, "8");
    assert!(
        !hip.contains(&forbidden_alias),
        "conflicting known sizes must NOT alias; got:\n{hip}"
    );
    let forbidden_guard = expected_contiguity_guard(a_id);
    assert!(
        !hip.contains(&forbidden_guard),
        "conflicting size must NOT emit contiguity guard; got:\n{hip}"
    );
}

/// Negative: Int32 and F32 have distinct exact representations even with
/// matching extents; representation is part of the alias proof.
///
/// Re-authored by chelis#730 Phase 1 (census row 5, chelis#689): the
/// former `elem_kind` F32 wildcard let this mixed-precision fused
/// fan-in EMIT (reading the int32 operand through the f32 kernel - the
/// corruption class chelis#689 runtime-confirmed), and the old
/// assertion only checked the alias line was absent in that corrupt
/// emission. There is no typed mixed-precision fused kernel, so HIP
/// codegen now rejects the DAG loudly - which also guarantees no alias
/// wrapper is emitted. The dtype half of the alias proof is thereby
/// subsumed by the rejection; the same-precision alias-gate rows above
/// keep the gate's live coverage.
#[test]
fn fan_in_different_precision_does_not_alias() {
    let (dag, _fused, _a) = fan_in_dag(
        vec_lit_i32(4),
        vec_lit_f32(4),
        vec_lit_f32(4),
        vec_lit_f32(4),
    );
    let err = codegen_hip(&dag, "test_fan_in_different_precision")
        .map(|_| ())
        .expect_err("a mixed int32/f32 fused fan-in has no typed HIP kernel");
    let rendered = err.to_string();
    assert!(
        rendered.starts_with("unsupported:") && rendered.contains("int32"),
        "the rejection must be branded and name the dtype; got: {rendered}"
    );
}

/// Negative: even with exact-equal shape and representation, an input with
/// more than one consumer
/// must NOT be aliased in-place. Mutating the shared buffer would
/// corrupt the second consumer's read.
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

    let result = codegen_hip(&dag, "test_fan_in_multi_consumer").unwrap();
    let hip = &result.c_source;
    let fused_id = fused.0;
    let a_id = a.0;

    let forbidden_alias = expected_in_place_view_alias(fused_id, a_id, "4");
    assert!(
        !hip.contains(&forbidden_alias),
        "multi-consumer reusable input must NOT alias even with exact shape; got:\n{hip}"
    );
    let forbidden_guard = expected_contiguity_guard(a_id);
    assert!(
        !hip.contains(&forbidden_guard),
        "multi-consumer must NOT emit contiguity guard; got:\n{hip}"
    );
}

/// Negative: equal total capacity does not make a mixed-positive-rank
/// same-shape producer valid. The malformed relation must be rejected before
/// storage planning can consider an in-place alias.
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
    let (dag, _, _) = fan_in_dag(ty_r1.clone(), ty_r2.clone(), ty_r2.clone(), ty_r2);
    let error = match chelis_ir::ownership::lower_dag_ownership(dag) {
        Ok(_) => panic!("mixed-positive-rank same-shape producer must be rejected"),
        Err(error) => error,
    };
    let rendered = error.to_string();
    assert!(
        rendered.contains("same-shape result at node 6")
            && rendered.contains("positive-rank operand")
            && rendered.contains("rank 1")
            && rendered.contains("expected rank 2"),
        "unexpected malformed-agreement rejection: {error}"
    );
}

/// Negative — no reusable input hint at all: the gate must reject and
/// the legacy slot-backed wrapper + non-`__restrict__` kernel shape
/// must be emitted. The in-place path is opt-in via the upstream
/// linearity hint, not the default.
#[test]
fn fan_in_no_reusable_input_keeps_slot_backed_path() {
    let mut dag = Dag::new();
    let a_ty = vec_lit_f32(4);
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
        a_ty.clone(),
        None,
    );
    let b = dag.add_node(RiscOp::Copy, vec![x_b], a_ty.clone(), None);

    let ops = vec![FusedStep {
        op: FusedStepOp::Add,
        input_indices: vec![FusedInput::External(0), FusedInput::External(1)],
    }];
    let fused = dag.add_node(RiscOp::FusedElem { ops }, vec![a, b], a_ty, None);
    // No set_reusable_input.
    dag.add_root(fused);

    let result = codegen_hip(&dag, "test_fan_in_no_reusable").unwrap();
    let hip = &result.c_source;
    let fused_id = fused.0;
    let a_id = a.0;

    let forbidden_alias = expected_in_place_view_alias(fused_id, a_id, "4");
    assert!(
        !hip.contains(&forbidden_alias),
        "no-reusable-input fused must NOT alias; got:\n{hip}"
    );
    let forbidden_guard = expected_contiguity_guard(a_id);
    assert!(
        !hip.contains(&forbidden_guard),
        "no-reusable-input must NOT emit contiguity guard; got:\n{hip}"
    );
    let forbidden_bare = expected_bare_declaration(fused_id);
    assert!(
        !hip.contains(&forbidden_bare),
        "no-reusable-input must NOT emit bare GPU tensor declaration; got:\n{hip}"
    );
    // Legacy kernel shape: no __restrict__ anywhere in the fused kernel.
    assert!(
        !hip.contains("__restrict__ ext0"),
        "no-reusable-input must not adopt in-place __restrict__ shape; got:\n{hip}"
    );
}
