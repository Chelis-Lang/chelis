//! Perf-F2(c): C in-place fused-elementwise aliasing must admit scoped
//! same-property `forall` / binder-equivalent aliases, not just literal
//! `DimInfo`-PartialEq matches.
//!
//! Wire format expectations are pinned with exact `contains` strings (not
//! pattern fragments) so a regression that drops the in-place alias or
//! reshapes the wrapper preamble is immediately visible.
//!
//! Boundary: this file exercises only the in-place fusion gate in
//! `chelis-backend-c::emit::fused_in_place_spec`. The slot planner in
//! `chelis-backend-c::memory` is the M2a peer and intentionally out of
//! scope; the fan-in shapes used here construct a deduplicated FusedElem
//! whose `reusable_input` is set by the test, mirroring what fusion does
//! after the linearity analyzer attaches the reuse hint.

use chelis_ir::dag::{Dag, DimInfo, FusedInput, FusedStep, FusedStepOp, RiscOp, TensorType};
use chelis_types::types::Prim;

use chelis_backend_c::emit::CEmitter;

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
/// input is intermediate (a `Realize` node so memory planning treats it
/// as SlotBacked, mirroring the copy-elision-probe fan-in shape).
///
/// `a_ty`, `b_ty`, `c_ty`, `out_ty` are independent so callers can probe
/// binder-equivalent vs literal-equal vs non-equivalent shapes without
/// rebuilding the chain.
fn fan_in_dag(
    a_ty: TensorType,
    b_ty: TensorType,
    c_ty: TensorType,
    out_ty: TensorType,
) -> (Dag, chelis_ir::dag::NodeId, chelis_ir::dag::NodeId) {
    let mut dag = Dag::new();
    // Sources: Load + Realize stages so each fan-in input is SlotBacked
    // (intermediate), not BorrowedLoad. This mirrors how `copy(x)` arms
    // materialize into distinct backing slots in the real probe.
    let x_a = dag.add_node(
        RiscOp::Load { name: "x_a".into() },
        vec![],
        a_ty.clone(),
        None,
    );
    let a = dag.add_node(RiscOp::Realize, vec![x_a], a_ty.clone(), None);

    let x_b = dag.add_node(
        RiscOp::Load { name: "x_b".into() },
        vec![],
        b_ty.clone(),
        None,
    );
    let b = dag.add_node(RiscOp::Realize, vec![x_b], b_ty.clone(), None);

    let x_c = dag.add_node(
        RiscOp::Load { name: "x_c".into() },
        vec![],
        c_ty.clone(),
        None,
    );
    let c = dag.add_node(RiscOp::Realize, vec![x_c], c_ty.clone(), None);

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

/// Pinning regression: fan-in with literal-equal shapes still aliases the
/// FusedElem's output to the reusable input's data. This is the existing
/// in-place wire format the gate already produces; it must not change.
#[test]
fn fan_in_literal_equal_shapes_aliases_reusable_input() {
    let (dag, fused, a) = fan_in_dag(
        vec_lit_f32(4),
        vec_lit_f32(4),
        vec_lit_f32(4),
        vec_lit_f32(4),
    );
    let c = CEmitter::emit_dag(&dag, "test_fan_in_literal").unwrap();
    let fused_id = fused.0;
    let a_id = a.0;

    // Exact-shape: the in-place wrapper must alias the FusedElem output
    // view to the reusable input's data buffer.
    let expected_alias =
        format!("t{fused_id} = chelis_alloc_view(1, (int[]){{ 4 }}, CHELIS_F32, t{a_id}->data);");
    assert!(
        c.contains(&expected_alias),
        "expected literal-shape fan-in to alias t{fused_id} onto t{a_id}->data; got:\n{c}"
    );

    // Exact-shape: the non-aliased fast-path output pointer must NOT use
    // restrict (it points at the same buffer as ext0).
    let expected_out = format!("float* __out_{fused_id} = t{fused_id}->data;");
    assert!(
        c.contains(&expected_out),
        "expected non-restrict __out_{fused_id} pointer; got:\n{c}"
    );

    // Exact-shape: the reusable input is the non-restrict ext0; the other
    // two external inputs remain restrict-qualified.
    let expected_ext0 = format!("const float* __ext0_{fused_id} = t{a_id}->data;");
    assert!(
        c.contains(&expected_ext0),
        "expected non-restrict __ext0_{fused_id} pointer for reusable input t{a_id}; got:\n{c}"
    );
}

/// Positive — binder-equivalent: FusedElem output uses
/// `Named("seq", Some(4))` while reusable input uses `Lit(4)` and the
/// other inputs use the matching `Named("seq", Some(4))`. PartialEq says
/// these `TensorType`s are NOT equal because `DimInfo::Lit != DimInfo::Named`.
/// The binder-equivalent extension must still admit the in-place alias.
///
/// This is the case the Perf-F2(c) closure narrative names: the IR has
/// proven the same forall-binder is in scope for both the producer (Lit
/// concrete after specialize / lower) and the consumer (Named, still
/// carrying the source binder name), and the C backend must not fall
/// back just because the structural `DimInfo` representations differ.
#[test]
fn fan_in_binder_equivalent_lit_to_named_aliases_reusable_input() {
    let (dag, fused, a) = fan_in_dag(
        vec_lit_f32(4),
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
    );
    let c = CEmitter::emit_dag(&dag, "test_fan_in_binder_lit_to_named").unwrap();
    let fused_id = fused.0;
    let a_id = a.0;

    // Shape literal: `Named("seq", Some(4))` lowers to `DimExpr::Concrete(4)`
    // so the emitted shape array is `(int[]){ 4 }`, not `(int[]){ seq }`.
    let expected_alias =
        format!("t{fused_id} = chelis_alloc_view(1, (int[]){{ 4 }}, CHELIS_F32, t{a_id}->data);");
    assert!(
        c.contains(&expected_alias),
        "binder-equivalent Lit->Named fan-in must alias t{fused_id} onto t{a_id}->data; got:\n{c}"
    );

    let expected_out = format!("float* __out_{fused_id} = t{fused_id}->data;");
    assert!(
        c.contains(&expected_out),
        "expected non-restrict __out_{fused_id} pointer for binder-equivalent in-place; got:\n{c}"
    );

    let expected_ext0 = format!("const float* __ext0_{fused_id} = t{a_id}->data;");
    assert!(
        c.contains(&expected_ext0),
        "expected non-restrict __ext0_{fused_id} pointer for binder-equivalent reusable input; got:\n{c}"
    );
}

/// Positive — binder-equivalent the other direction: FusedElem output
/// uses `Lit(4)` while reusable input is `Named("seq", Some(4))`. The
/// alias must still fire — the binder-equivalent predicate is symmetric.
#[test]
fn fan_in_binder_equivalent_named_to_lit_aliases_reusable_input() {
    let (dag, fused, a) = fan_in_dag(
        vec_named_f32("seq", 4),
        vec_lit_f32(4),
        vec_lit_f32(4),
        vec_lit_f32(4),
    );
    let c = CEmitter::emit_dag(&dag, "test_fan_in_binder_named_to_lit").unwrap();
    let fused_id = fused.0;
    let a_id = a.0;

    let expected_alias =
        format!("t{fused_id} = chelis_alloc_view(1, (int[]){{ 4 }}, CHELIS_F32, t{a_id}->data);");
    assert!(
        c.contains(&expected_alias),
        "binder-equivalent Named->Lit fan-in must alias t{fused_id} onto t{a_id}->data; got:\n{c}"
    );

    let expected_out = format!("float* __out_{fused_id} = t{fused_id}->data;");
    assert!(
        c.contains(&expected_out),
        "expected non-restrict __out_{fused_id} for binder-equivalent in-place; got:\n{c}"
    );
}

/// Positive — same Named binder name on both sides with matching known
/// size: this is the canonical binder-equivalent case and must alias.
#[test]
fn fan_in_same_named_binder_aliases_reusable_input() {
    let (dag, fused, a) = fan_in_dag(
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
    );
    let c = CEmitter::emit_dag(&dag, "test_fan_in_same_named_binder").unwrap();
    let fused_id = fused.0;
    let a_id = a.0;

    let expected_alias =
        format!("t{fused_id} = chelis_alloc_view(1, (int[]){{ 4 }}, CHELIS_F32, t{a_id}->data);");
    assert!(
        c.contains(&expected_alias),
        "same-binder Named fan-in must alias t{fused_id} onto t{a_id}->data; got:\n{c}"
    );
}

/// Positive — same Named binder name with unsized binder on the reusable
/// side: a `Named("seq", None)` (symbolic size, unresolved) must still be
/// binder-equivalent to `Named("seq", Some(4))` provided the binder
/// *name* matches (the alias proof relies on the binder identity, not
/// the resolved size that was concretized later).
#[test]
fn fan_in_named_binder_with_unknown_size_aliases() {
    let (dag, fused, a) = fan_in_dag(
        vec_named_unsized_f32("seq"),
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
    );
    let c = CEmitter::emit_dag(&dag, "test_fan_in_named_unsized").unwrap();
    let fused_id = fused.0;
    let a_id = a.0;

    // The FusedElem's output_type is `Named("seq", Some(4))`, so the
    // shape array lowers to `(int[]){ 4 }` from the FusedElem side.
    let expected_alias =
        format!("t{fused_id} = chelis_alloc_view(1, (int[]){{ 4 }}, CHELIS_F32, t{a_id}->data);");
    assert!(
        c.contains(&expected_alias),
        "binder-equivalent Named(unsized)->Named(sized) fan-in must alias t{fused_id} onto t{a_id}->data; got:\n{c}"
    );
}

/// Negative — different binder names: the alias proof does NOT consider
/// `Named("batch", Some(4))` equivalent to `Named("seq", Some(4))`, even
/// though both resolve to size 4. The extension must not aliasing-rename
/// distinct binder names.
#[test]
fn fan_in_different_named_binders_does_not_alias() {
    let (dag, fused, a) = fan_in_dag(
        vec_named_f32("batch", 4),
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 4),
    );
    let c = CEmitter::emit_dag(&dag, "test_fan_in_different_binders").unwrap();
    let fused_id = fused.0;
    let a_id = a.0;

    // No in-place alias: the FusedElem output must be slot-backed, not
    // view-aliased to the reusable input. Shape lowers to `(int[]){ 4 }`
    // because every Named dim has a known size of 4.
    let forbidden_alias =
        format!("t{fused_id} = chelis_alloc_view(1, (int[]){{ 4 }}, CHELIS_F32, t{a_id}->data);");
    assert!(
        !c.contains(&forbidden_alias),
        "different-binder fan-in must NOT alias t{fused_id} onto t{a_id}->data; got:\n{c}"
    );

    // The standard slot-backed wrapper must still be present.
    let expected_slot = format!("chelis_tensor *t{fused_id} = chelis_alloc_view");
    assert!(
        c.contains(&expected_slot),
        "expected slot-backed t{fused_id} wrapper in fall-back path; got:\n{c}"
    );

    // Fall-back path: output is restrict-qualified (no overlap with any input).
    let expected_out = format!("float* restrict __out_{fused_id} = t{fused_id}->data;");
    assert!(
        c.contains(&expected_out),
        "expected restrict __out_{fused_id} pointer in fall-back path; got:\n{c}"
    );
}

/// Negative — different concrete sizes on the same binder name: a
/// `Named("seq", Some(4))` is NOT binder-equivalent to
/// `Named("seq", Some(8))`. Same binder *name* alone is not enough when
/// both sides carry a known and conflicting size.
#[test]
fn fan_in_same_binder_different_known_size_does_not_alias() {
    let (dag, fused, a) = fan_in_dag(
        vec_named_f32("seq", 4),
        vec_named_f32("seq", 8),
        vec_named_f32("seq", 8),
        vec_named_f32("seq", 8),
    );
    let c = CEmitter::emit_dag(&dag, "test_fan_in_size_mismatch").unwrap();
    let fused_id = fused.0;
    let a_id = a.0;

    // FusedElem's output is `Named("seq", Some(8))` so it would lower
    // to `(int[]){ 8 }` if the in-place wrapper fired.
    let forbidden_alias =
        format!("t{fused_id} = chelis_alloc_view(1, (int[]){{ 8 }}, CHELIS_F32, t{a_id}->data);");
    assert!(
        !c.contains(&forbidden_alias),
        "binder with conflicting known sizes must NOT alias; got:\n{c}"
    );
}

/// Negative — different precision: F32 vs F64 is never binder-equivalent
/// even when dim shapes match. dtype is part of the alias proof.
#[test]
fn fan_in_different_precision_does_not_alias() {
    let (dag, fused, a) = fan_in_dag(
        vec_lit_f64(4),
        vec_lit_f32(4),
        vec_lit_f32(4),
        vec_lit_f32(4),
    );
    let c = CEmitter::emit_dag(&dag, "test_fan_in_different_precision").unwrap();
    let fused_id = fused.0;
    let a_id = a.0;

    let forbidden_alias =
        format!("t{fused_id} = chelis_alloc_view(1, (int[]){{ 4 }}, CHELIS_F32, t{a_id}->data);");
    assert!(
        !c.contains(&forbidden_alias),
        "different-precision fan-in must NOT alias t{fused_id} onto t{a_id}->data; got:\n{c}"
    );
}

/// Negative — multi-consumer reusable input: even when shapes are
/// binder-equivalent, an input with more than one consumer must NOT be
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
    let a = dag.add_node(RiscOp::Realize, vec![x_a], vec_named_f32("seq", 4), None);
    let x_b = dag.add_node(
        RiscOp::Load { name: "x_b".into() },
        vec![],
        vec_named_f32("seq", 4),
        None,
    );
    let b = dag.add_node(RiscOp::Realize, vec![x_b], vec_named_f32("seq", 4), None);

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

    // Second consumer of `a` forces fall-back even with binder-equivalent shapes.
    let other = dag.add_node(RiscOp::Neg, vec![a], vec_named_f32("seq", 4), None);
    dag.add_root(fused);
    dag.add_root(other);

    let c = CEmitter::emit_dag(&dag, "test_fan_in_multi_consumer").unwrap();
    let fused_id = fused.0;
    let a_id = a.0;

    let forbidden_alias =
        format!("t{fused_id} = chelis_alloc_view(1, (int[]){{ 4 }}, CHELIS_F32, t{a_id}->data);");
    assert!(
        !c.contains(&forbidden_alias),
        "multi-consumer reusable input must NOT alias even with binder-equivalent shapes; got:\n{c}"
    );
}

/// Negative — different rank: a rank-1 reusable input and a rank-2
/// FusedElem output are never binder-equivalent, regardless of total
/// element count.
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
    let c = CEmitter::emit_dag(&dag, "test_fan_in_different_rank").unwrap();
    let fused_id = fused.0;
    let a_id = a.0;

    // Forbidden aliases for both possible output shape literals.
    let forbidden_r2 = format!(
        "t{fused_id} = chelis_alloc_view(2, (int[]){{ 2, 4 }}, CHELIS_F32, t{a_id}->data);"
    );
    assert!(
        !c.contains(&forbidden_r2),
        "different-rank fan-in must NOT alias t{fused_id} onto t{a_id}->data; got:\n{c}"
    );
}
