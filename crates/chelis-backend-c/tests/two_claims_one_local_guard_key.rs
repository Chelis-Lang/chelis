//! chelis#1277 B2r: two DIFFERENT local extent claims on one axis are two
//! obligations, and the C emitter emits both.
//!
//! The two producers of `local_dim_guard_sites` key their sites the same way,
//! on the axis whose extent the guard reads. An equality class keys on the
//! member's own axis; a unit-extent claim keys on the `expand` operand's axis.
//! A `reshape` with a computed target, claimed by a signature and then
//! broadcast by a same-rank `expand`, is both at once, so the class's binder
//! and the unit claim's literal 1 land on the reshape's axis 0.
//!
//! Before this change the emitter refused that program with
//! `Unsupported::Construct("two disagreeing local extent guards on one axis")`,
//! argued from a pair believed impossible. This file is the construction, so
//! the refusal was refusing a program that checks clean. One comparison cannot
//! discharge two claims; the answer is two comparisons.
//!
//! EVIDENTIARY STATUS: regression test. On the tree before this change,
//! `emit_dag` returns `Err(Unsupported)` and both assertions below are
//! unreachable.

mod support;

use chelis_ir::dag::{Dag, DimInfo, RiscOp, RtDim, TensorType};
use chelis_types::types::Prim;
use support::emit_dag;

fn ty(dims: Vec<DimInfo>) -> TensorType {
    TensorType {
        dims,
        precision: Prim::F32,
    }
}

/// The extent-carrying scalar's type: C4 requires a node-valued reshape bound
/// to be int64.
fn i64_scalar() -> TensorType {
    TensorType {
        dims: vec![],
        precision: Prim::Int64,
    }
}

fn named(name: &str) -> DimInfo {
    DimInfo::Named(name.to_string(), None)
}

/// The reshape whose axis carries both claims, and the guards it owes.
#[test]
fn two_disagreeing_claims_on_one_axis_emit_two_guards() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        ty(vec![named("n")]),
        None,
    );
    // A rank-0 scalar this function COMPUTES, so the reshape target's
    // `ScalarInput` does not resolve through a `Load` and the class is Local.
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        i64_scalar(),
        None,
    );
    let size = dag.add_node(RiscOp::Mul, vec![a, a], i64_scalar(), None);
    // Claimed `n` by its output type, and sized from the computed scalar: an
    // equality-class member on axis 0, keyed (reshape, 0).
    let reshaped = dag.add_node(
        RiscOp::Reshape {
            new_shape: vec![RtDim::Node(1)],
        },
        vec![x, size],
        ty(vec![named("n")]),
        None,
    );
    // The same-rank broadcast, whose unit-extent claim is about the OPERAND's
    // axis 0: the same key.
    let expanded = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Lit(3),
        },
        vec![reshaped],
        ty(vec![DimInfo::Lit(3)]),
        None,
    );
    let _ = expanded;

    // The derivation's own view of the collision, so a failure says which half
    // moved.
    let sites: Vec<_> = chelis_ir::axis_sources::local_dim_guard_sites(&dag)
        .into_iter()
        .filter(|((node, axis), _)| *node == reshaped.0 && *axis == 0)
        .map(|(_, claim)| (claim.claim, claim.op))
        .collect();
    assert_eq!(
        sites,
        vec![("n".to_string(), "reshape"), ("1".to_string(), "expand"),],
        "the class binder and the unit claim both key on the reshape's axis 0",
    );

    let emitted = emit_dag(&dag, "two_claims")
        .expect("two disagreeing claims are two obligations, not an unsupported construct");
    assert!(
        emitted.contains("numeric trap: domain in reshape at int64"),
        "the class's guard is emitted:\n{emitted}"
    );
    assert!(
        emitted.contains("numeric trap: domain in expand at int64"),
        "and so is the unit claim's, at the same axis:\n{emitted}"
    );
}

/// The negative twin: EQUAL claims on one key still coalesce to one guard, so
/// the change above did not turn coalescing off.
///
/// The reshape's output carries a name of its own here, with no second witness,
/// so no equality class forms and the only sites on the key are the two
/// identical unit claims. That is what makes this a PURE equal pair: under the
/// refusal this change removes, this program was already accepted, and it stays
/// accepted with the same single guard.
///
/// EVIDENTIARY STATUS: disposition lock. It passes on both trees, which is the
/// point: it says the change altered only the disagreeing case.
#[test]
fn two_equal_claims_on_one_axis_emit_one_guard() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        ty(vec![named("n")]),
        None,
    );
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        i64_scalar(),
        None,
    );
    let size = dag.add_node(RiscOp::Mul, vec![a, a], i64_scalar(), None);
    // `m`, not `n`: a claim with a single witness is not a class (C2.4), so the
    // reshape's axis owes no class guard and the key carries unit claims only.
    let reshaped = dag.add_node(
        RiscOp::Reshape {
            new_shape: vec![RtDim::Node(1)],
        },
        vec![x, size],
        ty(vec![named("m")]),
        None,
    );
    // TWO same-rank broadcasts over the one operand axis: the identical unit
    // claim twice on one key.
    for extent in [3usize, 4usize] {
        dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: RtDim::Lit(extent),
            },
            vec![reshaped],
            ty(vec![DimInfo::Lit(extent)]),
            None,
        );
    }
    let on_key: Vec<_> = chelis_ir::axis_sources::local_dim_guard_sites(&dag)
        .into_iter()
        .filter(|((node, axis), _)| *node == reshaped.0 && *axis == 0)
        .map(|(_, claim)| claim.claim)
        .collect();
    assert_eq!(
        on_key,
        vec!["1".to_string(), "1".to_string()],
        "the derivation yields the identical claim once per expand, and nothing else",
    );

    let emitted = emit_dag(&dag, "two_equal_claims").expect("equal claims coalesce");
    assert_eq!(
        emitted
            .matches("numeric trap: domain in expand at int64")
            .count(),
        1,
        "but one comparison discharges both, so exactly one guard is emitted:\n{emitted}"
    );
}
