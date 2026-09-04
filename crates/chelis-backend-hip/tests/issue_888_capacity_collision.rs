//! Phase 0 planner-level witness for chelis#888, HIP lane.
//!
//! This is the twin of
//! `crates/chelis-backend-c/tests/issue_888_capacity_collision.rs`.
//! `crates/chelis-backend-hip/src/memory.rs` carries a verbatim copy of the C
//! backend's `capacity_fits` and `logical_elements`, so #888 is one defect
//! with two instances and a witness for only one of them understates the
//! class. Phase 1 has to repair both, and both witnesses have to invert
//! together.
//!
//! Every test in this file is a Phase 0 *characterization of a known-bad
//! behavior*. Each one asserts the wrong answer the HIP memory planner
//! produces today, so that Phase 1's exact `CapacityKey` cannot land
//! silently. When that repair lands these tests must **invert** (the
//! collision case must start allocating distinct slots) and move into the
//! Phase 1 positive controls. They must not be deleted.
//!
//! Mechanism, in one paragraph. `DimExprKey` folds concrete dimension factors
//! with `saturating_mul` (`chelis_ir::dag`), and saturation is not injective:
//! two products that both exceed `usize::MAX` clamp to the same
//! `usize::MAX`. `capacity_fits` in `crates/chelis-backend-hip/src/memory.rs`
//! falls back to raw key equality whenever either capacity is non-concrete,
//! which is exactly the case for any shape carrying a symbolic axis. So two
//! symbolic shapes whose true element counts differ by a factor of two
//! compare equal, and the slot-reuse search hands the larger value a buffer
//! sized for the smaller one.
//!
//! The HIP planner is not a byte-for-byte copy of the C one: it classifies
//! program inputs as `UniqueInput` slots rather than borrowed loads, and it
//! aliases repeated loads. Neither difference touches this DAG, which is a
//! const followed by two elementwise nodes, so the slot arithmetic below is
//! the C witness's arithmetic exercised through the HIP code path rather
//! than an assumption that the two files agree.

use chelis_unord::UnordSet;

use chelis_backend_hip::memory::{MemoryPlan, NodeMemoryKind};
use chelis_ir::dag::{Dag, DimExpr, DimExprKey, DimInfo, NodeId, RiscOp, TensorType};
use chelis_types::types::Prim;
mod support;

/// Tail extents chosen so the *concrete* part of the shape product,
/// `2^40 * TAIL`, exceeds `usize::MAX` and therefore saturates.
const SMALL_TAIL: usize = 1 << 40;
const LARGE_TAIL: usize = 1 << 41;
const HEAD: usize = 1 << 40;

/// `tensor[n, 2^40, tail, f32]`: one symbolic axis, so `as_concrete` returns
/// `None` and `capacity_fits` is forced onto the key-equality branch.
fn saturating_tensor(tail: usize) -> TensorType {
    TensorType {
        dims: vec![
            DimInfo::Named("n".to_string(), None),
            DimInfo::Lit(HEAD),
            DimInfo::Lit(tail),
        ],
        precision: Prim::F32,
    }
}

/// `tensor[n, extent, f32]`: the same symbolic-capacity shape below the
/// saturation threshold, for the positive control.
fn representable_tensor(extent: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Named("n".to_string(), None), DimInfo::Lit(extent)],
        precision: Prim::F32,
    }
}

/// The exact product of a shape's concrete extents, in `u128` so the test's
/// own arithmetic is never the thing that saturates.
///
/// Both shapes under test carry the same single symbolic axis `n`, so `n`
/// factors out: comparing the concrete products compares the true element
/// counts for every positive `n`.
fn exact_concrete_elements(ty: &TensorType) -> u128 {
    ty.dims
        .iter()
        .map(|dim| match dim {
            DimInfo::Lit(n) | DimInfo::Named(_, Some(n)) => *n as u128,
            DimInfo::Named(_, None) => 1,
        })
        .product()
}

/// `memory::logical_elements` is private, and it folds the extents left to
/// right into a `DimExpr::Mul` chain. Mirror that shape so the expected
/// `SlotPlan::capacity_elems` below is an exact value, not a re-derivation.
fn logical_elements(ty: &TensorType) -> DimExpr {
    ty.dims
        .iter()
        .map(|dim| match dim {
            DimInfo::Lit(n) | DimInfo::Named(_, Some(n)) => DimExpr::Concrete(*n),
            DimInfo::Named(name, None) => DimExpr::Sym(name.clone()),
        })
        .reduce(|lhs, rhs| DimExpr::Mul(Box::new(lhs), Box::new(rhs)))
        .expect("shapes under test have at least one axis")
}

/// Build the three-node chain `a -> b -> c` with `c` as the single output.
///
/// The lifetimes are what make the reuse search reachable: `a` dies at `b`,
/// `b` is still live when `c` is born, so `c` is offered slot 0 (a's slot)
/// and rejected from slot 1 (b's), and only `capacity_fits` decides whether
/// it takes slot 0. The third `build` argument is HIP's reduction-inlined
/// set; this DAG inlines nothing.
fn chain_plan(
    a_type: TensorType,
    b_type: TensorType,
    c_type: TensorType,
) -> (MemoryPlan, NodeId, NodeId, NodeId) {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::synth_const(a_type.precision, 1.0),
        vec![],
        a_type,
        None,
    );
    let b = dag.add_node(RiscOp::Neg, vec![a], b_type, None);
    let c = dag.add_node(RiscOp::Neg, vec![b], c_type, None);
    dag.add_root(c);
    let verified = support::verified_dag(&dag);
    let plan = MemoryPlan::build(verified.emission(), &[c], &UnordSet::new());
    (plan, a, b, c)
}

/// PHASE 0 CHARACTERIZATION OF A KNOWN-BAD BEHAVIOR.
///
/// This asserts the wrong plan: a value needing `n * 2^81` elements is given
/// the slot allocated for `n * 2^80`. Phase 1's exact `CapacityKey` must make
/// this test fail, at which point it inverts to assert three distinct slots.
/// Do not delete it.
#[test]
fn phase0_hip_planner_reuses_a_slot_sized_for_half_the_requested_capacity() {
    let small = saturating_tensor(SMALL_TAIL);
    let large = saturating_tensor(LARGE_TAIL);

    // The premise: the two shapes really do denote different sizes. Stated in
    // u128 so this comparison is exact rather than another saturating fold.
    let small_elems = exact_concrete_elements(&small);
    let large_elems = exact_concrete_elements(&large);
    assert_eq!(
        large_elems,
        2 * small_elems,
        "the larger shape must denote exactly twice the elements of the smaller"
    );
    assert!(
        small_elems > u128::from(u64::MAX),
        "both concrete products must exceed usize::MAX for the fold to saturate"
    );

    let (plan, a, b, c) = chain_plan(small.clone(), small.clone(), large.clone());

    // The observable wrong outcome: two slots, and the large value `c` is
    // assigned to slot 0, the slot the small value `a` created.
    assert_eq!(
        plan.slots().len(),
        2,
        "the collision hides the third capacity, so only two slots are created"
    );
    assert_eq!(plan.node_kind(a), &NodeMemoryKind::SlotBacked { slot: 0 });
    assert_eq!(plan.node_kind(b), &NodeMemoryKind::SlotBacked { slot: 1 });
    assert_eq!(
        plan.node_kind(c),
        &NodeMemoryKind::SlotBacked { slot: 0 },
        "chelis#888: the larger value reuses the smaller value's slot"
    );

    // And slot 0 really is sized for the smaller shape, so the reuse is an
    // under-allocation by exactly a factor of two, not a harmless alias.
    let slot = plan.slot(0);
    assert_eq!(slot.first_owner, a);
    assert_eq!(
        slot.capacity_elems,
        logical_elements(&small),
        "slot 0's recorded capacity is the smaller shape's element count"
    );
    assert_ne!(
        logical_elements(&small),
        logical_elements(&large),
        "the two capacities are structurally distinct expressions"
    );

    // Lower-level companion: the exact false equality that produced the plan
    // above. Both capacities saturate to the same key, and neither is
    // concrete, so `capacity_fits` takes its key-equality branch.
    let small_capacity = logical_elements(&small);
    let large_capacity = logical_elements(&large);
    assert_eq!(small_capacity.as_concrete(), None);
    assert_eq!(large_capacity.as_concrete(), None);
    let saturated = DimExprKey::Mul(vec![
        DimExprKey::Concrete(usize::MAX),
        DimExprKey::Sym("n".to_string()),
    ]);
    assert_eq!(small_capacity.normalized_key(), saturated);
    assert_eq!(large_capacity.normalized_key(), saturated);
}

/// Negative parity for the case above, and the proof that saturation is the
/// cause rather than a planner that ignores capacity in general.
///
/// Identical DAG shape and identical lifetimes, with the same symbolic axis,
/// but concrete factors small enough that the fold does not saturate. Here
/// the planner is correct: the larger value refuses the smaller slot and
/// takes a third one. This test must keep passing through Phase 1.
#[test]
fn representable_symbolic_capacities_do_not_share_a_hip_slot() {
    let small = representable_tensor(4);
    let large = representable_tensor(8);

    let (plan, a, b, c) = chain_plan(small.clone(), small, large);

    assert_eq!(
        plan.slots().len(),
        3,
        "distinct representable capacities must not be merged"
    );
    assert_eq!(plan.node_kind(a), &NodeMemoryKind::SlotBacked { slot: 0 });
    assert_eq!(plan.node_kind(b), &NodeMemoryKind::SlotBacked { slot: 1 });
    assert_eq!(
        plan.node_kind(c),
        &NodeMemoryKind::SlotBacked { slot: 2 },
        "a larger symbolic capacity must get its own slot"
    );
}

/// The reuse the planner is supposed to perform still happens, so the two
/// tests above are not just observing a planner that never reuses anything.
///
/// Same chain, one capacity throughout: `c` legitimately inherits `a`'s slot
/// because the capacities are genuinely equal. This test must keep passing
/// through Phase 1.
#[test]
fn equal_symbolic_capacities_still_share_a_hip_slot() {
    let same = representable_tensor(4);

    let (plan, a, b, c) = chain_plan(same.clone(), same.clone(), same);

    assert_eq!(plan.slots().len(), 2);
    assert_eq!(plan.node_kind(a), &NodeMemoryKind::SlotBacked { slot: 0 });
    assert_eq!(plan.node_kind(b), &NodeMemoryKind::SlotBacked { slot: 1 });
    assert_eq!(
        plan.node_kind(c),
        &NodeMemoryKind::SlotBacked { slot: 0 },
        "an equal capacity must still reuse the dead slot"
    );
}
