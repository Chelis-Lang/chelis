//! Exact-capacity planner regressions for chelis#888, HIP lane.
//!
//! This is the twin of
//! `crates/chelis-backend-c/tests/issue_888_capacity_collision.rs`.
//! The retired C and HIP backends carried independent lossy capacity
//! predicates, so #888 was one class with two production paths. Both
//! adapters now consume the same exact `CapacityKey` storage plan.
//!
//! Mechanism, in one paragraph. `DimExprKey` folds concrete dimension factors
//! with `saturating_mul` (`chelis_ir::dag`), and saturation is not injective:
//! two products that both exceed `usize::MAX` clamp to the same
//! `usize::MAX`. The retired HIP planner compared that lossy key when a
//! symbolic capacity was not concrete, allowing distinct capacities to share
//! one physical slot. The adapter now exposes only the shared exact plan.
//!
//! HIP still maps entry inputs to device mirror slots and aliases repeated
//! loads, but neither target mechanic is capacity authority.

use chelis_backend_hip::memory::{MemoryPlan, NodeMemoryKind};
use chelis_ir::dag::{Dag, DimExpr, DimExprKey, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir::ownership::plan_hip_storage;
use chelis_types::types::Prim;
mod support;

/// Tail extents chosen so the *concrete* part of the shape product,
/// `2^40 * TAIL`, exceeds `usize::MAX` and therefore saturates.
const SMALL_TAIL: usize = 1 << 40;
const LARGE_TAIL: usize = 1 << 41;
const HEAD: usize = 1 << 40;

/// `tensor[n, 2^40, tail, f32]`: one symbolic axis, so the retired lossy
/// capacity fold took its key-equality branch.
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

/// Mirror the adapter's renderable allocation expression so the expected
/// `SlotPlan::capacity_elems` is exact.
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
/// and rejected from slot 1 (b's). Exact capacity decides whether it may
/// take slot 0.
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
    let shared = plan_hip_storage(verified).expect("exact HIP storage plan");
    let plan = MemoryPlan::from_shared(&shared);
    (plan, a, b, c)
}

/// A value needing `n * 2^81` elements must not receive the slot allocated
/// for `n * 2^80`, even though the retired normalized keys collide.
#[test]
fn exact_hip_planner_separates_saturated_legacy_key_collision() {
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

    // The repaired outcome: all three concurrently incompatible capacities
    // have distinct slots.
    assert_eq!(
        plan.slots().len(),
        3,
        "the exact key must preserve the third distinct capacity"
    );
    assert_eq!(plan.node_kind(a), &NodeMemoryKind::SlotBacked { slot: 0 });
    assert_eq!(plan.node_kind(b), &NodeMemoryKind::SlotBacked { slot: 1 });
    assert_eq!(
        plan.node_kind(c),
        &NodeMemoryKind::SlotBacked { slot: 2 },
        "the larger value must not reuse the smaller value's slot"
    );

    // Slot 0 remains sized for the smaller shape; assigning c to it would be
    // an under-allocation by exactly a factor of two.
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

    // Historical premise: the retired lossy keys collide. This equality is
    // evidence for the regression and is no longer storage authority.
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
/// takes a third one.
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
/// because the capacities are genuinely equal.
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
