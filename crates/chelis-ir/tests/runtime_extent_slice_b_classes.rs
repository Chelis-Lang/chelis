//! Derived runtime-dimension equality classes (chelis#1277 Slice B, b2.1).
//!
//! Every assertion below is derived from a normative sentence, not from the
//! implementation it constrains. The sentence is named on each test. The two
//! controlling documents are `spec/04-type-system.md` section 4.7 (the runtime
//! extent guard paragraph, which decides guard placement and ordering) and
//! `spec/05-risc-primitives.md` section 2.4.1 (the `RtDim` carrier set), with
//! `spec/design/runtime_extents.md` C2.4 recording how the four ordering rules
//! follow from them.
//!
//! The defect class these lock out is recovery of a runtime extent's identity
//! by STRING NAME. The legacy grouping these replaced, deleted in chelis#665,
//! put occurrences in a
//! `BTreeMap<String, _>`, so two axes that happen to share a spelling are
//! identified with no guard between them, and two axes that are genuinely the
//! same extent are given different synthesized spellings and guarded twice or
//! not at all. The replacement groups by the STAMPED CLAIM and locates each
//! member's value through `output_axis_sources`, so a name can no longer
//! locate anything.
//!
//! Ordering note, because it is the rule most easily satisfied by accident:
//! section 4.7 requires interface guards "at entry, in declared signature
//! order", and gives "an entry that declares no signature ... its ABI
//! input-slot order" instead, closing with "Whatever rule assigns the slots,
//! the guard order follows the assigned slots, and never a separate traversal
//! by binding name, hash iteration, or node identity." In the IR both branches
//! collapse to one key: `lower_fn` registers parameter `Load`s in declared
//! order, and helpers extracted from declared functions retain that order.
//! Only signatureless subexpressions assign name-sorted slots for determinism.
//! The ABI input slot of the declaring `Load` is therefore the assigned slot in both
//! cases.
//!
//! That is the sentence's own conclusion and not an inference from the two
//! lowering paths, so it must not be "simplified" back to name order by a
//! reader who sees only the declared-signature branch. The closing clause
//! also names the failure mode by hand: a `BTreeMap<String, _>` grouping is a
//! traversal by binding name, which is exactly what the legacy grouping did.
//! Several tests below discriminate the slot key from name order.

use chelis_ir::axis_sources::{
    AxisSource, DimClaim, GuardPlacement, RuntimeDimClass, derive_runtime_dim_classes,
};
use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, RtAxis, RtDim, TensorType};
use chelis_types::types::Prim;

fn ty(dims: Vec<DimInfo>, precision: Prim) -> TensorType {
    TensorType { dims, precision }
}

fn named(name: &str) -> DimInfo {
    DimInfo::Named(name.into(), None)
}

fn f32_load(dag: &mut Dag, name: &str, dims: Vec<DimInfo>) -> NodeId {
    dag.add_node(
        RiscOp::Load { name: name.into() },
        vec![],
        ty(dims, Prim::F32),
        None,
    )
}

/// The claims of `classes`, in the order derivation produced them.
fn claims(classes: &[RuntimeDimClass]) -> Vec<DimClaim> {
    classes.iter().map(|class| class.claim.clone()).collect()
}

/// The `(node, axis)` members of the class whose claim is `claim`, in member
/// order. Panics when no such class exists, so a test that means "this class
/// has these members" cannot silently degrade into "there is no such class".
fn members_of(classes: &[RuntimeDimClass], claim: DimClaim) -> Vec<(NodeId, usize)> {
    let class = classes
        .iter()
        .find(|class| class.claim == claim)
        .unwrap_or_else(|| panic!("no class for claim {claim:?} in {:?}", claims(classes)));
    class
        .members
        .iter()
        .map(|member| (member.node, member.axis))
        .collect()
}

fn class_for(classes: &[RuntimeDimClass], claim: DimClaim) -> &RuntimeDimClass {
    classes
        .iter()
        .find(|class| class.claim == claim)
        .unwrap_or_else(|| panic!("no class for claim {claim:?} in {:?}", claims(classes)))
}

// ---------------------------------------------------------------------------
// C2.4 rule 1: canonical and member order.
//
// "the canonical member is the signature-first witness, so interface members
// order by declared signature index" (design C2.4), which realizes section
// 4.7's "in declared signature order" / "its ABI input-slot order" pair.
// ---------------------------------------------------------------------------

/// Two `Load`s declare the same name. The canonical member is the one in the
/// earlier ABI input slot, which here is the earlier-registered `Load`.
#[test]
fn interface_members_order_by_abi_input_slot() {
    let mut dag = Dag::new();
    let x = f32_load(&mut dag, "x", vec![named("n")]);
    let y = f32_load(&mut dag, "y", vec![named("n")]);
    let classes = derive_runtime_dim_classes(&dag);
    assert_eq!(
        members_of(&classes, DimClaim::Name("n".into())),
        vec![(x, 0), (y, 0)],
        "the signature-first witness is canonical",
    );
}

/// The discriminating twin of the test above: registering the same two `Load`s
/// in the opposite order moves the canonical member with the slot, because the
/// key is the assigned slot. Name order would report `x` both times, so a
/// by-name implementation passes the first test and fails this one. That is
/// exactly the `BTreeMap<String, _>` ordering the legacy grouping used.
#[test]
fn interface_canonical_follows_the_slot_and_not_the_name() {
    let mut dag = Dag::new();
    let y = f32_load(&mut dag, "y", vec![named("n")]);
    let x = f32_load(&mut dag, "x", vec![named("n")]);
    let classes = derive_runtime_dim_classes(&dag);
    assert_eq!(
        members_of(&classes, DimClaim::Name("n".into())),
        vec![(y, 0), (x, 0)],
        "`y` occupies slot 0 here, so `y` is canonical despite sorting after `x`",
    );
}

/// "a class with no interface member takes its canonical member by node
/// position" (design C2.4 rule 1).
#[test]
fn an_all_local_class_takes_its_canonical_by_node_position() {
    let mut dag = Dag::new();
    let base = f32_load(&mut dag, "b", vec![]);
    let size = f32_load(&mut dag, "k", vec![]);
    let first = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Node(1),
        },
        vec![base, size],
        ty(vec![named("d")], Prim::F32),
        None,
    );
    let second = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Node(1),
        },
        vec![base, size],
        ty(vec![named("d")], Prim::F32),
        None,
    );
    let classes = derive_runtime_dim_classes(&dag);
    assert_eq!(
        members_of(&classes, DimClaim::Name("d".into())),
        vec![(first, 0), (second, 0)],
        "no `Load` declares `d`, so node position orders the class",
    );
}

/// "which is also how local members order inside a class that has both"
/// (design C2.4 rule 1). The interface witness is canonical even when a local
/// member has the lower node id, and the local members keep node order behind
/// it.
#[test]
fn a_mixed_class_puts_every_interface_member_before_its_local_members() {
    let mut dag = Dag::new();
    let base = f32_load(&mut dag, "b", vec![]);
    let size = f32_load(&mut dag, "k", vec![]);
    let expanded = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Node(1),
        },
        vec![base, size],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    let declaring = f32_load(&mut dag, "x", vec![named("n")]);
    let classes = derive_runtime_dim_classes(&dag);
    assert_eq!(
        members_of(&classes, DimClaim::Name("n".into())),
        vec![(declaring, 0), (expanded, 0)],
        "the interface witness is canonical even with the higher node id",
    );
}

/// "Nothing orders by hash iteration or display name" (design C2.4 rule 1).
/// Two classes whose claim names sort opposite to their slot order: the class
/// list follows the slots.
#[test]
fn the_class_list_order_is_not_display_name_order() {
    let mut dag = Dag::new();
    let z0 = f32_load(&mut dag, "z0", vec![named("z")]);
    let z1 = f32_load(&mut dag, "z1", vec![named("z")]);
    let a0 = f32_load(&mut dag, "a0", vec![named("a")]);
    let a1 = f32_load(&mut dag, "a1", vec![named("a")]);
    let classes = derive_runtime_dim_classes(&dag);
    assert_eq!(
        claims(&classes),
        vec![DimClaim::Name("z".into()), DimClaim::Name("a".into())],
        "`z` occupies the earlier slot, so its entry guard runs first",
    );
    assert_eq!(
        members_of(&classes, DimClaim::Name("z".into())),
        vec![(z0, 0), (z1, 0)]
    );
    assert_eq!(
        members_of(&classes, DimClaim::Name("a".into())),
        vec![(a0, 0), (a1, 0)]
    );
}

// ---------------------------------------------------------------------------
// C2.4 rule 2: no guard is ever discharged.
// ---------------------------------------------------------------------------

/// "an all-interface class runs its guard at entry regardless of data use"
/// (design C2.4 rule 2). Neither `Load` feeds any computation here.
#[test]
fn an_all_interface_class_keeps_both_members_with_no_data_use() {
    let mut dag = Dag::new();
    let x = f32_load(&mut dag, "x", vec![named("n")]);
    let y = f32_load(&mut dag, "y", vec![named("n")]);
    let classes = derive_runtime_dim_classes(&dag);
    assert_eq!(
        members_of(&classes, DimClaim::Name("n".into())),
        vec![(x, 0), (y, 0)],
    );
}

/// A member's source is read from `output_axis_sources`, never from a string
/// search, so the declaring `Load` is named by exact `NodeId`. Two `Load`s
/// carrying one name can no longer be confused for one another.
#[test]
fn a_member_names_its_declaring_load_by_node_id() {
    let mut dag = Dag::new();
    let x = f32_load(&mut dag, "x", vec![named("n")]);
    let y = f32_load(&mut dag, "y", vec![named("n")]);
    let classes = derive_runtime_dim_classes(&dag);
    let class = class_for(&classes, DimClaim::Name("n".into()));
    assert_eq!(
        class.members[0].source,
        AxisSource::ExternalAxis { load: x, axis: 0 },
    );
    assert_eq!(
        class.members[1].source,
        AxisSource::ExternalAxis { load: y, axis: 0 },
    );
}

// ---------------------------------------------------------------------------
// C2.4 rule 3: two classes may share a node; one member per `(node, axis)`.
// ---------------------------------------------------------------------------

/// "two classes may share a node, each with its own guard" (design C2.4
/// rule 3).
///
/// `Reshape` sets every target axis, so one node carries a member of two
/// different classes. Each claim needs a second witness of its own, because
/// C2.4 makes a class out of a claim "the checker attached to more than one
/// witness" and a lone witness has nothing to disagree with.
#[test]
fn two_classes_sharing_one_node_each_keep_their_own_member() {
    let mut dag = Dag::new();
    let x = f32_load(&mut dag, "x", vec![named("n")]);
    let y = f32_load(&mut dag, "y", vec![named("m")]);
    let rows = f32_load(&mut dag, "r", vec![]);
    let cols = f32_load(&mut dag, "c", vec![]);
    let reshaped = dag.add_node(
        RiscOp::Reshape {
            new_shape: vec![RtDim::Node(1), RtDim::Node(2)],
        },
        vec![x, rows, cols],
        ty(vec![named("n"), named("m")], Prim::F32),
        None,
    );
    let classes = derive_runtime_dim_classes(&dag);
    assert_eq!(
        members_of(&classes, DimClaim::Name("n".into())),
        vec![(x, 0), (reshaped, 0)],
    );
    assert_eq!(
        members_of(&classes, DimClaim::Name("m".into())),
        vec![(y, 0), (reshaped, 1)],
    );
}

/// "derivation yields one member per `(node, axis)`, so `splice_dag` mapping
/// both parameters of `f(n, n)` to one `NodeId` produces one member" (design
/// C2.4 rule 3). Modelled directly: two `RtDim` slots of one node reference
/// the same producer.
#[test]
fn one_node_axis_yields_exactly_one_member_under_the_f_of_n_n_splice() {
    let mut dag = Dag::new();
    let base = f32_load(&mut dag, "b", vec![]);
    let shared = f32_load(&mut dag, "k", vec![]);
    let declaring = f32_load(&mut dag, "x", vec![named("n")]);
    // Both bound slots are the same node, the shape `splice_dag` produces for
    // `f(n, n)`.
    let expanded = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Node(1),
        },
        vec![base, shared, shared],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    let classes = derive_runtime_dim_classes(&dag);
    assert_eq!(
        members_of(&classes, DimClaim::Name("n".into())),
        vec![(declaring, 0), (expanded, 0)],
        "one member for the set axis, not one per referencing slot",
    );
}

// ---------------------------------------------------------------------------
// C2.4 rule 4: a node a member or an `RtDim` slot references survives.
// ---------------------------------------------------------------------------

/// The positive form: the referenced producer is present and the member reads
/// it as a scalar input.
#[test]
fn a_member_referencing_an_rtdim_slot_reads_that_slot() {
    let mut dag = Dag::new();
    let base = f32_load(&mut dag, "b", vec![]);
    let size = f32_load(&mut dag, "k", vec![]);
    let declaring = f32_load(&mut dag, "x", vec![named("n")]);
    let expanded = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Node(1),
        },
        vec![base, size],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    let classes = derive_runtime_dim_classes(&dag);
    let class = class_for(&classes, DimClaim::Name("n".into()));
    assert_eq!(class.members[0].node, declaring);
    assert_eq!(
        class.members[1].source,
        AxisSource::ScalarInput { input: 1 },
    );
    assert_eq!(class.members[1].node, expanded);
}

/// The negative twin. A slot index past the node's inputs is malformed IR.
///
/// C2.4 rule 4 puts slot validation on C2.1, not on this derivation: "a node
/// an `RtDim::Node` slot or a member references must survive as a node, which
/// C2.1's slot validation already enforces". So the contract here is that
/// derivation does not PANIC, and that the malformed node is caught by
/// [`check_axis_sources`] with the registered typed receipt rather than by an
/// internal compiler error, which is C4.3.
///
/// An earlier draft of this test additionally required the derivation to drop
/// the member. That over-specified the rule: it would have moved slot
/// validation into a second place, where it could disagree with the first.
#[test]
fn a_member_whose_rtdim_slot_is_absent_is_a_typed_receipt_not_a_panic() {
    let mut dag = Dag::new();
    let base = f32_load(&mut dag, "b", vec![]);
    let declaring = f32_load(&mut dag, "x", vec![named("n")]);
    let expanded = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            // Slot 1 does not exist: `inputs` has only the tensor operand.
            size: RtDim::Node(1),
        },
        vec![base],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    // The declaring `Load` is joined into the result rather than left
    // dangling, so both witnesses share one scope. C2.4 scopes a claim by the
    // results it reaches, and a witness no result reaches forms no class -
    // section 4.7's "regardless of data use" case, which this derivation
    // cannot honour and which C2.4 records as a residual owned by B2b. The
    // `Expand`'s inputs are untouched, so the absent bound slot this row
    // exists for is unchanged.
    let joined = dag.add_node(
        RiscOp::Add,
        vec![expanded, declaring],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    dag.add_root(joined);

    // Derivation is total on this graph rather than panicking.
    let classes = derive_runtime_dim_classes(&dag);
    assert!(
        members_of(&classes, DimClaim::Name("n".into())).contains(&(declaring, 0)),
        "the declaring Load is still a member",
    );

    // And the malformed slot is reported where the rule puts it.
    let receipt = chelis_ir::axis_sources::check_axis_sources(
        &dag,
        chelis_types::unsupported::Stage::Runtime,
    )
    .expect_err("an absent bound slot is a checked cardinality failure");
    let text = receipt.to_string();
    assert!(
        text.contains("reads input slot 1") && text.contains("has 1 input(s)"),
        "the receipt names the absent slot: {text}",
    );
    assert!(
        text.contains("chelis#1277"),
        "the receipt cites its owning issue: {text}",
    );
}

// ---------------------------------------------------------------------------
// Pass-through versus member.
//
// "Only an output axis that C4.2 maps to an unchanged input axis is
// pass-through and not a member" (design C2.4), realizing section 4.7's
// "symbolic-dim pass-through is identity-only: a `stride` axis with literal
// step 1 and a `pad` axis with zero padding keep the input's symbolic dim;
// every other movement axis ... types a fresh `(d-name {} *)`".
// ---------------------------------------------------------------------------

/// VACUOUS AT b2.1, and it must not be read as coverage on its own:
/// this asserts an EMPTY result, so it passes against the b2.1 shell's
/// empty derivation as readily as against a correct one. Its
/// discrimination is carried entirely by its twin,
/// `a_non_identity_stride_axis_under_the_inputs_name_is_a_member`,
/// which fails at b2.1 and must be green afterwards while this row stays
/// green. Deleting the twin leaves a test that proves nothing.
#[test]
fn an_identity_stride_axis_is_pass_through_and_not_a_member() {
    let mut dag = Dag::new();
    let x = f32_load(&mut dag, "x", vec![named("n")]);
    let strided = dag.add_node(
        RiscOp::Stride {
            strides: vec![RtDim::Lit(1)],
        },
        vec![x],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    let classes = derive_runtime_dim_classes(&dag);
    assert!(
        classes.is_empty(),
        "a literal step of one keeps the input's dim, so nothing is claimed twice: {:?}",
        claims(&classes),
    );
    let _ = strided;
}

/// VACUOUS AT b2.1, and it must not be read as coverage on its own:
/// this asserts an EMPTY result, so it passes against the b2.1 shell's
/// empty derivation as readily as against a correct one. Its
/// discrimination is carried entirely by its twin,
/// `a_full_axis_symbolic_shrink_mints_a_fresh_member`,
/// which fails at b2.1 and must be green afterwards while this row stays
/// green. Deleting the twin leaves a test that proves nothing.
#[test]
fn a_zero_pad_axis_is_pass_through_and_not_a_member() {
    let mut dag = Dag::new();
    let x = f32_load(&mut dag, "x", vec![named("n")]);
    let padded = dag.add_node(
        RiscOp::zero_pad(Prim::F32, vec![(RtDim::Lit(0), RtDim::Lit(0))]),
        vec![x],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    let classes = derive_runtime_dim_classes(&dag);
    assert!(
        classes.is_empty(),
        "zero padding keeps the input's dim: {:?}",
        claims(&classes),
    );
    let _ = padded;
}

/// The discriminating twin of the two above: a non-identity step produces a
/// fresh extent, so restating the input's name over it is a claim that must be
/// guarded. Without this test, an implementation that treats every movement
/// axis as pass-through passes both identity tests.
#[test]
fn a_non_identity_stride_axis_under_the_inputs_name_is_a_member() {
    let mut dag = Dag::new();
    let x = f32_load(&mut dag, "x", vec![named("n")]);
    let strided = dag.add_node(
        RiscOp::Stride {
            strides: vec![RtDim::Lit(2)],
        },
        vec![x],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    let classes = derive_runtime_dim_classes(&dag);
    assert_eq!(
        members_of(&classes, DimClaim::Name("n".into())),
        vec![(x, 0), (strided, 0)],
        "`stride(x, 2)`'s extent is ceil(n/2), not n, so the claim needs a guard",
    );
}

/// "a full-axis symbolic `shrink` `(0, ToEnd)` whose extent equals the input's
/// but whose identity and guard are fresh" (design C5 property 1), and section
/// 4.7's "`shrink` has no checker-detectable identity form for a symbolic
/// axis ... so its symbolic axes always mint fresh extents".
#[test]
fn a_full_axis_symbolic_shrink_mints_a_fresh_member() {
    let mut dag = Dag::new();
    let x = f32_load(&mut dag, "x", vec![named("n")]);
    let shrunk = dag.add_node(
        RiscOp::Shrink {
            bounds: vec![(RtDim::Lit(0), RtDim::ToEnd)],
        },
        vec![x],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    let classes = derive_runtime_dim_classes(&dag);
    assert_eq!(
        members_of(&classes, DimClaim::Name("n".into())),
        vec![(x, 0), (shrunk, 0)],
        "forwarding the input's class instead of minting a member is the C5 \
         property 3 failure",
    );
}

/// chelis#1376: the axis the operation SETS is a member whatever slot its
/// `InputAxis` names, so a same-tensor read under a FOREIGN claim is guarded.
/// `expand(x, 1, shape(x, 0))` declared `-> tensor[n, m, f32]` claims `m` for
/// an axis whose runtime extent is `shape(x, 0)`.
#[test]
fn a_same_tensor_read_under_a_foreign_claim_is_a_member() {
    let mut dag = Dag::new();
    let x = f32_load(&mut dag, "x", vec![named("n")]);
    let y = f32_load(&mut dag, "y", vec![named("m")]);
    let expanded = dag.add_node(
        RiscOp::Expand {
            axis: 1,
            size: RtDim::InputAxis {
                tensor: 0,
                axis: RtAxis::Lit(0),
            },
        },
        vec![x],
        ty(vec![named("n"), named("m")], Prim::F32),
        None,
    );
    let classes = derive_runtime_dim_classes(&dag);
    assert_eq!(
        members_of(&classes, DimClaim::Name("m".into())),
        vec![(y, 0), (expanded, 1)],
        "the set axis reads x's axis 0 but claims m, so it owes a guard",
    );
}

/// The positive twin: a same-tensor read that KEEPS the source dimension's
/// name is a proved identity, "so a same-tensor read under a proved identity
/// costs no guard because C1.2's static proof leaves no claim" (design C2.4).
#[test]
fn a_same_tensor_read_under_a_proved_identity_costs_no_guard() {
    let mut dag = Dag::new();
    let x = f32_load(&mut dag, "x", vec![named("n")]);
    let expanded = dag.add_node(
        RiscOp::Expand {
            axis: 1,
            size: RtDim::InputAxis {
                tensor: 0,
                axis: RtAxis::Lit(0),
            },
        },
        vec![x],
        ty(vec![named("n"), named("n")], Prim::F32),
        None,
    );
    let classes = derive_runtime_dim_classes(&dag);
    assert_eq!(
        members_of(&classes, DimClaim::Name("n".into())),
        vec![(x, 0), (expanded, 1)],
        "the set axis restates n over a read of n's own tensor",
    );
    let _ = expanded;
}

// ---------------------------------------------------------------------------
// Literal claims.
//
// "A literal claim is the class's canonical value itself. A literal claim on
// an anonymous runtime extent ... is therefore a class whose canonical value
// is the literal and whose one member is the scalar-sourced set axis"
// (design C2.4), realizing section 4.7.2's "When a declared or inferred result
// dimension claims a literal or named extent that is not statically proven
// equal to `size`, execution checks equality and traps `Domain` on mismatch."
// ---------------------------------------------------------------------------

/// chelis#1377: `-> tensor[4, f32]` over `expand(b, 0, shape(x, 0))`.
#[test]
fn a_literal_claim_over_a_read_is_a_one_member_class() {
    let mut dag = Dag::new();
    let base = f32_load(&mut dag, "b", vec![]);
    let x = f32_load(&mut dag, "x", vec![named("n")]);
    let expanded = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::InputAxis {
                tensor: 1,
                axis: RtAxis::Lit(0),
            },
        },
        vec![base, x],
        ty(vec![DimInfo::Lit(4)], Prim::F32),
        None,
    );
    let classes = derive_runtime_dim_classes(&dag);
    assert_eq!(
        members_of(&classes, DimClaim::Literal(4)),
        vec![(expanded, 0)],
        "the literal is the canonical value; the set axis is its one member",
    );
}

/// The discriminating twin: a literal claim over a LITERAL size is statically
/// proved, so section 4.7.2's "not statically proven equal" precondition fails
/// and no class exists. Without this, an implementation that mints a class for
/// every literal dim passes the test above and guards every static program.
/// VACUOUS AT b2.1, and it must not be read as coverage on its own:
/// this asserts an EMPTY result, so it passes against the b2.1 shell's
/// empty derivation as readily as against a correct one. Its
/// discrimination is carried entirely by its twin,
/// `a_literal_claim_over_a_read_is_a_one_member_class`,
/// which fails at b2.1 and must be green afterwards while this row stays
/// green. Deleting the twin leaves a test that proves nothing.
#[test]
fn a_literal_claim_matching_a_literal_size_yields_no_class() {
    let mut dag = Dag::new();
    let base = f32_load(&mut dag, "b", vec![]);
    let expanded = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Lit(4),
        },
        vec![base],
        ty(vec![DimInfo::Lit(4)], Prim::F32),
        None,
    );
    let classes = derive_runtime_dim_classes(&dag);
    assert!(
        classes.is_empty(),
        "a statically proved literal claim owes no guard: {:?}",
        claims(&classes),
    );
    let _ = expanded;
}

/// chelis#1374: "a cross-tensor folded read under a name is a class with the
/// declaring `Load` axis and the `InputAxis`-sourced set axis, exactly the two
/// guards `spec/04` section 4.7.2 and section 4.7.3 require" (design C2.4).
#[test]
fn a_cross_tensor_folded_read_under_a_name_has_exactly_two_members() {
    let mut dag = Dag::new();
    let base = f32_load(&mut dag, "b", vec![]);
    let x = f32_load(&mut dag, "x", vec![named("n")]);
    let y = f32_load(&mut dag, "y", vec![named("m")]);
    let expanded = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::InputAxis {
                tensor: 1,
                axis: RtAxis::Lit(0),
            },
        },
        vec![base, y],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    let classes = derive_runtime_dim_classes(&dag);
    assert_eq!(
        members_of(&classes, DimClaim::Name("n".into())),
        vec![(x, 0), (expanded, 0)],
        "the declaring Load and the set axis, and nothing else",
    );
    assert!(
        classes
            .iter()
            .all(|class| class.claim != DimClaim::Name("m".into())),
        "m has one witness, so it is not a class: {:?}",
        claims(&classes),
    );
}

// ---------------------------------------------------------------------------
// C1.3 placement: which guards run at entry and which at the introducing
// operation.
//
// Section 4.7: "A guard whose operands are all interface values (an input
// tensor's axis, a scalar parameter, or a literal) is evaluated at function
// entry ... A guard that compares a locally computed value ... takes the
// source position of the operation that introduces the guarded extent."
// ---------------------------------------------------------------------------

#[test]
fn an_all_interface_class_is_an_entry_guard() {
    let mut dag = Dag::new();
    f32_load(&mut dag, "x", vec![named("n")]);
    f32_load(&mut dag, "y", vec![named("n")]);
    let classes = derive_runtime_dim_classes(&dag);
    assert_eq!(
        class_for(&classes, DimClaim::Name("n".into())).placement(&dag),
        GuardPlacement::Entry,
    );
}

/// A folded `shape(...)` read is an INTERFACE value, so a class whose members
/// are all such reads is an ENTRY guard.
///
/// Section 4.7 keys on the guard's operands - "an input tensor's axis, a
/// scalar parameter, or a literal" - and `spec/05` section 2.4.1 admits
/// `InputAxis` as an `expand` extent read "directly from that tensor's shape
/// metadata". So the quantity compared is an input tensor's axis, which is
/// the first item in that list. An earlier draft of this row asserted
/// `Local`, confusing "is this axis a `Load`'s own" with "is this operand an
/// input's axis"; that reading made every folded cross-tensor read a local
/// guard and would have put chelis#1374's guard at the wrong place under the
/// wrong `<op>`.
#[test]
fn a_class_whose_members_are_folded_input_axis_reads_is_an_entry_guard() {
    let mut dag = Dag::new();
    let base = f32_load(&mut dag, "b", vec![]);
    let x = f32_load(&mut dag, "x", vec![named("n")]);
    dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::InputAxis {
                tensor: 1,
                axis: RtAxis::Lit(0),
            },
        },
        vec![base, x],
        ty(vec![DimInfo::Lit(4)], Prim::F32),
        None,
    );
    let classes = derive_runtime_dim_classes(&dag);
    assert_eq!(
        class_for(&classes, DimClaim::Literal(4)).placement(&dag),
        GuardPlacement::Entry,
    );
}

// ---------------------------------------------------------------------------
// C5 property 7: rebuild survival.
//
// "the stamped names survive on the rebuilt nodes so the derived classes are
// the same set in the same order".
// ---------------------------------------------------------------------------

#[test]
fn classes_survive_vectorize_axis0_as_the_same_set_in_the_same_order() {
    let mut dag = Dag::new();
    let x = f32_load(&mut dag, "x", vec![named("n")]);
    let y = f32_load(&mut dag, "y", vec![named("n")]);
    // ONE result reaching both witnesses, not one root each. C2.4 scopes a
    // claim by the results it reaches, so two roots are two signatures and
    // two claims sharing a spelling; joining them is what a single signature
    // with two parameters looks like in the DAG. The property this row names
    // is unchanged.
    let joined = dag.add_node(
        RiscOp::Add,
        vec![x, y],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    dag.add_root(joined);
    let before = derive_runtime_dim_classes(&dag);
    let batched = chelis_ir::vmap::vectorize_axis0(&dag, DimInfo::Lit(3))
        .expect("vectorize a two-Load graph");
    let after = derive_runtime_dim_classes(&batched);
    assert_eq!(
        claims(&before),
        claims(&after),
        "vmap shifts axes but must not change which claims are classes",
    );
    assert_eq!(after.len(), 1);
}

#[test]
fn classes_survive_bind_symbolic_dims_unchanged() {
    let mut dag = Dag::new();
    let x = f32_load(&mut dag, "x", vec![named("n")]);
    let y = f32_load(&mut dag, "y", vec![named("n")]);
    // ONE result reaching both witnesses, not one root each. C2.4 scopes a
    // claim by the results it reaches, so two roots are two signatures and
    // two claims sharing a spelling; joining them is what a single signature
    // with two parameters looks like in the DAG. The property this row names
    // is unchanged.
    let joined = dag.add_node(
        RiscOp::Add,
        vec![x, y],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    dag.add_root(joined);
    let before = derive_runtime_dim_classes(&dag);
    let bound = chelis_ir::dag::bind_symbolic_dims(
        &dag,
        &chelis_unord::UnordMap::from([("n".to_string(), 4usize)]),
        &chelis_unord::UnordSet::new(),
    )
    .expect("bind n");
    let after = derive_runtime_dim_classes(&bound);
    assert_eq!(claims(&before), claims(&after));
    assert_eq!(
        members_of(&before, DimClaim::Name("n".into())),
        members_of(&after, DimClaim::Name("n".into())),
        "binding preserves node ids, so the members are identical",
    );
}

/// The `(claim, interface members)` shape of a class list, with each member
/// named by the `Load` it reads rather than by node id.
///
/// Node ids are a rebuild's own business: a pass that renumbers is not a pass
/// that broke the derivation. What must survive is WHICH input tensor axes a
/// claim groups, and that is stable under renumbering because a `Load` carries
/// its name. Members that are not input axes are counted but not named, so a
/// pass that drops one still fails the comparison.
fn class_shape(dag: &Dag, classes: &[RuntimeDimClass]) -> Vec<(DimClaim, Vec<(String, usize)>)> {
    classes
        .iter()
        .map(|class| {
            let members = class
                .members
                .iter()
                .map(|member| match dag.get(member.node).map(|node| &node.op) {
                    Some(RiscOp::Load { name }) => (name.to_string(), member.axis),
                    _ => ("<computed>".to_string(), member.axis),
                })
                .collect();
            (class.claim.clone(), members)
        })
        .collect()
}

/// Every node's op and declared dims, in graph order. Two graphs with the same
/// structure rebuilt identically; two that differ here were REWRITTEN.
fn structure(dag: &Dag) -> Vec<(String, Vec<DimInfo>)> {
    dag.nodes()
        .iter()
        .map(|node| (format!("{:?}", node.op), node.output_type.dims.clone()))
        .collect()
}

fn find_load(dag: &Dag, name: &str) -> NodeId {
    dag.nodes()
        .iter()
        .find(|node| matches!(&node.op, RiscOp::Load { name: n } if n.as_str() == name))
        .unwrap_or_else(|| panic!("no Load named {name} survived the pass"))
        .id
}

/// `dag` with the stamped `name` removed from every axis of `id`, which is how
/// a rebuild pass that lost a name during a rewrite would leave the graph.
fn drop_stamped_name(dag: &Dag, id: NodeId, name: &str) -> Dag {
    let mut out = dag.clone();
    if let Some(node) = out.node_mut(id) {
        for dim in node.output_type.dims.iter_mut() {
            if let DimInfo::Named(stamped, _) = dim
                && stamped.as_str() == name
            {
                *dim = DimInfo::Lit(97);
            }
        }
    }
    out
}

/// The shared body of the rebuild row: one pass, its input, its output.
///
/// Three assertions, in the order that makes the third mean something.
///
/// 1. The pass REWROTE this fixture. Without it the two below hold over the
///    copy path and prove nothing about the transform the pass is named for,
///    which is the exact way this row's earlier fixture was vacuous.
/// 2. Every class that existed before survives, in order, grouping the same
///    input tensor axes. `added` states the classes the pass introduces, so a
///    pass that adds one silently cannot pass by being ignored.
/// 3. Dropping the claim's stamped name from a surviving witness IN THE PASS
///    OUTPUT changes the derived classes. This is the per-pass red-first
///    check: it makes assertion 2 an equality that a broken rebuild fails.
///
/// What assertion 3 does NOT cover, stated because the gap is real: dropping a
/// name from an axis the pass SYNTHESIZED (the fused node's output, the
/// specialized matmul's output) changes nothing, because such an axis is
/// op-computed and C2.4 makes it claim-supplied rather than a witness. The
/// derivation is insensitive to it by design, so this receipt is sensitive to
/// exactly what the property is about and no more.
fn assert_rebuild_preserves_classes(
    pass: &str,
    before: &Dag,
    after: &Dag,
    witness_load: &str,
    claim_name: &str,
    axis_shift: usize,
    added: &[DimClaim],
) {
    assert_ne!(
        structure(before),
        structure(after),
        "{pass}: the fixture must make this pass rewrite, or the equalities \
         below cover the rebuild-copy path only",
    );

    let before_classes = derive_runtime_dim_classes(before);
    let after_classes = derive_runtime_dim_classes(after);
    let before_claims = claims(&before_classes);

    let survived: Vec<_> = class_shape(after, &after_classes)
        .into_iter()
        .filter(|(claim, _)| before_claims.contains(claim))
        .collect();
    // `vectorize_axis0` inserts a batch axis at position 0, so every surviving
    // member's axis index moves right by one. That is the pass doing its job,
    // not the derivation losing a witness, so the shift is declared per pass
    // rather than absorbed by comparing something weaker.
    let expected: Vec<_> = class_shape(before, &before_classes)
        .into_iter()
        .map(|(claim, members)| {
            let shifted = members
                .into_iter()
                .map(|(load, axis)| (load, axis + axis_shift))
                .collect();
            (claim, shifted)
        })
        .collect();
    assert_eq!(
        expected, survived,
        "{pass}: every class present before the pass must survive it, in \
         order, grouping the same input tensor axes",
    );

    let introduced: Vec<DimClaim> = claims(&after_classes)
        .into_iter()
        .filter(|claim| !before_claims.contains(claim))
        .collect();
    assert_eq!(
        introduced,
        added.to_vec(),
        "{pass}: the classes this pass introduces are part of the record",
    );

    let witness = find_load(after, witness_load);
    let broken = drop_stamped_name(after, witness, claim_name);
    assert_ne!(
        claims(&derive_runtime_dim_classes(&broken)),
        claims(&after_classes),
        "{pass}: dropping `{claim_name}` from `{witness_load}` in the pass \
         output must change the derived classes, or the equality above is \
         satisfied by a derivation that sees nothing",
    );
}

/// Two witnesses of `nn` joined into one rooted result: the smallest graph
/// that derives a class at all. Each pass fixture below extends it with the
/// structure that makes that pass rewrite.
fn two_witness_base(dag: &mut Dag) -> (NodeId, NodeId, NodeId) {
    let x = f32_load(dag, "x", vec![named("nn")]);
    let y = f32_load(dag, "y", vec![named("nn")]);
    let joined = dag.add_node(
        RiscOp::Add,
        vec![x, y],
        ty(vec![named("nn")], Prim::F32),
        None,
    );
    (x, y, joined)
}

/// Oracle row `rebuild.classes_after_each_pass`.
///
/// C5 property 7 asks for the derived classes to be "the same set in the same
/// order" after each rebuild pass. The two tests above take one pass each on a
/// graph each of them genuinely transforms; this one is the row's receipt and
/// takes every rebuild pass `chelis-ir` exposes as a public entry point, EACH
/// ON A FIXTURE THAT MAKES IT REWRITE.
///
/// That last clause is the whole point of the fixtures below, and it is worth
/// the lines. An earlier version of this row ran all seven passes on
/// `Add(Load x, Load y)`, where nothing is constant, nothing is dead, there is
/// no common subexpression, no fusible chain and no BLAS pattern. Five of the
/// seven rebuilt that graph by copy, so `constant_fold`'s assertion reduced to
/// `derive(dag) == derive(dag)` and the other four proved the copy path rather
/// than the transform each is named for. A rewrite is precisely where a pass
/// would plausibly synthesize a fresh output type and lose a stamped name, so
/// the copy path is the half of the property that does not need proving.
/// `assert_rebuild_preserves_classes` now fails if a fixture ever stops making
/// its pass rewrite, so this cannot silently regress.
///
/// COVERAGE, stated exactly because the row's name says "each pass": the
/// passes exercised are `vmap::vectorize_axis0`, `dag::bind_symbolic_dims`,
/// `optimize::constant_fold`, `optimize::dead_code_eliminate`,
/// `optimize::common_subexpr_eliminate`, `fuse::fuse`, and
/// `specialize::specialize_for_blas`. `splice_dag` and graph cloning are
/// lowering-side and live in `chelis-types`, so they are not reachable from
/// this crate and are not covered here; C2.4's `f(n, n)` splice row covers the
/// splice's observable consequence instead.
///
/// MEASURED DEVIATION, recorded rather than hidden: `specialize_for_blas` does
/// not leave "the same set". Replacing `Sum(Mul(Expand a, Expand b))` with a
/// `BlasMatmul` gives the specialized node two op-computed output axes under
/// literal claims, so the pass ADDS `Literal(2)` and `Literal(3)` as one-member
/// classes that the reduction did not produce. It loses nothing: the `kk` class
/// survives with both of its input axes. The addition is asserted exactly here,
/// so it is locked and visible; whether the derivation should form those two
/// classes at all is not this row's question.
///
/// EVIDENTIARY STATUS: regression test for all five passes whose fixtures were
/// added here, each watched failing under the name drop in
/// `assert_rebuild_preserves_classes` before the equality above it was trusted;
/// disposition lock for `vectorize_axis0` and `bind_symbolic_dims`, whose two
/// neighbouring tests already covered them on transforming graphs.
#[test]
fn every_rebuild_pass_preserves_the_derived_classes() {
    // `bind_symbolic_dims` and `vectorize_axis0` transform the bare two-witness
    // graph, so they keep it. Node ids survive both, so these two assert the
    // stronger `(node, axis)` member equality that the projection cannot make.
    let mut dag = Dag::new();
    let (_, _, joined) = two_witness_base(&mut dag);
    dag.add_root(joined);
    let before = derive_runtime_dim_classes(&dag);
    assert_eq!(
        claims(&before),
        vec![DimClaim::Name("nn".into())],
        "the base fixture must derive its claim before any pass runs",
    );

    let bound = chelis_ir::dag::bind_symbolic_dims(
        &dag,
        &chelis_unord::UnordMap::from([("nn".to_string(), 4usize)]),
        &chelis_unord::UnordSet::new(),
    )
    .expect("bind nn");
    assert_rebuild_preserves_classes("bind_symbolic_dims", &dag, &bound, "y", "nn", 0, &[]);
    assert_eq!(
        members_of(&before, DimClaim::Name("nn".into())),
        members_of(
            &derive_runtime_dim_classes(&bound),
            DimClaim::Name("nn".into())
        ),
        "bind_symbolic_dims preserves node ids, so the members are identical",
    );

    let batched = chelis_ir::vmap::vectorize_axis0(&dag, DimInfo::Lit(3)).expect("vectorize");
    assert_rebuild_preserves_classes("vectorize_axis0", &dag, &batched, "y", "nn", 1, &[]);

    // `constant_fold` folds a binary op over two `Const` operands. It mutates
    // in place, so node ids survive and the member equality is exact.
    let mut folding = Dag::new();
    let (_, _, folding_joined) = two_witness_base(&mut folding);
    let five = folding.add_node(
        RiscOp::synth_const(Prim::F32, 5.0),
        vec![],
        ty(vec![], Prim::F32),
        None,
    );
    let two = folding.add_node(
        RiscOp::synth_const(Prim::F32, 2.0),
        vec![],
        ty(vec![], Prim::F32),
        None,
    );
    let difference = folding.add_node(RiscOp::Sub, vec![five, two], ty(vec![], Prim::F32), None);
    let scaled = folding.add_node(
        RiscOp::Add,
        vec![folding_joined, difference],
        ty(vec![named("nn")], Prim::F32),
        None,
    );
    folding.add_root(scaled);
    let mut folded = folding.clone();
    chelis_ir::optimize::constant_fold(&mut folded);
    assert_rebuild_preserves_classes("constant_fold", &folding, &folded, "y", "nn", 0, &[]);
    assert_eq!(
        members_of(
            &derive_runtime_dim_classes(&folding),
            DimClaim::Name("nn".into())
        ),
        members_of(
            &derive_runtime_dim_classes(&folded),
            DimClaim::Name("nn".into())
        ),
        "constant_fold replaces a node in place, so the members are identical",
    );

    // `dead_code_eliminate` needs a node no root reaches.
    let mut with_dead = Dag::new();
    let (dead_x, dead_y, dead_joined) = two_witness_base(&mut with_dead);
    with_dead.add_root(dead_joined);
    with_dead.add_node(
        RiscOp::Mul,
        vec![dead_x, dead_y],
        ty(vec![named("nn")], Prim::F32),
        None,
    );
    let live = chelis_ir::optimize::dead_code_eliminate(&with_dead);
    assert_rebuild_preserves_classes("dead_code_eliminate", &with_dead, &live, "y", "nn", 0, &[]);

    // `common_subexpr_eliminate` needs two structurally identical nodes.
    let mut with_duplicate = Dag::new();
    let (dup_x, dup_y, _) = two_witness_base(&mut with_duplicate);
    let first = with_duplicate.add_node(
        RiscOp::Mul,
        vec![dup_x, dup_y],
        ty(vec![named("nn")], Prim::F32),
        None,
    );
    let second = with_duplicate.add_node(
        RiscOp::Mul,
        vec![dup_x, dup_y],
        ty(vec![named("nn")], Prim::F32),
        None,
    );
    let combined = with_duplicate.add_node(
        RiscOp::Add,
        vec![first, second],
        ty(vec![named("nn")], Prim::F32),
        None,
    );
    with_duplicate.add_root(combined);
    let deduplicated = chelis_ir::optimize::common_subexpr_eliminate(&with_duplicate);
    assert_rebuild_preserves_classes(
        "common_subexpr_eliminate",
        &with_duplicate,
        &deduplicated,
        "y",
        "nn",
        0,
        &[],
    );

    // `fuse` needs an elementwise chain whose intermediate has one consumer,
    // so the two nodes merge into one `FusedElem`.
    let mut chained = Dag::new();
    let (_, _, chain_joined) = two_witness_base(&mut chained);
    let exponentiated = chained.add_node(
        RiscOp::Exp,
        vec![chain_joined],
        ty(vec![named("nn")], Prim::F32),
        None,
    );
    chained.add_root(exponentiated);
    let fused = chelis_ir::fuse::fuse(&chained);
    assert_rebuild_preserves_classes("fuse", &chained, &fused, "y", "nn", 0, &[]);

    // `specialize_for_blas` needs tier2 lowering's `Sum(Mul(Expand, Expand))`
    // shape, which its matmul detector rewrites to one `BlasMatmul`. The shared
    // contraction axis `kk` is the claim, so the class here is the one a real
    // matmul carries.
    let mut contraction = Dag::new();
    let a = f32_load(&mut contraction, "a", vec![DimInfo::Lit(2), named("kk")]);
    let b = f32_load(&mut contraction, "b", vec![named("kk"), DimInfo::Lit(3)]);
    let widened = ty(
        vec![DimInfo::Lit(2), named("kk"), DimInfo::Lit(3)],
        Prim::F32,
    );
    let expanded_a = contraction.add_node(
        RiscOp::Expand {
            axis: 2,
            size: RtDim::Lit(3),
        },
        vec![a],
        widened.clone(),
        None,
    );
    let expanded_b = contraction.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Lit(2),
        },
        vec![b],
        widened.clone(),
        None,
    );
    let product = contraction.add_node(RiscOp::Mul, vec![expanded_a, expanded_b], widened, None);
    let contracted = contraction.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F32,
        },
        vec![product],
        ty(vec![DimInfo::Lit(2), DimInfo::Lit(3)], Prim::F32),
        None,
    );
    contraction.add_root(contracted);
    let specialized = chelis_ir::specialize::specialize_for_blas(&contraction);
    assert!(
        specialized
            .nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::BlasMatmul { .. })),
        "the fixture must reach the matmul detector, not merely the cleanup legs",
    );
    assert_rebuild_preserves_classes(
        "specialize_for_blas",
        &contraction,
        &specialized,
        "b",
        "kk",
        0,
        &[DimClaim::Literal(2), DimClaim::Literal(3)],
    );
}

/// The discriminating twin for the two rebuild tests: dropping a stamped name
/// DOES change the derived classes. Without it, a derivation that returned an
/// empty list for every graph would pass both survival tests.
#[test]
fn dropping_a_stamped_name_changes_the_derived_classes() {
    let mut dag = Dag::new();
    f32_load(&mut dag, "x", vec![named("n")]);
    f32_load(&mut dag, "y", vec![named("n")]);
    let before = derive_runtime_dim_classes(&dag);
    assert_eq!(before.len(), 1, "the fixture must start with one class");

    let mut stripped = Dag::new();
    f32_load(&mut stripped, "x", vec![named("n")]);
    f32_load(&mut stripped, "y", vec![DimInfo::Lit(4)]);
    let after = derive_runtime_dim_classes(&stripped);
    assert!(
        after.is_empty(),
        "one witness for n is not a class: {:?}",
        claims(&after),
    );
}

// ---------------------------------------------------------------------------
// An axis SIZED BY the claim is not a witness of it.
//
// A `Const` declares a shape it does not compute: its named dimension is
// supplied by whatever the class resolves to. `declared_shape_sources` said as
// much in prose before there was a kind for it - "supplied by the equality
// class that name groups, which is the operation's own extent as far as this
// derivation is concerned" - and returned `OpComputed` only because
// `AxisSource` had no claim-supplied variant. Guarding such an axis against
// the class's canonical member would compare a value with itself.
// ---------------------------------------------------------------------------

#[test]
fn a_const_sized_by_its_claim_is_neither_a_member_nor_a_class() {
    let mut dag = Dag::new();
    let x = f32_load(&mut dag, "x", vec![named("n")]);
    let fill = dag.add_node(
        RiscOp::synth_const(Prim::F32, 0.0),
        vec![],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    let sum = dag.add_node(
        RiscOp::Add,
        vec![x, fill],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    dag.add_root(sum);

    let classes = derive_runtime_dim_classes(&dag);
    assert!(
        classes.is_empty(),
        "the Load declares `n` and the Const consumes it, so there is nothing \
         to guard: {:?}",
        claims(&classes),
    );
}

/// The discriminating twin: the `Const`'s axis carries the claim-supplied
/// source kind rather than `OpComputed`, so the rule above is decided on the
/// SOURCE and not by a second list of operations. Without this, an
/// implementation that excluded `Const` by matching the op would pass the row
/// above while leaving "what is a witness" decided in two places.
#[test]
fn a_const_named_axis_reports_the_claim_supplied_source_kind() {
    let mut dag = Dag::new();
    let fill = dag.add_node(
        RiscOp::synth_const(Prim::F32, 0.0),
        vec![],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    assert_eq!(
        chelis_ir::axis_sources::output_axis_sources(&dag, fill),
        vec![AxisSource::ClassSupplied { op: fill, axis: 0 }],
    );
}

// ---------------------------------------------------------------------------
// Symbolic-parameter ORDER.
//
// `symbolic_params` feeds `CodegenResult.symbolic_dims`, which three CLI sites
// print verbatim as `Symbolic dims: <joined>` and two compiler-api sites
// publish on a public result field. Its order is therefore observable, and
// NOTHING in the tree covered it: the only two assertions on `symbolic_dims`
// are single-element vectors, which cannot fail on a reordering.
//
// `spec/04-type-system.md` section 4.7 is the authority for what the order
// must be: "Whatever rule assigns the slots, the guard order follows the
// assigned slots, and never a separate traversal by binding name, hash
// iteration, or node identity." Grouping in a `BTreeMap<String, _>` is a
// traversal by binding name.
// ---------------------------------------------------------------------------

/// Two symbols on ONE input, whose axis order is the reverse of their name
/// order, so the two rules give different answers and the row can fail.
#[test]
fn symbolic_params_follow_assigned_slots_not_binding_names() {
    let mut dag = Dag::new();
    let x = f32_load(&mut dag, "x", vec![named("n"), named("m")]);
    dag.add_root(x);
    assert_eq!(
        chelis_ir::dag::symbolic_params(&dag),
        vec!["n".to_string(), "m".to_string()],
        "`n` is axis 0 of the only input, so it comes first; name order would \
         report `m` first, which is the traversal section 4.7 forbids",
    );
}

// ---------------------------------------------------------------------------
// A scalar PARAMETER is an interface value; a computed scalar is not.
//
// Section 4.7's interface list is "an input tensor's axis, a scalar parameter,
// or a literal", against "checked integer arithmetic, a user-function result,
// or an extent an operation computes". `AxisSource::ScalarInput` covers both
// sides of that line, because it records only WHICH SLOT holds the extent.
// The distinction survives in the DAG: the slot names a node, and that node
// is either the `Load` of a scalar parameter or the arithmetic that produced
// it, so the derivation can tell them apart without a second representation.
// ---------------------------------------------------------------------------

#[test]
fn a_class_fed_by_a_scalar_parameter_is_an_entry_guard() {
    let mut dag = Dag::new();
    let base = f32_load(&mut dag, "b", vec![]);
    let k = dag.add_node(
        RiscOp::Load { name: "k".into() },
        vec![],
        ty(vec![], Prim::Int64),
        None,
    );
    let declaring = f32_load(&mut dag, "x", vec![named("k")]);
    dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Node(1),
        },
        vec![base, k],
        ty(vec![named("k")], Prim::F32),
        None,
    );
    let classes = derive_runtime_dim_classes(&dag);
    assert_eq!(
        class_for(&classes, DimClaim::Name("k".into())).placement(&dag),
        GuardPlacement::Entry,
        "a bare scalar parameter is section 4.7's \"a scalar parameter\"",
    );
    let _ = declaring;
}

/// The discriminating twin: the same class fed by CHECKED ARITHMETIC over that
/// parameter is locally computed, so its guard takes the introducing
/// operation's position. Without this row, treating every `ScalarInput` as
/// interface would satisfy the row above and hoist an arithmetic guard to
/// entry, where its operand does not yet exist.
#[test]
fn a_class_fed_by_computed_arithmetic_is_a_local_guard() {
    let mut dag = Dag::new();
    let base = f32_load(&mut dag, "b", vec![]);
    let k = dag.add_node(
        RiscOp::Load { name: "k".into() },
        vec![],
        ty(vec![], Prim::Int64),
        None,
    );
    let doubled = dag.add_node(RiscOp::Mul, vec![k, k], ty(vec![], Prim::Int64), None);
    let declaring = f32_load(&mut dag, "x", vec![named("k")]);
    dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Node(1),
        },
        vec![base, doubled],
        ty(vec![named("k")], Prim::F32),
        None,
    );
    let classes = derive_runtime_dim_classes(&dag);
    assert_eq!(
        class_for(&classes, DimClaim::Name("k".into())).placement(&dag),
        GuardPlacement::Local,
        "checked arithmetic is a locally computed value",
    );
    let _ = declaring;
}

// ---------------------------------------------------------------------------
// A claim on a DEAD local intermediate owes no guard.
//
// C2.4 rule 2 forces only INTERFACE witnesses live. A local member's guard
// exists only if the operation introducing the extent is in the DAG a lane
// consumes after the last rewrite (C4.5), so a claim on an intermediate that
// nothing reaches produces no guard: the value it claims is never produced,
// and there is nothing to compare. Forcing local members live instead makes
// liveness circular, because a dead node carrying a claim becomes a member and
// the membership then keeps it alive.
// ---------------------------------------------------------------------------

/// A dead ascribed intermediate contributes no member, so its claim is left
/// with the declaring `Load` alone and forms no class.
#[test]
fn a_claim_on_a_dead_local_intermediate_owes_no_guard() {
    let mut dag = Dag::new();
    let x = f32_load(&mut dag, "x", vec![named("n")]);
    let base = f32_load(&mut dag, "b", vec![]);
    let size = f32_load(&mut dag, "k", vec![]);
    let dead = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Node(1),
        },
        vec![base, size],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    // Only `x` is a root; the ascribed intermediate reaches nothing.
    dag.add_root(x);
    let pruned = chelis_ir::optimize::dead_code_eliminate(&dag);
    let classes = derive_runtime_dim_classes(&pruned);
    assert!(
        classes.is_empty(),
        "the dead intermediate is gone, so `n` has one witness: {:?}",
        claims(&classes),
    );
    let _ = dead;
}

/// The discriminating twin: the SAME ascription on a live intermediate keeps
/// its member and its local guard. Without this row, a rule that dropped every
/// local member would satisfy the row above and delete chelis#1379's guard.
#[test]
fn the_same_claim_on_a_live_local_intermediate_keeps_its_guard() {
    let mut dag = Dag::new();
    let x = f32_load(&mut dag, "x", vec![named("n")]);
    let base = f32_load(&mut dag, "b", vec![]);
    let size = f32_load(&mut dag, "k", vec![]);
    let live = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Node(1),
        },
        vec![base, size],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    // ONE result reaching both witnesses, not one root each. C2.4 scopes a
    // claim by the results it reaches, so two roots are two signatures and
    // two claims sharing a spelling; joining them is what a single signature
    // with two parameters looks like in the DAG. The property this row names
    // is unchanged.
    let joined = dag.add_node(
        RiscOp::Add,
        vec![live, x],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    dag.add_root(joined);
    let pruned = chelis_ir::optimize::dead_code_eliminate(&dag);
    let classes = derive_runtime_dim_classes(&pruned);
    assert_eq!(
        members_of(&classes, DimClaim::Name("n".into())).len(),
        2,
        "the declaring Load and the live intermediate both witness `n`",
    );
}

/// An `InputAxis` is an interface value only when the slot it names holds an
/// input tensor.
///
/// Section 4.7's list says "an input tensor's AXIS". A folded read of a
/// COMPUTED tensor's axis is not that: the value does not exist until the
/// producing operation runs, so its guard cannot be evaluated at entry
/// "before any other operation of the function". Treating every `InputAxis`
/// as interface would hoist such a guard to a point where its operand has not
/// been produced.
#[test]
fn an_input_axis_naming_a_computed_tensor_is_a_local_guard() {
    let mut dag = Dag::new();
    let base = f32_load(&mut dag, "b", vec![]);
    let x = f32_load(&mut dag, "x", vec![named("n")]);
    // A computed tensor, not an input: its axis is not available at entry.
    let computed = dag.add_node(RiscOp::Neg, vec![x], ty(vec![named("n")], Prim::F32), None);
    dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::InputAxis {
                tensor: 1,
                axis: RtAxis::Lit(0),
            },
        },
        vec![base, computed],
        ty(vec![DimInfo::Lit(4)], Prim::F32),
        None,
    );
    let classes = derive_runtime_dim_classes(&dag);
    assert_eq!(
        class_for(&classes, DimClaim::Literal(4)).placement(&dag),
        GuardPlacement::Local,
        "the read tensor is computed, so its axis is not an interface value",
    );
}

// ---------------------------------------------------------------------------
// "A `cast` takes the placement of the value it casts" (`spec/04` §4.7).
//
// The rule is in the same paragraph as the interface list, and a cast is how a
// scalar parameter of the wrong width reaches an extent: `reshape(x, [cast(m,
// int64)])` for an `int32` parameter `m` lands its `RtDim::Node` at the Cast,
// not at the `Load`. Classifying by the slot's immediate producer would make
// that Local while the bare parameter is Entry, which is the same claim placed
// two different ways depending on a width conversion.
// ---------------------------------------------------------------------------

#[test]
fn a_cast_of_a_scalar_parameter_takes_the_parameters_placement() {
    let mut dag = Dag::new();
    let base = f32_load(&mut dag, "b", vec![]);
    let m = dag.add_node(
        RiscOp::Load { name: "m".into() },
        vec![],
        ty(vec![], Prim::Int32),
        None,
    );
    let widened = dag.add_node(
        RiscOp::Cast {
            new_precision: Prim::Int64,
        },
        vec![m],
        ty(vec![], Prim::Int64),
        None,
    );
    f32_load(&mut dag, "x", vec![named("k")]);
    dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Node(1),
        },
        vec![base, widened],
        ty(vec![named("k")], Prim::F32),
        None,
    );
    let classes = derive_runtime_dim_classes(&dag);
    assert_eq!(
        class_for(&classes, DimClaim::Name("k".into())).placement(&dag),
        GuardPlacement::Entry,
        "a cast takes the placement of the value it casts, and that value is a \
         scalar parameter",
    );
}

/// The discriminating twin: a cast of CHECKED ARITHMETIC takes the
/// arithmetic's placement, which is local. Without it, looking through every
/// `Cast` unconditionally would satisfy the row above and hoist a computed
/// guard to entry, where its operand has not been produced.
#[test]
fn a_cast_of_computed_arithmetic_takes_the_arithmetics_placement() {
    let mut dag = Dag::new();
    let base = f32_load(&mut dag, "b", vec![]);
    let m = dag.add_node(
        RiscOp::Load { name: "m".into() },
        vec![],
        ty(vec![], Prim::Int32),
        None,
    );
    let doubled = dag.add_node(RiscOp::Mul, vec![m, m], ty(vec![], Prim::Int32), None);
    let widened = dag.add_node(
        RiscOp::Cast {
            new_precision: Prim::Int64,
        },
        vec![doubled],
        ty(vec![], Prim::Int64),
        None,
    );
    f32_load(&mut dag, "x", vec![named("k")]);
    dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Node(1),
        },
        vec![base, widened],
        ty(vec![named("k")], Prim::F32),
        None,
    );
    let classes = derive_runtime_dim_classes(&dag);
    assert_eq!(
        class_for(&classes, DimClaim::Name("k".into())).placement(&dag),
        GuardPlacement::Local,
        "the value cast is checked arithmetic, so the cast is local too",
    );
}

// ---------------------------------------------------------------------------
// C1.3 local guard sites over a node-valued extent (chelis#1375).
//
// Section 4.7 places a guard "at the source position of the operation that
// introduces the guarded extent" when it "compares a locally computed value
// (checked integer arithmetic, a user-function result, or an extent an
// operation computes)". `spec/05` section 2.4.1 admits a rank-0 `Node` scalar
// as a `reshape` target carrier, so a `reshape` whose target is computed is
// exactly that sentence's operation, and the axis it sets is exactly that
// sentence's extent.
// ---------------------------------------------------------------------------

/// The guard sites of `dag`, as `(node, axis, claim, op)`.
fn guard_sites(dag: &Dag) -> Vec<(usize, usize, String, &'static str)> {
    chelis_ir::axis_sources::local_dim_guard_sites(dag)
        .into_iter()
        .map(|((node, axis), claim)| (node, axis, claim.claim, claim.op))
        .collect()
}

/// A `reshape` whose target is a computed rank-0 scalar, under a claim the
/// operand's `Load` also declares. The class has two members and the computed
/// one is a guard site: nothing else in the program compares the extent the
/// operation produced against the extent the signature claimed.
#[test]
fn a_node_valued_reshape_target_is_a_local_guard_site() {
    let mut dag = Dag::new();
    let x = f32_load(&mut dag, "x", vec![named("n")]);
    let read = dag.add_node(
        RiscOp::Shape { axis: 0 },
        vec![x],
        ty(vec![], Prim::Int64),
        None,
    );
    let extent = dag.add_node(RiscOp::Mul, vec![read, read], ty(vec![], Prim::Int64), None);
    let reshaped = dag.add_node(
        RiscOp::Reshape {
            new_shape: vec![RtDim::Node(1), RtDim::Lit(2)],
        },
        vec![x, extent],
        ty(vec![named("n"), DimInfo::Lit(2)], Prim::F32),
        None,
    );
    assert_eq!(
        class_for(
            &derive_runtime_dim_classes(&dag),
            DimClaim::Name("n".into())
        )
        .placement(&dag),
        GuardPlacement::Local,
        "the target is produced by arithmetic, so the class is not all-interface",
    );
    assert_eq!(
        guard_sites(&dag),
        vec![(reshaped.0, 0, "n".to_string(), "reshape")],
        "the computed axis owes a guard at the reshape that introduces it",
    );
}

/// The discriminating twin. A `Sym` target RESTATES a symbol declared
/// elsewhere rather than computing one, so C2.4 makes it no more a witness
/// than a passed-through axis: it forms no member, so there is no site. A
/// derivation that admitted every reshape target axis would pass the test
/// above and fail this one.
#[test]
fn a_sym_reshape_target_is_not_a_local_guard_site() {
    let mut dag = Dag::new();
    let x = f32_load(&mut dag, "x", vec![named("n")]);
    dag.add_node(
        RiscOp::Reshape {
            new_shape: vec![RtDim::Sym("n".into()), RtDim::Lit(1)],
        },
        vec![x],
        ty(vec![named("n"), DimInfo::Lit(1)], Prim::F32),
        None,
    );
    assert_eq!(
        guard_sites(&dag),
        vec![],
        "a restated symbol computes no extent, so it owes no comparison",
    );
}

/// Round 1's P2-1. The `Expand` arm of the derivation's carrier read is NOT
/// dead: an `expand`/`insert` whose size reads the shape of a tensor this
/// function COMPUTED carries an `InputAxis` carrier whose producer is not a
/// `Load`, so the class is `Local` and the axis is a site. The reviewer's two
/// spellings both miss it, and both for real reasons: chelis#469 rejects an
/// arithmetic size at check time, and a size reading a PARAMETER's shape is an
/// interface value, so its class runs at entry.
///
/// Measured on the shipped compiler with
/// `def f(x: tensor[n, f32]) -> tensor[n, f32] = { y = mul(x, x); b = sum(x, 0);
/// insert(b, 0, shape(y, 0)) }`, whose emitted C carries
/// `numeric trap: domain in expand at int64` at the operation. Removing the arm
/// would make the eval lane skip a site the C lane guards, which is the lane
/// divergence this slice exists to remove, so the arm stays and this row is
/// what says so.
///
/// EVIDENTIARY STATUS: disposition lock on a live arm.
#[test]
fn an_expand_sized_from_a_computed_tensor_is_a_local_guard_site() {
    let mut dag = Dag::new();
    let x = f32_load(&mut dag, "x", vec![named("n")]);
    let computed = dag.add_node(
        RiscOp::Mul,
        vec![x, x],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    let base = dag.add_node(
        RiscOp::Mul,
        vec![x, x],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    let inserted = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::InputAxis {
                tensor: 1,
                axis: RtAxis::Lit(0),
            },
        },
        vec![base, computed],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    assert_eq!(
        class_for(
            &derive_runtime_dim_classes(&dag),
            DimClaim::Name("n".into())
        )
        .placement(&dag),
        GuardPlacement::Local,
        "the size reads a computed tensor, so the class is not all-interface",
    );
    assert_eq!(
        guard_sites(&dag),
        vec![(inserted.0, 0, "n".to_string(), "expand")],
        "the expand that introduces the extent owes the guard",
    );
    // The carrier is no longer readable through a public helper: the derivation
    // states it in the site, which is the point of the unification. So the
    // assertion reads the site's own read instruction.
    let observed = chelis_ir::axis_sources::local_dim_guard_sites(&dag)
        .into_iter()
        .find(|((node, axis), _)| *node == inserted.0 && *axis == 0)
        .map(|(_, claim)| claim.observed)
        .expect("the expand's axis is a site");
    assert!(
        matches!(
            observed,
            chelis_ir::axis_sources::LocalGuardObservation::Carrier(RtDim::InputAxis { .. })
        ),
        "and the derivation hands both lanes the carrier to evaluate, not a \
         realized shape read: {observed:?}",
    );
}

/// Every class site's read instruction is the carrier its own source names.
///
/// The derivation admits a class member only when its source is `InputAxis` or
/// `ScalarInput`, and `rt_dim_source` mints exactly those two from
/// `RtDim::InputAxis` and `RtDim::Node`. So the source and the carrier are one
/// fact read two ways, and the `else { continue }` guarding the carrier read in
/// `local_dim_guard_sites` is unreachable rather than a silent drop.
///
/// That correspondence is why no class site needs a "cannot be observed"
/// state: every admitted source names a readable quantity. If a future change
/// admits a source without giving it a read instruction, sites would vanish
/// from BOTH lanes with nothing to show for it, so the property is asserted
/// rather than trusted. The `OpComputed` admission added its own read
/// instruction rather than widening the filter alone, which is the shape this
/// row asks a future widening to take.
///
/// EVIDENTIARY STATUS: disposition lock. It cannot fail on today's tree; it
/// fails on the tree that widens the source filter without widening the read.
#[test]
fn every_local_class_site_carries_the_carrier_its_source_names() {
    // A class with both kinds of admitted source: a node-valued reshape target
    // (`ScalarInput`) and an expand sized from a computed tensor's shape
    // (`InputAxis`).
    let mut dag = Dag::new();
    let x = f32_load(&mut dag, "x", vec![named("n")]);
    let size = f32_load(&mut dag, "k", vec![]);
    let reshaped = dag.add_node(
        RiscOp::Reshape {
            new_shape: vec![RtDim::Node(1)],
        },
        vec![x, size],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    let computed = dag.add_node(
        RiscOp::Mul,
        vec![x, x],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    let expanded = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::InputAxis {
                tensor: 3,
                axis: RtAxis::Lit(0),
            },
        },
        vec![reshaped, computed],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    let sites = chelis_ir::axis_sources::local_dim_guard_sites(&dag);
    assert!(
        !sites.is_empty(),
        "the fixture must produce sites for the property to say anything",
    );
    let _ = expanded;
    for ((node, axis), claim) in sites {
        match claim.observed {
            chelis_ir::axis_sources::LocalGuardObservation::Carrier(_) => {}
            chelis_ir::axis_sources::LocalGuardObservation::RealizedExtent
            | chelis_ir::axis_sources::LocalGuardObservation::ComputedExtent(_) => {
                // A unit-extent site legitimately has no carrier, and an
                // op-computed site legitimately computes its own; this fixture
                // builds neither, so reaching here means the class loop took
                // the wrong branch.
                panic!("node {node} axis {axis} is a class site with no carrier");
            }
        }
    }
}

/// Round 1's P2-2. The eval consumer's `CanonicalExtent::Resolved` arm: a
/// class whose canonical value the checker already resolved to a literal is
/// compared against that literal, not against a binder no lane declares.
///
/// The resolved size belongs to the interface member in this fixture. A
/// resolved size on the local member is also a claim, and is exercised by
/// the backend's numeric_local_extent_claims_execute_exactly matrix. Built directly because
/// the surface spelling that resolves an input extent this way is the
/// interface half of chelis#1377, which reaches the ENTRY path instead.
///
/// EVIDENTIARY STATUS: regression test. Watched failing with the arm returning
/// the binder instead of the resolved size, where evaluation succeeded.
#[test]
fn a_resolved_canonical_traps_on_eval_against_its_literal() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        ty(vec![DimInfo::Named("n".into(), Some(4))], Prim::F32),
        None,
    );
    let read = dag.add_node(
        RiscOp::Shape { axis: 0 },
        vec![x],
        ty(vec![], Prim::Int64),
        None,
    );
    let doubled = dag.add_node(RiscOp::Mul, vec![read, read], ty(vec![], Prim::Int64), None);
    let reshaped = dag.add_node(
        RiscOp::Reshape {
            new_shape: vec![RtDim::Node(1)],
        },
        vec![x, doubled],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    let sites = chelis_ir::axis_sources::local_dim_guard_sites(&dag);
    assert_eq!(
        sites
            .iter()
            .map(|((node, axis), claim)| (*node, *axis, claim.canonical.clone()))
            .collect::<Vec<_>>(),
        vec![(
            reshaped.0,
            0,
            chelis_ir::axis_sources::CanonicalExtent::Resolved(4)
        )],
        "the checker resolved the claim, so the canonical value is that literal",
    );
    let err = chelis_ir::eval::eval_tensor_with(&dag, |name| {
        (name == "x").then(|| chelis_ir::eval::TensorValue::from_vec(vec![4], vec![1.0; 4]))
    })
    .expect_err("the reshape computes 16 against a claim of 4");
    assert!(
        err.contains("numeric trap: domain in reshape at int64"),
        "the eval lane renders [04-NUM-9] for a resolved canonical too: {err}"
    );
    assert!(
        err.contains("extent `n`: claimed = 4,"),
        "and compares against the literal, not a binder: {err}"
    );
}

/// Folded shape reads remain interface witnesses. Their introducing operation
/// order must not reverse the declaring parameters (or axes within one input).
#[test]
fn folded_interface_claims_follow_declaring_slots_and_axes() {
    for same_input in [false, true] {
        let mut dag = Dag::new();
        let b = f32_load(&mut dag, "b", vec![]);
        let z = f32_load(&mut dag, "z", vec![named("rows"), named("cols")]);
        let a = if same_input {
            z
        } else {
            f32_load(&mut dag, "a", vec![named("other")])
        };
        let inner = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: RtDim::InputAxis {
                    tensor: 1,
                    axis: RtAxis::Lit(if same_input { 1 } else { 0 }),
                },
            },
            vec![b, a],
            ty(vec![DimInfo::Lit(3)], Prim::F32),
            None,
        );
        let outer = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: RtDim::InputAxis {
                    tensor: 1,
                    axis: RtAxis::Lit(0),
                },
            },
            vec![inner, z],
            ty(vec![DimInfo::Lit(4), DimInfo::Lit(3)], Prim::F32),
            None,
        );
        dag.add_root(outer);
        let classes = derive_runtime_dim_classes(&dag);
        assert_eq!(
            claims(&classes),
            vec![DimClaim::Literal(4), DimClaim::Literal(3)]
        );
        assert!(
            classes
                .iter()
                .all(|class| class.placement(&dag) == GuardPlacement::Entry)
        );
    }
}

/// chelis#1800's derivation change, and the two halves that must NOT form a
/// class.
///
/// A `Name` class needs two members, because one witness has nothing to
/// disagree with. The one exception this change adds is a single member whose
/// source is [`AxisSource::OpComputed`] and whose dim carries a RESOLVED name:
/// that member has both a number to compare against and an extent the
/// operation computes to compare, so it is a complete guard, exactly as C2.4
/// makes a literal its own canonical value. The positive half is observable
/// from the CLI through
/// `runtime_extent_slice_b::a_nested_named_result_claim_is_enforced_through_its_resolved_binder`;
/// these are the negatives, at the layer that decides them.
///
/// Both halves matter because the rule was written narrow on purpose. An
/// unresolved single member has no canonical value for anything to compare
/// against, and admitting a resolved member with any other source would mint
/// a class for every literal-shaped input, which is the mint `is_member`'s
/// `ExternalAxis` rule exists to prevent.
///
/// EVIDENTIARY STATUS: regression tests for a rule this change introduces, at
/// the derivation layer the suite header names as the lock on the derivation.
/// Neither can fail on the tree before the change, because the single-member
/// admission did not exist there; each fails on a tree that widens the
/// admission past the source or past the resolved dim.
#[test]
fn a_single_resolved_op_computed_member_forms_a_class_and_its_two_negatives() {
    // Positive control: one member, `OpComputed` source, resolved dim.
    let mut dag = Dag::new();
    let operand = f32_load(&mut dag, "x", vec![named("r")]);
    let shrunk = dag.add_node(
        RiscOp::Shrink {
            bounds: vec![(
                RtDim::Lit(1),
                RtDim::InputAxis {
                    tensor: 0,
                    axis: RtAxis::Lit(0),
                },
            )],
        },
        vec![operand],
        ty(vec![DimInfo::Named("n".into(), Some(2))], Prim::F32),
        None,
    );
    let _ = shrunk;
    assert_eq!(
        claims(&derive_runtime_dim_classes(&dag)),
        vec![DimClaim::Name("n".into())],
        "a resolved op-computed member is its own canonical value"
    );
    assert_eq!(
        members_of(
            &derive_runtime_dim_classes(&dag),
            DimClaim::Name("n".into())
        ),
        vec![(shrunk, 0)],
        "and the class holds exactly that one member"
    );

    // Negative one: the SAME member unresolved. Nothing supplies a canonical
    // value, so C2.4's two-witness rule stands and no class forms.
    let mut dag = Dag::new();
    let operand = f32_load(&mut dag, "x", vec![named("r")]);
    dag.add_node(
        RiscOp::Shrink {
            bounds: vec![(
                RtDim::Lit(1),
                RtDim::InputAxis {
                    tensor: 0,
                    axis: RtAxis::Lit(0),
                },
            )],
        },
        vec![operand],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    assert!(
        !claims(&derive_runtime_dim_classes(&dag)).contains(&DimClaim::Name("n".into())),
        "an unresolved single member has no canonical value and forms no class"
    );

    // Negative two: a RESOLVED single member whose source is a `Load` axis.
    // Admitting it would mint a class for every literal-shaped input.
    let mut dag = Dag::new();
    let declared = f32_load(&mut dag, "x", vec![DimInfo::Named("m".into(), Some(4))]);
    let _ = declared;
    assert!(
        !claims(&derive_runtime_dim_classes(&dag)).contains(&DimClaim::Name("m".into())),
        "a resolved member whose source is an external axis forms no class"
    );
}
