//! The DAG evaluator's runtime-extent guard (chelis#1277 Slice B, b2.1).
//!
//! ## Why this file exists instead of a CLI row
//!
//! `chelis eval --file` cannot reach `chelis_ir::eval` for any fixture form.
//! Measured on `801f92c02`: a `def main()` root, a top-level value binding,
//! and a def-free top-level tensor expression all yield ZERO `Lane::Tensor`
//! roots, so every root falls to the host interpreter. A value binding keeps
//! the strict lowering classification, because chelis#218's `to_tensor`
//! exemption applies to fn-def bodies only (`lower.rs:3040`); a zero-argument
//! fn root is host-applied; and the CLI supplies no input bindings for a
//! parameterized tensor entry (`main.rs:9876`, `bindings: BTreeMap::new()`).
//!
//! So the honest granularity for this pull request is "the DAG evaluator
//! places a conforming guard, proved with inputs bound at the API boundary".
//! The CLI eval rows in `crates/chelis-cli/tests/runtime_extent_slice_b.rs`
//! stay red and are owned by B2h, which routes lowered-def application
//! through this evaluator. Claiming them here would be claiming a lane this
//! change does not reach.
//!
//! ## What is asserted
//!
//! `spec/04-type-system.md` section 4.7 makes a runtime extent guard a typed
//! operation-precondition guard under [04-NUM-9], so the complete line is
//! `numeric trap: domain in <op> at int64` with no prefix and no suffix, and
//! section 4.7's placement rule puts it "after every value it compares is
//! available and before the first allocation or element access whose shape
//! depends on the guarded extent".
//!
//! Both halves are RED on the unfixed tree, and for different reasons. The
//! existing check (`eval.rs`, the op-declared runtime-dim comparison) renders
//! `runtime dim \`n\` mismatch: node N axis A computed X, but an earlier
//! declaration bound Y`, which is not [04-NUM-9]'s line; and it runs AFTER
//! the node's value is produced, which is after the allocation the rule puts
//! it before.

use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, RtAxis, RtDim, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict};
use chelis_types::types::Prim;

fn ty(dims: Vec<DimInfo>, precision: Prim) -> TensorType {
    TensorType { dims, precision }
}

fn named(name: &str) -> DimInfo {
    DimInfo::Named(name.into(), None)
}

fn load(dag: &mut Dag, name: &str, dims: Vec<DimInfo>) -> NodeId {
    dag.add_node(
        RiscOp::Load { name: name.into() },
        vec![],
        ty(dims, Prim::F32),
        None,
    )
}

/// The exact [04-NUM-9] line an extent guard renders.
fn domain_trap_line(op: &str) -> String {
    format!("numeric trap: domain in {op} at int64")
}

/// chelis#1374's shape, built directly so the inputs can be bound at the API
/// boundary: `x` declares `n`, `y` declares `m`, and the `Expand` sets an axis
/// claimed as `n` whose extent is read from `y`'s axis 0. Binding `x` at
/// extent 2 and `y` at extent 3 makes the claim false.
fn cross_tensor_claim_dag() -> (Dag, NodeId) {
    let mut dag = Dag::new();
    let base = load(&mut dag, "b", vec![]);
    let x = load(&mut dag, "x", vec![named("n")]);
    let y = load(&mut dag, "y", vec![named("m")]);
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
    // `x` is READ, not merely declared. C2.4 scopes a claim by the results it
    // reaches, so a witness no root reaches forms no class and gets no guard -
    // section 4.7's "regardless of data use" case, which this derivation
    // cannot honour and which C2.4 records as a residual owned by B2b. The
    // driven C rows in `chelis-backend-c/tests/exec_compile.rs` were rebuilt
    // the same way and for the same reason. The entry guard runs before any
    // node evaluates, so consuming `x` does not let the elementwise operand
    // check preempt it, which the disagreeing rows measure rather than assume.
    let out = dag.add_node(
        RiscOp::Add,
        vec![expanded, x],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    dag.add_root(out);
    (dag, out)
}

fn bind(name: &str, extent: usize) -> Option<TensorValue> {
    match extent {
        0 => Some(TensorValue::scalar(1.0)),
        n => Some(TensorValue::from_vec(vec![n], vec![1.0; n])),
    }
    .filter(|_| !name.is_empty())
}

/// A disagreeing claim traps, and renders [04-NUM-9]'s exact line.
///
/// EVIDENTIARY STATUS: regression test. On the unfixed tree the evaluator
/// either accepts this silently or reports its own non-conforming string;
/// either way this assertion fails, and it was watched failing at b2.1.
#[test]
fn a_disagreeing_class_traps_with_the_numeric_trap_line() {
    let (dag, root) = cross_tensor_claim_dag();
    let result = eval_tensor_roots_with_strict(&dag, &[root], |name| match name {
        "b" => Some(TensorValue::scalar(1.0)),
        "x" => bind("x", 2),
        "y" => bind("y", 3),
        _ => None,
    });
    let err = result.expect_err("n = 2 claimed over a read of m = 3 must not evaluate");
    // `load`, not `expand`. Section 4.7 fixes the slot for this shape:
    // "for a guard whose operands are all interface values, the `load`
    // primitive of the later witness in signature order". Both witnesses here
    // are input tensor axes - `x`'s own, and the `expand` size's folded read
    // of `y` - so the class is all-interface and its guard runs at entry. The
    // b2.1 stub wrote `expand` under the pre-refinement belief that an
    // `InputAxis` member makes a class local; the spec text is the oracle and
    // it says otherwise.
    assert!(
        err.contains(&domain_trap_line("load")),
        "section 4.7 makes this an [04-NUM-9] typed precondition guard, got: {err}",
    );
}

/// The disagreeing sources, the axis, and each observed value are conveyed.
///
/// Section 4.7 binds "the information conveyed and not the bytes rendered",
/// so this checks content rather than an invented format.
#[test]
fn a_disagreeing_class_conveys_its_sources_axis_and_values() {
    let (dag, root) = cross_tensor_claim_dag();
    let err = eval_tensor_roots_with_strict(&dag, &[root], |name| match name {
        "b" => Some(TensorValue::scalar(1.0)),
        "x" => bind("x", 2),
        "y" => bind("y", 3),
        _ => None,
    })
    .expect_err("must trap");
    assert!(err.contains('n'), "the claim is named: {err}");
    assert!(err.contains('2'), "the claimed extent is reported: {err}");
    assert!(err.contains('3'), "the observed extent is reported: {err}");
}

/// The positive twin: an agreeing claim evaluates and keeps its shape.
///
/// Without this, a guard that trapped on every class would satisfy the two
/// rows above.
#[test]
fn an_agreeing_class_evaluates() {
    let (dag, root) = cross_tensor_claim_dag();
    let values = eval_tensor_roots_with_strict(&dag, &[root], |name| match name {
        "b" => Some(TensorValue::scalar(1.0)),
        "x" => bind("x", 3),
        "y" => bind("y", 3),
        _ => None,
    })
    .expect("n = m = 3 agrees and must evaluate");
    assert_eq!(values[&root].shape, vec![3]);
}

/// C1.3 placement: the guard runs BEFORE the allocation whose shape depends on
/// the guarded extent.
///
/// The discriminator is the allocated shape itself. A guard placed after the
/// movement node has already built its value has, by definition, allocated at
/// the WRONG extent first; a guard placed before it never allocates. So the
/// observable is that no value for the guarded node exists once the guard has
/// fired, which `eval_tensor_roots_with_strict` reports by returning `Err`
/// rather than a map containing a mis-sized entry.
///
/// EVIDENTIARY STATUS: regression test. The existing check runs after the
/// node's value is produced, so on the unfixed tree the mis-sized value is
/// built before anything compares it.
#[test]
fn the_guard_precedes_the_allocation_it_protects() {
    let (dag, root) = cross_tensor_claim_dag();
    let result = eval_tensor_roots_with_strict(&dag, &[root], |name| match name {
        "b" => Some(TensorValue::scalar(1.0)),
        "x" => bind("x", 2),
        "y" => bind("y", 3),
        _ => None,
    });
    match result {
        Err(err) => assert!(
            err.contains(&domain_trap_line("load")),
            "the guard, not a later shape error, must be what fails: {err}",
        ),
        Ok(values) => panic!(
            "the guarded node was allocated at {:?} before anything compared it",
            values.get(&root).map(|value| value.shape.clone()),
        ),
    }
}

/// Two independent roots whose signatures both spell a binder `seq`, at
/// different extents. C2.4 scopes a claim by the results it reaches, so these
/// are two claims sharing a spelling and not one class.
///
/// Asserted on the DERIVATION rather than through `eval_tensor_roots_with_strict`,
/// and the reason is worth recording: this program is already rejected on
/// `main` by an older mechanism. `infer_symbolic_bindings_from_inputs`
/// (`eval.rs:1738`) reads the legacy name-grouped `symbolic_bindings` and errs
/// with `symbolic dimension \`seq\` mismatch: canonical x[0] = 3, but y[1] = 2`
/// before any guard this slice places can run. That is a pre-existing
/// same-spelling defect on the eval lane, out of this slice's scope and not
/// introduced by it, so the eval lane cannot host a row that isolates the
/// class derivation's behaviour. The derivation is what this slice changed and
/// is what this row measures.
///
/// EVIDENTIARY STATUS: regression test. Round 2 found `root_reach` applied in
/// `derive_dim_witnesses` alone, so the C prologue's `Name` guards were scoped
/// while `derive_runtime_dim_classes` - which the entry and local guard sites
/// and the eval guard all read - was not. Measured on `3a0703027`: one `Entry`
/// class pairing `x` axis 0 with `y` axis 1.
#[test]
fn two_roots_spelling_one_binder_are_two_claims_not_one_class() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec![named("seq")]);
    let y = load(&mut dag, "y", vec![named("batch"), named("seq")]);
    let from_x = dag.add_node(
        RiscOp::Neg,
        vec![x],
        ty(vec![named("seq")], Prim::F32),
        None,
    );
    let from_y = dag.add_node(
        RiscOp::Neg,
        vec![y],
        ty(vec![named("batch"), named("seq")], Prim::F32),
        None,
    );
    dag.add_root(from_x);
    dag.add_root(from_y);

    for class in chelis_ir::axis_sources::derive_runtime_dim_classes(&dag) {
        let chelis_ir::axis_sources::DimClaim::Name(name) = &class.claim else {
            continue;
        };
        assert_ne!(
            (name.as_str(), class.members.len()),
            ("seq", 2),
            "two signatures spelling `seq` are two claims, not one class: {:?}",
            class.members,
        );
    }
}

// ===========================================================================
// S2b: the same-rank `expand`'s unit-extent claim on this evaluator.
//
// The CLI `.eval` receipt in `crates/chelis-cli/tests/runtime_extent_slice_b.rs`
// proves the HOST interpreter, which is the lane `chelis eval` reaches for a
// program of that shape. It is not evidence about the DAG evaluator's own
// guard, so that guard gets its own row here, with inputs bound at the API
// boundary, for exactly the reason this file's header gives.
// ===========================================================================

/// A same-rank `Expand` over a symbolic operand: the claim is that `x`'s axis
/// 0 is 1, and nothing in the graph proves it.
fn unit_extent_claim_dag() -> (Dag, NodeId) {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec![named("n")]);
    let out = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Lit(3),
        },
        vec![x],
        ty(vec![DimInfo::Lit(3)], Prim::F32),
        None,
    );
    dag.add_root(out);
    (dag, out)
}

/// An operand extent that refutes the claim traps, with [04-NUM-9]'s line.
///
/// `spec/05-risc-primitives.md` section 2.4.1: the same-rank form "is a claim
/// that the operand's extent at `axis` is 1 [...] A symbolic or runtime
/// operand extent at `axis` other than 1 fails that claim's runtime extent
/// guard and traps `Domain`."
///
/// `<op>` is `load` rather than `expand` for the same reason the class rows
/// above give: the only quantity this guard reads is an input tensor's axis,
/// which section 4.7 lists first among interface values, so the guard is
/// placed at entry and takes the `load` primitive.
///
/// EVIDENTIARY STATUS: regression test. On the tree without
/// `derive_unit_extent_claims` this program evaluates silently and returns a
/// tensor built by reading index 0 of an axis with more than one element,
/// which is the `silent_unguarded` baseline the corpus records.
#[test]
fn a_non_unit_operand_extent_refutes_the_unit_claim_and_traps() {
    let (dag, root) = unit_extent_claim_dag();
    let err = eval_tensor_roots_with_strict(&dag, &[root], |name| bind(name, 2))
        .expect_err("an operand extent of 2 must not broadcast under a unit claim");
    assert!(
        err.contains(&domain_trap_line("load")),
        "the claim's failure is an [04-NUM-9] typed precondition guard, got: {err}",
    );
    assert!(
        err.contains("claimed = 1") && err.contains("x axis 0 = 2"),
        "section 4.7 also requires the axis and the value observed for it: {err}",
    );
}

/// The control: an operand that satisfies the claim broadcasts, and the guard
/// stays out of the way.
///
/// This is what separates a guard from a rejection of the symbolic spelling.
/// Without it the row above would pass just as well against an evaluator that
/// refused every unproven extent.
#[test]
fn a_unit_operand_extent_satisfies_the_claim_and_broadcasts() {
    let (dag, root) = unit_extent_claim_dag();
    let values = eval_tensor_roots_with_strict(&dag, &[root], |name| bind(name, 1))
        .expect("a unit operand extent satisfies the claim");
    let out = &values[&root];
    assert_eq!(out.shape, vec![3], "the broadcast sets the claimed axis");
    assert_eq!(
        out.to_f64_lossy_vec(),
        vec![1.0, 1.0, 1.0],
        "and repeats the single element across it"
    );
}

/// A LOCALLY placed unit-extent claim, on this evaluator.
///
/// The operand's extent is computed by a `Shrink` with a node-valued end, so
/// no input carries it and the entry loop cannot see it. `spec/04-type-system.md`
/// section 4.7 places such a guard "after its producers and takes the source
/// position of the operation that introduces the guarded extent", which is the
/// `expand`, and gives its `<op>` slot the same name.
///
/// EVIDENTIARY STATUS: regression test. The first cut of S2b derived the Local
/// case and wired only the entry consumer, so this program evaluated to a
/// broadcast of element 0 of a two-element axis.
fn local_unit_extent_claim_dag(end_slot: usize) -> (Dag, NodeId) {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec![named("n")]);
    // A movement bound source is extent-domain and therefore exactly `int64`
    // ([05-DIM-1]); the f32 helper above is for tensor operands.
    let end = dag.add_node(
        RiscOp::Load { name: "end".into() },
        vec![],
        ty(vec![], Prim::Int64),
        None,
    );
    let shrunk = dag.add_node(
        RiscOp::Shrink {
            bounds: vec![(RtDim::Lit(0), RtDim::Node(end_slot))],
        },
        vec![x, end],
        ty(vec![named("_rt_shrink")], Prim::F32),
        None,
    );
    let out = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Lit(3),
        },
        vec![shrunk],
        ty(vec![DimInfo::Lit(3)], Prim::F32),
        None,
    );
    dag.add_root(out);
    (dag, out)
}

#[test]
fn a_locally_placed_unit_claim_traps_at_the_expand_that_makes_it() {
    let (dag, root) = local_unit_extent_claim_dag(1);
    let err = eval_tensor_roots_with_strict(&dag, &[root], |name| match name {
        "x" => Some(TensorValue::from_vec(vec![3], vec![7.0, 9.0, 11.0])),
        // Shrink to two elements, which refutes the claim.
        "end" => Some(TensorValue::scalar(2.0)),
        _ => None,
    })
    .expect_err("a computed operand extent of 2 must not broadcast under a unit claim");
    assert!(
        err.contains(&domain_trap_line("expand")),
        "a locally placed guard names the operation that introduces the claim, \
         not `load`: {err}",
    );
    assert!(
        err.contains("claimed = 1") && err.contains("axis 0 = 2"),
        "with the axis and the value observed for it: {err}",
    );
}

/// The control: the same graph shrunk to one element satisfies the claim.
#[test]
fn a_locally_placed_unit_claim_that_holds_broadcasts() {
    let (dag, root) = local_unit_extent_claim_dag(1);
    let values = eval_tensor_roots_with_strict(&dag, &[root], |name| match name {
        "x" => Some(TensorValue::from_vec(vec![3], vec![7.0, 9.0, 11.0])),
        "end" => Some(TensorValue::scalar(1.0)),
        _ => None,
    })
    .expect("a computed operand extent of 1 satisfies the claim");
    let out = &values[&root];
    assert_eq!(out.shape, vec![3]);
    assert_eq!(out.to_f64_lossy_vec(), vec![7.0, 7.0, 7.0]);
}

// ===========================================================================
// chelis#1566: the binding inference identifies extents by SPELLING.
//
// `infer_symbolic_bindings_from_inputs` groups occurrences with the legacy
// name-keyed `dag::symbolic_bindings`, a plain `BTreeMap<String, _>`, so two
// independent signatures that merely spell a binder `seq` land in one group
// and the equality loop rejects a correct merged kernel. The C and HIP
// prologues do not, because `derive_dim_witnesses` runs `split_by_scope`.
//
// The `.ch` shape is `rank_poly_tier3::named_axis_eval_parity_corners`, whose
// `total(x: &tensor[seq, f32])` and `use2(x: &tensor[batch, seq, f32])` merge
// into one kernel. The CLI cannot host this row: `runtime_extent_claim_
// preparation.rs`'s `scope.independent` cell passes there because both roots
// are value bindings the host interpreter applies, so it never reaches this
// evaluator's inputs map. The witness therefore binds the inputs at the API
// boundary, which is the same reason this file's header gives for every other
// row in it.
// ===========================================================================

/// Two independent roots, each from its own signature, both spelling `seq`,
/// at DIFFERENT extents. Both must evaluate: `seq` in one signature and `seq`
/// in the other are two claims sharing a spelling, not one extent.
///
/// EVIDENTIARY STATUS: regression test. Measured RED on `3dc3f54f6`, where
/// the evaluator returns `symbolic dimension `seq` mismatch: canonical x[0] =
/// 3, but y[1] = 2`.
#[test]
fn two_signatures_spelling_one_binder_bind_independently_on_eval() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec![named("seq")]);
    let y = load(&mut dag, "y", vec![named("batch"), named("seq")]);
    let from_x = dag.add_node(
        RiscOp::Neg,
        vec![x],
        ty(vec![named("seq")], Prim::F32),
        None,
    );
    let from_y = dag.add_node(
        RiscOp::Neg,
        vec![y],
        ty(vec![named("batch"), named("seq")], Prim::F32),
        None,
    );
    dag.add_root(from_x);
    dag.add_root(from_y);

    let values = eval_tensor_roots_with_strict(&dag, &[from_x, from_y], |name| match name {
        "x" => Some(TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0])),
        "y" => Some(TensorValue::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])),
        _ => None,
    })
    .expect("two scopes spelling `seq` are two claims, not one extent");
    assert_eq!(
        values.get(&from_x).map(|value| value.shape.clone()),
        Some(vec![3]),
        "`total`'s root keeps its own extent",
    );
    assert_eq!(
        values.get(&from_y).map(|value| value.shape.clone()),
        Some(vec![2, 2]),
        "`use2`'s root keeps its own extent",
    );
}

/// The negative twin. One scope, two `Load` axes spelling `seq` at
/// disagreeing extents, both reachable from the SAME root: that is one claim
/// with two witnesses and it must still refuse. Scoping a claim by the
/// results it reaches must not be mistaken for dropping the equality check.
///
/// EVIDENTIARY STATUS: disposition lock. Measured GREEN on `3dc3f54f6` for
/// the same reason it must stay green afterwards, through a different
/// mechanism: today the legacy name grouping refuses it, and after the switch
/// the scoped derivation must.
#[test]
fn one_scope_with_two_disagreeing_witnesses_of_a_binder_still_refuses_on_eval() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec![named("seq")]);
    let y = load(&mut dag, "y", vec![named("seq")]);
    let sum = dag.add_node(
        RiscOp::Add,
        vec![x, y],
        ty(vec![named("seq")], Prim::F32),
        None,
    );
    dag.add_root(sum);

    let err = eval_tensor_roots_with_strict(&dag, &[sum], |name| match name {
        "x" => Some(TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0])),
        "y" => Some(TensorValue::from_vec(vec![2], vec![1.0, 2.0])),
        _ => None,
    })
    .expect_err("one claim with two disagreeing witnesses must refuse");
    assert!(
        err.contains("seq"),
        "the refusal names the claim whose witnesses disagree: {err}"
    );
}
