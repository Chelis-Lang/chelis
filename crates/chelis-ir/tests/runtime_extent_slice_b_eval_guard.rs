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
    let _x = load(&mut dag, "x", vec![named("n")]);
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
    dag.add_root(expanded);
    (dag, expanded)
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
