//! Contract test for the first-class replace-scatter (last-write-wins)
//! sparse op `RiscOp::Scatter { axis }` introduced by W2-A (M4-residual).
//!
//! Three-part oracle, enforced by this file:
//!
//! 1. **Forward semantics.** Last-write-wins over duplicate target
//!    indices, with the deterministic order spec-defined in
//!    `spec/05-risc-primitives.md` §3.5 (updates-tensor row-major flat
//!    iteration). Duplicate indices in the index tensor cause the
//!    write with the larger flat index in `updates` to be the final
//!    value at the target cell. This is structurally distinct from
//!    `scatter_add`'s commutative accumulation.
//!
//! 2. **AD policy is fail-closed with a structured error.** Reverse-
//!    mode AD over `RiscOp::Scatter` must return exactly
//!    `AdError::NotSupported { op: "scatter_replace", reason:
//!    AdRejectionReason::NonDeterministicAtDuplicateIndices }`. The
//!    test asserts this by pattern-matching on the enum variant —
//!    NOT by `.contains()` on the rendered `Display` string. This is
//!    the contract downstream consumers (compiler-api, examples,
//!    cli) pattern-match against, so a silent regression to free-text
//!    rejection or to silent-zero adjoints would re-arm the
//!    "naive backward drops gradients" failure mode that motivated
//!    the linearity feature in the first place.
//!
//! 3. **`ScatterAdd`'s AD path is unchanged.** A regression assertion
//!    that `RiscOp::ScatterAdd` still produces an accumulating
//!    adjoint via `Gather` — the existing `grad_gather_contract.rs`
//!    coverage of duplicate-index accumulation must not regress
//!    because this branch added a sibling op.

use chelis_unord::UnordMap;

use chelis_ir::dag::{Dag, RiscOp};
use chelis_ir::eval::{TensorValue, eval_tensor_with};
use chelis_ir::grad::{AdError, AdRejectionReason, grad_dag, grad_dag_checked};
use chelis_ir::{DimInfo, TensorType};
use chelis_types::types::Prim;

fn t(dims: Vec<usize>) -> TensorType {
    TensorType {
        dims: dims.into_iter().map(DimInfo::Lit).collect(),
        precision: Prim::F32,
    }
}

fn t_i32(dims: Vec<usize>) -> TensorType {
    TensorType {
        dims: dims.into_iter().map(DimInfo::Lit).collect(),
        precision: Prim::Int32,
    }
}

/// Part (a): forward semantics. Last-write-wins over duplicate
/// indices, deterministic-order: updates-tensor row-major flat
/// iteration ⇒ the largest flat index in `updates` is the final value
/// at any colliding target cell.
#[test]
fn scatter_replace_forward_last_write_wins_with_deterministic_order() {
    // target: shape [3, 2], initialized via Const to all-zero with the
    // appropriate dtype.
    let mut dag = Dag::new();
    let target = dag.add_node(
        RiscOp::Load {
            name: "target".into(),
        },
        vec![],
        t(vec![3, 2]),
        None,
    );
    // indices: shape [4], all pointing to row 1 — the "all-duplicate"
    // stress case. With deterministic-order updates-flat iteration,
    // updates[3, :] is the last write and wins per cell.
    let indices = dag.add_node(
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        t_i32(vec![4]),
        None,
    );
    // updates: shape [4, 2] — four 2-element rows, all going to row 1.
    let updates = dag.add_node(
        RiscOp::Load {
            name: "updates".into(),
        },
        vec![],
        t(vec![4, 2]),
        None,
    );
    let scatter = dag.add_node(
        RiscOp::Scatter { axis: 0 },
        vec![target, indices, updates],
        t(vec![3, 2]),
        None,
    );

    let fwd_errs = chelis_ir::verify::verify(&dag);
    assert!(
        fwd_errs.is_empty(),
        "forward DAG verification errors: {fwd_errs:?}"
    );

    let mut inputs: UnordMap<String, TensorValue> = UnordMap::new();
    inputs.insert(
        "target".to_string(),
        TensorValue::from_vec(vec![3, 2], vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0]),
    );
    inputs.insert(
        "indices".to_string(),
        TensorValue::from_vec(vec![4], vec![1.0, 1.0, 1.0, 1.0]),
    );
    inputs.insert(
        "updates".to_string(),
        // Row 0: [10, 11]
        // Row 1: [20, 21]
        // Row 2: [30, 31]
        // Row 3: [40, 41] ← LAST in flat order, must win.
        TensorValue::from_vec(
            vec![4, 2],
            vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0],
        ),
    );
    let vals = eval_tensor_with(&dag, |n| inputs.get(n).cloned()).expect("scatter_replace eval");
    let out = &vals[&scatter];

    // target was [[100,200],[300,400],[500,600]]; row 0 + row 2
    // untouched, row 1 overwritten by updates[3, :] = [40, 41].
    assert_eq!(out.shape, vec![3, 2]);
    let expected = [100.0, 200.0, 40.0, 41.0, 500.0, 600.0];
    for (i, want) in expected.iter().enumerate() {
        assert!(
            (out.to_f64_lossy_vec()[i] - want).abs() < 1e-9,
            "scatter_replace last-write-wins at flat index {i}: \
             expected {want}, got {} (deterministic-order rule violated)",
            out.to_f64_lossy_vec()[i]
        );
    }
}

/// Part (a) — companion: duplicate-index resolution depends ONLY on
/// updates-flat-order, not on values in the data, by construction. A
/// distinct-indices stress case keeps each updates row contributing
/// to a different target row — no collision — and the test confirms
/// every cell is overwritten exactly once.
#[test]
fn scatter_replace_forward_distinct_indices_writes_each_cell_once() {
    let mut dag = Dag::new();
    let target = dag.add_node(
        RiscOp::Load {
            name: "target".into(),
        },
        vec![],
        t(vec![3, 2]),
        None,
    );
    let indices = dag.add_node(
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        t_i32(vec![3]),
        None,
    );
    let updates = dag.add_node(
        RiscOp::Load {
            name: "updates".into(),
        },
        vec![],
        t(vec![3, 2]),
        None,
    );
    let scatter = dag.add_node(
        RiscOp::Scatter { axis: 0 },
        vec![target, indices, updates],
        t(vec![3, 2]),
        None,
    );

    let mut inputs: UnordMap<String, TensorValue> = UnordMap::new();
    inputs.insert(
        "target".to_string(),
        TensorValue::from_vec(vec![3, 2], vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0]),
    );
    // 2 → 0 → 1: row order in target after scatter is updates[1] /
    // updates[2] / updates[0].
    inputs.insert(
        "indices".to_string(),
        TensorValue::from_vec(vec![3], vec![2.0, 0.0, 1.0]),
    );
    inputs.insert(
        "updates".to_string(),
        TensorValue::from_vec(vec![3, 2], vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0]),
    );
    let vals = eval_tensor_with(&dag, |n| inputs.get(n).cloned()).expect("scatter_replace eval");
    let out = &vals[&scatter];

    // updates[0] → target[2,:], updates[1] → target[0,:], updates[2] → target[1,:]
    let expected = [20.0, 21.0, 30.0, 31.0, 10.0, 11.0];
    for (i, want) in expected.iter().enumerate() {
        assert!(
            (out.to_f64_lossy_vec()[i] - want).abs() < 1e-9,
            "scatter_replace distinct-indices at flat {i}: expected {want}, got {}",
            out.to_f64_lossy_vec()[i]
        );
    }
}

/// Part (b): AD over `Scatter` must return exactly
/// `AdError::NotSupported { op: "scatter_replace", reason:
/// AdRejectionReason::NonDeterministicAtDuplicateIndices }`. The
/// assertion is a structural pattern-match on the enum variant +
/// fields. NOT `.contains()` on a rendered string.
#[test]
fn scatter_replace_ad_returns_structured_not_supported_error() {
    let mut dag = Dag::new();
    let target = dag.add_node(
        RiscOp::Load {
            name: "target".into(),
        },
        vec![],
        t(vec![3, 2]),
        None,
    );
    let indices = dag.add_node(
        RiscOp::synth_const(t_i32(vec![2]).precision, 0.0),
        vec![],
        t_i32(vec![2]),
        None,
    );
    let updates = dag.add_node(
        RiscOp::Load {
            name: "updates".into(),
        },
        vec![],
        t(vec![2, 2]),
        None,
    );
    let scatter = dag.add_node(
        RiscOp::Scatter { axis: 0 },
        vec![target, indices, updates],
        t(vec![3, 2]),
        None,
    );
    // Collapse to scalar so grad_dag_checked has a scalar-float output.
    let s1 = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![scatter],
        t(vec![2]),
        None,
    );
    let out = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![s1],
        TensorType::scalar_f32(),
        None,
    );

    let result = grad_dag_checked(&dag, out, &[target, updates]);
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!(
            "AD over RiscOp::Scatter must fail-closed with \
             AdError::NotSupported; got Ok(_)"
        ),
    };

    // Structural pattern match on the enum variant + fields.
    // Downstream consumers (compiler-api, cli) pattern-match on
    // exactly this shape; weakening it to a `.contains()` check on
    // the rendered string would silently break their integration.
    assert!(
        matches!(
            err,
            AdError::NotSupported {
                op: "scatter_replace",
                reason: AdRejectionReason::NonDeterministicAtDuplicateIndices,
            }
        ),
        "AD must return exactly AdError::NotSupported {{ \
         op: \"scatter_replace\", reason: NonDeterministicAtDuplicateIndices \
         }}; got: {err:?}"
    );

    // Defense in depth: also assert via destructuring (not contains()).
    match err {
        AdError::NotSupported { op, reason } => {
            assert_eq!(op, "scatter_replace");
            assert_eq!(
                reason,
                AdRejectionReason::NonDeterministicAtDuplicateIndices
            );
        }
    }
}

/// Part (b) — companion: the legacy unchecked `grad_dag` entry point
/// must also fail-closed (return `None`) rather than synthesize a
/// silent-zero adjoint. This locks down the
/// `feedback_ad_reduction_pitfalls.md` regression class:
/// silent-zero AD drops are the recurring failure pattern.
#[test]
fn scatter_replace_unchecked_grad_dag_returns_none_not_silent_zero() {
    let mut dag = Dag::new();
    let target = dag.add_node(
        RiscOp::Load {
            name: "target".into(),
        },
        vec![],
        t(vec![3, 2]),
        None,
    );
    let indices = dag.add_node(
        RiscOp::synth_const(t_i32(vec![2]).precision, 0.0),
        vec![],
        t_i32(vec![2]),
        None,
    );
    let updates = dag.add_node(
        RiscOp::Load {
            name: "updates".into(),
        },
        vec![],
        t(vec![2, 2]),
        None,
    );
    let scatter = dag.add_node(
        RiscOp::Scatter { axis: 0 },
        vec![target, indices, updates],
        t(vec![3, 2]),
        None,
    );
    let s1 = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![scatter],
        t(vec![2]),
        None,
    );
    let out = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![s1],
        TensorType::scalar_f32(),
        None,
    );

    let result = grad_dag(&dag, out, &[target, updates]);
    assert!(
        result.is_none(),
        "grad_dag (unchecked) over RiscOp::Scatter must return None, \
         not synthesize a silent-zero adjoint"
    );
}

/// Part (c): the `ScatterAdd` AD path is unchanged. Mirrors the
/// existing `first_class_gather_adjoint_scatter_add_accumulates_duplicate_indices`
/// in `grad_gather_contract.rs` but lives here to lock the
/// regression: introducing the sibling `RiscOp::Scatter` must not
/// affect `RiscOp::ScatterAdd`'s adjoint or its accumulating
/// behavior. A duplicate-index stress through `Gather` (whose
/// adjoint IS `ScatterAdd`) must still accumulate.
#[test]
fn scatter_add_ad_path_unchanged_after_scatter_landed() {
    let mut dag = Dag::new();
    let table = dag.add_node(
        RiscOp::Load {
            name: "table".into(),
        },
        vec![],
        t(vec![2, 2]),
        None,
    );
    // All indices point to row 0 — three duplicate consumers of
    // table[0, :]. The gather-then-sum scalar gradient should
    // accumulate to dtable[0, *] = 3, dtable[1, *] = 0.
    let indices = dag.add_node(
        RiscOp::synth_const(t_i32(vec![3]).precision, 0.0),
        vec![],
        t_i32(vec![3]),
        None,
    );
    let gathered = dag.add_node(
        RiscOp::Gather { axis: 0 },
        vec![table, indices],
        t(vec![3, 2]),
        None,
    );
    let s1 = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![gathered],
        t(vec![2]),
        None,
    );
    let out = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![s1],
        TensorType::scalar_f32(),
        None,
    );

    let grad = grad_dag_checked(&dag, out, &[table])
        .expect("ScatterAdd AD path must remain intact (Gather adjoint is ScatterAdd)");
    let grad_node = grad.grad_nodes[&table];

    // Inspect the constructed adjoint DAG: it must contain at least
    // one RiscOp::ScatterAdd node. Without this assertion, a
    // regression that quietly retargeted the Gather adjoint to
    // something else would still produce numerically correct values
    // for this specific test but would silently bypass the
    // accumulating contract.
    let adj_dag = &grad.dag;
    let has_scatter_add = adj_dag
        .nodes()
        .iter()
        .any(|n| matches!(n.op, RiscOp::ScatterAdd { .. }));
    assert!(
        has_scatter_add,
        "Gather's adjoint must remain RiscOp::ScatterAdd; \
         this regression test guards against W2-A inadvertently \
         retargeting the adjoint to the new Scatter (replace) op"
    );

    // Also: ScatterAdd nodes must NOT be RiscOp::Scatter — the two
    // are intentionally distinct.
    let has_replace_scatter = adj_dag
        .nodes()
        .iter()
        .any(|n| matches!(n.op, RiscOp::Scatter { .. }));
    assert!(
        !has_replace_scatter,
        "Gather's adjoint must NOT use RiscOp::Scatter (replace-scatter); \
         that op has no AD adjoint by design. Got a Scatter node in the \
         backward DAG."
    );

    // Verify the duplicate-index accumulation contract numerically.
    let mut inputs: UnordMap<String, TensorValue> = UnordMap::new();
    inputs.insert(
        "table".to_string(),
        TensorValue::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
    );
    let vals =
        eval_tensor_with(&grad.dag, |n| inputs.get(n).cloned()).expect("backward eval succeeded");
    let dtable = &vals[&grad_node];
    assert_eq!(dtable.shape, vec![2, 2]);
    let expected = [3.0, 3.0, 0.0, 0.0];
    for (i, want) in expected.iter().enumerate() {
        assert!(
            (dtable.to_f64_lossy_vec()[i] - want).abs() < 1e-6,
            "ScatterAdd accumulation regressed: expected dtable[{i}]={want}, got {}",
            dtable.to_f64_lossy_vec()[i]
        );
    }
}

/// Part (c) — companion: the verifier accepts `RiscOp::Scatter` with
/// well-formed shapes and rejects axis-out-of-bounds. Mirrors the
/// existing ScatterAdd verifier coverage.
#[test]
fn scatter_replace_verifier_rejects_out_of_bounds_axis() {
    let mut dag = Dag::new();
    let target = dag.add_node(
        RiscOp::Load {
            name: "target".into(),
        },
        vec![],
        t(vec![3, 2]),
        None,
    );
    let indices = dag.add_node(
        RiscOp::synth_const(t_i32(vec![2]).precision, 0.0),
        vec![],
        t_i32(vec![2]),
        None,
    );
    let updates = dag.add_node(
        RiscOp::Load {
            name: "updates".into(),
        },
        vec![],
        t(vec![2, 2]),
        None,
    );
    // axis: 7 — out of bounds for rank-2 target. Note: output type
    // must still equal target (verifier checks both axis AND output
    // shape; we want the axis error to fire and not get masked).
    let _scatter = dag.add_node(
        RiscOp::Scatter { axis: 7 },
        vec![target, indices, updates],
        t(vec![3, 2]),
        None,
    );

    let errs = chelis_ir::verify::verify(&dag);
    assert!(
        errs.iter()
            .any(|e| e.contains("scatter_replace") && e.contains("out of bounds")),
        "verifier must reject scatter_replace with out-of-bounds axis; got: {errs:?}"
    );
}
