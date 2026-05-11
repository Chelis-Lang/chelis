//! Wave 4 / W4-A — M5(c) IR-level structured-rejection oracle.
//!
//! Complements the CLI-level oracle in
//! `crates/chelis-cli/tests/cross_library_semantic_gap_diagnostics.rs`
//! by driving the structured sparse-helper recognizer
//! (`try_summarize_sparse_helper_for_test`) on **synthetic helper
//! DAGs** that the Surf surface cannot construct today. Together the
//! two files lock every W3-B-enumerated rejection category as a
//! structured `SummaryRejection`.
//!
//! ## Categories covered here
//!
//! 1. `MultipleRoots` — synthetic two-root helper.
//! 5. `IndicesDTypeMismatch` — synthetic helper with f32 indices.
//! 6. `PayloadDTypeMismatch` — synthetic helper with f64 updates on an
//!    f32 target/output.
//!
//! ## Categories covered at the CLI level
//!
//!   2. `MultipleReturnPaths`
//!   3. `NonLoadOperand`
//!   4. `PostProcessingAfterSparseOp`
//!   7. `WildcardDim`
//!
//! ## Pattern-match contract
//!
//! Every test pattern-matches on:
//!   * the `SparseSummaryAttempt::Rejected(_)` enum variant,
//!   * the `HelperSummaryRejection::rejection_class` enum,
//!   * the `HelperSummaryRejection::detail` enum variant + structured fields.
//!
//! `contains()` on the rendered `Display` string is explicitly
//! rejected, per the W5 red-team thoroughness contract.

use chelis_ir::dag::{Dag, RiscOp};
use chelis_ir::host::{
    HelperSummaryRejection, HostTensorInput, PayloadRole, SparseOpKind, SparseSummaryAttempt,
    SummaryRejectionClass, SummaryRejectionDetail, WildcardLocation,
    try_summarize_sparse_helper_for_test,
};
use chelis_ir::{DimInfo, TensorType};
use chelis_types::types::Prim;

fn t_f32(dims: Vec<usize>) -> TensorType {
    TensorType {
        dims: dims.into_iter().map(DimInfo::Lit).collect(),
        precision: Prim::F32,
    }
}

fn t_i64(dims: Vec<usize>) -> TensorType {
    TensorType {
        dims: dims.into_iter().map(DimInfo::Lit).collect(),
        precision: Prim::Int64,
    }
}

fn input(name: &str, ty: TensorType) -> HostTensorInput {
    HostTensorInput {
        name: name.to_string(),
        ty,
    }
}

/// Unwrap a `SparseSummaryAttempt::Rejected(_)` arm and return its
/// carried `HelperSummaryRejection`. Panics with a clear message if
/// the attempt is `Ok(_)` or `NotEligible` — those represent test
/// fixture bugs.
fn expect_rejected(
    attempt: Result<chelis_ir::host::HostTensorSpecialization, SparseSummaryAttempt>,
) -> HelperSummaryRejection {
    match attempt {
        Ok(spec) => panic!("expected SparseSummaryAttempt::Rejected, got Ok({spec:?})"),
        Err(SparseSummaryAttempt::NotEligible) => {
            panic!("expected SparseSummaryAttempt::Rejected, got NotEligible")
        }
        Err(SparseSummaryAttempt::Rejected(r)) => r,
    }
}

// =========================================================================
// Category 1: MultipleRoots
// =========================================================================

#[test]
fn multiple_roots_synthetic_helper_emits_structured_rejection() {
    // Two roots: one Gather, one a sibling sum-shaped result.
    let mut dag = Dag::new();
    let values_ty = t_f32(vec![1000, 128]);
    let indices_ty = t_i64(vec![64]);
    let output_ty = t_f32(vec![64, 128]);
    let values = dag.add_node(
        RiscOp::Load {
            name: "table".into(),
        },
        vec![],
        values_ty.clone(),
        None,
    );
    let indices = dag.add_node(
        RiscOp::Load { name: "idx".into() },
        vec![],
        indices_ty.clone(),
        None,
    );
    let root_a = dag.add_node(
        RiscOp::Gather { axis: 0 },
        vec![values, indices],
        output_ty.clone(),
        None,
    );
    let root_b = dag.add_node(
        RiscOp::Gather { axis: 0 },
        vec![values, indices],
        output_ty.clone(),
        None,
    );
    dag.add_root(root_a);
    dag.add_root(root_b);

    let inputs = vec![input("table", values_ty), input("idx", indices_ty)];
    let attempt = try_summarize_sparse_helper_for_test(&dag, &inputs, &output_ty);
    let rejection = expect_rejected(attempt);

    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::MultipleRoots,
    );
    let SummaryRejectionDetail::MultipleRoots { root_count } = rejection.detail else {
        panic!(
            "expected SummaryRejectionDetail::MultipleRoots, got {:?}",
            rejection.detail,
        );
    };
    assert_eq!(root_count, 2);
}

// =========================================================================
// Category 5: IndicesDTypeMismatch
// =========================================================================

#[test]
fn indices_dtype_mismatch_gather_emits_structured_rejection() {
    // Indices precision is f32 (not int32 / int64).
    let mut dag = Dag::new();
    let values_ty = t_f32(vec![1000, 128]);
    let bogus_indices_ty = t_f32(vec![64]);
    let output_ty = t_f32(vec![64, 128]);
    let values = dag.add_node(
        RiscOp::Load {
            name: "table".into(),
        },
        vec![],
        values_ty.clone(),
        None,
    );
    let indices = dag.add_node(
        RiscOp::Load { name: "idx".into() },
        vec![],
        bogus_indices_ty.clone(),
        None,
    );
    let root = dag.add_node(
        RiscOp::Gather { axis: 0 },
        vec![values, indices],
        output_ty.clone(),
        None,
    );
    dag.add_root(root);

    let inputs = vec![input("table", values_ty), input("idx", bogus_indices_ty)];
    let attempt = try_summarize_sparse_helper_for_test(&dag, &inputs, &output_ty);
    let rejection = expect_rejected(attempt);

    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::IndicesDTypeMismatch,
    );
    let SummaryRejectionDetail::IndicesDTypeMismatch { op, observed } = rejection.detail else {
        panic!(
            "expected SummaryRejectionDetail::IndicesDTypeMismatch, got {:?}",
            rejection.detail,
        );
    };
    assert_eq!(op, SparseOpKind::Gather);
    assert_eq!(observed, Prim::F32);
}

#[test]
fn indices_dtype_mismatch_scatter_add_emits_structured_rejection() {
    // Indices precision is bool (not int32 / int64).
    let mut dag = Dag::new();
    let target_ty = t_f32(vec![10, 4]);
    let bogus_indices_ty = TensorType {
        dims: vec![DimInfo::Lit(64)],
        precision: Prim::Bool,
    };
    let updates_ty = t_f32(vec![64, 4]);
    let output_ty = target_ty.clone();
    let target = dag.add_node(
        RiscOp::Load {
            name: "base".into(),
        },
        vec![],
        target_ty.clone(),
        None,
    );
    let indices = dag.add_node(
        RiscOp::Load {
            name: "bin_ids".into(),
        },
        vec![],
        bogus_indices_ty.clone(),
        None,
    );
    let updates = dag.add_node(
        RiscOp::Load { name: "upd".into() },
        vec![],
        updates_ty.clone(),
        None,
    );
    let root = dag.add_node(
        RiscOp::ScatterAdd { axis: 0 },
        vec![target, indices, updates],
        output_ty.clone(),
        None,
    );
    dag.add_root(root);

    let inputs = vec![
        input("base", target_ty),
        input("bin_ids", bogus_indices_ty),
        input("upd", updates_ty),
    ];
    let attempt = try_summarize_sparse_helper_for_test(&dag, &inputs, &output_ty);
    let rejection = expect_rejected(attempt);

    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::IndicesDTypeMismatch,
    );
    let SummaryRejectionDetail::IndicesDTypeMismatch { op, observed } = rejection.detail else {
        panic!(
            "expected SummaryRejectionDetail::IndicesDTypeMismatch, got {:?}",
            rejection.detail,
        );
    };
    assert_eq!(op, SparseOpKind::ScatterAdd);
    assert_eq!(observed, Prim::Bool);
}

// =========================================================================
// Category 6: PayloadDTypeMismatch
// =========================================================================

#[test]
fn payload_dtype_mismatch_scatter_add_updates_emits_structured_rejection() {
    // ScatterAdd's `updates` is f64 but `target` and `output` are f32.
    // The summarizer must name `Updates` as the mismatching role.
    let mut dag = Dag::new();
    let target_ty = t_f32(vec![10, 4]);
    let indices_ty = t_i64(vec![64]);
    let bogus_updates_ty = TensorType {
        dims: vec![DimInfo::Lit(64), DimInfo::Lit(4)],
        precision: Prim::F64,
    };
    let output_ty = target_ty.clone();
    let target = dag.add_node(
        RiscOp::Load {
            name: "base".into(),
        },
        vec![],
        target_ty.clone(),
        None,
    );
    let indices = dag.add_node(
        RiscOp::Load {
            name: "bin_ids".into(),
        },
        vec![],
        indices_ty.clone(),
        None,
    );
    let updates = dag.add_node(
        RiscOp::Load { name: "upd".into() },
        vec![],
        bogus_updates_ty.clone(),
        None,
    );
    let root = dag.add_node(
        RiscOp::ScatterAdd { axis: 0 },
        vec![target, indices, updates],
        output_ty.clone(),
        None,
    );
    dag.add_root(root);

    let inputs = vec![
        input("base", target_ty),
        input("bin_ids", indices_ty),
        input("upd", bogus_updates_ty),
    ];
    let attempt = try_summarize_sparse_helper_for_test(&dag, &inputs, &output_ty);
    let rejection = expect_rejected(attempt);

    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::PayloadDTypeMismatch,
    );
    let SummaryRejectionDetail::PayloadDTypeMismatch {
        op,
        which,
        expected,
        observed,
    } = rejection.detail
    else {
        panic!(
            "expected SummaryRejectionDetail::PayloadDTypeMismatch, got {:?}",
            rejection.detail,
        );
    };
    assert_eq!(op, SparseOpKind::ScatterAdd);
    assert_eq!(which, PayloadRole::Updates);
    assert_eq!(expected, Prim::F32);
    assert_eq!(observed, Prim::F64);
}

#[test]
fn payload_dtype_mismatch_gather_values_emits_structured_rejection() {
    // Gather's `values` payload is f64 but the output is f32. The
    // summarizer must name `Values` as the mismatching role with
    // expected=f32, observed=f64.
    let mut dag = Dag::new();
    let bogus_values_ty = TensorType {
        dims: vec![DimInfo::Lit(1000), DimInfo::Lit(128)],
        precision: Prim::F64,
    };
    let indices_ty = t_i64(vec![64]);
    let output_ty = t_f32(vec![64, 128]);
    let values = dag.add_node(
        RiscOp::Load {
            name: "table".into(),
        },
        vec![],
        bogus_values_ty.clone(),
        None,
    );
    let indices = dag.add_node(
        RiscOp::Load { name: "idx".into() },
        vec![],
        indices_ty.clone(),
        None,
    );
    let root = dag.add_node(
        RiscOp::Gather { axis: 0 },
        vec![values, indices],
        output_ty.clone(),
        None,
    );
    dag.add_root(root);

    let inputs = vec![input("table", bogus_values_ty), input("idx", indices_ty)];
    let attempt = try_summarize_sparse_helper_for_test(&dag, &inputs, &output_ty);
    let rejection = expect_rejected(attempt);

    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::PayloadDTypeMismatch,
    );
    let SummaryRejectionDetail::PayloadDTypeMismatch {
        op,
        which,
        expected,
        observed,
    } = rejection.detail
    else {
        panic!(
            "expected SummaryRejectionDetail::PayloadDTypeMismatch, got {:?}",
            rejection.detail,
        );
    };
    assert_eq!(op, SparseOpKind::Gather);
    assert_eq!(which, PayloadRole::Values);
    assert_eq!(expected, Prim::F32);
    assert_eq!(observed, Prim::F64);
}

// =========================================================================
// Additional structural coverage:
//
//   Non-load operand at each operand position (gather + scatter_add)
//   Post-processing detection picks the deepest sparse op for `op`
//   Wildcard at Output vs Input(i)
// =========================================================================

#[test]
fn non_load_operand_synthetic_helper_gather_locks_operand_index() {
    // The `indices` operand of Gather is `Add(idx, idx)` — not a
    // direct Load. The summarizer must reject with NonLoadOperand at
    // operand_index = 1.
    let mut dag = Dag::new();
    let values_ty = t_f32(vec![1000, 128]);
    let indices_ty = t_i64(vec![64]);
    let output_ty = t_f32(vec![64, 128]);
    let values = dag.add_node(
        RiscOp::Load {
            name: "table".into(),
        },
        vec![],
        values_ty.clone(),
        None,
    );
    let idx = dag.add_node(
        RiscOp::Load { name: "idx".into() },
        vec![],
        indices_ty.clone(),
        None,
    );
    let doubled = dag.add_node(RiscOp::Add, vec![idx, idx], indices_ty.clone(), None);
    let root = dag.add_node(
        RiscOp::Gather { axis: 0 },
        vec![values, doubled],
        output_ty.clone(),
        None,
    );
    dag.add_root(root);

    let inputs = vec![input("table", values_ty), input("idx", indices_ty)];
    let attempt = try_summarize_sparse_helper_for_test(&dag, &inputs, &output_ty);
    let rejection = expect_rejected(attempt);

    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::NonLoadOperand,
    );
    let SummaryRejectionDetail::NonLoadOperand { op, operand_index } = rejection.detail else {
        panic!(
            "expected SummaryRejectionDetail::NonLoadOperand, got {:?}",
            rejection.detail,
        );
    };
    assert_eq!(op, SparseOpKind::Gather);
    assert_eq!(operand_index, 1);
}

#[test]
fn post_processing_after_sparse_op_synthetic_helper_locks_tail_op_name() {
    // Helper post-processes ScatterAdd with an elementwise add.
    let mut dag = Dag::new();
    let target_ty = t_f32(vec![10, 4]);
    let indices_ty = t_i64(vec![64]);
    let updates_ty = t_f32(vec![64, 4]);
    let zero_ty = target_ty.clone();
    let output_ty = target_ty.clone();
    let target = dag.add_node(
        RiscOp::Load {
            name: "base".into(),
        },
        vec![],
        target_ty.clone(),
        None,
    );
    let indices = dag.add_node(
        RiscOp::Load {
            name: "bin_ids".into(),
        },
        vec![],
        indices_ty.clone(),
        None,
    );
    let updates = dag.add_node(
        RiscOp::Load { name: "upd".into() },
        vec![],
        updates_ty.clone(),
        None,
    );
    let zero = dag.add_node(
        RiscOp::Load {
            name: "zero".into(),
        },
        vec![],
        zero_ty.clone(),
        None,
    );
    let scattered = dag.add_node(
        RiscOp::ScatterAdd { axis: 0 },
        vec![target, indices, updates],
        target_ty.clone(),
        None,
    );
    let root = dag.add_node(RiscOp::Add, vec![scattered, zero], output_ty.clone(), None);
    dag.add_root(root);

    let inputs = vec![
        input("base", target_ty),
        input("bin_ids", indices_ty),
        input("upd", updates_ty),
        input("zero", zero_ty),
    ];
    let attempt = try_summarize_sparse_helper_for_test(&dag, &inputs, &output_ty);
    let rejection = expect_rejected(attempt);

    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::PostProcessingAfterSparseOp,
    );
    let SummaryRejectionDetail::PostProcessingAfterSparseOp { op, tail_op } = rejection.detail
    else {
        panic!(
            "expected SummaryRejectionDetail::PostProcessingAfterSparseOp, got {:?}",
            rejection.detail,
        );
    };
    assert_eq!(op, SparseOpKind::ScatterAdd);
    assert_eq!(tail_op, "add");
}

#[test]
fn wildcard_dim_synthetic_helper_locks_output_location() {
    // Output carries a wildcard `Named("*", None)` dim.
    let mut dag = Dag::new();
    let values_ty = t_f32(vec![1000, 128]);
    let indices_ty = t_i64(vec![64]);
    let output_ty = TensorType {
        dims: vec![DimInfo::Lit(64), DimInfo::Named("*".to_string(), None)],
        precision: Prim::F32,
    };
    let values = dag.add_node(
        RiscOp::Load {
            name: "table".into(),
        },
        vec![],
        values_ty.clone(),
        None,
    );
    let indices = dag.add_node(
        RiscOp::Load { name: "idx".into() },
        vec![],
        indices_ty.clone(),
        None,
    );
    let root = dag.add_node(
        RiscOp::Gather { axis: 0 },
        vec![values, indices],
        output_ty.clone(),
        None,
    );
    dag.add_root(root);

    let inputs = vec![input("table", values_ty), input("idx", indices_ty)];
    let attempt = try_summarize_sparse_helper_for_test(&dag, &inputs, &output_ty);
    let rejection = expect_rejected(attempt);

    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::WildcardDim,
    );
    let SummaryRejectionDetail::WildcardDim { location } = rejection.detail else {
        panic!(
            "expected SummaryRejectionDetail::WildcardDim, got {:?}",
            rejection.detail,
        );
    };
    assert_eq!(location, WildcardLocation::Output);
}

#[test]
fn wildcard_dim_synthetic_helper_locks_input_location() {
    // Input[0] (values) carries a wildcard dim.
    let mut dag = Dag::new();
    let wildcard_values_ty = TensorType {
        dims: vec![
            DimInfo::Named("*".to_string(), None),
            DimInfo::Named("*".to_string(), None),
        ],
        precision: Prim::F32,
    };
    let indices_ty = t_i64(vec![64]);
    let output_ty = t_f32(vec![64, 128]);
    let values = dag.add_node(
        RiscOp::Load {
            name: "table".into(),
        },
        vec![],
        wildcard_values_ty.clone(),
        None,
    );
    let indices = dag.add_node(
        RiscOp::Load { name: "idx".into() },
        vec![],
        indices_ty.clone(),
        None,
    );
    let root = dag.add_node(
        RiscOp::Gather { axis: 0 },
        vec![values, indices],
        output_ty.clone(),
        None,
    );
    dag.add_root(root);

    let inputs = vec![input("table", wildcard_values_ty), input("idx", indices_ty)];
    let attempt = try_summarize_sparse_helper_for_test(&dag, &inputs, &output_ty);
    let rejection = expect_rejected(attempt);

    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::WildcardDim,
    );
    let SummaryRejectionDetail::WildcardDim { location } = rejection.detail else {
        panic!(
            "expected SummaryRejectionDetail::WildcardDim, got {:?}",
            rejection.detail,
        );
    };
    // Output is checked first; if neither output nor any input has a
    // wildcard, location is None. Here output is concrete and inputs[0]
    // has wildcards — the recognizer must report Input(0).
    assert_eq!(location, WildcardLocation::Input(0));
}

// =========================================================================
// Negative: NotEligible vs Rejected
//
// A helper DAG that contains no sparse op at all (e.g. a single
// elementwise Add) must NOT produce a rejection — it returns
// `NotEligible`. The CLI then falls back to BLAS recognition / no
// specialization without emitting any diagnostic.
// =========================================================================

#[test]
fn non_sparse_helper_dag_returns_not_eligible_not_rejected() {
    let mut dag = Dag::new();
    let in_ty = t_f32(vec![4, 4]);
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        in_ty.clone(),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        in_ty.clone(),
        None,
    );
    let root = dag.add_node(RiscOp::Add, vec![a, b], in_ty.clone(), None);
    dag.add_root(root);

    let inputs = vec![input("a", in_ty.clone()), input("b", in_ty.clone())];
    let attempt = try_summarize_sparse_helper_for_test(&dag, &inputs, &in_ty);
    match attempt {
        Err(SparseSummaryAttempt::NotEligible) => {}
        Err(SparseSummaryAttempt::Rejected(r)) => {
            panic!("non-sparse helper must be NotEligible, not Rejected({r:?})")
        }
        Ok(spec) => panic!("non-sparse helper must NOT be summarized; got {spec:?}"),
    }
}

// =========================================================================
// HelperSummaryRejection structural sanity
// =========================================================================

#[test]
fn rejection_helper_body_span_is_threaded_through_when_present() {
    // Construct a Gather whose root node carries a span_id. The
    // recognizer must surface it as `helper_body_span`.
    let mut dag = Dag::new();
    let values_ty = t_f32(vec![1000, 128]);
    let indices_ty = t_i64(vec![64]);
    let output_ty = t_f32(vec![64, 128]);
    let values = dag.add_node(
        RiscOp::Load {
            name: "table".into(),
        },
        vec![],
        values_ty.clone(),
        None,
    );
    let indices = dag.add_node(
        RiscOp::Load { name: "idx".into() },
        vec![],
        indices_ty.clone(),
        None,
    );
    let root = dag.add_node(
        RiscOp::Gather { axis: 0 },
        vec![values, indices],
        output_ty.clone(),
        Some("surf:42..99".to_string()),
    );
    dag.add_root(root);
    // Now add a sibling root so the multi-root rejection fires; the
    // helper body span should still be the deepest sparse-node's
    // span_id (the Gather's `surf:42..99`).
    let root_b = dag.add_node(
        RiscOp::Gather { axis: 0 },
        vec![values, indices],
        output_ty.clone(),
        None,
    );
    dag.add_root(root_b);

    let inputs = vec![input("table", values_ty), input("idx", indices_ty)];
    let attempt = try_summarize_sparse_helper_for_test(&dag, &inputs, &output_ty);
    let rejection = expect_rejected(attempt);
    assert_eq!(
        rejection.rejection_class,
        SummaryRejectionClass::MultipleRoots,
    );
    assert_eq!(
        rejection.helper_body_span.as_deref(),
        Some("surf:42..99"),
        "helper_body_span must be threaded through when the helper root carries one",
    );
}
