//! IR-level lock for the W3-B compiler-derived host sparse summary
//! recognizer.
//!
//! The cli-level `cross_library_sparse_summaries.rs` test suite locks
//! the C codegen consequences for the two sparse ops a Surf author
//! can reach today (`gather` and `scatter_replace`). `RiscOp::ScatterAdd`
//! has no Surf surface path — it is produced exclusively by the AD
//! adjoint of `gather` (see `crates/chelis-ir/src/grad.rs`'s
//! `gather` adjoint). This file fills the surface-coverage gap by
//! driving the recognizer directly on a synthetic `ScatterAdd` helper
//! DAG, so the W3-B contract for all three sparse ops is locked.
//!
//! Both positive and negative `ScatterAdd` recognition cases are
//! pinned here:
//!   * a single `ScatterAdd` whose operands are direct Loads recognizes
//!   * a helper with a post-processing Add on the result rejects
//!   * a helper with mismatched indices precision rejects
//!   * a helper with mismatched payload precision rejects
//!
//! The Gather and Scatter-replace positive cases are also exercised
//! here mechanically (in addition to the cli-level structural checks)
//! to keep the per-op invariants symmetric in the IR test surface.

use chelis_ir::dag::{Dag, RiscOp};
use chelis_ir::host::{
    HostSparseOpSummary, HostTensorInput, HostTensorSpecialization,
    summarize_sparse_helper_for_test,
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

fn t_i32(dims: Vec<usize>) -> TensorType {
    TensorType {
        dims: dims.into_iter().map(DimInfo::Lit).collect(),
        precision: Prim::Int32,
    }
}

fn input(name: &str, ty: TensorType) -> HostTensorInput {
    HostTensorInput {
        name: name.to_string(),
        ty,
    }
}

// =========================================================================
// Positive recognition: all three sparse ops
// =========================================================================

#[test]
fn gather_helper_with_load_operands_is_summarized() {
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
        None,
    );
    dag.add_root(root);

    let inputs = vec![input("table", values_ty), input("idx", indices_ty)];
    let summary = summarize_sparse_helper_for_test(&dag, &inputs, &output_ty);
    let Some(HostTensorSpecialization::SparseGather(HostSparseOpSummary {
        axis,
        input_indices,
        ..
    })) = summary
    else {
        panic!("expected SparseGather summary, got {summary:?}");
    };
    assert_eq!(axis, 0);
    // input_indices = [values_pos, indices_pos]
    assert_eq!(input_indices, vec![0, 1]);
}

#[test]
fn scatter_add_helper_with_load_operands_is_summarized() {
    // ScatterAdd is the AD-only sparse op. Constructing the helper
    // DAG mechanically locks the W3-B contract that the summary
    // recognizer covers all three sparse RiscOps symmetrically, not
    // just the two with Surf-reachable surface forms.
    let mut dag = Dag::new();
    let target_ty = t_f32(vec![10, 4]);
    let indices_ty = t_i64(vec![64]);
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
        indices_ty.clone(),
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
        input("bin_ids", indices_ty),
        input("upd", updates_ty),
    ];
    let summary = summarize_sparse_helper_for_test(&dag, &inputs, &output_ty);
    let Some(HostTensorSpecialization::SparseScatterAdd(HostSparseOpSummary {
        axis,
        input_indices,
        ..
    })) = summary
    else {
        panic!("expected SparseScatterAdd summary, got {summary:?}");
    };
    assert_eq!(axis, 0);
    assert_eq!(input_indices, vec![0, 1, 2]);
}

#[test]
fn scatter_replace_helper_with_load_operands_is_summarized() {
    let mut dag = Dag::new();
    let target_ty = t_f32(vec![3, 2]);
    let indices_ty = t_i32(vec![4]);
    let updates_ty = t_f32(vec![4, 2]);
    let output_ty = target_ty.clone();
    let target = dag.add_node(
        RiscOp::Load {
            name: "table".into(),
        },
        vec![],
        target_ty.clone(),
        None,
    );
    let indices = dag.add_node(
        RiscOp::Load { name: "idx".into() },
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
    let root = dag.add_node(
        RiscOp::Scatter { axis: 0 },
        vec![target, indices, updates],
        output_ty.clone(),
        None,
    );
    dag.add_root(root);

    let inputs = vec![
        input("table", target_ty),
        input("idx", indices_ty),
        input("upd", updates_ty),
    ];
    let summary = summarize_sparse_helper_for_test(&dag, &inputs, &output_ty);
    assert!(
        matches!(
            summary,
            Some(HostTensorSpecialization::SparseScatterReplace(_))
        ),
        "expected SparseScatterReplace summary, got {summary:?}"
    );
}

// =========================================================================
// Negative recognition: each rejection class is locked here
// =========================================================================

#[test]
fn helper_with_post_processing_add_after_scatter_add_is_rejected() {
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
    let summary = summarize_sparse_helper_for_test(&dag, &inputs, &output_ty);
    assert!(
        summary.is_none(),
        "post-processed scatter_add helper MUST NOT be summarized; got {summary:?}"
    );
}

#[test]
fn helper_with_non_load_operand_for_scatter_add_is_rejected() {
    // The `updates` operand of ScatterAdd is `Add(updates_in,
    // updates_in)` — not a direct Load. The summarizer requires
    // each sparse-op operand to be a Load referencing a helper
    // input.
    let mut dag = Dag::new();
    let target_ty = t_f32(vec![10, 4]);
    let indices_ty = t_i64(vec![64]);
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
        indices_ty.clone(),
        None,
    );
    let updates_in = dag.add_node(
        RiscOp::Load { name: "upd".into() },
        vec![],
        updates_ty.clone(),
        None,
    );
    let doubled = dag.add_node(
        RiscOp::Add,
        vec![updates_in, updates_in],
        updates_ty.clone(),
        None,
    );
    let root = dag.add_node(
        RiscOp::ScatterAdd { axis: 0 },
        vec![target, indices, doubled],
        output_ty.clone(),
        None,
    );
    dag.add_root(root);

    let inputs = vec![
        input("base", target_ty),
        input("bin_ids", indices_ty),
        input("upd", updates_ty),
    ];
    let summary = summarize_sparse_helper_for_test(&dag, &inputs, &output_ty);
    assert!(
        summary.is_none(),
        "scatter_add helper with non-Load operand MUST NOT be summarized; got {summary:?}"
    );
}

#[test]
fn helper_with_mismatched_indices_precision_is_rejected() {
    // Indices must be int32 or int64. An f32 "indices" load should
    // reject (this is not reachable from a well-typed Surf program,
    // but is locked here against ad-hoc DAG construction).
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
    let summary = summarize_sparse_helper_for_test(&dag, &inputs, &output_ty);
    assert!(
        summary.is_none(),
        "gather helper with f32 indices MUST NOT be summarized; got {summary:?}"
    );
}

#[test]
fn helper_with_mismatched_payload_precision_is_rejected() {
    // ScatterAdd's `updates` is f64 but `target` is f32. The
    // summarizer requires all payload precisions to match.
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
    let summary = summarize_sparse_helper_for_test(&dag, &inputs, &output_ty);
    assert!(
        summary.is_none(),
        "scatter_add helper with mismatched updates precision MUST NOT be summarized; got {summary:?}"
    );
}

#[test]
fn helper_with_wildcard_dim_is_rejected() {
    // A wildcard Named("*", None) dim is a type-inference
    // placeholder. All wildcards across a helper share the same
    // string name; the recognizer rejects so contract assertions
    // do not collapse distinct unknown axes onto a single symbol.
    let mut dag = Dag::new();
    let wildcard_values_ty = TensorType {
        dims: vec![
            DimInfo::Named("*".to_string(), None),
            DimInfo::Named("*".to_string(), None),
        ],
        precision: Prim::F32,
    };
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
    let summary = summarize_sparse_helper_for_test(&dag, &inputs, &output_ty);
    assert!(
        summary.is_none(),
        "gather helper with wildcard dims MUST NOT be summarized; got {summary:?}"
    );
}

#[test]
fn helper_with_multiple_roots_is_rejected() {
    // The summarizer requires exactly one DAG root. A two-output
    // helper (e.g. compute both gather and a sibling sum) cannot be
    // condensed into a single sparse-op summary.
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
    let summary = summarize_sparse_helper_for_test(&dag, &inputs, &output_ty);
    assert!(
        summary.is_none(),
        "multi-root helper MUST NOT be summarized; got {summary:?}"
    );
}
