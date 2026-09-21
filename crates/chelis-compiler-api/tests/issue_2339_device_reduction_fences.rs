use chelis_compiler_api::compiler::{
    BuildTarget, CompilerError, reject_inexact_device_reduction_cells, reject_unsupported_hip_ops,
    reject_unsupported_metal_ops,
};
use chelis_ir::{
    dag::{Dag, DimInfo, ReduceWindowKind, RiscOp, TensorType},
    grad::grad_dag_checked,
    tier2::lower_mean,
};
use chelis_types::{
    types::Prim,
    unsupported::{RejectionAuthorityKind, Stage, UnsupportedKind},
};

fn ty(dims: &[usize], precision: Prim) -> TensorType {
    TensorType {
        dims: dims.iter().copied().map(DimInfo::Lit).collect(),
        precision,
    }
}

fn mean_dag() -> Dag {
    let mut dag = Dag::new();
    let input = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        ty(&[4], Prim::F32),
        Some("surf:mean".into()),
    );
    let output = lower_mean(&mut dag, input, 0, &ty(&[4], Prim::F32), Some("surf:mean"));
    dag.add_root(output);
    dag
}

fn simple_reduction_dag(op: RiscOp, output_precision: Prim) -> Dag {
    let mut dag = Dag::new();
    let input = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        ty(&[4], Prim::F32),
        Some("surf:reduction".into()),
    );
    let output = dag.add_node(
        op,
        vec![input],
        ty(&[], output_precision),
        Some("surf:reduction".into()),
    );
    dag.add_root(output);
    dag
}

fn window_dag(reducer: ReduceWindowKind) -> Dag {
    let mut dag = Dag::new();
    let input = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        ty(&[4], Prim::F32),
        Some("surf:window".into()),
    );
    let output = dag.add_node(
        RiscOp::ReduceWindow {
            reducer,
            window_shape: vec![2],
            strides: vec![1],
        },
        vec![input],
        ty(&[3], Prim::F32),
        Some("surf:window".into()),
    );
    dag.add_root(output);
    dag
}

fn window_grad_dag(reducer: ReduceWindowKind) -> Dag {
    let mut dag = Dag::new();
    let input = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        ty(&[4], Prim::F32),
        Some("surf:window-grad".into()),
    );
    let cotangent = dag.add_node(
        RiscOp::Load { name: "g".into() },
        vec![],
        ty(&[3], Prim::F32),
        Some("surf:window-grad".into()),
    );
    let output = dag.add_node(
        RiscOp::ReduceWindowGrad {
            reducer,
            window_shape: vec![2],
            strides: vec![1],
        },
        vec![input, cotangent],
        ty(&[4], Prim::F32),
        Some("surf:window-grad".into()),
    );
    dag.add_root(output);
    dag
}

fn reduction_cases() -> Vec<(&'static str, Dag)> {
    vec![
        ("mean", mean_dag()),
        (
            "max_reduce",
            simple_reduction_dag(RiscOp::MaxReduce { axis: 0 }, Prim::F32),
        ),
        (
            "min_reduce",
            simple_reduction_dag(RiscOp::MinReduce { axis: 0 }, Prim::F32),
        ),
        (
            "argmax_reduce",
            simple_reduction_dag(RiscOp::Argmax { axis: 0 }, Prim::Int64),
        ),
        (
            "argmin_reduce",
            simple_reduction_dag(RiscOp::Argmin { axis: 0 }, Prim::Int64),
        ),
        ("reduce_window_max", window_dag(ReduceWindowKind::Max)),
        ("reduce_window_min", window_dag(ReduceWindowKind::Min)),
        (
            "reduce_window_max adjoint",
            window_grad_dag(ReduceWindowKind::Max),
        ),
        (
            "reduce_window_min adjoint",
            window_grad_dag(ReduceWindowKind::Min),
        ),
    ]
}

fn assert_issue_2339_receipt(
    target: &'static str,
    expected_operation: &str,
    error: CompilerError,
) {
    let diagnostic = error.errors.first().expect("one typed diagnostic");
    assert_eq!(
        diagnostic.kind(),
        chelis_vocab::DiagnosticKind::UnsupportedFeature
    );
    let identity = diagnostic
        .unsupported_identity()
        .expect("device fences must preserve typed unsupported metadata");
    assert_eq!(identity.brand, "unsupported:");
    assert_eq!(identity.kind.as_str(), "unsupported_feature");
    assert_eq!(identity.payload.stage, Stage::Codegen(target));
    assert_eq!(
        identity.payload.context,
        format!("`chelis build --target {target}` early capability gate")
    );
    assert_eq!(
        identity.payload.disposition,
        RejectionAuthorityKind::Unimplemented
    );
    assert_eq!(identity.payload.atom, None);
    assert_eq!(
        identity.payload.tracking_issue.map(|issue| issue.number()),
        Some(2339)
    );
    match &identity.payload.what {
        UnsupportedKind::Construct(message) => assert!(
            message.contains(expected_operation),
            "typed operation payload `{message}` did not name `{expected_operation}`"
        ),
        other => panic!("expected a typed construct rejection, got {other:?}"),
    }
}

#[test]
fn c_target_keeps_every_issue_2339_operation_admitted() {
    for (operation, dag) in reduction_cases() {
        reject_inexact_device_reduction_cells(&dag, BuildTarget::C)
            .unwrap_or_else(|error| panic!("C must keep `{operation}` admitted: {error:?}"));
    }
}

#[test]
fn hip_and_metal_reject_every_inexact_reduction_cell_with_typed_issue_metadata() {
    for (operation, dag) in reduction_cases() {
        let hip = match reject_unsupported_hip_ops(&dag) {
            Ok(()) => panic!("HIP must fence `{operation}`"),
            Err(error) => error,
        };
        assert_issue_2339_receipt("hip", operation, hip);

        let metal = match reject_unsupported_metal_ops(&dag) {
            Ok(()) => panic!("Metal must fence `{operation}`"),
            Err(error) => error,
        };
        assert_issue_2339_receipt("metal", operation, metal);
    }
}

#[test]
fn global_extrema_adjoint_dags_retain_the_issue_2339_fence() {
    for (operation, op) in [
        ("max_reduce adjoint", RiscOp::MaxReduce { axis: 0 }),
        ("min_reduce adjoint", RiscOp::MinReduce { axis: 0 }),
    ] {
        let forward = simple_reduction_dag(op, Prim::F32);
        let output = forward.roots()[0];
        let input = forward.nodes()[0].id;
        let adjoint = grad_dag_checked(&forward, output, &[input])
            .unwrap_or_else(|error| panic!("construct `{operation}`: {error}"));

        let hip = match reject_unsupported_hip_ops(&adjoint.dag) {
            Ok(()) => panic!("HIP must fence `{operation}`"),
            Err(error) => error,
        };
        assert_issue_2339_receipt("hip", operation.trim_end_matches(" adjoint"), hip);

        let metal = match reject_unsupported_metal_ops(&adjoint.dag) {
            Ok(()) => panic!("Metal must fence `{operation}`"),
            Err(error) => error,
        };
        assert_issue_2339_receipt("metal", operation.trim_end_matches(" adjoint"), metal);
    }
}

#[test]
fn issue_2339_fence_does_not_capture_sum_prod_count_or_window_sum_mean() {
    let unrelated = [
        simple_reduction_dag(RiscOp::sum_default(0, Prim::F32).unwrap(), Prim::F32),
        simple_reduction_dag(RiscOp::ProdReduce { axis: 0 }, Prim::F32),
        simple_reduction_dag(RiscOp::Count { axes: vec![0] }, Prim::Int64),
        window_dag(ReduceWindowKind::Sum),
        window_dag(ReduceWindowKind::Mean),
        window_grad_dag(ReduceWindowKind::Sum),
        window_grad_dag(ReduceWindowKind::Mean),
    ];

    for target in [BuildTarget::C, BuildTarget::Hip, BuildTarget::Metal] {
        for dag in &unrelated {
            reject_inexact_device_reduction_cells(dag, target)
                .expect("the #2339 fence must remain narrow");
        }
    }
}
