//! Production paths through the retained semantic core on explicit native
//! services: checking, lowering, evaluation, reverse AD and independent
//! ownership verification, plus cancellation arriving during a run, reuse
//! after rejections, and concurrent invocations.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use chelis_ir::capacity_key::CapacityScopeSupply;
use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict};
use chelis_ir::grad::{AdError, grad_dag_checked};
use chelis_ir::lowering_runtime::{LoweringRuntime, with_native_lowering_runtime};
use chelis_ir::ownership::{lower_dag_ownership, verify_ownership};
use chelis_pipeline_core::{
    CheckedCompilation, CoreLowerError, IrName, LoweredCompilation, LoweringMode, PreparedProgram,
    PreparedTypeAnalysisOutcome, SemanticContext, analyze_prepared, complete_checks, lower_checked,
};
use chelis_types::types::Prim;
use chelis_types::{CancelToken, EVAL_CANCELLED_MSG, NativeServices, ProgramOrigin};

/// `loss(theta) = sum(factor * theta)` with `factor` = `theta` (squared
/// norm) or `floor(theta)` (not differentiable).
fn loss_program(factor: &str) -> String {
    let factor = match factor {
        "theta" => "(copy {} (var {} theta))",
        "floor" => "(app {} (var {} floor) (copy {} (var {} theta)))",
        other => panic!("unknown factor {other}"),
    };
    format!(
        "(defsig {{}} loss
           (t-fn {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} f32)) (t-prim {{}} f32)))
         (def {{}} loss
           (fn {{}} (params {{}} (theta {{type: (t-var {{}} _)}}))
             (app {{}} (var {{}} tensor_to_scalar)
               (app {{}} (var {{}} sum)
                 (app {{}} (var {{}} mul) {factor} (var {{}} theta))
                 (lit {{type: (t-prim {{}} i32)}} 0)))))"
    )
}

fn checked(
    source: &str,
    runtime: &mut LoweringRuntime,
    native: &mut NativeServices,
) -> CheckedCompilation {
    let prepared = PreparedProgram::from_expanded_deep(
        chelis_deep::parser::parse_str(source).expect("the Deep fixture parses"),
        ProgramOrigin::Unlinked,
    );
    let PreparedTypeAnalysisOutcome::Accepted(analysis) = analyze_prepared(prepared, native) else {
        panic!("the fixture type-checks");
    };
    complete_checks(*analysis, SemanticContext::Isolated, runtime, native)
        .expect("the fixture passes semantic checks")
}

/// The lowered `loss` entry: its output root and its `theta` input load.
fn lowered_loss(
    factor: &str,
    runtime: &mut LoweringRuntime,
    native: &mut NativeServices,
) -> (LoweredCompilation, NodeId, NodeId) {
    let lowered = lower_checked(
        checked(&loss_program(factor), runtime, native),
        LoweringMode::Strict,
        runtime,
        native,
    )
    .expect("the loss entry lowers");
    let output = *lowered
        .named_roots()
        .get(&IrName::new("loss"))
        .expect("loss root");
    let theta = *lowered
        .forward_node_index()
        .get(&IrName::new("theta"))
        .expect("theta load");
    (lowered, output, theta)
}

fn point(a: f64, b: f64) -> TensorValue {
    TensorValue::from_vec(vec![2], vec![a, b])
}

/// Evaluate the primal, differentiate with the production AD entry,
/// evaluate the gradient, and independently verify the ownership plan of
/// the combined graph.
fn primal_and_gradient(
    a: f64,
    b: f64,
    runtime: &mut LoweringRuntime,
    native: &mut NativeServices,
) -> Result<(f64, Vec<f64>), String> {
    let (lowered, output, theta) = lowered_loss("theta", runtime, native);
    let input = point(a, b);
    let primal =
        eval_tensor_roots_with_strict(lowered.dag(), &[output], |_| Some(input.clone()), native)?;
    let gradient =
        grad_dag_checked(lowered.dag(), output, &[theta]).map_err(|error| format!("{error:?}"))?;
    let gradient_node = gradient.grad_nodes[&theta];
    let values = eval_tensor_roots_with_strict(
        &gradient.dag,
        &[gradient.output_node, gradient_node],
        |_| Some(input.clone()),
        native,
    )?;
    assert_eq!(
        values[&gradient.output_node].to_f64_lossy_vec(),
        primal[&output].to_f64_lossy_vec(),
        "the combined graph preserves the primal"
    );
    let verified = verify_ownership(
        lower_dag_ownership(gradient.dag.clone(), &mut CapacityScopeSupply::native())
            .map_err(|error| format!("{error:?}"))?,
    );
    assert!(
        verified.is_ok(),
        "independent ownership verification: {verified:?}"
    );
    Ok((
        primal[&output].to_f64_lossy_vec()[0],
        values[&gradient_node].to_f64_lossy_vec(),
    ))
}

#[test]
fn squared_norm_is_checked_lowered_evaluated_differentiated_and_verified() {
    let mut native = NativeServices::new(None);
    let result =
        with_native_lowering_runtime(|runtime| primal_and_gradient(2.0, 3.0, runtime, &mut native));
    assert_eq!(result, Ok((13.0, vec![4.0, 6.0])));
}

#[test]
fn rejections_leave_the_runtime_and_services_reusable() {
    let mut native = NativeServices::new(None);
    with_native_lowering_runtime(|runtime| {
        // A lowering rejection returned through the runtime's catch.
        let unsupported = checked(
            include_str!("fixtures/lower_rejection.dp"),
            runtime,
            &mut native,
        );
        assert!(matches!(
            lower_checked(unsupported, LoweringMode::Strict, runtime, &mut native),
            Err(CoreLowerError::Lower(_))
        ));
        // An AD rejection of a piecewise-constant operation.
        let (lowered, output, theta) = lowered_loss("floor", runtime, &mut native);
        assert!(matches!(
            grad_dag_checked(lowered.dag(), output, &[theta]),
            Err(AdError::NotSupported { op: "floor", .. })
        ));
        assert_eq!(
            primal_and_gradient(2.0, 3.0, runtime, &mut native),
            Ok((13.0, vec![4.0, 6.0]))
        );
    });
}

/// `n` chained additions over one loaded vector: long enough that a
/// cancellation raised after the run starts lands between node visits.
fn long_chain(n: usize, width: usize) -> (Dag, NodeId) {
    let ty = TensorType {
        dims: vec![DimInfo::Lit(width)],
        precision: Prim::F64,
    };
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
    let mut acc = x;
    for _ in 0..n {
        acc = dag.add_node(RiscOp::Add, vec![acc, x], ty.clone(), None);
    }
    dag.add_root(acc);
    (dag, acc)
}

#[test]
fn evaluation_observes_cancellation_raised_during_the_run() {
    const WIDTH: usize = 4096;
    let (dag, root) = long_chain(20_000, WIDTH);
    let token = CancelToken::new();
    let canceller = token.clone();
    let (started, signal) = mpsc::channel();
    let cancel = thread::spawn(move || {
        signal.recv().expect("evaluation start signal");
        thread::sleep(Duration::from_millis(50));
        canceller.cancel();
    });
    let mut native = NativeServices::new(Some(token));
    let input = TensorValue::from_vec(vec![WIDTH], vec![1.0; WIDTH]);
    started.send(()).expect("canceller is waiting");
    let outcome =
        eval_tensor_roots_with_strict(&dag, &[root], |_| Some(input.clone()), &mut native);
    cancel.join().expect("canceller thread");
    // A flag read once at entry would let the run complete.
    assert_eq!(outcome.err().as_deref(), Some(EVAL_CANCELLED_MSG));

    // Fresh services on the same thread evaluate normally afterwards.
    let (small, small_root) = long_chain(3, 2);
    let values = eval_tensor_roots_with_strict(
        &small,
        &[small_root],
        |_| Some(point(1.0, 2.0)),
        &mut NativeServices::new(None),
    )
    .expect("uncancelled evaluation");
    assert_eq!(values[&small_root].to_f64_lossy_vec(), vec![4.0, 8.0]);
}

#[test]
fn concurrent_pipelines_keep_independent_results_and_cancellation() {
    let workers: Vec<_> = (0..4u8)
        .map(|index| {
            thread::spawn(move || {
                let a = f64::from(index) + 1.0;
                let b = f64::from(index) + 2.0;
                let token = CancelToken::new();
                let mut native = NativeServices::new(Some(token.clone()));
                with_native_lowering_runtime(|runtime| {
                    if index == 3 {
                        // Checking and lowering finish before the token trips;
                        // the evaluation then observes it.
                        let (lowered, output, _) = lowered_loss("theta", runtime, &mut native);
                        token.cancel();
                        return eval_tensor_roots_with_strict(
                            lowered.dag(),
                            &[output],
                            |_| Some(point(a, b)),
                            &mut native,
                        )
                        .map(|values| (values[&output].to_f64_lossy_vec()[0], Vec::new()));
                    }
                    primal_and_gradient(a, b, runtime, &mut native)
                })
            })
        })
        .collect();
    let outcomes: Vec<_> = workers
        .into_iter()
        .map(|worker| worker.join().expect("worker thread"))
        .collect();
    assert_eq!(
        outcomes,
        vec![
            Ok((5.0, vec![2.0, 4.0])),
            Ok((13.0, vec![4.0, 6.0])),
            Ok((25.0, vec![6.0, 8.0])),
            Err(EVAL_CANCELLED_MSG.to_string()),
        ]
    );
}
