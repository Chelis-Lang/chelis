//! Native emission observations, not compiler certification or numerical tests.
#![cfg(feature = "emission-observer")]

use chelis_compiler_api::compiler::{
    EntryLaneDecline, compile_for_execution, compile_for_execution_with_observer,
};
use chelis_compiler_api::emission_observer::SelectedEmission;
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, SourceKind};
use chelis_ir::dag::RiscOp;
use chelis_ir::ownership::VerifiedDagAction;

fn request(source: &str, target: CompileTarget, entry: Option<&str>) -> CompileRequest {
    CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        target,
        entry_name: entry.map(str::to_owned),
    }
}

const SOURCE: &str = r#"
def helper(x: tensor[2, f32]) -> tensor[2, f32] = mul(x, x)
def main(a: tensor[2, f32], b: tensor[2, f32]) -> tensor[2, f32] = add(helper(a), b)
"#;

#[test]
fn rejected_resource_region_produces_no_emission_observation() {
    for (target, device, allowed) in [
        (CompileTarget::C, "cpu", true),
        (CompileTarget::C, "cpu:author-device", false),
        (CompileTarget::C, "cpu:socket_9", false),
        (CompileTarget::C, "cuda:0", false),
        (CompileTarget::C, "metal", false),
        (CompileTarget::C, "cpu:", false),
        (CompileTarget::C, "gpu:0", false),
        (CompileTarget::Hip, "gpu:0", true),
        (CompileTarget::Hip, "cpu", false),
    ] {
        let source = format!(
            "def main(x: tensor[2, f32]) -> tensor[2, f32] = \
             with device(\"{device}\") {{ mul(x, x) }}"
        );
        let mut observations = 0;
        let result = compile_for_execution_with_observer(
            request(&source, target, Some("main")),
            &mut |_| observations += 1,
        );
        if allowed {
            result.unwrap();
            assert_eq!(observations, 1);
        } else {
            let error = result.map(|_| ()).unwrap_err();
            assert!(
                error
                    .errors
                    .iter()
                    .all(|d| d.kind() == chelis_vocab::DiagnosticKind::BuildTargetMismatch)
            );
            assert!(
                error
                    .errors
                    .iter()
                    .any(|d| d.message.contains("cannot satisfy resource region")),
                "{error:?}"
            );
            assert_eq!(observations, 0);
        }
    }
}

#[test]
fn observes_selected_entry_and_preserves_complete_c_artifact() {
    let ordinary = compile_for_execution(request(SOURCE, CompileTarget::C, Some("main"))).unwrap();
    let mut count = 0;
    let observed = compile_for_execution_with_observer(
        request(SOURCE, CompileTarget::C, Some("main")),
        &mut |observation| {
            count += 1;
            assert!(!observation.program.checked().exprs().is_empty());
            let SelectedEmission::Dag { unfused, selected } = observation.selected else {
                panic!("tensor entry must select a standalone DAG");
            };
            let mut loads = selected
                .nodes()
                .iter()
                .filter_map(|node| match &node.op {
                    RiscOp::Load { name } => Some(name.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            loads.sort();
            assert_eq!(loads, ["a", "b"]);
            assert_eq!(unfused.roots().len(), 1);
            assert_eq!(selected.roots().len(), 1);
            let actions = selected.actions().collect::<Vec<_>>();
            assert!(
                actions
                    .iter()
                    .any(|action| matches!(action, VerifiedDagAction::BorrowLoad { .. }))
            );
            assert!(actions.iter().any(|action| matches!(
                action,
                VerifiedDagAction::Root { .. } | VerifiedDagAction::RootClone { .. }
            )));
            assert!(!actions.is_empty());
        },
    )
    .unwrap();
    assert_eq!(count, 1);
    assert_eq!(
        serde_json::to_value(ordinary).unwrap(),
        serde_json::to_value(observed).unwrap()
    );
}

#[test]
fn observes_hip_dag_without_executing_a_gpu() {
    let source = "def main(x: tensor[2, f32]) -> tensor[2, f32] = mul(x, x)";
    let ordinary = compile_for_execution(request(source, CompileTarget::Hip, None)).unwrap();
    let mut count = 0;
    let observed = compile_for_execution_with_observer(
        request(source, CompileTarget::Hip, None),
        &mut |observation| {
            count += 1;
            let SelectedEmission::Dag { unfused, selected } = observation.selected else {
                panic!("expected HIP DAG");
            };
            assert_eq!(unfused.roots().len(), 1);
            assert_eq!(selected.roots().len(), 1);
            assert!(selected.actions().len() >= selected.nodes().len());
        },
    )
    .unwrap();
    assert_eq!(count, 1);
    assert_eq!(
        serde_json::to_value(ordinary).unwrap(),
        serde_json::to_value(observed).unwrap()
    );
}

#[test]
fn observes_host_payload_not_an_empty_library_dag() {
    let source = "def main(x: f32) -> f32 = x + 1.0f32";
    for target in [CompileTarget::C, CompileTarget::Hip] {
        let ordinary = compile_for_execution(request(source, target, None)).unwrap();
        let mut count = 0;
        let observed = compile_for_execution_with_observer(
            request(source, target, None),
            &mut |observation| {
                count += 1;
                let SelectedEmission::Host(selected) = observation.selected else {
                    panic!("expected host payload");
                };
                assert!(selected.function_count() > 0);
                assert!(selected.sites().len() > 0);
            },
        )
        .unwrap();
        assert_eq!(count, 1);
        assert_eq!(
            serde_json::to_value(ordinary).unwrap(),
            serde_json::to_value(observed).unwrap()
        );
    }
}

#[test]
fn rejection_is_identical_and_does_not_certify_empty_observations() {
    for source in [SOURCE, "def main(x: f32) -> f32 = missing(x)"] {
        let ordinary =
            compile_for_execution(request(source, CompileTarget::C, Some("absent"))).unwrap_err();
        let mut count = 0;
        let observed = compile_for_execution_with_observer(
            request(source, CompileTarget::C, Some("absent")),
            &mut |_| count += 1,
        )
        .unwrap_err();
        assert_eq!(count, 0);
        assert_eq!(ordinary.stage, observed.stage);
        assert_eq!(
            serde_json::to_value(ordinary.errors).unwrap(),
            serde_json::to_value(observed.errors).unwrap()
        );
    }
}

#[test]
fn gradient_host_observation_exposes_verified_nested_helpers() {
    let source = r#"
def loss(x: tensor[2, f32], w: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(mul(x, w), 0))
def dloss(x: tensor[2, f32], w: tensor[2, f32]) -> tensor[2, f32] = (grad(loss)(x, w)).0
"#;
    let ordinary = compile_for_execution(request(source, CompileTarget::C, Some("dloss"))).unwrap();
    let mut count = 0;
    let mut helpers = 0;
    let observed = compile_for_execution_with_observer(
        request(source, CompileTarget::C, Some("dloss")),
        &mut |observation| {
            count += 1;
            let SelectedEmission::Host(selected) = observation.selected else {
                panic!("gradient entry must preserve host routing");
            };
            for i in 0..selected.function_count() {
                let function = selected.function(i).unwrap();
                for j in 0..function.tensor_helper_count() {
                    let dag = function.tensor_helper(j).unwrap().dag();
                    assert!(!dag.roots().is_empty());
                    assert!(dag.actions().len() >= dag.nodes().len());
                    helpers += 1;
                }
            }
        },
    )
    .unwrap();
    assert_eq!(count, 1);
    assert!(helpers > 0);
    assert_eq!(
        serde_json::to_value(ordinary).unwrap(),
        serde_json::to_value(observed).unwrap()
    );
}

#[test]
fn tuple_gradient_retains_actual_lowering_without_retaining_dead_emitted_functions() {
    for (body, has_product) in [
        ("tensor_to_scalar(sum(mul(x, y), 0))", true),
        ("tensor_to_scalar(sum(x, 0))", false),
        ("3.0", false),
    ] {
        let source = format!(
            "def loss(x: tensor[2, f32], y: tensor[2, f32]) -> f32 = {body}\n\
             def derivative(x: tensor[2, f32], y: tensor[2, f32]) -> \
             (tensor[2, f32], tensor[2, f32]) = grad(loss)(x, y)"
        );
        let ordinary =
            compile_for_execution(request(&source, CompileTarget::C, Some("derivative"))).unwrap();
        let mut count = 0;
        let observed = compile_for_execution_with_observer(
            request(&source, CompileTarget::C, Some("derivative")),
            &mut |observation| {
                count += 1;
                assert_eq!(observation.program.checked().exprs().len(), 4);
                let lowered = observation
                    .lowered_host
                    .expect("actual initial host lowering");
                assert_eq!(
                    lowered
                        .functions
                        .iter()
                        .map(|f| f.name.as_str())
                        .collect::<Vec<_>>(),
                    ["loss", "derivative"]
                );
                let loss = &lowered.functions[0];
                assert_eq!(loss.tensor_helpers.len(), usize::from(body != "3.0"));
                assert_eq!(
                    loss.params
                        .iter()
                        .map(|p| p.name.as_str())
                        .collect::<Vec<_>>(),
                    ["x", "y"]
                );
                assert_eq!(
                    loss.tensor_helpers.iter().any(|helper| helper
                        .dag
                        .nodes()
                        .iter()
                        .any(|node| matches!(node.op, RiscOp::Mul))),
                    has_product,
                    "the snapshot must retain this source's primal, not a previous loss"
                );
                let SelectedEmission::Host(selected) = observation.selected else {
                    panic!("tuple derivative selects the host lane");
                };
                assert_eq!(selected.function_count(), 1);
                let derivative = selected.function(0).unwrap();
                assert_eq!(derivative.name(), "derivative");
                assert_eq!(derivative.tensor_helper_count(), 1);
                assert_eq!(derivative.tensor_helper(0).unwrap().dag().roots().len(), 2);
                // The unverified observation can be copied and inspected, but
                // changing that copy cannot change the selected verified view.
                let mut copy = lowered.clone();
                copy.functions.clear();
                assert!(copy.functions.is_empty());
                assert_eq!(selected.function_count(), 1);
                assert_eq!(lowered.functions.len(), 2);
            },
        )
        .unwrap();
        assert_eq!(count, 1);
        assert!(matches!(
            &observed.entry_lane_decline,
            Some(EntryLaneDecline::NotTensorSignature { entry }) if entry == "derivative"
        ));
        assert_eq!(
            serde_json::to_value(ordinary).unwrap(),
            serde_json::to_value(observed).unwrap(),
            "observation must not change entry selection, emitted files or metadata"
        );
    }
}

#[test]
fn observation_does_not_turn_a_later_failure_into_success() {
    let source = r#"
def process(x: tensor[4, f32]) -> tensor[4, f32] = relu(x)
def batch_process(xs: tensor[8, 4, f32]) -> tensor[8, 4, f32] = xs |> vmap(process)
"#;
    let ordinary = compile_for_execution(request(source, CompileTarget::C, Some("batch_process")))
        .unwrap_err();
    let mut count = 0;
    let observed = compile_for_execution_with_observer(
        request(source, CompileTarget::C, Some("batch_process")),
        &mut |_| count += 1,
    )
    .unwrap_err();
    assert_eq!(
        count, 1,
        "the legacy fallback is observed before the strict entry decline"
    );
    assert_eq!(ordinary.stage, observed.stage);
    assert_eq!(
        serde_json::to_value(ordinary.errors).unwrap(),
        serde_json::to_value(observed.errors).unwrap()
    );
}

#[test]
fn ordinary_nested_compilation_does_not_inherit_the_observer() {
    let source = "def main(x: tensor[2, f32]) -> tensor[2, f32] = x";
    let ordinary = compile_for_execution(request(source, CompileTarget::C, None)).unwrap();
    let mut count = 0;
    let observed =
        compile_for_execution_with_observer(request(source, CompileTarget::C, None), &mut |_| {
            count += 1;
            let nested = compile_for_execution(request(source, CompileTarget::C, None)).unwrap();
            assert_eq!(
                serde_json::to_value(&ordinary).unwrap(),
                serde_json::to_value(nested).unwrap()
            );
        })
        .unwrap();
    assert_eq!(count, 1);
    assert_eq!(
        serde_json::to_value(ordinary).unwrap(),
        serde_json::to_value(observed).unwrap()
    );
}
