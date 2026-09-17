//! Spec-derived acceptance tests for the opt-in trace, not an AD certificate.

#[cfg(feature = "lowering-trace")]
use chelis_ir::dag::{Dag, RiscOp};
#[cfg(feature = "lowering-trace")]
use chelis_ir::lower::try_lower_program_to_library;
#[cfg(feature = "lowering-trace")]
use chelis_ir::lowering_trace::{
    BoundaryKind, ContextId, ContextKind, try_lower_program_to_library_with_trace,
};
#[cfg(feature = "lowering-trace")]
use chelis_surf::{desugar::desugar_program, parser::parse_str};
#[cfg(feature = "lowering-trace")]
use chelis_types::{CheckedProgram, check_typed_program};

#[cfg(feature = "lowering-trace")]
const SQUARE: &str = r#"
def loss(x: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(mul(x, x), 0))
def derivative(x: tensor[3, f32]) -> tensor[3, f32] = grad(loss)(x)
"#;

#[cfg(feature = "lowering-trace")]
fn checked(source: &str) -> CheckedProgram {
    let deep = desugar_program(&parse_str(source).expect("parse fixture"));
    let program = check_typed_program(&deep).expect("check fixture types");
    let program = chelis_effects::check_program(&program).expect("check fixture effects");
    chelis_types::check_linearity(&program).expect("check fixture linearity")
}

// The existing DAG codec compares every stored bit and metadata field. This
// helper is not a new trace wire format and does not serialize LoweringTrace.
#[cfg(feature = "lowering-trace")]
fn same_dag(left: &Dag, right: &Dag) {
    assert_eq!(
        bincode::serialize(left).unwrap(),
        bincode::serialize(right).unwrap()
    );
}

// The feature-enabled support lane below owns the trace oracle. The generic
// change-owned lane lists directly modified integration targets without
// features, so retain one active feature-off test instead of compiling an
// empty integration binary.
#[cfg(not(feature = "lowering-trace"))]
#[test]
fn feature_off_lane_defers_to_the_feature_owned_trace_oracle() {
    assert!(!cfg!(feature = "lowering-trace"));
}

#[cfg(feature = "lowering-trace")]
#[test]
fn execution_trace_captures_the_actual_ad_call_without_replaying_seed_controls() {
    use chelis_ir::execution_spine::{SourceKind, Step};
    use chelis_ir::lower::try_lower_program_to_evaluation_library;
    use chelis_ir::lowering_trace::try_lower_program_to_evaluation_library_with_trace;
    let program = checked(
        r#"
def loss(x: tensor[32, f32]) -> f32 = with seed(42i64) {
  identity = with seed(42i64) { x }
  tensor_to_scalar(sum(dropout(identity, 0.5f32), 0))
}
def derivative(x: tensor[32, f32]) -> tensor[32, f32] = grad(loss)(x)
"#,
    );
    let ordinary = try_lower_program_to_evaluation_library(&program).unwrap();
    let (observed, trace) = try_lower_program_to_evaluation_library_with_trace(&program).unwrap();
    assert_eq!(
        bincode::serialize(ordinary.library_for_inspection()).unwrap(),
        bincode::serialize(observed.library_for_inspection()).unwrap()
    );
    assert_eq!(trace.executions.len(), 1);
    let execution = &trace.executions[0];
    let gradient = &trace.lowering.gradients[execution.gradient];
    same_dag(execution.forward.dag_for_inspection(), &gradient.forward);
    same_dag(execution.backward.dag_for_inspection(), &gradient.backward);
    assert_eq!(execution.forward.source_for_inspection().len(), 5);
    let source_kinds = |plan: &chelis_ir::evaluation::EvaluationPlan| {
        plan.source_for_inspection()
            .iter()
            .map(|event| match event.kind {
                SourceKind::Forward { draw, scope, .. } => (0, draw.index(), scope.index()),
                SourceKind::Control(chelis_ir::execution_spine::Control::Enter {
                    scope, ..
                }) => (1, scope.index(), 0),
                SourceKind::Control(chelis_ir::execution_spine::Control::Leave { scope }) => {
                    (2, scope.index(), 0)
                }
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        source_kinds(&execution.forward),
        source_kinds(&execution.backward)
    );
    assert_eq!(
        execution
            .backward
            .steps_for_inspection()
            .iter()
            .filter(|step| matches!(step, Step::Control { .. }))
            .count(),
        4
    );
    assert_eq!(
        execution
            .backward
            .dag_for_inspection()
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::Dropout { .. }))
            .count(),
        2
    );
    // Execute the captured backward graph itself. Replay does not require an
    // active outer Random effect after its local seed handler has left.
    let mut context =
        chelis_ir::evaluation::RandomExecutionContext::new(chelis_ir::host::RandomLoweringState {
            seed: None,
            counter: 0,
        });
    let values =
        chelis_ir::eval::eval_tensor_plan_with_strict(&execution.backward, &mut context, |name| {
            (name == "x").then(|| chelis_ir::eval::TensorValue::from_vec(vec![32], vec![1.0; 32]))
        })
        .unwrap();
    let derivative = &values[&gradient.gradients[&gradient.wrt[0]]];
    let bits = derivative
        .to_f64_lossy_vec()
        .iter()
        .map(|value| (*value as f32).to_bits())
        .collect::<Vec<_>>();
    // Independent BigInt splitmix/rotate reference, high53 -> f32 comparison;
    // pinned full output, not expectations recovered from the candidate.
    let expected = [
        0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2,
        2, 2,
    ];
    assert_eq!(bits, expected.map(|value| (value as f32).to_bits()));
    assert_eq!(context.state().seed, None);
    assert_eq!(context.state().counter, 0);
}

#[cfg(feature = "lowering-trace")]
#[test]
fn selected_source_census_keeps_draw_free_dependencies_and_excludes_siblings() {
    use chelis_ir::execution_spine::{Control, SourceKind};
    let program = checked(
        r#"
x: tensor[32, f32] = x
empty_scope = with seed(7i64) { x }
unrelated = with seed(99i64) { x }
sample = with seed(42i64) { dropout(empty_scope, 0.0f32) }
"#,
    );
    let library = chelis_ir::lower::try_lower_program_to_evaluation_library(&program).unwrap();
    let selected = library.program().select_roots(&["sample".into()]).unwrap();
    let seeds = selected
        .plan()
        .source_for_inspection()
        .iter()
        .filter_map(|event| match event.kind {
            SourceKind::Control(Control::Enter { seed, .. }) => Some(seed),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(seeds, [7, 42]);
    assert_eq!(selected.plan().source_for_inspection().len(), 5);
    let mut context =
        chelis_ir::evaluation::RandomExecutionContext::new(chelis_ir::host::RandomLoweringState {
            seed: None,
            counter: 0,
        });
    let values =
        chelis_ir::eval::eval_tensor_plan_with_strict(selected.plan(), &mut context, |name| {
            (name == "x").then(|| chelis_ir::eval::TensorValue::from_vec(vec![32], vec![1.0; 32]))
        })
        .unwrap();
    assert_eq!(
        values[&selected.roots()["sample"]].to_f64_lossy_vec(),
        vec![1.0; 32]
    );
}

#[cfg(feature = "lowering-trace")]
#[test]
fn selecting_a_draw_free_region_retains_controls_without_changing_dispatch() {
    use chelis_ir::evaluation::{EvaluationProfile, LegacyEvaluationReason};
    use chelis_ir::execution_spine::{Control, SourceKind};
    let program = checked(
        r#"
x: tensor[2, f32] = x
selected = with seed(7i64) { with seed(7i64) { x } }
unrelated = with seed(99i64) { x }
"#,
    );
    let library = chelis_ir::lower::try_lower_program_to_evaluation_library(&program).unwrap();
    let names = ["selected".into()];
    assert_eq!(
        library.program().profile_for_roots(&names).unwrap(),
        EvaluationProfile::Legacy(LegacyEvaluationReason::NoDropout)
    );
    let selected = library.program().select_roots(&names).unwrap();
    let source = selected.plan().source_for_inspection();
    assert_eq!(source.len(), 4);
    assert_eq!(
        source
            .iter()
            .filter_map(|event| match event.kind {
                SourceKind::Control(Control::Enter { seed, .. }) => Some(seed),
                _ => None,
            })
            .collect::<Vec<_>>(),
        [7, 7]
    );
    let mut context =
        chelis_ir::evaluation::RandomExecutionContext::new(chelis_ir::host::RandomLoweringState {
            seed: None,
            counter: 23,
        });
    let values =
        chelis_ir::eval::eval_tensor_plan_with_strict(selected.plan(), &mut context, |name| {
            (name == "x").then(|| chelis_ir::eval::TensorValue::from_vec(vec![2], vec![3.0, 5.0]))
        })
        .unwrap();
    assert_eq!(
        values[&selected.roots()["selected"]].to_f64_lossy_vec(),
        [3.0, 5.0]
    );
    assert_eq!((context.state().seed, context.state().counter), (None, 23));
    assert!(library.program().select_roots(&["missing".into()]).is_err());

    let excluded = checked(
        "def choose(x: tensor[2, f32], flag: bool) -> tensor[2, f32] = if flag then x else x\n",
    );
    let library = chelis_ir::lower::try_lower_program_to_evaluation_library(&excluded).unwrap();
    assert!(!library.library_for_inspection().lowered_names()["choose"]);
    assert!(library.program().select_roots(&["choose".into()]).is_err());
}

#[cfg(feature = "lowering-trace")]
#[test]
fn context_composition_retains_declared_controls_without_confusing_local_aliases() {
    use chelis_ir::execution_spine::{Control, SourceKind};
    let source = r#"
x: tensor[32, f32] = x
empty_scope = with seed(7i64) { x }
unrelated = with seed(99i64) { x }
"#;
    let deep = desugar_program(&parse_str(source).unwrap());
    let env = chelis_types::build_type_env_from_library(&deep).unwrap();
    let library =
        chelis_ir::lower::try_lower_program_to_evaluation_library(&checked(source)).unwrap();
    let original = bincode::serialize(library.library_for_inspection()).unwrap();
    for (body, expected) in [
        (
            "sample = with seed(42i64) { dropout(empty_scope, 0.0f32) }",
            vec![7, 42],
        ),
        (
            "sample = {\n empty_scope = x\n with seed(42i64) { dropout(empty_scope, 0.0f32) }\n}",
            vec![42],
        ),
    ] {
        let deep = desugar_program(&parse_str(body).unwrap());
        let program = chelis_types::check_ir_with_context(&env, &deep).unwrap();
        let program = chelis_effects::check_program(&program).unwrap();
        let program = chelis_types::check_linearity(&program).unwrap();
        let composed =
            chelis_ir::lower::try_lower_program_with_evaluation_context(&library, &program)
                .unwrap();
        let selected = composed.select_roots(&["sample".into()]).unwrap();
        let seeds = selected
            .plan()
            .source_for_inspection()
            .iter()
            .filter_map(|event| match event.kind {
                SourceKind::Control(Control::Enter { seed, .. }) => Some(seed),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(seeds, expected);
        let mut context = chelis_ir::evaluation::RandomExecutionContext::new(
            chelis_ir::host::RandomLoweringState {
                seed: None,
                counter: 0,
            },
        );
        let values =
            chelis_ir::eval::eval_tensor_plan_with_strict(selected.plan(), &mut context, |name| {
                (name == "x")
                    .then(|| chelis_ir::eval::TensorValue::from_vec(vec![32], vec![1.0; 32]))
            })
            .unwrap();
        assert_eq!(
            values[&selected.roots()["sample"]].to_f64_lossy_vec(),
            vec![1.0; 32]
        );
        assert_eq!(
            bincode::serialize(library.library_for_inspection()).unwrap(),
            original
        );
    }
}

#[cfg(feature = "lowering-trace")]
#[test]
fn trace_preserves_the_ordinary_library_and_actual_normalization() {
    let program = checked(&format!(
        r#"{SQUARE}
def pair(x: tensor[3, f32]) -> (tensor[3, f32], tensor[3, f32]) = (x, x)
def dead(x: tensor[3, f32]) -> tensor[3, f32] = {{
  unused = mul(x, x)
  x
}}
"#
    ));
    let ordinary = try_lower_program_to_library(&program).unwrap();
    let (observed, trace) = try_lower_program_to_library_with_trace(&program).unwrap();
    assert_eq!(
        bincode::serialize(&ordinary).unwrap(),
        bincode::serialize(&observed).unwrap()
    );
    let normalization = trace.normalization;
    same_dag(&normalization.after_drops, observed.dag());
    // These are genuinely different stages, not four copies of the result.
    assert!(
        normalization
            .after_copies
            .nodes()
            .iter()
            .any(|node| node.op == RiscOp::Copy)
    );
    assert!(
        !normalization
            .after_copies
            .nodes()
            .iter()
            .any(|node| node.op == RiscOp::Drop)
    );
    assert!(
        normalization
            .after_drops
            .nodes()
            .iter()
            .any(|node| node.op == RiscOp::Drop)
    );
    assert!(normalization.after_dce.len() < normalization.before_dce.len());
    assert_eq!(
        normalization.copy_remap.len(),
        normalization.after_dce.len()
    );
    for (old, new) in &normalization.dce_remap {
        let before = normalization.before_dce.get(*old).unwrap();
        let after = normalization.after_dce.get(*new).unwrap();
        assert_eq!(before.op, after.op);
        assert_eq!(before.output_type, after.output_type);
    }
}

#[cfg(feature = "lowering-trace")]
#[test]
fn trace_captures_actual_ad_ports_roots_and_ordered_wrt() {
    let (_, trace) = try_lower_program_to_library_with_trace(&checked(SQUARE)).unwrap();
    assert_eq!(trace.gradients.len(), 1);
    let grad = &trace.gradients[0];
    assert_eq!(trace.contexts[grad.context.0].kind, ContextKind::Gradient);
    assert_eq!(trace.contexts[grad.context.0].parent, Some(ContextId(0)));
    assert_eq!(grad.wrt.len(), 1);
    assert!(
        matches!(&grad.forward.get(grad.wrt[0]).unwrap().op, RiscOp::Load { name } if name.as_str() == "x")
    );
    let mul = grad
        .forward
        .nodes()
        .iter()
        .find(|node| node.op == RiscOp::Mul)
        .unwrap();
    assert_eq!(mul.inputs, vec![grad.wrt[0], grad.wrt[0]]);
    assert_eq!(grad.forward.roots(), &[grad.output]);
    assert!(grad.backward.is_root(grad.backward_output));
    assert!(grad.backward.is_root(grad.gradients[&grad.wrt[0]]));
    assert!(grad.backward.len() > grad.forward.len());
    assert!(trace.boundaries.is_empty());
    let application = grad.application.as_ref().expect("completed application");
    same_dag(&application.specialized, &grad.backward);
    assert_eq!(application.remap.len(), grad.backward.len());
    assert_eq!(application.wrt_actuals, vec![application.arguments["x"]]);
    for node in application.specialized.nodes() {
        let mapped = application
            .after_splice
            .get(application.remap[&node.id])
            .unwrap();
        match &node.op {
            RiscOp::Load { name } if application.arguments.contains_key(name.as_str()) => {
                assert_eq!(mapped.id, application.arguments[name.as_str()]);
                assert!(application.before_splice.get(mapped.id).is_some());
            }
            _ => {
                assert_eq!(mapped.op, node.op);
                assert_eq!(mapped.output_type, node.output_type);
                assert_eq!(
                    mapped.inputs,
                    node.inputs
                        .iter()
                        .map(|id| application.remap[id])
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    mapped.shape_deps,
                    node.shape_deps
                        .iter()
                        .map(|id| application.remap[id])
                        .collect::<Vec<_>>()
                );
            }
        }
    }
    let expected = application.remap[&grad.gradients[&grad.wrt[0]]];
    assert_eq!(
        application.result,
        chelis_ir::lowering_trace::Value::Node(expected)
    );
    assert!(application.after_packing.get(expected).is_some());
}

#[cfg(feature = "lowering-trace")]
#[test]
fn empty_and_repeated_invocations_do_not_leak_capture_state() {
    for source in [SQUARE, "", SQUARE, ""] {
        let program = checked(source);
        let (library, trace) = try_lower_program_to_library_with_trace(&program).unwrap();
        let ordinary = try_lower_program_to_library(&program).unwrap();
        same_dag(library.dag(), ordinary.dag());
        assert_eq!(trace.contexts[0].kind, ContextKind::Library);
        assert_eq!(trace.contexts[0].parent, None);
        assert_eq!(trace.gradients.len(), usize::from(!source.is_empty()));
        assert_eq!(trace.contexts.len(), if source.is_empty() { 1 } else { 2 });
    }
}

#[cfg(feature = "lowering-trace")]
#[test]
fn rejected_ad_keeps_its_diagnostic_instead_of_returning_a_trace() {
    let source = r#"
def loss(x: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(floor(x), 0))
def derivative(x: tensor[3, f32]) -> tensor[3, f32] = grad(loss)(x)
"#;
    let program = checked(source);
    let ordinary = try_lower_program_to_library(&program).unwrap_err();
    let observed = try_lower_program_to_library_with_trace(&program).unwrap_err();
    assert_eq!(ordinary, observed);
    assert!(ordinary.to_string().contains("floor"));
    let (_, next) = try_lower_program_to_library_with_trace(&checked(SQUARE)).unwrap();
    assert_eq!(next.gradients.len(), 1);
    assert_eq!(next.contexts.len(), 2);
}

#[cfg(feature = "lowering-trace")]
#[test]
fn multiple_wrt_inputs_keep_their_forward_order_and_distinct_results() {
    let source = r#"
def loss(x: tensor[3, f32], y: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(mul(x, y), 0))
def derivative(x: tensor[3, f32], y: tensor[3, f32]) -> (tensor[3, f32], tensor[3, f32]) = grad(loss)(x, y)
"#;
    let (_, trace) = try_lower_program_to_library_with_trace(&checked(source)).unwrap();
    let grad = &trace.gradients[0];
    assert_eq!(grad.wrt.len(), 2);
    for (id, expected_name) in grad.wrt.iter().zip(["x", "y"]) {
        assert!(
            matches!(&grad.forward.get(*id).unwrap().op, RiscOp::Load { name } if name.as_str() == expected_name)
        );
        assert!(grad.backward.is_root(grad.gradients[id]));
    }
    assert_ne!(grad.gradients[&grad.wrt[0]], grad.gradients[&grad.wrt[1]]);
    let application = grad.application.as_ref().unwrap();
    assert_eq!(
        application.result,
        chelis_ir::lowering_trace::Value::Tuple(
            grad.wrt
                .iter()
                .map(|id| chelis_ir::lowering_trace::Value::Node(
                    application.remap[&grad.gradients[id]]
                ))
                .collect()
        )
    );
    assert_eq!(
        application.wrt_actuals,
        vec![application.arguments["x"], application.arguments["y"]]
    );
    assert_ne!(application.wrt_actuals[0], application.wrt_actuals[1]);
}

#[cfg(feature = "lowering-trace")]
#[test]
fn nested_gradients_have_distinct_contexts_and_actual_parent_links() {
    let source = format!(
        r#"{SQUARE}
def first(x: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(grad(loss)(x), 0))
def second(x: tensor[3, f32]) -> tensor[3, f32] = grad(first)(x)
"#
    );
    let (_, trace) = try_lower_program_to_library_with_trace(&checked(&source)).unwrap();
    assert_eq!(trace.gradients.len(), 4);
    let nested = trace
        .gradients
        .iter()
        .find(|grad| trace.contexts[grad.context.0].parent != Some(ContextId(0)))
        .unwrap();
    let parent = trace.contexts[nested.context.0].parent.unwrap();
    assert_ne!(parent, nested.context);
    assert!(trace.gradients.iter().any(|grad| grad.context == parent));
    assert_eq!(trace.contexts[parent.0].parent, Some(ContextId(0)));
    for grad in &trace.gradients {
        assert!(
            grad.application.is_some(),
            "nested observations are completed independently"
        );
        assert!(trace.contexts[grad.context.0].parent.is_some());
    }
}

#[cfg(feature = "lowering-trace")]
#[test]
fn shaped_zero_materialization_is_not_fabricated_in_the_raw_ad_result() {
    let source = r#"
def constant(x: tensor[3, f32]) -> f32 = 1.0f32
def derivative(x: tensor[3, f32]) -> tensor[3, f32] = grad(constant)(x)
"#;
    let (library, trace) = try_lower_program_to_library_with_trace(&checked(source)).unwrap();
    let grad = &trace.gradients[0];
    assert_eq!(grad.wrt.len(), 1);
    assert!(
        grad.gradients.is_empty(),
        "the AD pass itself does not materialize this zero"
    );
    assert!(!library.rootless_defs().contains("derivative"));
    let root = library.symbol_table()["derivative"];
    assert!(
        library.dag().is_root(root),
        "source packing subsequently materializes it"
    );
    assert_eq!(
        library.dag().get(root).unwrap().output_type.dims,
        vec![chelis_ir::dag::DimInfo::Lit(3)]
    );
    same_dag(&trace.normalization.after_drops, library.dag());
    let application = grad.application.as_ref().unwrap();
    let chelis_ir::lowering_trace::Value::Node(zero) = application.result else {
        panic!("single disconnected cotangent remains a tensor result");
    };
    assert!(application.after_splice.get(zero).is_none());
    assert!(application.after_packing.get(zero).is_some());
    assert!(application.after_packing.len() > application.after_splice.len());
}

#[cfg(feature = "lowering-trace")]
#[test]
fn host_only_definitions_do_not_fabricate_graph_observations() {
    let source = r#"
def filled[n](x: tensor[n, f32]) -> tensor[n, f32] =
  expand(to_tensor([3.0f32]), 0, cast(shape(&x, 0), i64))
"#;
    let (library, trace) = try_lower_program_to_library_with_trace(&checked(source)).unwrap();
    assert!(library.dag().nodes().is_empty());
    assert!(trace.gradients.is_empty());
    assert_eq!(trace.boundaries.len(), 1);
    assert_eq!(trace.boundaries[0].kind, BoundaryKind::UnloweredDefinitions);
    same_dag(&trace.normalization.after_drops, library.dag());
}

#[cfg(feature = "lowering-trace")]
#[test]
fn application_preserves_formal_names_when_caller_names_differ() {
    let source = r#"
def loss(x: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(mul(x, x), 0))
def derivative(y: tensor[3, f32]) -> tensor[3, f32] = grad(loss)(y)
"#;
    let program = checked(source);
    let (library, trace) = try_lower_program_to_library_with_trace(&program).unwrap();
    same_dag(
        library.dag(),
        try_lower_program_to_library(&program).unwrap().dag(),
    );
    let grad = &trace.gradients[0];
    let application = grad.application.as_ref().unwrap();
    let actual = application.arguments["x"];
    assert!(matches!(&application.before_splice.get(actual).unwrap().op,
        RiscOp::Load { name } if name.as_str() == "y"));
    assert_eq!(application.wrt_actuals, vec![actual]);
    let chelis_ir::lowering_trace::Value::Node(result) = application.result else {
        panic!("one tensor result");
    };
    let dce = trace.normalization.dce_remap[&result];
    let copied = trace.normalization.copy_remap[&dce];
    assert_eq!(library.symbol_table()["derivative"], copied);
    assert_ne!(
        result, actual,
        "returned cotangent is not its primal argument"
    );
}

#[cfg(feature = "lowering-trace")]
#[test]
fn host_structured_results_are_not_reported_as_observed_applications() {
    let source = r#"
type Mixed = | Mixed { t: tensor[2, f32], n: i32 }
def loss(p: Mixed) -> f32 = match p with {
  | Mixed { t, n: _ } => tensor_to_scalar(sum(t, 0))
}
def derivative(x: tensor[2, f32]) -> tensor[2, f32] = {
  g = grad(loss)(Mixed { t: x, n: 3 })
  g.t
}
"#;
    let program = checked(source);
    let (library, trace) = try_lower_program_to_library_with_trace(&program).unwrap();
    same_dag(
        library.dag(),
        try_lower_program_to_library(&program).unwrap().dag(),
    );
    assert!(trace.gradients.is_empty());
    assert!(
        trace
            .boundaries
            .iter()
            .any(|b| b.kind == BoundaryKind::UnloweredDefinitions)
    );
    assert!(!library.lowered_names()["derivative"]);
}

#[cfg(feature = "lowering-trace")]
#[test]
fn specialization_records_both_named_and_actual_dimension_types() {
    let source = r#"
def loss[n](x: tensor[n, f32]) -> f32 = tensor_to_scalar(sum(mul(x, x), 0))
def derivative(y: tensor[3, f32]) -> tensor[3, f32] = grad(loss)(y)
"#;
    let (_, trace) = try_lower_program_to_library_with_trace(&checked(source)).unwrap();
    assert_eq!(trace.gradients.len(), 1);
    let application = trace.gradients[0].application.as_ref().unwrap();
    assert_eq!(application.formal_types.len(), 1);
    assert_eq!(application.actual_types.len(), 1);
    assert_ne!(
        application.formal_types[0].dims,
        application.actual_types[0].dims
    );
    assert_eq!(
        application.actual_types[0].dims,
        vec![chelis_ir::dag::DimInfo::Lit(3)]
    );
    assert!(
        application
            .specialized
            .nodes()
            .iter()
            .any(|node| node.output_type.dims == application.actual_types[0].dims)
    );
}

#[cfg(feature = "lowering-trace")]
#[test]
fn selected_wrt_does_not_drop_the_other_argument_binding() {
    let source = r#"
def loss(x: tensor[3, f32], y: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(mul(x, y), 0))
def derivative(x: tensor[3, f32], y: tensor[3, f32]) -> tensor[3, f32] = grad(loss, wrt=x)(x, y)
"#;
    let (_, trace) = try_lower_program_to_library_with_trace(&checked(source)).unwrap();
    let grad = &trace.gradients[0];
    let application = grad.application.as_ref().unwrap();
    assert_eq!(grad.wrt.len(), 1);
    assert_eq!(application.wrt_actuals, vec![application.arguments["x"]]);
    assert_ne!(application.arguments["x"], application.arguments["y"]);
    assert_eq!(application.formal_types.len(), 2);
    assert_eq!(application.actual_types.len(), 2);
    assert!(
        application
            .specialized
            .nodes()
            .iter()
            .any(|node| matches!(&node.op, RiscOp::Load { name } if name.as_str() == "y"))
    );
}

#[cfg(feature = "lowering-trace")]
#[test]
fn completed_application_snapshots_do_not_alias_later_mutations() {
    let (_, mut trace) = try_lower_program_to_library_with_trace(&checked(SQUARE)).unwrap();
    let grad = &mut trace.gradients[0];
    let backward = bincode::serialize(&grad.backward).unwrap();
    let application = grad.application.as_mut().unwrap();
    let specialized = bincode::serialize(&application.specialized).unwrap();
    let spliced = bincode::serialize(&application.after_splice).unwrap();
    let chelis_ir::lowering_trace::Value::Node(result) = application.result else {
        panic!("single gradient result");
    };
    application
        .after_packing
        .node_mut(result)
        .unwrap()
        .shape_deps
        .push(result);
    application
        .after_packing
        .node_mut(result)
        .unwrap()
        .merged_spans
        .push("later mutation".into());
    assert_eq!(bincode::serialize(&grad.backward).unwrap(), backward);
    assert_eq!(
        bincode::serialize(&application.specialized).unwrap(),
        specialized
    );
    assert_eq!(
        bincode::serialize(&application.after_splice).unwrap(),
        spliced
    );
    assert_ne!(
        bincode::serialize(&application.after_packing).unwrap(),
        spliced
    );
}

#[cfg(feature = "lowering-trace")]
#[test]
fn full_integer_constants_keep_bits_beyond_the_f64_exact_range() {
    let source = "def values() -> tensor[2, i64] = [9007199254740993, -9007199254740993]\n";
    let (library, trace) = try_lower_program_to_library_with_trace(&checked(source)).unwrap();
    for dag in [
        &trace.normalization.before_dce,
        &trace.normalization.after_dce,
        &trace.normalization.after_copies,
        &trace.normalization.after_drops,
        library.dag(),
    ] {
        let payload = dag
            .nodes()
            .iter()
            .find_map(|node| match &node.op {
                RiscOp::ConstTensor { data } => Some(data),
                _ => None,
            })
            .expect("nonuniform complete tensor literal");
        assert_eq!(
            payload.scalar_at(0),
            chelis_types::scalar_from_i64(
                "trace test",
                chelis_types::types::Prim::Int64,
                9_007_199_254_740_993
            )
            .unwrap()
        );
        assert_eq!(
            payload.scalar_at(1),
            chelis_types::scalar_from_i64(
                "trace test",
                chelis_types::types::Prim::Int64,
                -9_007_199_254_740_993
            )
            .unwrap()
        );
    }
}

#[cfg(feature = "lowering-trace")]
#[test]
fn unresolved_and_vectorized_gradients_are_explicit_boundaries() {
    let source = r#"
def unknown(model: tensor[3, f32] -> f32, x: tensor[3, f32]) -> tensor[3, f32] = {
  target = fn (v: tensor[3, f32]) -> model(v)
  grad(target, wrt=v)(x)
}
"#;
    let (library, trace) = try_lower_program_to_library_with_trace(&checked(source)).unwrap();
    assert!(library.rootless_defs().contains("unknown"));
    assert!(trace.gradients.is_empty());
    assert_eq!(trace.boundaries.len(), 1);
    assert_eq!(
        trace.boundaries[0].kind,
        BoundaryKind::UnresolvedCallableGradient
    );
    assert_eq!(
        trace.contexts[trace.boundaries[0].context.0].kind,
        ContextKind::Gradient
    );

    for (transform, result_type, boundary, context) in [
        (
            "vmap(loss)",
            "tensor[2, f32]",
            BoundaryKind::Vmap,
            ContextKind::Vmap,
        ),
        (
            "vmap(grad(loss))",
            "tensor[2, 3, f32]",
            BoundaryKind::VmapGradient,
            ContextKind::VmapGradient,
        ),
    ] {
        let source = format!(
            r#"
def loss(x: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(mul(x, x), 0))
def mapped(x: tensor[2, 3, f32]) -> {result_type} = x |> {transform}
"#
        );
        let program = checked(&source);
        let (library, trace) = try_lower_program_to_library_with_trace(&program).unwrap();
        same_dag(
            library.dag(),
            try_lower_program_to_library(&program).unwrap().dag(),
        );
        assert!(trace.gradients.is_empty());
        assert_eq!(trace.boundaries.len(), 1);
        assert_eq!(trace.boundaries[0].kind, boundary);
        assert_eq!(trace.contexts[1].kind, context);
    }
}
