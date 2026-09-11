//! Same-compilation helper trace acceptance tests.
//!
//! These tests exercise the opt-in host-helper ingress. The trace is an
//! observation of the helper that host lowering actually retained, not a
//! second library lowering or an independently pairable graph.
#![cfg(feature = "lowering-trace")]

use chelis_effects::realizability::{compute_root_manifest, infer_realizability};
use chelis_ir::dag::RiscOp;
use chelis_ir::host::{
    try_lower_manifested_execution_program, try_lower_manifested_execution_program_with_trace,
};
use chelis_ir::lowering_trace::Value;
use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::manifest::ManifestedProgram;
use chelis_types::types::{Prim, Target};
use chelis_types::{CheckedProgram, check_linearity, check_typed_program};

const C_TENSOR_PRIMS: &[Prim] = &[
    Prim::F32,
    Prim::Bool,
    Prim::Bf16,
    Prim::F16,
    Prim::Int32,
    Prim::Int64,
];

fn manifested(source: &str) -> ManifestedProgram {
    let deep = desugar_program(&parse_str(source).expect("parse fixture"));
    let checked: CheckedProgram = check_typed_program(&deep).expect("check fixture types");
    let checked = chelis_effects::check_program(&checked).expect("check fixture effects");
    let checked = check_linearity(&checked).expect("check fixture linearity");
    let realizability = infer_realizability(&checked, C_TENSOR_PRIMS);
    let manifest = compute_root_manifest(&checked, &realizability);
    ManifestedProgram::new(checked, manifest, Target::C)
}

fn same_dag(left: &chelis_ir::Dag, right: &chelis_ir::Dag) {
    assert_eq!(
        bincode::serialize(left).unwrap(),
        bincode::serialize(right).unwrap()
    );
}

#[test]
fn traced_helper_is_the_ordinary_retained_helper_and_projects_with_its_function() {
    let program = manifested(
        r#"
def selected(x: tensor[3, f32], flag: bool) -> (tensor[3, f32], bool) =
  (add(x, x), flag)
def discarded(x: tensor[3, f32], flag: bool) -> (tensor[3, f32], bool) =
  (mul(x, x), flag)
"#,
    );
    let (_, ordinary) = try_lower_manifested_execution_program(&program).unwrap();
    let (_, traced) = try_lower_manifested_execution_program_with_trace(&program).unwrap();
    let ordinary = ordinary.expect("ordinary host plan");
    let traced = traced.expect("traced host plan");

    for name in ["selected", "discarded"] {
        let ordinary_function = ordinary
            .program()
            .functions
            .iter()
            .find(|function| function.name == name)
            .expect("ordinary function");
        let traced_function = traced
            .program()
            .functions
            .iter()
            .find(|function| function.name == name)
            .expect("traced function");
        assert_eq!(ordinary_function.tensor_helpers.len(), 1);
        assert_eq!(traced_function.tensor_helpers.len(), 1);
        same_dag(
            &ordinary_function.tensor_helpers[0].dag,
            &traced_function.tensor_helpers[0].dag,
        );
        let trace = traced
            .function_helper_trace(name, 0)
            .unwrap()
            .expect("opt-in helper trace");
        same_dag(
            trace
                .after_dimension_rebinding
                .as_ref()
                .expect("host helper records final dimension rebinding"),
            &traced_function.tensor_helpers[0].dag,
        );
    }
    assert!(!ordinary.has_helper_traces());
    assert!(
        ordinary
            .function_helper_trace("selected", 0)
            .unwrap()
            .is_none()
    );
    assert!(traced.has_helper_traces());

    let projected = traced.project_functions(&["selected".into()]).unwrap();
    assert!(
        projected
            .function_helper_trace("selected", 0)
            .unwrap()
            .is_some()
    );
    assert!(projected.function_helper_trace("discarded", 0).is_err());
    assert!(projected.into_ordinary().is_err());

    let (_, traced) = try_lower_manifested_execution_program_with_trace(&program).unwrap();
    let ordinary = traced
        .unwrap()
        .discard_helper_traces()
        .into_ordinary()
        .expect("explicit trace opt-out retains no execution metadata");
    assert!(
        ordinary
            .functions
            .iter()
            .any(|function| function.name == "selected")
    );
}

#[test]
fn helper_trace_retains_actual_ad_and_ordered_duplicate_root_packing() {
    let program = manifested(
        r#"
def loss(x: tensor[3, f32], y: tensor[3, f32]) -> f32 =
  tensor_to_scalar(sum(x, 0))
def derivative(x: tensor[3, f32], y: tensor[3, f32])
  -> (tensor[3, f32], tensor[3, f32]) = grad(loss)(x, y)
"#,
    );
    let (_, plan) = try_lower_manifested_execution_program_with_trace(&program).unwrap();
    let plan = plan.expect("host execution plan");
    let function = plan
        .program()
        .functions
        .iter()
        .find(|function| function.name == "derivative")
        .expect("derivative function");
    let helper = &function.tensor_helpers[0];
    let trace = plan
        .function_helper_trace("derivative", 0)
        .unwrap()
        .expect("derivative helper trace");

    assert_eq!(trace.lowering.gradients.len(), 1);
    let gradient = &trace.lowering.gradients[0];
    assert_eq!(gradient.wrt.len(), 2);
    assert_eq!(gradient.gradients.len(), 1, "raw AD keeps y missing");
    let application = gradient.application.as_ref().expect("packed application");
    let Value::Tuple(results) = &application.result else {
        panic!("two ordered cotangents stay a tuple");
    };
    assert_eq!(results.len(), 2);
    let Value::Node(disconnected) = results[1] else {
        panic!("disconnected tensor cotangent is a shaped zero");
    };
    assert!(application.after_splice.get(disconnected).is_none());
    assert_eq!(
        application
            .after_packing
            .get(disconnected)
            .unwrap()
            .output_type
            .dims,
        vec![chelis_ir::DimInfo::Lit(3)]
    );
    assert_eq!(helper.dag.roots().len(), 2);
    assert_ne!(helper.dag.roots()[0], helper.dag.roots()[1]);
    assert_eq!(
        trace.packed_roots,
        trace.lowering.normalization.before_dce.roots()
    );
    assert_eq!(
        trace.packed_result,
        Value::Tuple(
            trace
                .packed_roots
                .iter()
                .copied()
                .map(Value::Node)
                .collect()
        )
    );
    same_dag(
        trace.after_dimension_rebinding.as_ref().unwrap(),
        &helper.dag,
    );
}

#[test]
fn named_axis_helper_records_dimension_rebinding_without_rewriting_pass_snapshots() {
    let program = manifested(
        r#"
def selected(
  x: tensor[batch, inner, f32],
  w: tensor[inner, output, f32],
  flag: bool
) -> (tensor[batch, output, f32], bool) = (matmul(x, w), flag)
"#,
    );
    let (_, plan) = try_lower_manifested_execution_program_with_trace(&program).unwrap();
    let plan = plan.expect("host execution plan");
    let function = plan
        .program()
        .functions
        .iter()
        .find(|function| function.name == "selected")
        .expect("selected function");
    let helper = &function.tensor_helpers[0];
    let trace = plan
        .function_helper_trace("selected", 0)
        .unwrap()
        .expect("selected helper trace");
    let before = &trace.lowering.normalization.after_drops;
    let after = trace
        .after_dimension_rebinding
        .as_ref()
        .expect("actual host rebinding output");
    same_dag(after, &helper.dag);
    assert_ne!(
        bincode::serialize(before).unwrap(),
        bincode::serialize(after).unwrap(),
        "the discriminator must exercise a nonidentity dimension rebinding"
    );
    let before_again = &trace.lowering.normalization.after_drops;
    same_dag(before_again, before);
}

#[test]
fn helper_packing_keeps_duplicate_cotangent_slots_as_distinct_ordered_roots() {
    let program = manifested(
        r#"
def duplicated(x: tensor[3, f32], flag: bool)
  -> (tensor[3, f32], tensor[3, f32], bool) = (x, x, flag)
"#,
    );
    let (_, plan) = try_lower_manifested_execution_program_with_trace(&program).unwrap();
    let plan = plan.expect("host execution plan");
    let function = plan
        .program()
        .functions
        .iter()
        .find(|function| function.name == "duplicated")
        .expect("duplicated function");
    let helper = &function.tensor_helpers[0];
    let trace = plan
        .function_helper_trace("duplicated", 0)
        .unwrap()
        .expect("duplicated helper trace");
    assert_eq!(
        trace.packed_roots,
        trace.lowering.normalization.before_dce.roots()
    );
    assert_eq!(trace.packed_roots.len(), 2);
    assert_ne!(trace.packed_roots[0], trace.packed_roots[1]);
    assert!(matches!(
        trace
            .lowering
            .normalization
            .before_dce
            .get(trace.packed_roots[1])
            .unwrap()
            .op,
        RiscOp::Copy
    ));
    assert_eq!(
        trace.packed_result,
        Value::Tuple(
            trace
                .packed_roots
                .iter()
                .copied()
                .map(Value::Node)
                .collect()
        )
    );
}

#[test]
fn fixed_control_helper_trace_captures_actual_pre_and_post_ad_execution() {
    let program = manifested(
        r#"
def loss(x: tensor[32, f32]) -> f32 = with seed(42i64) {
  draw_free = with seed(7i64) { x }
  tensor_to_scalar(sum(dropout(draw_free, 0.5f32), 0))
}
def derivative(x: tensor[32, f32]) -> tensor[32, f32] = grad(loss)(x)
"#,
    );
    let (_, plan) = try_lower_manifested_execution_program_with_trace(&program).unwrap();
    let plan = plan.expect("host execution plan");
    let trace = plan
        .function_helper_trace("derivative", 0)
        .unwrap()
        .expect("fixed-control helper trace");
    assert_eq!(trace.executions.len(), 1);
    let execution = &trace.executions[0];
    let gradient = &trace.lowering.gradients[execution.gradient];
    same_dag(execution.forward.dag_for_inspection(), &gradient.forward);
    same_dag(execution.backward.dag_for_inspection(), &gradient.backward);
    assert_eq!(
        execution
            .forward
            .dag_for_inspection()
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::Dropout { .. }))
            .count(),
        1
    );
    assert_eq!(
        execution
            .backward
            .dag_for_inspection()
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::Dropout { .. }))
            .count(),
        2,
        "post-AD capture includes the actual backward replay"
    );
    let application = &trace.applications[0];
    assert_eq!(application.gradient, execution.gradient);
    assert_eq!(application.remap.occurrences.len(), 5);
    assert_eq!(application.remap.draws.len(), 1);
    assert_eq!(application.remap.scopes.len(), 3);
    for (source, target) in &application.remap.occurrences {
        assert!(
            execution
                .backward
                .source_for_inspection()
                .iter()
                .any(|occurrence| occurrence.id == *source)
        );
        assert!(
            application
                .after_splice
                .source
                .iter()
                .any(|occurrence| occurrence.id == *target)
        );
    }
    let normalization = trace
        .normalization
        .as_ref()
        .expect("fixed-control helper records execution normalization");
    assert!(!normalization.before_dce.steps.is_empty());
    assert!(!normalization.after_drops.steps.is_empty());
}

#[test]
fn failed_helper_attempt_returns_no_trace_and_cannot_pollute_the_next_compilation() {
    let rejected = manifested(
        r#"
def loss(x: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(floor(x), 0))
def derivative(x: tensor[3, f32]) -> tensor[3, f32] = grad(loss)(x)
"#,
    );
    let error = try_lower_manifested_execution_program_with_trace(&rejected)
        .expect_err("the actual AD pass rejects floor");
    assert!(error.to_string().contains("floor"));

    let accepted = manifested(
        r#"
def loss(x: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(mul(x, x), 0))
def derivative(x: tensor[3, f32]) -> tensor[3, f32] = grad(loss)(x)
"#,
    );
    let (_, plan) = try_lower_manifested_execution_program_with_trace(&accepted).unwrap();
    let plan = plan.expect("successful host plan");
    let trace = plan
        .function_helper_trace("derivative", 0)
        .unwrap()
        .expect("successful helper trace");
    assert_eq!(trace.lowering.gradients.len(), 1);
    assert!(trace.executions.is_empty());
    assert!(trace.normalization.is_none());
}
