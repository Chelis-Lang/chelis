//! Same actual compilation and final-success pairing; not a certificate.
#![cfg(feature = "compilation-trace")]

use chelis_compiler_api::compilation_trace::SelectedLowering;
use chelis_compiler_api::compiler::{compile_for_execution, compile_for_execution_with_trace};
use chelis_compiler_api::emission_observer::SelectedEmission;
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, SourceKind};

fn request(source: &str, entry: &str) -> CompileRequest {
    CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        target: CompileTarget::C,
        entry_name: Some(entry.into()),
    }
}

#[test]
fn direct_fixed_entry_pairs_actual_helper_normalization_and_complete_artifact() {
    let source =
        "def main(x: tensor[4,f32]) -> tensor[4,f32] = with seed(42i64) { dropout(x,0.5f32) }";
    let ordinary = compile_for_execution(request(source, "main")).unwrap();
    let captured = compile_for_execution_with_trace(request(source, "main"), |observation| {
        let SelectedLowering::Dag(trace) = observation.lowering else {
            panic!("actual direct entry capture");
        };
        let SelectedEmission::Dag { unfused, selected } = observation.emission.selected else {
            panic!("actual direct entry emission");
        };
        assert_eq!(
            bincode::serialize(&trace.lowering.normalization.after_drops).unwrap(),
            bincode::serialize(unfused).unwrap()
        );
        assert_eq!(
            bincode::serialize(unfused.nodes()).unwrap(),
            bincode::serialize(selected.nodes()).unwrap()
        );
        assert!(
            !trace
                .normalization
                .as_ref()
                .unwrap()
                .after_drops
                .source
                .is_empty()
        );
        selected.roots().len()
    })
    .unwrap();
    assert_eq!(*captured.projection(), 1);
    assert_eq!(
        serde_json::to_value(&ordinary).unwrap(),
        serde_json::to_value(captured.artifact()).unwrap()
    );
}

#[test]
fn gradient_host_capture_comes_from_the_selected_helpers_actual_ad() {
    for (source, input_count) in [
        (
            r#"
def loss(x: tensor[4,f32]) -> f32 = tensor_to_scalar(sum(dropout(x,0.5f32),0))
def derivative(x: tensor[4,f32]) -> tensor[4,f32] = with seed(42i64) { grad(loss)(x) }
"#,
            1,
        ),
        (
            r#"
def loss(x: tensor[4,f32], y: tensor[4,f32]) -> f32 =
  tensor_to_scalar(sum(dropout(mul(x,y),0.5f32),0))
def derivative(x: tensor[4,f32], y: tensor[4,f32]) -> (tensor[4,f32],tensor[4,f32]) =
  with seed(42i64) { grad(loss)(x,y) }
"#,
            2,
        ),
    ] {
        let ordinary = compile_for_execution(request(source, "derivative")).unwrap();
        let captured = compile_for_execution_with_trace(request(source, "derivative"), |observation| {
        let SelectedLowering::Host(traces) = observation.lowering else {
            panic!("actual selected host captures");
        };
        let SelectedEmission::Host(host) = observation.emission.selected else {
            panic!("selected host");
        };
        let mut gradients = 0;
        for (index, (name, helpers)) in traces.functions().enumerate() {
            let function = host.function(index).unwrap();
            assert_eq!(name, function.name());
            assert_eq!(helpers.len(), function.tensor_helper_count());
            for (helper_index, trace) in helpers.iter().enumerate() {
                if let Some(trace) = trace {
                    let helper = function.tensor_helper(helper_index).unwrap();
                    let captured = trace.after_dimension_rebinding.as_ref().unwrap();
                    assert_eq!(
                        bincode::serialize(&(captured.nodes(), captured.roots())).unwrap(),
                        bincode::serialize(&(helper.dag().nodes(), helper.dag().roots())).unwrap(),
                        "the complete selected graph includes ordered roots, not just nodes"
                    );
                    gradients += trace.executions.len();
                    assert_eq!(trace.executions.len(), trace.applications.len());
                    for execution in &trace.executions {
                        assert_eq!(
                            trace.lowering.gradients[execution.gradient].wrt.len(),
                            input_count,
                            "the selected capture retains every ordered differentiated input"
                        );
                    }
                }
            }
        }
        gradients
    })
    .unwrap();
        assert!(
            *captured.projection() > 0,
            "empty captures cannot stand in for actual AD"
        );
        assert_eq!(
            serde_json::to_value(&ordinary).unwrap(),
            serde_json::to_value(captured.artifact()).unwrap()
        );
    }
}

#[test]
fn ordinary_compilation_keeps_its_lane_and_explicitly_missing_capture() {
    for source in [
        "def main(x: tensor[4,f32]) -> tensor[4,f32] = mul(x,x)",
        "def main(x: f32) -> f32 = x + 1.0f32",
    ] {
        let ordinary = compile_for_execution(request(source, "main")).unwrap();
        let captured = compile_for_execution_with_trace(request(source, "main"), |observation| {
            assert!(matches!(
                observation.lowering,
                SelectedLowering::Unavailable
            ));
            // A nested ordinary compile must not inherit a process-global collector.
            compile_for_execution(request("def main(x: f32) -> f32 = x", "main")).unwrap();
        })
        .unwrap();
        assert_eq!(
            serde_json::to_value(&ordinary).unwrap(),
            serde_json::to_value(captured.artifact()).unwrap()
        );
    }
}

#[test]
fn tracing_preserves_forward_failure_in_ordinary_host_control() {
    for source in [
        "def main(x: tensor[4,f32], flag: bool) -> tensor[4,f32] = if flag then fail(\"stop\") else x",
        "def bad(x: tensor[4,f32]) -> tensor[4,f32] = fail(\"stop\")\ndef main(x: tensor[4,f32], flag: bool) -> tensor[4,f32] = if flag then bad(x) else x",
    ] {
        let ordinary = compile_for_execution(request(source, "main")).unwrap();
        let traced = compile_for_execution_with_trace(request(source, "main"), |_| ()).unwrap();
        let c = |artifact: &chelis_compiler_api::compiler::CompiledExecutionArtifact| {
            artifact
                .compile_result
                .files
                .iter()
                .find(|file| file.path == "chelis_main.c")
                .unwrap()
                .contents
                .clone()
        };
        assert!(c(&ordinary).contains("chelis_fail("));
        assert!(c(traced.artifact()).contains("chelis_fail("));
        assert_eq!(
            serde_json::to_value(ordinary).unwrap(),
            serde_json::to_value(traced.artifact()).unwrap()
        );
    }
}

#[test]
fn tracing_preserves_forward_failure_beside_planned_dropout() {
    let source = "def bad(x: tensor[4,f32]) -> tensor[4,f32] = if tensor_to_scalar(sum(x,0)) > 0.0f32 then fail(\"stop\") else x\ndef random(x: tensor[4,f32]) -> tensor[4,f32] = with seed(42i64) { dropout(x,0.5f32) }\ndef main(x: tensor[4,f32], flag: bool) -> (tensor[4,f32],bool) = (bad(random(x)),flag)";
    let ordinary = compile_for_execution(request(source, "main")).unwrap();
    let traced = compile_for_execution_with_trace(request(source, "main"), |_| ()).unwrap();
    assert_eq!(
        serde_json::to_value(ordinary).unwrap(),
        serde_json::to_value(traced.artifact()).unwrap()
    );
}

#[test]
fn later_compiler_failure_never_returns_an_earlier_projection() {
    let source = "def process(x: tensor[4,f32]) -> tensor[4,f32] = relu(x)\ndef batch_process(xs: tensor[8,4,f32]) -> tensor[8,4,f32] = xs |> vmap(process)";
    let ordinary = compile_for_execution(request(source, "batch_process")).unwrap_err();
    let mut calls = 0;
    let error = compile_for_execution_with_trace(request(source, "batch_process"), |_| {
        calls += 1;
        "apparently valid projection"
    })
    .unwrap_err();
    assert_eq!(
        calls, 1,
        "exercise a failure after observation, not an early rejection"
    );
    assert_eq!(ordinary.stage, error.stage);
    assert_eq!(
        serde_json::to_value(ordinary.errors).unwrap(),
        serde_json::to_value(error.errors).unwrap()
    );
}

#[test]
fn source_rejection_never_invokes_projection() {
    let source =
        "def main(x: tensor[4,f32]) -> tensor[4,f32] = with device(\"gpu:0\") { mul(x,x) }";
    let ordinary = compile_for_execution(request(source, "main")).unwrap_err();
    let error = compile_for_execution_with_trace(request(source, "main"), |_| {
        panic!("rejected source observed")
    })
    .unwrap_err();
    assert_eq!(ordinary.stage, error.stage);
    assert_eq!(
        serde_json::to_value(ordinary.errors).unwrap(),
        serde_json::to_value(error.errors).unwrap()
    );
}
