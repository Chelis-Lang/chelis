//! Same actual compilation and final-success pairing; not a certificate.
#![cfg(feature = "compilation-trace")]

use chelis_compiler_api::compilation_trace::SelectedLowering;
use chelis_compiler_api::compiler::{compile_for_execution, compile_for_execution_with_trace};
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, SourceKind};

fn request(source: &str, entry: &str) -> CompileRequest {
    CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        target: CompileTarget::C,
        entry_name: Some(entry.into()),
    }
}

/// Every entry takes its ordinary lane, and its capture is explicitly
/// unavailable. That includes a keyed draw and a gradient through one: the
/// dropout-only entry and host lanes whose lowering the trace captured were
/// selected by the retired counter-stream draw key.
#[test]
fn ordinary_compilation_keeps_its_lane_and_explicitly_missing_capture() {
    for source in [
        "def main(x: tensor[4,f32]) -> tensor[4,f32] = mul(x,x)",
        "def main(x: f32) -> f32 = x + 1.0f32",
        "def main(x: tensor[4,f32]) -> tensor[4,f32] = with device(\"cpu\") { mul(x,x) }",
        "def main(x: tensor[4,f32]) -> tensor[4,f32] = dropout(key_from_seed(42i64),x,0.5f32)",
        "def loss(k: key, x: tensor[4,f32]) -> f32 = tensor_to_scalar(sum(dropout(k,x,0.5f32),0))\ndef main(x: tensor[4,f32]) -> tensor[4,f32] = grad(loss, wrt=x)(key_from_seed(42i64),x)",
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
fn direct_gradient_with_unused_gpu_sibling_is_an_explicit_selection_boundary() {
    let source = r#"
def loss(k: key, x: tensor[4,f32]) -> f32 = with device("cpu") {
  tensor_to_scalar(sum(dropout(k,x,0.5f32),0))
}
def derivative(x: tensor[4,f32]) -> tensor[4,f32] = grad(loss, wrt=x)(key_from_seed(42i64),x)
def unused_gpu(x: tensor[4,f32]) -> tensor[4,f32] = with device("gpu:0") { mul(x,x) }
"#;
    let ordinary = compile_for_execution(request(source, "derivative")).unwrap_err();
    let mut projections = 0;
    let traced = compile_for_execution_with_trace(request(source, "derivative"), |_| {
        projections += 1;
    })
    .unwrap_err();
    assert_eq!(projections, 0);
    assert_eq!(ordinary.stage, "effects");
    assert_eq!(traced.stage, ordinary.stage);
    assert!(ordinary.errors.iter().all(|error| {
        error.kind() == chelis_vocab::DiagnosticKind::BuildTargetMismatch
            && error.message.contains("resource region `gpu:0`")
            && !error.message.contains("resource region `cpu`")
    }));
    assert_eq!(
        serde_json::to_value(traced.errors).unwrap(),
        serde_json::to_value(ordinary.errors).unwrap()
    );
}

#[test]
fn selected_gpu_resource_rejects_before_trace_projection() {
    let source = r#"
def loss(k: key, x: tensor[4,f32]) -> f32 = with device("gpu:0") {
  tensor_to_scalar(sum(dropout(k,x,0.5f32),0))
}
def derivative(x: tensor[4,f32]) -> tensor[4,f32] = grad(loss, wrt=x)(key_from_seed(42i64),x)
"#;
    let ordinary = compile_for_execution(request(source, "derivative")).unwrap_err();
    let mut projections = 0;
    let traced = compile_for_execution_with_trace(request(source, "derivative"), |_| {
        projections += 1;
    })
    .unwrap_err();
    assert_eq!(projections, 0);
    assert_eq!(ordinary.stage, "effects");
    assert_eq!(traced.stage, ordinary.stage);
    assert_eq!(
        serde_json::to_value(traced.errors).unwrap(),
        serde_json::to_value(ordinary.errors).unwrap()
    );
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
    let source = "def bad(x: tensor[4,f32]) -> tensor[4,f32] = if tensor_to_scalar(sum(x,0)) > 0.0f32 then fail(\"stop\") else x\ndef random(x: tensor[4,f32]) -> tensor[4,f32] = dropout(key_from_seed(42i64),x,0.5f32)\ndef main(x: tensor[4,f32], flag: bool) -> (tensor[4,f32],bool) = (bad(random(x)),flag)";
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
