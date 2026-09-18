//! Resource regions must be checked against the actual selected build target.
//! These are compilation/admission tests, not GPU execution claims.

use chelis_compiler_api::compiler::{compile, compile_for_execution};
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, SourceKind};

fn request(source: &str, target: CompileTarget) -> CompileRequest {
    CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        target,
        entry_name: Some("main".into()),
    }
}

fn function(device: &str) -> String {
    format!(
        "def main(x: tensor[2, f32]) -> tensor[2, f32] = \
         with device(\"{device}\") {{ mul(x, x) }}"
    )
}

#[test]
fn ordinary_and_callable_apis_enforce_the_selected_target() {
    for (target, device, allowed) in [
        (CompileTarget::C, "cpu", true),
        (CompileTarget::C, "cpu:author-device", false),
        (CompileTarget::C, "cpu:socket_9", false),
        (CompileTarget::C, "gpu:0", false),
        (CompileTarget::Hip, "gpu:0", true),
        (CompileTarget::Hip, "cpu", false),
    ] {
        let source = function(device);
        for result in [
            compile(request(&source, target)).map(|_| ()),
            compile_for_execution(request(&source, target)).map(|_| ()),
        ] {
            if allowed {
                result.unwrap();
            } else {
                let error = result.unwrap_err();
                assert_eq!(error.stage, "effects");
                assert!(error.transcript.is_empty());
                assert!(
                    error
                        .errors
                        .iter()
                        .all(|d| d.kind() == chelis_vocab::DiagnosticKind::BuildTargetMismatch)
                );
                assert!(error.errors.iter().all(|d| !d.suggestions.is_empty()));
                assert!(
                    error
                        .errors
                        .iter()
                        .any(|d| d.message.contains("cannot satisfy resource region")),
                    "{error:?}"
                );
            }
        }
    }
}

#[test]
fn c_apis_reject_every_non_host_or_malformed_device_before_emission() {
    for device in [
        "cpu:0",
        "cpu:author-device",
        "cpu:socket_9",
        "cpu:HOST_2",
        "cuda:0",
        "metal",
        "rocm",
        "xpu:1",
        "Gpu:0",
        "gpu:0",
        "host",
        "",
        "cpu:",
        "cpu:two words",
        "cpu:/0",
    ] {
        let source = function(device);
        for result in [
            compile(request(&source, CompileTarget::C)).map(|_| ()),
            compile_for_execution(request(&source, CompileTarget::C)).map(|_| ()),
        ] {
            let error = result.unwrap_err();
            assert_eq!(error.stage, "effects", "{device}: {error:?}");
            assert!(error.transcript.is_empty(), "{device}: {error:?}");
            assert_eq!(error.errors.len(), 1, "{device}: {error:?}");
            assert_eq!(
                error.errors[0].kind(),
                chelis_vocab::DiagnosticKind::BuildTargetMismatch,
                "{device}"
            );
            assert_eq!(
                error.errors[0].message,
                format!(
                    "`chelis build --target c` cannot satisfy resource region `{device}`: \
                     host C accepts only exact `cpu`"
                ),
                "{device}"
            );
            assert_eq!(error.errors[0].suggestions.len(), 1, "{device}");
        }
    }
}

#[test]
fn c_apis_preserve_unpinned_and_explicit_host_programs() {
    for source in [
        "def main(x: tensor[2, f32]) -> tensor[2, f32] = mul(x, x)",
        "def main(x: tensor[2, f32]) -> tensor[2, f32] = with device(\"cpu\") { mul(x, x) }",
        "def main(x: tensor[2, f32]) -> tensor[2, f32] = with device(\"cpu\") { with device(\"cpu\") { mul(x, x) } }",
    ] {
        compile(request(source, CompileTarget::C)).unwrap_or_else(|error| {
            panic!("{source}: {error:?}");
        });
        compile_for_execution(request(source, CompileTarget::C)).unwrap_or_else(|error| {
            panic!("{source}: {error:?}");
        });
    }
}

#[test]
fn deep_api_rejects_non_host_device_with_the_same_typed_diagnostic() {
    for device in ["cuda:0", "cpu:author-device", "cpu:socket_9"] {
        let source = r#"(def {} main
      (fn {}
        (params {} (x {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))}))
        (handle-effect {effect: resource}
          (lit {type: (t-prim {} string)} "__DEVICE__")
          (app {} (var {} mul) (var {} x) (var {} x)))))"#
            .replace("__DEVICE__", device);
        let error = compile(CompileRequest {
            source_kind: SourceKind::Deep,
            source,
            target: CompileTarget::C,
            entry_name: Some("main".into()),
        })
        .unwrap_err();
        assert_eq!(error.stage, "effects", "{device}: {error:?}");
        assert!(error.transcript.is_empty(), "{device}: {error:?}");
        assert_eq!(error.errors.len(), 1, "{device}: {error:?}");
        assert_eq!(
            error.errors[0].kind(),
            chelis_vocab::DiagnosticKind::BuildTargetMismatch
        );
        assert_eq!(
            error.errors[0].message,
            format!(
                "`chelis build --target c` cannot satisfy resource region `{device}`: \
                 host C accepts only exact `cpu`"
            )
        );
    }
}

#[test]
fn legacy_value_root_compilation_checks_resource_regions() {
    for (device, allowed) in [("cpu", true), ("gpu:0", false)] {
        let source = format!("x: i32 = with device(\"{device}\") {{ 1 }}");
        let result = compile(request(&source, CompileTarget::C));
        if allowed {
            result.unwrap();
        } else {
            let error = result.map(|_| ()).unwrap_err();
            assert!(
                error
                    .errors
                    .iter()
                    .any(|d| d.message.contains("cannot satisfy resource region")),
                "{error:?}"
            );
        }
    }
}

#[test]
fn selected_tensor_entry_checks_only_its_reachable_resource_regions() {
    let source = "def gpu_helper(x: tensor[2, f32]) -> tensor[2, f32] = with device(\"gpu:0\") { mul(x, x) }\n\
                  def cpu_helper(x: tensor[2, f32]) -> tensor[2, f32] = with device(\"cpu\") { mul(x, x) }\n\
                  def main(x: tensor[2, f32]) -> tensor[2, f32] = cpu_helper(x)\n";
    compile_for_execution(request(source, CompileTarget::C)).unwrap();
    let source = source.replace("= cpu_helper(x)", "= gpu_helper(x)");
    let error = compile_for_execution(request(&source, CompileTarget::C))
        .map(|_| ())
        .unwrap_err();
    assert!(
        error
            .errors
            .iter()
            .any(|d| d.message.contains("cannot satisfy resource region")),
        "{error:?}"
    );
}

#[test]
fn a_local_parameter_does_not_select_a_same_named_resource_helper() {
    let source = "def gpu_helper(x: tensor[2, f32]) -> tensor[2, f32] = with device(\"gpu:0\") { mul(x, x) }\n\
                  def main(gpu_helper: tensor[2, f32]) -> tensor[2, f32] = mul(gpu_helper, gpu_helper)\n";
    compile_for_execution(request(source, CompileTarget::C)).unwrap();
    let local_binding = "def gpu_helper(x: tensor[2, f32]) -> tensor[2, f32] = with device(\"gpu:0\") { mul(x, x) }\n\
                         def main(x: tensor[2, f32]) -> tensor[2, f32] = {\n\
                         \x20\x20gpu_helper = x\n\
                         \x20\x20mul(gpu_helper, gpu_helper)\n}\n";
    compile_for_execution(request(local_binding, CompileTarget::C)).unwrap();
}

#[test]
fn selected_scalar_entry_preserves_legacy_whole_program_distinction() {
    let source = "def gpu_helper(x: f32) -> f32 = with device(\"gpu:0\") { x * x }\n\
                  def main(x: f32) -> f32 = with device(\"cpu\") { x * x }\n";
    compile_for_execution(request(source, CompileTarget::C)).unwrap();
    // This surface emits the whole host program, unlike the strict callable API.
    let error = compile(request(source, CompileTarget::C))
        .map(|_| ())
        .unwrap_err();
    assert!(
        error
            .errors
            .iter()
            .any(|d| d.message.contains("cannot satisfy resource region")),
        "{error:?}"
    );
    let called = "def gpu_helper(x: f32) -> f32 = with device(\"gpu:0\") { x * x }\n\
                  def bridge(x: f32) -> f32 = gpu_helper(x)\n\
                  def main(x: f32) -> f32 = bridge(x)\n";
    let error = compile_for_execution(request(called, CompileTarget::C))
        .map(|_| ())
        .unwrap_err();
    assert!(
        error
            .errors
            .iter()
            .any(|d| d.message.contains("cannot satisfy resource region")),
        "{error:?}"
    );
}

#[test]
fn resource_requirements_survive_imports_and_context_serialization() {
    use chelis_compiler_api::compiler::compile_for_execution_in_context;
    use chelis_compiler_api::{COMPILER_VERSION, CompiledContext, compile_reef_context};
    use std::fs;

    let dir = tempfile::TempDir::new().unwrap();
    let root = dir.path();
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("reef.toml"),
        format!("[package]\nname = \"resource-probe\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Probe\"\n"),
    ).unwrap();
    fs::write(
        root.join("src/lib.ch"),
        "module Probe.Lib\nexport (cpu_helper, gpu_helper)\n\
         def cpu_helper(x: tensor[2, f32]) -> tensor[2, f32] = mul(x, x)\n\
         def gpu_helper(x: tensor[2, f32]) -> tensor[2, f32] = with device(\"gpu:0\") { mul(x, x) }\n",
    ).unwrap();
    let context = compile_reef_context(&root.join("reef-home"), root).unwrap();
    let restored = CompiledContext::decode(&context.encode().unwrap()).unwrap();
    for context in [&context, &restored] {
        for (helper, allowed) in [("cpu_helper", true), ("gpu_helper", false)] {
            let source = format!(
                "module Probe.Eval\nimport Probe.Lib ({helper})\n\
                 def main(x: tensor[2, f32]) -> tensor[2, f32] = {helper}(x)\n"
            );
            let result =
                compile_for_execution_in_context(context, &source, CompileTarget::C, Some("main"));
            if allowed {
                result.unwrap();
            } else {
                let error = result.map(|_| ()).unwrap_err();
                // The contextual lane currently declines region-bearing helpers
                // before selecting a callable tensor DAG. Do not turn that
                // independent limitation into a claim of Resource validation.
                assert!(
                    error
                        .errors
                        .iter()
                        .any(|d| d.message.contains("no callable tensor-kernel form")),
                    "{error:?}"
                );
            }
        }
    }
}

#[test]
fn executable_cpu_example_is_target_checked() {
    let source = include_str!("../../../examples/resource_target_cpu.ch");
    compile(request(source, CompileTarget::C)).unwrap();
    let error = compile(request(source, CompileTarget::Hip))
        .map(|_| ())
        .unwrap_err();
    assert!(
        error
            .errors
            .iter()
            .any(|d| d.message.contains("cannot satisfy resource region")),
        "{error:?}"
    );
}
