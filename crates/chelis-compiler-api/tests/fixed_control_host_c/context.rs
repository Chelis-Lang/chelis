//! #1876: selected fixed-control source execution survives linked context transport.
use super::ownership_support;

use chelis_compiler_api::compiler::{CompiledExecutionArtifact, compile_for_execution_in_context};
use chelis_compiler_api::context::CompiledContext;
use chelis_compiler_api::schema::CompileTarget;
use chelis_compiler_api::{COMPILER_VERSION, compile_reef_context};

fn context() -> CompiledContext {
    context_with_library(
        r#"
module Probe.Draw
export (keep, loss, matrix, empty, single)
def keep(x: tensor[4, f32], rate: f32) -> tensor[4, f32] = dropout(x, rate)
def loss(x: tensor[4, f32]) -> tensor[f32] = sum(keep(x, 0.5f32), 0i32)
def matrix(x: tensor[2, 2, f32]) -> tensor[2, 2, f32] = dropout(x, 0.5f32)
def empty(x: tensor[0, f32]) -> tensor[0, f32] = dropout(x, 0.5f32)
def single(x: tensor[1, f32]) -> tensor[1, f32] = dropout(x, 0.5f32)
"#,
    )
}

fn context_with_library(library: &str) -> CompiledContext {
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("src")).unwrap();
    std::fs::write(directory.path().join("reef.toml"), format!("[package]\nname = \"context_draw\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Probe\"\n")).unwrap();
    std::fs::write(directory.path().join("src/draw.ch"), library).unwrap();
    compile_reef_context(directory.path(), directory.path()).unwrap()
}

fn source(body: &str, shape: &str) -> String {
    format!(
        "module Probe.Client\nimport Probe.Draw (keep, loss, matrix, empty, single)\n\
         def main(x: tensor[{shape}, f32]) -> tensor[{shape}, f32] = with seed(42i64) {{ {body} }}\n"
    )
}

fn native(artifact: &CompiledExecutionArtifact, dims: &[usize], expected: &[f32]) {
    assert_eq!(artifact.host_entry_name, "chelis_main");
    assert!(artifact.entry_lane_decline.is_none());
    assert_eq!(artifact.inputs.len(), 1);
    assert_eq!(artifact.inputs[0].name, "x");
    assert_eq!(artifact.outputs.len(), 1);
    assert!(artifact.symbolic_dims.is_empty());
    for spec in [&artifact.inputs[0], &artifact.outputs[0]] {
        assert_eq!(spec.dtype, "f32");
        assert_eq!(spec.dims.len(), dims.len());
        for (actual, expected) in spec.dims.iter().zip(dims) {
            assert_eq!(serde_json::to_value(actual).unwrap()["size"], *expected);
        }
    }
    let c = artifact
        .compile_result
        .files
        .iter()
        .find(|file| file.path == "chelis_main.c")
        .unwrap()
        .contents
        .clone();
    let header = artifact
        .compile_result
        .files
        .iter()
        .find(|file| file.path == "chelis_main.h")
        .unwrap()
        .contents
        .clone();
    let generated = ownership_support::GeneratedProgram::new(c, header);
    let shape = dims
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let bits = expected
        .iter()
        .map(|value| format!("{}u", value.to_bits()))
        .chain(std::iter::once("0u".to_string()))
        .collect::<Vec<_>>()
        .join(",");
    let driver = format!(
        r#"
static void check_result(const chelis_tensor *tensor, const uint32_t *bits) {{
    const int64_t shape[] = {{{shape}}};
    assert(chelis_tensor_rank(tensor) == {rank});
    for (int axis = 0; axis < {rank}; ++axis)
        assert(chelis_tensor_shape(tensor, axis) == shape[axis]);
    assert(chelis_tensor_numel(tensor) == {count});
    chelis_read_view view = chelis_tensor_read_view(tensor);
    assert(view.dtype == CHELIS_DTYPE_F32 && view.count == {count});
    for (int64_t i = 0; i < {count}; ++i)
        assert(memcmp((const float *)view.data + i, bits + i, sizeof(float)) == 0);
}}
int main(void) {{
    const int64_t shape[] = {{{shape}}};
    chelis_tensor *x = chelis_alloc({rank}, shape, CHELIS_DTYPE_F32);
    uint32_t original[{storage}];
    chelis_tensor_write *guard = chelis_tensor_begin_write(x);
    chelis_write_view view = chelis_tensor_write_view(guard);
    for (int64_t i = 0; i < {count}; ++i) {{
        float value = (float)(2*i-3);
        ((float *)view.data)[i] = value;
        memcpy(original+i, &value, sizeof(value));
    }}
    chelis_tensor_end_write(guard);
    const uint32_t expected[] = {{{bits}}};
    chelis_tensor *inputs[] = {{x}};
    for (int repeat = 0; repeat < 3; ++repeat) {{
        chelis_tensor *outputs[] = {{NULL}};
        chelis_main(inputs, 1, outputs, 1);
        check_result(outputs[0], expected);
        check_result(x, original);
        chelis_tensor_release(outputs[0]);
    }}
    chelis_tensor_release(x);
    return 0;
}}
"#,
        rank = dims.len(),
        count = expected.len(),
        storage = expected.len().max(1)
    );
    ownership_support::balanced(&ownership_support::run(&generated, &driver));
}

#[test]
fn context_native_source_order_saved_mask_and_shapes_survive_decode() {
    let original = context();
    let decoded = CompiledContext::decode(&original.encode().unwrap()).unwrap();
    // [05-RNG-1]'s seed42 masks at ordinals 0/1/2 are [0,2,0,0],
    // [2,0,0,0], [2,2,0,0]. These exact dyadic expectations are shared
    // with the committed independent oracle in dropout_fixed_stream_api.
    let cases: &[(&str, &str, &[usize], &[f32])] = &[
        ("keep(x, 0.5f32)", "4", &[4], &[0.0, -2.0, 0.0, 0.0]),
        (
            "Probe.Draw.keep(x, 0.5f32)",
            "4",
            &[4],
            &[0.0, -2.0, 0.0, 0.0],
        ),
        (
            "unused = keep(x, 0.0f32)\n _ = drop(unused)\n keep(x, 0.5f32)",
            "4",
            &[4],
            &[-6.0, 0.0, 0.0, 0.0],
        ),
        (
            "unused = keep(x, 0.5f32)\n _ = drop(unused)\n x",
            "4",
            &[4],
            &[-3.0, -1.0, 1.0, 3.0],
        ),
        (
            "g = grad(loss)(x)\n add(g, keep(x, 0.5f32))",
            "4",
            &[4],
            &[-6.0, 2.0, 0.0, 0.0],
        ),
        (
            "nested = with seed(99i64) { keep(x, 0.5f32) }\n add(nested, keep(x, 0.5f32))",
            "4",
            &[4],
            &[-6.0, -4.0, 2.0, 6.0],
        ),
        ("matrix(x)", "2, 2", &[2, 2], &[0.0, -2.0, 0.0, 0.0]),
        ("empty(x)", "0", &[0], &[]),
        ("single(x)", "1", &[1], &[0.0]),
    ];
    for context in [&original, &decoded] {
        for &(body, shape, dims, expected) in cases {
            let source = source(body, shape);
            let artifact =
                compile_for_execution_in_context(context, &source, CompileTarget::C, None)
                    .unwrap_or_else(|error| panic!("{body}: {error:?}"));
            native(&artifact, dims, expected);
        }
    }
}

#[test]
fn context_source_selection_preserves_siblings_and_ordinary_fallback() {
    let original = context();
    let decoded = CompiledContext::decode(&original.encode().unwrap()).unwrap();
    for context in [&original, &decoded] {
        let source = format!(
            "{}\ndef sibling(y: tensor[2, f32], z: tensor[2, f32]) -> tensor[2, f32] = add(y, z)\n",
            source("keep(x, 0.5f32)", "4")
        );
        let artifact =
            compile_for_execution_in_context(context, &source, CompileTarget::C, None).unwrap();
        native(&artifact, &[4], &[0.0, -2.0, 0.0, 0.0]);
        let ordinary =
            compile_for_execution_in_context(context, &source, CompileTarget::C, Some("sibling"))
                .unwrap();
        assert_eq!(
            ordinary
                .inputs
                .iter()
                .map(|input| input.name.as_str())
                .collect::<Vec<_>>(),
            ["y", "z"]
        );
        assert!(
            !ordinary
                .compile_result
                .files
                .iter()
                .any(|file| file.contents.contains("chelis_dropout_unit"))
        );
        for entry in ["keep", "Probe.Draw.keep", "ain"] {
            let error =
                compile_for_execution_in_context(context, &source, CompileTarget::C, Some(entry))
                    .unwrap_err();
            assert!(
                format!("{error:?}").contains("unknown entry_name"),
                "{error:?}"
            );
        }
    }
}

#[test]
fn context_rejects_unbound_names_runtime_controls_and_rootless_entries() {
    let original = context();
    let decoded = CompiledContext::decode(&original.encode().unwrap()).unwrap();
    let cases = [
        (source("keep(x, missing)", "4"), "unbound variable"),
        (source("keep(x, 0.5f32)", "4").replace("import Probe.Draw (keep, loss, matrix, empty, single)\n", ""), "unbound variable"),
        (source("rate = tensor_to_scalar(sum(x, 0i32))\n keep(x, rate)", "4"), "no callable tensor-kernel form"),
        (source("rate = tensor_to_scalar(sum(x, 0i32))\n dead = keep(x, rate)\n _ = drop(dead)\n x", "4"), "no callable tensor-kernel form"),
        (source("keep(x, 0.5f32)", "4").replace("with seed(42i64) { keep(x, 0.5f32) }", "keep(x, 0.5f32)"), "Random"),
        ("module Probe.Client\nimport Probe.Draw (keep)\ndef main(x: tensor[4, f32], rate: f32) -> tensor[4, f32] = with seed(42i64) { keep(x, rate) }\n".into(), "no callable tensor-kernel form"),
    ];
    for context in [&original, &decoded] {
        for (source, reason) in &cases {
            let error =
                compile_for_execution_in_context(context, source, CompileTarget::C, Some("main"))
                    .err()
                    .unwrap_or_else(|| panic!("expected {reason} rejection for {source}"));
            assert!(error.transcript.is_empty());
            assert!(
                format!("{error:?}").contains(reason),
                "expected {reason}: {error:?}"
            );
        }
    }
}

#[test]
fn context_imports_enforce_module_exports_before_checking_or_emission() {
    let original = context_with_library(
        "module Probe.Draw\nexport (keep)\ndef keep(x: tensor[4, f32], rate: f32) -> tensor[4, f32] = dropout(x, rate)\ndef private_keep(x: tensor[4, f32]) -> tensor[4, f32] = dropout(x, 0.5f32)\n",
    );
    let decoded = CompiledContext::decode(&original.encode().unwrap()).unwrap();
    for context in [&original, &decoded] {
        for import in [
            "import Probe.Draw",
            "import Probe.Draw (keep)",
            "import Probe.Draw (..)",
        ] {
            for (call, allowed) in [
                ("Probe.Draw.keep(x, 0.5f32)", true),
                ("Probe.Draw.private_keep(x)", false),
            ] {
                let source = format!(
                    "module Probe.Client\n{import}\ndef main(x: tensor[4, f32]) -> tensor[4, f32] = with seed(42i64) {{ {call} }}\n"
                );
                let checked = chelis_compiler_api::compiler::check_in_context(context, &source);
                let emitted = compile_for_execution_in_context(
                    context,
                    &source,
                    CompileTarget::C,
                    Some("main"),
                );
                if allowed {
                    let checked = checked.unwrap();
                    assert!(checked.errors.is_empty(), "{checked:?}");
                    emitted.unwrap();
                } else {
                    for error in [checked.unwrap_err(), emitted.unwrap_err()] {
                        let message = format!("{error:?}");
                        assert!(
                            message.contains("does not export `private_keep`"),
                            "{message}"
                        );
                    }
                }
            }
        }
        let source = "module Probe.Client\nimport Probe.Draw (private_keep)\ndef main(x: tensor[4, f32]) -> tensor[4, f32] = with seed(42i64) { private_keep(x) }\n";
        let error = chelis_compiler_api::compiler::check_in_context(context, source).unwrap_err();
        assert!(format!("{error:?}").contains("does not export `private_keep`"));
    }
}
