//! C3: HIP artifact source closure contains the separately compiled owner.
use chelis_compiler_api::compiler::{compile, compile_for_execution};
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, SourceKind};
fn request(source: &str, target: CompileTarget) -> CompileRequest {
    CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        target,
        entry_name: None,
    }
}
fn require_companion(files: &[chelis_compiler_api::schema::GeneratedFile]) {
    for (name, contents) in [
        (
            "chelis_device_owner.cpp",
            include_str!("../../chelis-backend-hip/runtime/chelis_device_owner.cpp"),
        ),
        (
            "chelis_device_owner.h",
            include_str!("../../chelis-backend-hip/runtime/chelis_device_owner.h"),
        ),
        (
            "chelis_device_descriptor.h",
            include_str!("../../chelis-backend-hip/runtime/chelis_device_descriptor.h"),
        ),
    ] {
        let matches: Vec<_> = files.iter().filter(|file| file.path == name).collect();
        assert_eq!(matches.len(), 1, "missing/duplicate {name}");
        assert_eq!(matches[0].contents, contents, "stale {name}");
    }
}
#[test]
fn hip_tensor_and_host_artifacts_materialize_exact_companion_inputs() {
    let tensor = compile_for_execution(request(
        "def main(x: tensor[2, f32]) -> tensor[2, f32] = mul(x, x)",
        CompileTarget::Hip,
    ))
    .unwrap();
    require_companion(&tensor.compile_result.files);
    let host = compile(request("message = \"hello\"\n", CompileTarget::Hip)).unwrap();
    require_companion(&host.files);
}
#[test]
fn c_artifacts_do_not_gain_an_unlinked_device_companion() {
    let host = compile(request("message = \"hello\"\n", CompileTarget::C)).unwrap();
    assert!(
        !host
            .files
            .iter()
            .any(|file| file.path.starts_with("chelis_device_"))
    );
}
