//! Public compiler-API device host-helper support for first-class Count (chelis#1291).

use chelis_compiler_api::compiler::compile;
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, SourceKind};

const COUNT_SOURCE: &str = include_str!("../../../examples/count_bool_axes.ch");
const COUNT_DEVICE_ENTRY: &str = include_str!("../../../examples/count_bool_device_entry.ch");
const TWO_COUNT_HELPERS: &str = "\
module Example.TwoCounts\n\
mask: tensor[2, 3, bool] = [[true, false, true], [false, true, true]]\n\
rows: tensor[2, i64] = count(&mask, 1)\n\
columns: tensor[3, i64] = count(&mask, 0)\n";

#[test]
fn hip_compile_api_emits_host_helper_count_on_the_device() {
    let result = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: COUNT_SOURCE.to_string(),
        target: CompileTarget::Hip,
        entry_name: None,
    })
    .expect("HIP must compile host-program Count through a device helper");

    let host = result
        .files
        .iter()
        .find(|file| file.path == "chelis_main_hip.cpp")
        .expect("HIP artifact contains the host wrapper")
        .contents
        .as_str();
    let helper = result
        .files
        .iter()
        .find(|file| file.path.contains("global__tensor_0_hip.cpp"))
        .expect("HIP artifact contains a separate device helper")
        .contents
        .as_str();

    assert!(host.contains("chelis_main__global__tensor_0("), "{host}");
    assert!(
        !host.contains("static void chelis_main__global__tensor_0("),
        "the host wrapper must declare, not embed, the device Count helper:\n{host}"
    );
    assert!(!host.contains("unimplemented chelis#1291"), "{host}");
    assert!(helper.contains("kernel_count_"), "{helper}");
    assert!(helper.contains("chelis_main__global__tensor_0"), "{helper}");
    for forbidden in [
        "kernel_sum",
        "kernel_cast",
        "count_host_fallback",
        "unimplemented stub",
    ] {
        assert!(
            !helper.contains(forbidden),
            "device Count helper emitted forbidden fallback {forbidden:?}:\n{helper}"
        );
    }
}

#[test]
fn hip_compile_api_emits_the_dedicated_count_tensor_entry() {
    let result = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: COUNT_DEVICE_ENTRY.to_string(),
        target: CompileTarget::Hip,
        entry_name: Some("main".to_string()),
    })
    .expect("parameterized Count entry must compile to the dedicated HIP kernel");
    let source = result
        .files
        .iter()
        .find(|file| file.path.ends_with("_hip.cpp"))
        .expect("HIP artifact contains generated source")
        .contents
        .as_str();

    assert!(source.contains("kernel_count_"), "{source}");
    assert!(source.contains("chelis_main_device"), "{source}");
    for forbidden in ["kernel_sum", "kernel_cast", "count_host_fallback"] {
        assert!(
            !source.contains(forbidden),
            "dedicated HIP Count entry emitted forbidden fallback {forbidden:?}:\n{source}"
        );
    }
}

#[test]
fn hip_compile_api_isolates_multiple_host_count_helpers() {
    let result = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: TWO_COUNT_HELPERS.to_string(),
        target: CompileTarget::Hip,
        entry_name: None,
    })
    .expect("each host-program Count must compile as a device helper");
    let helpers = result
        .files
        .iter()
        .filter(|file| file.path.contains("global__tensor_") && file.path.ends_with("_hip.cpp"))
        .collect::<Vec<_>>();
    assert_eq!(
        helpers.len(),
        2,
        "generated files: {:?}",
        result
            .files
            .iter()
            .map(|file| &file.path)
            .collect::<Vec<_>>()
    );
    assert_ne!(helpers[0].path, helpers[1].path);
    for helper in helpers {
        assert!(
            helper.contents.contains("kernel_count_"),
            "{}",
            helper.contents
        );
        assert!(
            !helper.contents.contains("count_host_fallback"),
            "{}",
            helper.contents
        );
    }
}
