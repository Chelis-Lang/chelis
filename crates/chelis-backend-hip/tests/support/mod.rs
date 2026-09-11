#![allow(dead_code)]

use chelis_backend_hip::HipCodegenResult;
use chelis_ir::dag::Dag;
use chelis_types::unsupported::Unsupported;

pub fn codegen_hip(dag: &Dag, name: &str) -> Result<HipCodegenResult, Unsupported> {
    let verified = verified_dag(dag);
    chelis_backend_hip::codegen_hip(verified, name)
}

pub fn verified_dag(dag: &Dag) -> chelis_ir::ownership::VerifiedDagProgram {
    let selected = chelis_backend_hip::prepare_dag_for_codegen(dag.clone());
    chelis_ir::ownership::verify_ownership(
        chelis_ir::ownership::lower_dag_ownership(selected)
            .expect("backend test DAG must lower ownership"),
    )
    .expect("backend test DAG must satisfy ownership verification")
}

/// Stage the actual owner implementation separately from the support header.
/// A header-only mock or legacy helper cannot satisfy this dependency closure.
pub fn stage_device_runtime(destination: &std::path::Path) {
    let runtime = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("runtime");
    for name in [
        "chelis_device_owner.h",
        "chelis_device_owner.cpp",
        "chelis_device_descriptor.h",
    ] {
        std::fs::copy(runtime.join(name), destination.join(name))
            .unwrap_or_else(|error| panic!("stage device runtime {name}: {error}"));
    }
}

/// Lower Surf source through the front end exactly as the build drivers do,
/// returning the wrapper's helper manifest (read before C payload selection)
/// beside the verified host program.
pub fn lowered_host_program(
    source: &str,
    func_name: &str,
) -> (
    Vec<chelis_backend_c::HostTensorHelperCodegen>,
    chelis_ir::ownership::VerifiedHostProgram,
) {
    let declarations = chelis_surf::parser::parse_str(source).expect("parse host source");
    let deep = chelis_surf::desugar::desugar_program(&declarations);
    let checked = chelis_types::check_typed_program(&deep)
        .unwrap_or_else(|errors| panic!("check host source: {:?}", errors.errors));
    let checked = chelis_effects::check_program(&checked).expect("effects host source");
    let checked = chelis_types::check_linearity(&checked).expect("linearity host source");
    let realizability = chelis_effects::realizability::infer_realizability(
        &checked,
        chelis_backend_hip::TENSOR_CAPABLE_PRIMS,
    );
    let manifest = chelis_effects::realizability::compute_root_manifest(&checked, &realizability);
    let lowered = chelis_ir::host::try_lower_compiled_program_with_manifest(&checked, &manifest)
        .expect("lower host source");
    let raw = lowered.host.expect("source uses the host lane");
    let (helpers, raw) =
        chelis_backend_c::host_tensor_helper_codegen(raw, func_name).expect("host helper manifest");
    let selected =
        chelis_backend_c::prepare_host_program_for_codegen(raw).expect("select C host payload");
    let manifested = chelis_types::manifest::ManifestedProgram::new(
        checked,
        manifest,
        chelis_types::types::Target::Hip,
    );
    let verified = chelis_ir::ownership::verify_ownership(
        chelis_ir::ownership::lower_host_ownership(&manifested, selected)
            .expect("lower host ownership"),
    )
    .expect("verify host ownership");
    (helpers, verified)
}
