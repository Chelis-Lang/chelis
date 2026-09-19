#![allow(dead_code)]

use chelis_backend_metal::MetalCodegenResult;
use chelis_ir::dag::Dag;
use chelis_types::unsupported::Unsupported;

pub fn codegen_metal(dag: &Dag, name: &str) -> MetalCodegenResult {
    try_codegen_metal(dag, name).expect("backend test DAG must be supported by Metal")
}

pub fn try_codegen_metal(dag: &Dag, name: &str) -> Result<MetalCodegenResult, Unsupported> {
    let verified = verified_dag(dag);
    let plan = chelis_backend_metal::plan_metal(verified);
    chelis_backend_metal::codegen_metal(plan, name)
}

pub fn verified_dag(dag: &Dag) -> chelis_ir::ownership::VerifiedDagProgram {
    chelis_ir::ownership::verify_ownership(
        chelis_ir::ownership::lower_dag_ownership(dag.clone())
            .expect("backend test DAG must lower ownership"),
    )
    .expect("backend test DAG must satisfy ownership verification")
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
    let deep =
        chelis_surf::desugar::desugar_program(&declarations).expect("Surf fixture must desugar");
    let checked = chelis_types::check_typed_program(&deep)
        .unwrap_or_else(|errors| panic!("check host source: {:?}", errors.errors));
    let checked = chelis_effects::check_program(&checked).expect("effects host source");
    let checked = chelis_types::check_linearity(&checked).expect("linearity host source");
    let realizability = chelis_effects::realizability::infer_realizability(
        &checked,
        chelis_backend_metal::TENSOR_CAPABLE_PRIMS,
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
        chelis_types::types::Target::Metal,
    );
    let verified = chelis_ir::ownership::verify_ownership(
        chelis_ir::ownership::lower_host_ownership(&manifested, selected)
            .expect("lower host ownership"),
    )
    .expect("verify host ownership");
    (helpers, verified)
}
