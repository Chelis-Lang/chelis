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
