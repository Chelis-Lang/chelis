#![allow(dead_code)]

use chelis_backend_metal::MetalCodegenResult;
use chelis_ir::dag::Dag;
use chelis_types::unsupported::Unsupported;

pub fn codegen_metal(dag: &Dag, name: &str) -> MetalCodegenResult {
    try_codegen_metal(dag, name).expect("backend test DAG must be supported by Metal")
}

pub fn try_codegen_metal(dag: &Dag, name: &str) -> Result<MetalCodegenResult, Unsupported> {
    let verified = verified_dag(dag);
    let plan = chelis_backend_metal::plan_metal(&verified);
    chelis_backend_metal::codegen_metal(&plan, name)
}

pub fn verified_dag(dag: &Dag) -> chelis_ir::ownership::VerifiedDagProgram {
    chelis_ir::ownership::verify_ownership(
        chelis_ir::ownership::lower_dag_ownership(dag.clone())
            .expect("backend test DAG must lower ownership"),
    )
    .expect("backend test DAG must satisfy ownership verification")
}
