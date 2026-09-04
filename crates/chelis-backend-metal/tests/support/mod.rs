#![allow(dead_code)]

use chelis_backend_metal::MetalCodegenResult;
use chelis_ir::dag::Dag;

pub fn codegen_metal(dag: &Dag, name: &str) -> MetalCodegenResult {
    let verified = verified_dag(dag);
    chelis_backend_metal::codegen_metal(&verified, name)
}

pub fn verified_dag(dag: &Dag) -> chelis_ir::ownership::VerifiedDagProgram {
    chelis_ir::ownership::verify_ownership(
        chelis_ir::ownership::lower_dag_ownership(dag.clone())
            .expect("backend test DAG must lower ownership"),
    )
    .expect("backend test DAG must satisfy ownership verification")
}
