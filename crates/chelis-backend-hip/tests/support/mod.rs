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
