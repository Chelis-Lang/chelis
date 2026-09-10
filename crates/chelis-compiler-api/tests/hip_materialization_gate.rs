//! Spec08 materialization barrier and spec04 exact dtype preservation.
//! Planned Realize transports stored bits; admitting it does not admit new arithmetic.
use chelis_compiler_api::compiler::reject_unsupported_hip_ops;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;
fn dag(precision: Prim, materialization: bool) -> Dag {
    let mut dag = Dag::new();
    let ty = TensorType {
        dims: vec![DimInfo::Lit(4)],
        precision,
    };
    let input = dag.add_node(
        RiscOp::Load {
            name: "input".into(),
        },
        vec![],
        ty.clone(),
        None,
    );
    let (op, inputs) = if materialization {
        (RiscOp::Realize, vec![input])
    } else {
        (RiscOp::Add, vec![input, input])
    };
    let result = dag.add_node(op, inputs, ty, None);
    dag.add_root(result);
    dag
}
#[test]
fn hip_gate_accepts_exact_narrow_float_materialization() {
    for precision in [Prim::Bf16, Prim::F16] {
        let prepared = chelis_backend_hip::prepare_dag_for_codegen(dag(precision, true));
        reject_unsupported_hip_ops(&prepared).expect("planned exact-bit Realize is supported");
    }
}
#[test]
fn hip_materialization_does_not_authorize_unimplemented_narrow_float_arithmetic() {
    for precision in [Prim::Bf16, Prim::F16] {
        let error = reject_unsupported_hip_ops(&dag(precision, false))
            .expect_err("Realize support must not admit unimplemented arithmetic");
        assert!(
            error.errors.iter().any(|diagnostic| diagnostic.message.contains("narrow-float compute")),
            "{error:?}"
        );
    }
}
