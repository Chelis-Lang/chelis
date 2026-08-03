use chelis_backend_metal::emit::emit_dag;
use chelis_ir::dag::{Dag, DimInfo, FusedInput, FusedStep, FusedStepOp, RiscOp, TensorType};
use chelis_types::types::Prim;

fn vec_i64(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::Int64,
    }
}

#[test]
fn integer_abs_is_rejected_before_the_float_unary_template() {
    let ty = vec_i64(1);

    let mut direct = Dag::new();
    let x = direct.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
    let out = direct.add_node(RiscOp::Abs, vec![x], ty.clone(), None);
    direct.set_roots(vec![out]);
    let err = match emit_dag(&direct, "integer_abs") {
        Err(error) => error,
        Ok(_) => panic!("integer abs must not enter the Metal fabs template"),
    };
    assert!(err.contains("unsupported: op `Abs`"));

    let mut fused = Dag::new();
    let x = fused.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
    let out = fused.add_node(
        RiscOp::FusedElem {
            ops: vec![FusedStep {
                op: FusedStepOp::Abs,
                input_indices: vec![FusedInput::External(0)],
            }],
        },
        vec![x],
        ty,
        None,
    );
    fused.set_roots(vec![out]);
    let err = match emit_dag(&fused, "fused_integer_abs") {
        Err(error) => error,
        Ok(_) => panic!("fused integer abs must not bypass the Metal guard"),
    };
    assert!(err.contains("unsupported: op `Abs`"));
}
