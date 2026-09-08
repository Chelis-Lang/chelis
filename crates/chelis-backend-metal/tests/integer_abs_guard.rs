mod support;
use chelis_ir::dag::{Dag, DimInfo, FusedInput, FusedStep, FusedStepOp, RiscOp, TensorType};
use chelis_types::types::Prim;
use support::try_codegen_metal;

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
    let error = try_codegen_metal(&direct, "integer_abs").unwrap_err();
    assert!(error.to_string().contains("unsupported: op `Abs`"));

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
    let error = try_codegen_metal(&fused, "fused_integer_abs").unwrap_err();
    assert!(error.to_string().contains("unsupported: op `Abs`"));
}
