use chelis_dag_core::dag::{Dag, DimInfo, ExtentWitnessSite, FusedInput, FusedStep,
    FusedStepOp, NodeId, Owner, RiscOp, RtAxis, RtDim, TensorType};
use chelis_dag_core::load_store_name::LoadStoreName;
use chelis_dag_core::{analysis, fuse, grad, optimize, verify};
use chelis_numeric_core::prim::Prim;
use chelis_numeric_core::dtype_semantics::{scalar_from_i64, tensor_from_scalars};

fn main() {
    let mut dag = Dag::new();
    let decl = dag.declare("entry");
    dag.declare_value("constant");
    let activation = dag.add_node(decl, RiscOp::Load { name: "active".into() }, vec![],
        TensorType { dims: vec![], precision: Prim::Bool }, Some("active".into()));
    let owner = Owner::new(decl, Some(activation));
    let ty = TensorType { dims: vec![DimInfo::Lit(4)], precision: Prim::F32 };
    let x = dag.add_node(owner, RiscOp::Load { name: "x".into() }, vec![], ty.clone(), Some("input".into()));
    let claim = dag.add_node(owner, RiscOp::ExtentWitness {
        site: ExtentWitnessSite::LiteralResultClaim, parameter: String::new(), axis: RtAxis::Lit(0),
        requirements: vec![scalar_from_i64("fixture", Prim::Int64, 4).unwrap()], claims: vec![],
    }, vec![], TensorType { dims: vec![], precision: Prim::Int64 }, Some("claim".into()));
    let square = dag.add_node(owner, RiscOp::Mul, vec![x, x], ty.clone(), Some("square".into()));
    dag.add_result_claim_dep(square, claim);
    dag.set_reusable_input(square, x);
    dag.node_mut(square).unwrap().merged_spans = vec!["extra".into()];
    let loss = dag.add_node(owner, RiscOp::sum_default(0, Prim::F32).unwrap(), vec![square],
        TensorType::scalar_f32(), Some("loss".into()));
    dag.set_roots(vec![loss, square]);
    assert!(verify::verify(&dag).is_empty(), "{:?}", verify::verify(&dag));
    let exact = scalar_from_i64("fixture", Prim::Int64, 9_007_199_254_740_993).unwrap();
    let payload = tensor_from_scalars(Prim::Int64, &[exact]);
    let ops = vec![RiscOp::Neg, RiscOp::Relu,
        RiscOp::FloatClassify(chelis_numeric_core::dtype_semantics::FloatClassify::Nan),
        RiscOp::FloatClassify(chelis_numeric_core::dtype_semantics::FloatClassify::Finite),
        RiscOp::FloatClassify(chelis_numeric_core::dtype_semantics::FloatClassify::Infinite), RiscOp::MinElem, RiscOp::MaxElem,
        RiscOp::Load { name: LoadStoreName::top_level("module.value") },
        RiscOp::Store { name: "grads.1".into() }, RiscOp::Const { value: exact },
        RiscOp::Shrink { bounds: vec![(RtDim::Lit(0), RtDim::Node(1))] },
        RiscOp::FusedElem { ops: vec![FusedStep { op: FusedStepOp::Neg,
            input_indices: vec![FusedInput::External(0)] }, FusedStep { op: FusedStepOp::Exp,
            input_indices: vec![FusedInput::PreviousStep(0)] }] }];
    let mut bad = dag.clone();
    bad.node_mut(x).unwrap().inputs = vec![NodeId(99), x];
    let (compact, remap) = optimize::dead_code_eliminate_with_remap(&dag);
    let fused = fuse::fuse_with_remap(&dag);
    let gradient = grad::grad_dag_checked(&dag, loss, &[x]).unwrap();
    let copy = dag.add_node(owner, RiscOp::Copy, vec![x], ty, None);
    let costs = analysis::analyze_copy_costs_for_roots(&dag, &[("entry".into(), vec![copy, copy])]);
    println!("DAG_SNAPSHOT {}", serde_json::json!({
        "graph_json": serde_json::to_string(&dag).unwrap(),
        "graph_binary": bincode::serialize(&dag).unwrap(),
        "op_json": serde_json::to_string(&ops).unwrap(), "op_binary": bincode::serialize(&ops).unwrap(),
        "payload_json": serde_json::to_string(&payload).unwrap(), "payload_binary": bincode::serialize(&payload).unwrap(),
        "diagnostics": verify::verify(&bad), "copy_costs": serde_json::to_string(&costs).unwrap(),
        "dce": compact, "dce_map": remap.into_sorted(),
        "fusion": fused.dag, "fusion_map": fused.old_to_new.into_sorted(),
        "ad": gradient.dag, "gradients": gradient.grad_nodes.into_sorted(),
    }));
}
