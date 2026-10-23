use chelis_dag_core::dag::{
    Dag, DimExpr, DimInfo, FusedInput, FusedStep, FusedStepOp, NodeId, RiscOp, RtDim, TensorType,
};
use chelis_dag_core::{analysis, fuse, grad, optimize, ownership, span_merge, verify};
use chelis_numeric_core::prim::Prim;

fn vector(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn load(dag: &mut Dag, name: &str, ty: TensorType) -> NodeId {
    dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Load { name: name.into() },
        vec![],
        ty,
        Some(name.into()),
    )
}

fn rejects(dag: &Dag) {
    let errors = verify::verify(dag);
    assert!(!errors.is_empty());
    assert_eq!(errors, verify::verify(dag));
    let ownership_errors = verify::verify_ownership_input(dag);
    assert!(!ownership_errors.is_empty());
    assert_eq!(ownership_errors, verify::verify_ownership_input(dag));
}

#[test]
fn invalid_topology_returns_deterministic_diagnostics() {
    let mut dag = Dag::new();
    dag.declare("fixture");
    let out = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Add,
        vec![NodeId(99), NodeId(0)],
        vector(3),
        None,
    );
    dag.add_root(out);
    rejects(&dag);
}

#[test]
fn reference_errors_retain_independent_memory_arity_diagnostics() {
    use chelis_numeric_core::dtype_semantics::{scalar_from_f64, tensor_from_scalars};

    let value = scalar_from_f64("fixture", Prim::F32, 1.0).unwrap();
    let data = tensor_from_scalars(Prim::F32, core::slice::from_ref(&value));
    let mut topology = Dag::new();
    let decl = topology.declare("fixture");
    let root = topology.add_node(
        decl,
        RiscOp::Add,
        vec![NodeId(99), NodeId(0)],
        TensorType::scalar_f32(),
        None,
    );
    topology.add_root(root);
    let reference_count = verify::verify(&topology).len();
    assert!(reference_count > 0);
    for (op, independent_count) in [
        (RiscOp::Const { value }, 1),
        (RiscOp::ConstTensor { data }, 1),
        (RiscOp::Load { name: "x".into() }, 1),
        (RiscOp::Store { name: "out".into() }, 2),
    ] {
        let mut dag = Dag::new();
        let decl = dag.declare("fixture");
        let root = dag.add_node(
            decl,
            op,
            vec![NodeId(99), NodeId(0)],
            TensorType::scalar_f32(),
            None,
        );
        dag.add_root(root);
        assert_eq!(
            verify::verify(&dag).len(),
            reference_count + independent_count,
        );
        assert_eq!(
            verify::verify_ownership_input(&dag).len(),
            reference_count + independent_count,
        );
        let mut mixed = Dag::new();
        let decl = mixed.declare("mixed");
        let input = load(&mut mixed, "input", TensorType::scalar_f32());
        let root = mixed.add_node(
            decl,
            dag.get(root).unwrap().op.clone(),
            vec![input, input],
            TensorType::scalar_f32(),
            None,
        );
        mixed.add_root(root);
        let independent = verify::verify(&mixed);
        assert_eq!(independent.len(), independent_count);
        mixed.set_roots(vec![root, NodeId(99)]);
        let errors = verify::verify(&mixed);
        assert_eq!(errors.len(), independent.len() + 1);
        assert!(errors.ends_with(&independent));
    }
}

#[test]
fn malformed_graph_references_fail_without_operator_execution() {
    let mut dag = Dag::new();
    dag.declare("fixture");
    let x = load(&mut dag, "x", vector(3));
    dag.add_root(x);
    assert!(verify::verify(&dag).is_empty());
    let mut bad = dag.clone();
    bad.set_roots(vec![x, NodeId(99)]);
    rejects(&bad);
    let mut bad = dag.clone();
    bad.node_mut(x).unwrap().inputs = vec![NodeId(99)];
    rejects(&bad);
    let mut bad = dag.clone();
    bad.add_shape_dep(x, NodeId(99));
    rejects(&bad);
    let mut bad = dag.clone();
    bad.add_shape_dep(x, x);
    rejects(&bad);
    let mut bad = dag.clone();
    bad.set_reusable_input(x, NodeId(99));
    rejects(&bad);
    let mut bad = dag.clone();
    let y = bad.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Neg,
        vec![x],
        vector(3),
        None,
    );
    bad.node_mut(x).unwrap().inputs = vec![y];
    bad.set_roots(vec![y]);
    rejects(&bad);
    for op in [
        RiscOp::Neg,
        RiscOp::Shrink {
            bounds: vec![(RtDim::Lit(0), RtDim::Lit(1))],
        },
        RiscOp::Reshape {
            new_shape: vec![RtDim::Lit(3)],
        },
        RiscOp::Shape { axis: 0 },
        RiscOp::Permute { axes: vec![0] },
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Lit(3),
        },
        RiscOp::OneHot { vocab: 3 },
        RiscOp::zero_pad(Prim::F32, vec![(RtDim::Lit(0), RtDim::Lit(1))]),
        RiscOp::Stride {
            strides: vec![RtDim::Lit(1)],
        },
        RiscOp::SplitN {
            count: RtDim::Node(1),
        },
    ] {
        let mut bad = Dag::new();
        bad.declare("fixture");
        let root = bad.add_node(
            chelis_dag_core::dag::DeclId(0),
            RiscOp::Neg,
            vec![NodeId(99)],
            vector(3),
            None,
        );
        bad.node_mut(root).unwrap().op = op;
        bad.add_root(root);
        rejects(&bad);
    }
    let mut dag = Dag::new();
    let decl = dag.declare("no_execution");
    let condition = dag.add_node(
        decl,
        RiscOp::synth_const(Prim::Bool, 1.0),
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::Bool,
        },
        None,
    );
    let fallback = dag.add_node(
        decl,
        RiscOp::synth_const(Prim::F32, 2.0),
        vec![],
        TensorType::scalar_f32(),
        None,
    );
    let abort = dag.add_node(
        decl,
        RiscOp::GuardedFail {
            message: "operator must not execute".into(),
            trap_on_true: true,
        },
        vec![condition, fallback],
        TensorType::scalar_f32(),
        None,
    );
    dag.add_root(abort);
    assert!(verify::verify(&dag).is_empty());
    assert!(verify::verify_ownership_input(&dag).is_empty());
    dag.node_mut(abort).unwrap().inputs[1] = NodeId(99);
    rejects(&dag);
}

#[test]
fn bad_arity_precision_shapes_and_fused_references_fail() {
    let mut dag = Dag::new();
    dag.declare("fixture");
    let x = load(&mut dag, "x", vector(3));
    let root = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Add,
        vec![x],
        vector(3),
        None,
    );
    dag.add_root(root);
    rejects(&dag);
    for ty in [
        TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F64,
        },
        vector(4),
    ] {
        let mut bad = Dag::new();
        bad.declare("fixture");
        let x = load(&mut bad, "x", vector(3));
        let y = load(&mut bad, "y", ty);
        let out = bad.add_node(
            chelis_dag_core::dag::DeclId(0),
            RiscOp::Add,
            vec![x, y],
            vector(3),
            None,
        );
        bad.add_root(out);
        rejects(&bad);
    }
    for reference in [
        FusedInput::External(1),
        FusedInput::PreviousStep(0),
        FusedInput::PreviousStep(99),
    ] {
        let mut bad = Dag::new();
        bad.declare("fixture");
        let x = load(&mut bad, "x", vector(3));
        let out = bad.add_node(
            chelis_dag_core::dag::DeclId(0),
            RiscOp::FusedElem {
                ops: vec![FusedStep {
                    op: FusedStepOp::Neg,
                    input_indices: vec![reference],
                }],
            },
            vec![x],
            vector(3),
            None,
        );
        bad.add_root(out);
        rejects(&bad);
    }
}

#[test]
fn runtime_bounds_use_absolute_integer_input_slots() {
    let mut dag = Dag::new();
    dag.declare("fixture");
    let x = load(&mut dag, "x", vector(3));
    let bound = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Shape { axis: 0 },
        vec![x],
        TensorType {
            dims: vec![],
            precision: Prim::Int64,
        },
        Some("shape".into()),
    );
    let out = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Shrink {
            bounds: vec![(RtDim::Lit(0), RtDim::Node(1))],
        },
        vec![x, bound],
        vector(3),
        Some("slice".into()),
    );
    dag.add_root(out);
    assert!(verify::verify(&dag).is_empty());
    let (rebuilt, remap) = optimize::dead_code_eliminate_with_remap(&dag);
    assert_eq!(
        rebuilt.get(remap[&out]).unwrap().inputs,
        vec![remap[&x], remap[&bound]]
    );
    assert!(matches!(&rebuilt.get(remap[&out]).unwrap().op,
        RiscOp::Shrink { bounds } if bounds == &vec![(RtDim::Lit(0), RtDim::Node(1))]));
    for slot in [0, 2, 99] {
        let mut bad = dag.clone();
        bad.node_mut(out).unwrap().op = RiscOp::Shrink {
            bounds: vec![(RtDim::Lit(0), RtDim::Node(slot))],
        };
        rejects(&bad);
    }
    let mut bad = dag.clone();
    bad.node_mut(bound).unwrap().output_type.precision = Prim::F32;
    rejects(&bad);
    let mut bad = dag;
    bad.node_mut(bound).unwrap().output_type.dims = vec![DimInfo::Lit(1)];
    rejects(&bad);
}

#[test]
fn dce_keeps_shape_sources_root_order_reuse_and_spans() {
    let mut dag = Dag::new();
    dag.declare("fixture");
    let dead = load(&mut dag, "dead", vector(7));
    let shape = load(
        &mut dag,
        "shape_source",
        TensorType {
            dims: vec![DimInfo::Named("batch".into(), None)],
            precision: Prim::F32,
        },
    );
    let x = load(&mut dag, "x", vector(3));
    let y = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Neg,
        vec![x],
        vector(3),
        Some("canonical".into()),
    );
    dag.node_mut(y).unwrap().merged_spans = vec!["extra".into()];
    dag.add_shape_dep(y, shape);
    dag.set_reusable_input(y, x);
    dag.set_roots(vec![y, x]);
    let (rebuilt, remap) = optimize::dead_code_eliminate_with_remap(&dag);
    assert!(!remap.contains_key(&dead));
    assert_eq!(rebuilt.len(), 3);
    assert_eq!(rebuilt.roots(), &[remap[&y], remap[&x]]);
    let node = rebuilt.get(remap[&y]).unwrap();
    assert_eq!(node.shape_deps, vec![remap[&shape]]);
    assert_eq!(node.reusable_input, Some(remap[&x]));
    assert_eq!(node.span_id.as_deref(), Some("canonical"));
    assert_eq!(node.merged_spans, vec!["extra"]);
    assert!(verify::verify(&rebuilt).is_empty());
    let batch = DimExpr::Sym("batch".into());
    let sequence = DimExpr::Sym("seq".into());
    assert_ne!(batch, sequence);
    let bindings = chelis_unord::UnordMap::from([("batch".to_string(), 3), ("seq".to_string(), 5)]);
    assert_eq!(batch.evaluate(&bindings), Ok(3));
    assert_eq!(sequence.evaluate(&bindings), Ok(5));
}

#[test]
fn ownership_remaps_copy_fanout_and_drop_boundaries() {
    let mut dag = Dag::new();
    dag.declare("fixture");
    let x = load(&mut dag, "x", vector(3));
    let a = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Realize,
        vec![x],
        vector(3),
        Some("a".into()),
    );
    let b = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Realize,
        vec![x],
        vector(3),
        Some("b".into()),
    );
    dag.add_shape_dep(b, x);
    dag.set_reusable_input(b, x);
    dag.node_mut(b).unwrap().merged_spans = vec!["b.extra".into()];
    dag.set_roots(vec![b, a]);
    let result = ownership::normalize_ownership(&dag);
    assert_eq!(result.old_to_new.len(), dag.len());
    assert_eq!(
        result.dag.roots(),
        &[result.old_to_new[&b], result.old_to_new[&a]]
    );
    let copies = result
        .dag
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, RiscOp::Copy))
        .collect::<Vec<_>>();
    assert_eq!(copies.len(), 1);
    assert_eq!(copies[0].inputs, vec![result.old_to_new[&x]]);
    assert_eq!(copies[0].span_id.as_deref(), Some("a"));
    assert_eq!(
        result.dag.get(result.old_to_new[&a]).unwrap().inputs,
        vec![copies[0].id]
    );
    assert_eq!(
        result.dag.get(result.old_to_new[&b]).unwrap().inputs,
        vec![result.old_to_new[&x]]
    );
    assert_eq!(
        result.dag.get(result.old_to_new[&b]).unwrap().shape_deps,
        vec![result.old_to_new[&x]]
    );
    assert_eq!(
        result.dag.get(result.old_to_new[&b]).unwrap().merged_spans,
        vec!["b.extra"]
    );
    assert_eq!(
        result
            .dag
            .get(result.old_to_new[&b])
            .unwrap()
            .reusable_input,
        Some(result.old_to_new[&x])
    );
    let (copied, copy_map) = ownership::insert_copy_nodes_for_consuming_fanout(&dag);
    assert_eq!(copy_map, result.old_to_new);
    let dropped = ownership::insert_drop_nodes_for_unconsumed_values(copied);
    assert_eq!(
        serde_json::to_string(&dropped).unwrap(),
        serde_json::to_string(&result.dag).unwrap()
    );
    assert!(verify::verify(&result.dag).is_empty());
    let cost = analysis::analyze_copy_costs_for_roots(
        &result.dag,
        &[("main".into(), result.dag.roots().to_vec())],
    );
    assert_eq!(cost.total_copy_count, 1);
    assert_eq!(cost.total_bytes_copied, Some(12));
    let mut with_dead = dag.clone();
    let unused = with_dead.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Neg,
        vec![x],
        vector(3),
        Some("unused".into()),
    );
    let normalized = ownership::normalize_ownership(&with_dead);
    assert!(
        normalized
            .dag
            .nodes()
            .iter()
            .any(|n| matches!(n.op, RiscOp::Drop)
                && n.inputs == vec![normalized.old_to_new[&unused]])
    );
    let stripped_result = ownership::strip_drop_nodes(&normalized.dag);
    let (stripped, map) = (stripped_result.dag, stripped_result.old_to_new);
    assert!(
        !stripped
            .nodes()
            .iter()
            .any(|n| matches!(n.op, RiscOp::Drop))
    );
    assert_eq!(
        stripped.roots(),
        &[
            map[&normalized.old_to_new[&b]],
            map[&normalized.old_to_new[&a]]
        ]
    );
    let node = stripped.get(map[&normalized.old_to_new[&b]]).unwrap();
    assert_eq!(node.reusable_input, Some(map[&normalized.old_to_new[&x]]));
    assert_eq!(node.shape_deps, vec![map[&normalized.old_to_new[&x]]]);
    assert_eq!(node.span_id.as_deref(), Some("b"));
    assert_eq!(node.merged_spans, vec!["b.extra"]);
    for node in normalized.dag.nodes() {
        assert_eq!(map.contains_key(&node.id), !matches!(node.op, RiscOp::Drop));
    }
}

#[test]
fn fusion_keeps_rooted_intermediates_and_span_contributors() {
    let mut dag = Dag::new();
    dag.declare("fixture");
    let shape = load(
        &mut dag,
        "shape",
        TensorType {
            dims: vec![DimInfo::Named("batch".into(), None)],
            precision: Prim::F32,
        },
    );
    let x = load(&mut dag, "x", vector(3));
    let a = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Neg,
        vec![x],
        vector(3),
        Some("a".into()),
    );
    let b = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Exp,
        vec![a],
        vector(3),
        Some("b".into()),
    );
    dag.node_mut(a).unwrap().merged_spans = vec!["z".into(), "a".into(), "z".into()];
    dag.node_mut(b).unwrap().merged_spans = vec!["a".into(), "c".into()];
    dag.add_shape_dep(a, shape);
    dag.add_shape_dep(b, shape);
    dag.set_reusable_input(a, x);
    dag.add_root(b);
    let fused = fuse::fuse_with_remap(&dag);
    let root = fused.dag.get(fused.old_to_new[&b]).unwrap();
    assert!(matches!(root.op, RiscOp::FusedElem { .. }));
    assert_eq!(root.span_id.as_deref(), Some("a"));
    assert_eq!(root.merged_spans, vec!["b", "c", "z"]);
    assert_eq!(root.shape_deps, vec![fused.old_to_new[&shape]]);
    assert_eq!(root.reusable_input, Some(fused.old_to_new[&x]));
    assert!(verify::verify(&fused.dag).is_empty());
    dag.set_roots(vec![a, b]);
    let fused = fuse::fuse_with_remap(&dag);
    assert_ne!(fused.old_to_new[&a], fused.old_to_new[&b]);
    assert_eq!(
        fused.dag.roots(),
        &[fused.old_to_new[&a], fused.old_to_new[&b]]
    );
    assert!(matches!(
        fused.dag.get(fused.old_to_new[&a]).unwrap().op,
        RiscOp::Neg
    ));
    assert!(verify::verify(&fused.dag).is_empty());
}

#[test]
fn span_merge_and_folds_preserve_operation_provenance() {
    let mut dag = Dag::new();
    dag.declare("fixture");
    let a = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::synth_const(Prim::F32, 2.0),
        vec![],
        TensorType::scalar_f32(),
        Some("a".into()),
    );
    let b = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::synth_const(Prim::F32, 3.0),
        vec![],
        TensorType::scalar_f32(),
        Some("b".into()),
    );
    let out = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Add,
        vec![a, b],
        TensorType::scalar_f32(),
        Some("op".into()),
    );
    span_merge::append_spans_to_node(&mut dag, out, &["z".into(), "op".into(), "z".into()]);
    optimize::constant_fold(&mut dag);
    let node = dag.get(out).unwrap();
    assert!(
        matches!(&node.op, RiscOp::Const { value } if value.as_f64_lossy() == 5.0 && value.prim() == Prim::F32)
    );
    assert_eq!(node.span_id.as_deref(), Some("op"));
    assert_eq!(node.merged_spans, vec!["a", "b", "z"]);
    let mut trap = Dag::new();
    trap.declare("fixture");
    let x = trap.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::synth_const(Prim::Int8, 127.0),
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::Int8,
        },
        None,
    );
    let out = trap.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Add,
        vec![x, x],
        TensorType {
            dims: vec![],
            precision: Prim::Int8,
        },
        Some("trap".into()),
    );
    optimize::constant_fold(&mut trap);
    assert!(matches!(trap.get(out).unwrap().op, RiscOp::Add));
    assert_eq!(trap.get(out).unwrap().inputs, vec![x, x]);
}

#[test]
fn symbolic_copy_costs_deduplicate_shared_roots_and_saturate() {
    let mut dag = Dag::new();
    dag.declare("fixture");
    let ty = TensorType {
        dims: vec![DimInfo::Named("batch".into(), None), DimInfo::Lit(4)],
        precision: Prim::F32,
    };
    let x = load(&mut dag, "x", ty.clone());
    let copy = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Copy,
        vec![x],
        ty,
        None,
    );
    let cost = analysis::analyze_copy_costs_for_roots(&dag, &[("main".into(), vec![copy, copy])]);
    assert_eq!(cost.total_copy_count, 1);
    assert_eq!(cost.total_bytes_copied, None);
    assert_eq!(cost.total_byte_formula.as_deref(), Some("batch * 4 * 4"));
    assert_eq!(
        serde_json::to_string(&cost).unwrap(),
        "{\"functions\":[{\"name\":\"main\",\"copy_count\":1,\"bytes_copied\":null}],\"total_copy_count\":1,\"total_bytes_copied\":null}"
    );
    let huge = TensorType {
        dims: vec![DimInfo::Lit(usize::MAX), DimInfo::Lit(2)],
        precision: Prim::F64,
    };
    dag.node_mut(copy).unwrap().output_type = huge;
    let cost = analysis::analyze_copy_costs(&dag, &[("main".into(), copy)]);
    assert_eq!(cost.total_bytes_copied, Some(usize::MAX));
}

#[test]
fn value_dependent_symbolic_adjoint_returns_a_diagnostic() {
    let mut dag = Dag::new();
    dag.declare("fixture");
    let ty = TensorType {
        dims: vec![DimInfo::Named("batch".into(), None)],
        precision: Prim::F32,
    };
    let x = load(&mut dag, "x", ty);
    let loss = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::ProdReduce { axis: 0 },
        vec![x],
        TensorType::scalar_f32(),
        Some("product".into()),
    );
    dag.add_root(loss);
    assert!(grad::grad_dag_checked(&dag, loss, &[x]).is_err());
}

#[test]
fn folds_decline_inexact_wide_images_and_unsupported_operators() {
    let mut dag = Dag::new();
    dag.declare("fixture");
    let ty = TensorType {
        dims: vec![],
        precision: Prim::Int64,
    };
    let value = chelis_numeric_core::dtype_semantics::scalar_from_i64(
        "fixture",
        Prim::Int64,
        9_007_199_254_740_993,
    )
    .unwrap();
    let a = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Const { value },
        vec![],
        ty.clone(),
        Some("exact".into()),
    );
    let zero =
        chelis_numeric_core::dtype_semantics::scalar_from_i64("fixture", Prim::Int64, 0).unwrap();
    let b = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Const { value: zero },
        vec![],
        ty.clone(),
        None,
    );
    let out = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Add,
        vec![a, b],
        ty,
        Some("add".into()),
    );
    dag.add_root(out);
    optimize::constant_fold(&mut dag);
    assert!(matches!(dag.get(out).unwrap().op, RiscOp::Add));
    assert_eq!(dag.get(out).unwrap().inputs, vec![a, b]);
    assert!(
        matches!(&dag.get(a).unwrap().op, RiscOp::Const { value } if value.as_i64_exact() == Some(9_007_199_254_740_993))
    );
    assert_eq!(dag.get(out).unwrap().span_id.as_deref(), Some("add"));
    assert!(verify::verify(&dag).is_empty());
    let mut dag = Dag::new();
    dag.declare("fixture");
    let a = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::synth_const(Prim::F32, 2.0),
        vec![],
        TensorType::scalar_f32(),
        None,
    );
    let b = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::synth_const(Prim::F32, 3.0),
        vec![],
        TensorType::scalar_f32(),
        None,
    );
    let out = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Div,
        vec![a, b],
        TensorType::scalar_f32(),
        Some("divide".into()),
    );
    dag.add_root(out);
    optimize::constant_fold(&mut dag);
    assert!(matches!(dag.get(out).unwrap().op, RiscOp::Div));
    assert_eq!(dag.get(out).unwrap().inputs, vec![a, b]);
    assert!(verify::verify(&dag).is_empty());
}

#[test]
fn symbolic_sum_adjoint_keeps_a_shape_only_load_and_scalar_seed() {
    let mut dag = Dag::new();
    dag.declare("fixture");
    let ty = TensorType {
        dims: vec![DimInfo::Named("batch".into(), None)],
        precision: Prim::F32,
    };
    let x = load(&mut dag, "x", ty.clone());
    let loss = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::sum_default(0, Prim::F32).unwrap(),
        vec![x],
        TensorType::scalar_f32(),
        Some("loss".into()),
    );
    dag.add_root(loss);
    let gradient = grad::grad_dag_checked(&dag, loss, &[x]).unwrap();
    assert_eq!(gradient.grad_nodes.len(), 1);
    let root = gradient.grad_nodes[&x];
    assert_eq!(gradient.dag.get(root).unwrap().output_type, ty);
    assert!(gradient.dag.nodes().iter().any(|node| matches!(&node.op,
        RiscOp::Const { value } if node.output_type.dims.is_empty() && value.as_f64_lossy() == 1.0)));
    for node in &gradient.dag.nodes()[dag.len()..] {
        assert_eq!(node.span_id.as_deref(), Some("__synthesized_grad__"));
        assert!(!node.merged_spans.is_empty());
    }
    let mut backward = gradient.dag.clone();
    backward.set_roots(vec![root]);
    let (compact, remap) = optimize::dead_code_eliminate_with_remap(&backward);
    assert!(remap.contains_key(&x));
    assert!(!remap.contains_key(&loss));
    assert!(compact.nodes().iter().any(|node| {
        node.shape_deps.contains(&remap[&x])
            || node
                .inputs
                .iter()
                .enumerate()
                .any(|(slot, input)| *input == remap[&x] && !node.reads_input_payload(slot))
    }));
    assert_eq!(compact.roots(), &[remap[&root]]);
    assert!(verify::verify(&compact).is_empty());
}

#[test]
fn every_rebuild_preserves_claims_owners_and_activation_edges() {
    use chelis_dag_core::dag::{DeclId, ExtentWitnessSite, Owner, RtAxis};
    let mut dag = Dag::new();
    let decl = dag.declare("entry");
    let other = dag.declare("other");
    let dead = dag.add_node(
        other,
        RiscOp::Load {
            name: "dead".into(),
        },
        vec![],
        vector(3),
        None,
    );
    let activation = dag.add_node(
        decl,
        RiscOp::Load {
            name: "active".into(),
        },
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::Bool,
        },
        Some("activation".into()),
    );
    let owner = Owner::new(decl, Some(activation));
    let claim = dag.add_node(
        owner,
        RiscOp::ExtentWitness {
            site: ExtentWitnessSite::LiteralResultClaim,
            parameter: String::new(),
            axis: RtAxis::Lit(0),
            requirements: vec![
                chelis_numeric_core::dtype_semantics::scalar_from_i64("fixture", Prim::Int64, 3)
                    .unwrap(),
            ],
            claims: vec![],
        },
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::Int64,
        },
        Some("claim".into()),
    );
    let input = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        vector(3),
        Some("input".into()),
    );
    let shape = dag.add_node(
        owner,
        RiscOp::Load {
            name: "shape".into(),
        },
        vec![],
        vector(3),
        None,
    );
    let output = dag.add_node(
        owner,
        RiscOp::Neg,
        vec![input],
        vector(3),
        Some("output".into()),
    );
    dag.add_result_claim_dep(output, claim);
    dag.add_shape_dep(output, shape);
    dag.set_reusable_input(output, input);
    dag.node_mut(output).unwrap().merged_spans = vec!["extra".into()];
    dag.set_roots(vec![output, input]);
    // Drop stripping and fusion retain unreachable inputs before ownership closes them.
    assert!(
        verify::verify_ownership_input(&dag).is_empty(),
        "{:?}",
        verify::verify_ownership_input(&dag)
    );
    let (compact, map) = optimize::dead_code_eliminate_with_remap(&dag);
    assert!(!map.contains_key(&dead));
    let fused = fuse::fuse_with_remap(&dag);
    let normalized = ownership::normalize_ownership(&dag);
    let stripped_result = ownership::strip_drop_nodes(&dag);
    let (stripped, strip_map) = (stripped_result.dag, stripped_result.old_to_new);
    for (rebuilt, map) in [
        (&compact, &map),
        (&fused.dag, &fused.old_to_new),
        (&normalized.dag, &normalized.old_to_new),
        (&stripped, &strip_map),
    ] {
        assert_eq!(rebuilt.declarations(), dag.declarations());
        assert_eq!(rebuilt.roots(), &[map[&output], map[&input]]);
        let node = rebuilt.get(map[&output]).unwrap();
        assert_eq!(node.owner, Owner::new(decl, Some(map[&activation])));
        assert_eq!(node.result_claim_deps, vec![map[&claim]]);
        assert_eq!(node.shape_deps, vec![map[&shape]]);
        assert_eq!(node.reusable_input, Some(map[&input]));
        assert_eq!(node.span_id.as_deref(), Some("output"));
        assert_eq!(node.merged_spans, vec!["extra"]);
        assert!(
            verify::verify_ownership_input(rebuilt).is_empty(),
            "{:?}",
            verify::verify_ownership_input(rebuilt)
        );
    }
    let common = optimize::common_subexpr_eliminate(&dag);
    assert_eq!(common.declarations(), dag.declarations());
    let node = common.get(common.roots()[0]).unwrap();
    assert_eq!(node.owner.decl, DeclId(0));
    assert_eq!(node.result_claim_deps.len(), 1);
    assert_eq!(node.shape_deps.len(), 1);
    assert!(node.owner.activation.is_some());
    assert!(verify::verify_ownership_input(&common).is_empty());
    assert!(verify::verify(&compact).is_empty());
    assert!(!verify::verify(&normalized.dag).is_empty());
    assert!(verify::verify_ownership_input(&normalized.dag).is_empty());
}

#[test]
fn malformed_declarations_activations_and_claim_edges_fail() {
    use chelis_dag_core::dag::{DeclId, Owner};
    let mut dag = Dag::new();
    let decl = dag.declare("entry");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vector(3),
        None,
    );
    let y = dag.add_node(decl, RiscOp::Neg, vec![x], vector(3), None);
    dag.add_root(y);
    let mut bad = dag.clone();
    bad.node_mut(y).unwrap().id = NodeId(99);
    rejects(&bad);
    let mut bad = dag.clone();
    bad.set_reusable_input(x, y);
    rejects(&bad);
    let mut bad = dag.clone();
    bad.node_mut(y).unwrap().owner = Owner::unconditional(DeclId(99));
    rejects(&bad);
    for activation in [x, y, NodeId(99)] {
        let mut bad = dag.clone();
        bad.node_mut(y).unwrap().owner.activation = Some(activation);
        rejects(&bad);
    }
    for claim in [y, NodeId(99)] {
        let mut bad = dag.clone();
        bad.add_result_claim_dep(y, claim);
        rejects(&bad);
    }
}
