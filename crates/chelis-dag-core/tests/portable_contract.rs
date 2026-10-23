use chelis_dag_core::{Dag, DimInfo, LoadStoreName, NodeId, RiscOp, TensorType};
use chelis_dag_core::{analysis, fuse, grad, optimize, ownership, tier2, verify};
use chelis_numeric_core::prim::Prim;

fn vector() -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(3)],
        precision: Prim::F32,
    }
}

#[test]
fn checked_names_reject_constructor_and_codec_inputs() {
    for name in ["", "1x", "x\0", "x\n", "x\r", "x%", "x/", "é", "x\u{7f}"] {
        assert!(LoadStoreName::new(name).is_err());
        assert!(
            serde_json::from_str::<LoadStoreName>(&serde_json::to_string(name).unwrap()).is_err()
        );
        assert!(bincode::deserialize::<LoadStoreName>(&bincode::serialize(name).unwrap()).is_err());
    }
    let name = LoadStoreName::new("grads.1").unwrap();
    assert_eq!(serde_json::to_string(&name).unwrap(), "\"grads.1\"");
    assert_eq!(
        bincode::serialize(&name).unwrap(),
        bincode::serialize("grads.1").unwrap()
    );
}

#[test]
fn shared_ad_fusion_and_ownership_have_real_graphs() {
    let mut dag = Dag::new();
    dag.declare("fixture");
    let x = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Load { name: "x".into() },
        vec![],
        vector(),
        Some("a:1".into()),
    );
    let square = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Mul,
        vec![x, x],
        vector(),
        Some("a:2".into()),
    );
    let loss = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![square],
        TensorType {
            dims: vec![],
            precision: Prim::F32,
        },
        Some("a:3".into()),
    );
    dag.add_root(loss);
    assert!(verify::verify(&dag).is_empty());
    let gradient = grad::grad_dag_checked(&dag, loss, &[x]).unwrap();
    assert!(gradient.grad_nodes.contains_key(&x));
    assert!(verify::verify(&gradient.dag).is_empty());
    let fused = fuse::fuse_with_remap(&gradient.dag);
    assert!(verify::verify(&fused.dag).is_empty());
    let normalized = ownership::normalize_ownership(&fused.dag);
    assert!(verify::verify(&normalized.dag).is_empty());
    let roots = normalized
        .dag
        .roots()
        .iter()
        .copied()
        .map(|root| ("loss".into(), root))
        .collect::<Vec<_>>();
    assert_eq!(
        analysis::analyze_copy_costs(&normalized.dag, &roots).total_copy_count,
        0
    );
}

#[test]
fn topology_errors_are_deterministic_and_do_not_panic() {
    let mut dag = Dag::new();
    dag.declare("fixture");
    dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Add,
        vec![NodeId(99), NodeId(0)],
        vector(),
        None,
    );
    dag.add_root(NodeId(88));
    let first = verify::verify(&dag);
    assert!(!first.is_empty());
    assert_eq!(first, verify::verify(&dag));
}

#[test]
fn tier2_and_declined_folds_preserve_operators() {
    let mut dag = Dag::new();
    dag.declare("fixture");
    let x = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Load { name: "x".into() },
        vec![],
        vector(),
        None,
    );
    let y = tier2::lower_relu(
        chelis_dag_core::dag::Owner::unconditional(chelis_dag_core::dag::DeclId(0)),
        &mut dag,
        x,
        &vector(),
        Some("a:4"),
    );
    dag.add_root(y);
    optimize::constant_fold(&mut dag);
    assert!(matches!(dag.get(y).unwrap().op, RiscOp::Relu));
    assert!(verify::verify(&dag).is_empty());
}

#[test]
fn malformed_shape_arithmetic_returns_diagnostics() {
    let mut dag = Dag::new();
    dag.declare("fixture");
    let x = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Load { name: "x".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(usize::MAX), DimInfo::Lit(2)],
            precision: Prim::F32,
        },
        None,
    );
    dag.add_root(x);
    assert!(!verify::verify(&dag).is_empty());
    let mut padded = Dag::new();
    padded.declare("fixture");
    let x = padded.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Load { name: "x".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(usize::MAX)],
            precision: Prim::F32,
        },
        None,
    );
    let y = padded.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::zero_pad(
            Prim::F32,
            vec![(
                chelis_dag_core::dag::RtDim::Lit(1),
                chelis_dag_core::dag::RtDim::Lit(1),
            )],
        ),
        vec![x],
        TensorType {
            dims: vec![DimInfo::Lit(1)],
            precision: Prim::F32,
        },
        None,
    );
    padded.add_root(y);
    assert!(!verify::verify(&padded).is_empty());
}

#[test]
fn consuming_reuse_hint_tracks_the_inserted_copy() {
    let mut dag = Dag::new();
    dag.declare("fixture");
    let x = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Load { name: "x".into() },
        vec![],
        vector(),
        None,
    );
    let first = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Realize,
        vec![x],
        vector(),
        None,
    );
    dag.set_reusable_input(first, x);
    let second = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Realize,
        vec![x],
        vector(),
        None,
    );
    dag.add_root(first);
    dag.add_root(second);
    let result = ownership::normalize_ownership(&dag);
    let first = result.dag.get(result.old_to_new[&first]).unwrap();
    assert_eq!(first.reusable_input, first.inputs.first().copied());
    assert!(matches!(
        result.dag.get(first.inputs[0]).unwrap().op,
        RiscOp::Copy
    ));
    assert!(verify::verify(&result.dag).is_empty());
}

#[test]
fn ownership_preserves_upstream_activation_reuse() {
    let lowerers = [tier2::lower_sigmoid, tier2::lower_silu];
    for lower in lowerers {
        let mut dag = Dag::new();
        dag.declare("fixture");
        let input = dag.add_node(
            chelis_dag_core::dag::DeclId(0),
            RiscOp::Load { name: "x".into() },
            vec![],
            vector(),
            None,
        );
        let owned = dag.add_node(
            chelis_dag_core::dag::DeclId(0),
            RiscOp::Realize,
            vec![input],
            vector(),
            None,
        );
        let output = lower(
            chelis_dag_core::dag::Owner::unconditional(chelis_dag_core::dag::DeclId(0)),
            &mut dag,
            owned,
            &vector(),
            Some("activation"),
        );
        dag.set_reusable_input(output, owned);
        dag.add_root(output);
        let normalized = ownership::normalize_ownership(&dag);
        let output = normalized.old_to_new[&output];
        let owned = normalized.old_to_new[&owned];
        assert_eq!(
            normalized.dag.get(output).unwrap().reusable_input,
            Some(owned)
        );
        assert!(verify::verify(&normalized.dag).is_empty());
        let fused = fuse::fuse_with_remap(&normalized.dag);
        let output = fused.old_to_new[&output];
        let owned = fused.old_to_new[&owned];
        assert_eq!(fused.dag.get(output).unwrap().reusable_input, Some(owned));
        assert!(verify::verify(&fused.dag).is_empty());
    }
}

#[test]
fn cse_keeps_distinct_reuse_contracts() {
    let mut dag = Dag::new();
    dag.declare("fixture");
    let x = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Load { name: "x".into() },
        vec![],
        vector(),
        None,
    );
    let y = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Load { name: "y".into() },
        vec![],
        vector(),
        None,
    );
    let a = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Add,
        vec![x, y],
        vector(),
        None,
    );
    dag.set_reusable_input(a, x);
    let b = dag.add_node(
        chelis_dag_core::dag::DeclId(0),
        RiscOp::Add,
        vec![x, y],
        vector(),
        None,
    );
    dag.set_reusable_input(b, y);
    dag.set_roots(vec![a, b]);
    let output = optimize::common_subexpr_eliminate(&dag);
    assert_ne!(output.roots()[0], output.roots()[1]);
    assert_eq!(
        output.get(output.roots()[0]).unwrap().reusable_input,
        Some(x)
    );
    assert_eq!(
        output.get(output.roots()[1]).unwrap().reusable_input,
        Some(y)
    );
    assert!(verify::verify(&output).is_empty());
}

#[test]
fn payload_reads_distinguish_shape_carriers_and_runtime_bounds() {
    use chelis_dag_core::dag::{RtAxis, RtDim};
    let mut dag = Dag::new();
    let owner = dag.declare("fixture");
    let data_type = TensorType {
        dims: vec![DimInfo::Lit(9)],
        precision: Prim::F32,
    };
    let data = dag.add_node(
        owner,
        RiscOp::Load {
            name: "data".into(),
        },
        vec![],
        data_type,
        None,
    );
    let shape = dag.add_node(
        owner,
        RiscOp::Load {
            name: "shape".into(),
        },
        vec![],
        vector(),
        None,
    );
    let extent_type = TensorType {
        dims: vec![],
        precision: Prim::Int64,
    };
    let extent = dag.add_node(
        owner,
        RiscOp::synth_const(Prim::Int64, 3.0),
        vec![],
        extent_type.clone(),
        None,
    );
    let reshape = dag.add_node(
        owner,
        RiscOp::Reshape {
            new_shape: vec![
                RtDim::InputAxis {
                    tensor: 1,
                    axis: RtAxis::Lit(0),
                },
                RtDim::Node(2),
            ],
        },
        vec![data, shape, extent],
        TensorType {
            dims: vec![DimInfo::Lit(3), DimInfo::Lit(3)],
            precision: Prim::F32,
        },
        None,
    );
    dag.add_root(reshape);
    assert!(verify::verify(&dag).is_empty());
    let node = dag.get(reshape).unwrap();
    assert!(node.reads_input_payload(0));
    assert!(!node.reads_input_payload(1));
    assert!(node.reads_input_payload(2));
    let shape_read = dag.add_node(
        owner,
        RiscOp::Shape { axis: 0 },
        vec![data],
        extent_type,
        None,
    );
    assert!(!dag.get(shape_read).unwrap().reads_input_payload(0));
    let drop = dag.add_node(owner, RiscOp::Drop, vec![data], vector(), None);
    assert!(!dag.get(drop).unwrap().reads_input_payload(0));
}

#[test]
fn dce_remaps_reuse_after_removing_an_earlier_node() {
    let mut dag = Dag::new();
    let owner = dag.declare("fixture");
    dag.add_node(
        owner,
        RiscOp::synth_const(Prim::F32, 0.0),
        vec![],
        vector(),
        None,
    );
    let input = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        vector(),
        None,
    );
    let output = dag.add_node(owner, RiscOp::Realize, vec![input], vector(), None);
    dag.set_reusable_input(output, input);
    dag.add_root(output);
    let (pruned, remap) = optimize::dead_code_eliminate_with_remap(&dag);
    assert_eq!(pruned.len(), 2);
    assert_eq!(pruned.roots(), &[remap[&output]]);
    assert_eq!(
        pruned.get(remap[&output]).unwrap().reusable_input,
        Some(remap[&input])
    );
    assert!(verify::verify(&pruned).is_empty());
}

#[test]
#[allow(clippy::disallowed_methods)]
fn wide_image_folds_preserve_native_stored_bits() {
    use chelis_numeric_core::dtype_semantics::{
        FloatUnOp, ScalarValue, WideBinaryFold, WideUnaryFold, float_unop, fold_wide_binary,
        fold_wide_unary, scalar_from_f64, scalar_from_i64,
    };
    use chelis_numeric_core::observation::ElementRef;
    use std::io::Write;

    #[derive(serde::Serialize)]
    struct FoldComparison {
        operation: &'static str,
        lhs: (Prim, u64),
        rhs: Option<(Prim, u64)>,
        destination: Prim,
        actual: Option<(Prim, u64)>,
        expected: Option<(Prim, u64)>,
    }

    fn record(output: &mut Option<std::io::BufWriter<std::fs::File>>, comparison: FoldComparison) {
        if let Some(output) = output {
            serde_json::to_writer(&mut *output, &comparison).unwrap();
            output.write_all(b"\n").unwrap();
        }
    }

    fn image(value: ScalarValue) -> Option<f64> {
        if let Some(integer) = value.as_i64_exact()
            && (integer as f64) as i128 != integer as i128
        {
            return None;
        }
        Some(value.as_f64_lossy())
    }

    fn stored(value: ScalarValue) -> (Prim, u64) {
        let bits = match value.element_ref() {
            ElementRef::F16(value) => u64::from(value.to_bits()),
            ElementRef::Bf16(value) => u64::from(value.to_bits()),
            ElementRef::F32(value) => u64::from(value.to_bits()),
            ElementRef::F64(value) => value.to_bits(),
            _ => value
                .as_i64_exact()
                .map(|value| value as u64)
                .unwrap_or_else(|| u64::from(value.as_bool_exact().unwrap())),
        };
        (value.prim(), bits)
    }

    let mut inputs = [
        4.0_f64,
        -4.0,
        0.0,
        -0.0,
        2.5,
        -2.5,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x7ff8_0000_0000_0042),
        f64::from_bits(0xfff8_0000_0000_0042),
        f64::from_bits(0x7ff0_0000_0000_0042),
        f64::from_bits(0xfff0_0000_0000_0042),
    ]
    .map(|value| scalar_from_f64("fixture", Prim::F64, value).unwrap())
    .to_vec();
    for value in [
        i64::MIN,
        i64::MAX,
        -9_007_199_254_740_993,
        9_007_199_254_740_993,
        -9_007_199_254_740_992,
        9_007_199_254_740_992,
        -1,
        0,
        1,
    ] {
        inputs.push(scalar_from_i64("fixture", Prim::Int64, value).unwrap());
    }
    for value in [0, 1] {
        inputs.push(scalar_from_i64("fixture", Prim::Bool, value).unwrap());
    }
    let destinations = [
        Prim::F64,
        Prim::F32,
        Prim::F16,
        Prim::Bf16,
        Prim::Int64,
        Prim::Int32,
        Prim::Int16,
        Prim::Int8,
        Prim::Bool,
    ];
    let mut evidence = std::env::var_os("CHELIS_DAG_FOLD_BITS_OUTPUT")
        .map(|path| std::io::BufWriter::new(std::fs::File::create(path).unwrap()));
    let mut differences = Vec::new();
    for (name, operation, fold) in [
        ("neg", RiscOp::Neg, WideUnaryFold::Neg),
        ("sqrt", RiscOp::Sqrt, WideUnaryFold::Sqrt),
        ("abs", RiscOp::Abs, WideUnaryFold::Abs),
        ("floor", RiscOp::Floor, WideUnaryFold::Floor),
        ("ceil", RiscOp::Ceil, WideUnaryFold::Ceil),
        ("round", RiscOp::Round, WideUnaryFold::Round),
    ] {
        for &input in &inputs {
            for destination in destinations {
                let expected = image(input).and_then(|value| {
                    let raw = match fold {
                        WideUnaryFold::Neg => -value,
                        WideUnaryFold::Sqrt => value.sqrt(),
                        WideUnaryFold::Abs => value.abs(),
                        WideUnaryFold::Floor => value.floor(),
                        WideUnaryFold::Ceil => value.ceil(),
                        WideUnaryFold::Round => value.round_ties_even(),
                    };
                    scalar_from_f64("const", destination, raw).ok()
                });
                let actual = fold_wide_unary(fold, input, destination);
                record(
                    &mut evidence,
                    FoldComparison {
                        operation: name,
                        lhs: stored(input),
                        rhs: None,
                        destination,
                        actual: actual.map(stored),
                        expected: expected.map(stored),
                    },
                );
                if actual.map(stored) != expected.map(stored) {
                    differences.push(format!(
                        "{fold:?}({:?}) -> {destination:?}: {:?} != {:?}",
                        stored(input),
                        actual.map(stored),
                        expected.map(stored)
                    ));
                }
                if destination == Prim::F64 {
                    let mut dag = Dag::new();
                    let owner = dag.declare("fixture");
                    let input_ty = TensorType {
                        dims: vec![],
                        precision: input.prim(),
                    };
                    let output_ty = TensorType {
                        dims: vec![],
                        precision: destination,
                    };
                    let source = dag.add_node(
                        owner,
                        RiscOp::Const { value: input },
                        vec![],
                        input_ty,
                        None,
                    );
                    let output =
                        dag.add_node(owner, operation.clone(), vec![source], output_ty, None);
                    dag.add_root(output);
                    optimize::constant_fold(&mut dag);
                    match (&dag.get(output).unwrap().op, expected) {
                        (RiscOp::Const { value }, Some(expected)) => {
                            if stored(*value) != stored(expected) {
                                differences.push(format!(
                                    "graph {fold:?}: {:?} != {:?}",
                                    stored(*value),
                                    stored(expected)
                                ));
                            }
                        }
                        (actual, None) => assert_eq!(actual, &operation),
                        (actual, Some(_)) => panic!("wide fold declined: {actual:?}"),
                    }
                }
            }
        }
    }
    for (name, operation, fold) in [
        ("add", RiscOp::Add, WideBinaryFold::Add),
        ("mul", RiscOp::Mul, WideBinaryFold::Mul),
    ] {
        for &lhs in &inputs {
            for &rhs in &inputs {
                for destination in destinations {
                    let expected = image(lhs).zip(image(rhs)).and_then(|(lhs, rhs)| {
                        let raw = match fold {
                            WideBinaryFold::Add => lhs + rhs,
                            WideBinaryFold::Mul => lhs * rhs,
                        };
                        scalar_from_f64("const", destination, raw).ok()
                    });
                    let actual = fold_wide_binary(fold, lhs, rhs, destination);
                    record(
                        &mut evidence,
                        FoldComparison {
                            operation: name,
                            lhs: stored(lhs),
                            rhs: Some(stored(rhs)),
                            destination,
                            actual: actual.map(stored),
                            expected: expected.map(stored),
                        },
                    );
                    if actual.map(stored) != expected.map(stored) {
                        differences.push(format!(
                            "{fold:?}({:?}, {:?}) -> {destination:?}: {:?} != {:?}",
                            stored(lhs),
                            stored(rhs),
                            actual.map(stored),
                            expected.map(stored)
                        ));
                    }
                    if destination == Prim::F64 {
                        let mut dag = Dag::new();
                        let owner = dag.declare("fixture");
                        let left = dag.add_node(
                            owner,
                            RiscOp::Const { value: lhs },
                            vec![],
                            TensorType {
                                dims: vec![],
                                precision: lhs.prim(),
                            },
                            None,
                        );
                        let right = dag.add_node(
                            owner,
                            RiscOp::Const { value: rhs },
                            vec![],
                            TensorType {
                                dims: vec![],
                                precision: rhs.prim(),
                            },
                            None,
                        );
                        let output = dag.add_node(
                            owner,
                            operation.clone(),
                            vec![left, right],
                            TensorType {
                                dims: vec![],
                                precision: destination,
                            },
                            None,
                        );
                        dag.add_root(output);
                        optimize::constant_fold(&mut dag);
                        match (&dag.get(output).unwrap().op, expected) {
                            (RiscOp::Const { value }, Some(expected)) => {
                                if stored(*value) != stored(expected) {
                                    differences.push(format!(
                                        "graph {fold:?}: {:?} != {:?}",
                                        stored(*value),
                                        stored(expected)
                                    ));
                                }
                            }
                            (actual, None) => assert_eq!(actual, &operation),
                            (actual, Some(_)) => panic!("wide fold declined: {actual:?}"),
                        }
                    }
                }
            }
        }
    }
    let nan = scalar_from_f64("fixture", Prim::F64, f64::from_bits(0x7ff8_0000_0000_0042)).unwrap();
    assert_eq!(
        stored(float_unop(FloatUnOp::Neg, nan).unwrap()),
        (Prim::F64, 0x7ff8_0000_0000_0000),
    );
    if let Some(mut evidence) = evidence {
        evidence.flush().unwrap();
    }
    assert!(
        differences.is_empty(),
        "numeric provider differences: {differences:#?}"
    );
}
