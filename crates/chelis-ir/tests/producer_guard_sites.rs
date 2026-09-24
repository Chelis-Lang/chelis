//! Spec/04 §4.7: claim ownership is independent of extent-value readiness.
use chelis_ir::axis_sources::{entry_extent_guards, local_dim_guard_sites};
use chelis_ir::dag::{Dag, DimInfo, ExtentWitnessSite, RiscOp, RtAxis, RtDim, TensorType};
use chelis_types::{ScalarValue, types::Prim};

fn scalar(value: i64) -> ScalarValue {
    chelis_types::scalar_from_i64("const", Prim::Int64, value).unwrap()
}

fn graph(axis: i32) -> Dag {
    let mut dag = Dag::new();
    let scalar_type = TensorType {
        dims: vec![],
        precision: Prim::Int64,
    };
    let source = dag.add_node(
        RiscOp::Load {
            name: "source".into(),
        },
        vec![],
        TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::Int64,
        },
        None,
    );
    let input = dag.add_node(
        RiscOp::Const { value: scalar(1) },
        vec![],
        scalar_type.clone(),
        None,
    );
    let claim = dag.add_node(
        RiscOp::ExtentWitness {
            site: ExtentWitnessSite::LiteralResultClaim,
            parameter: String::new(),
            axis: RtAxis::Lit(axis),
            requirements: vec![scalar(3)],
            claims: vec![],
        },
        vec![],
        scalar_type,
        None,
    );
    let producer = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::InputAxis {
                tensor: 1,
                axis: RtAxis::Lit(0),
            },
        },
        vec![input, source],
        TensorType {
            dims: vec![DimInfo::Named(String::new(), None)],
            precision: Prim::Int64,
        },
        None,
    );
    dag.add_shape_dep(producer, claim);
    dag.add_root(producer);
    dag
}

#[test]
fn interface_sized_insert_has_one_local_result_claim() {
    for (physical_claim, token) in [(None, true), (Some(3), false), (Some(3), true)] {
        let mut dag = graph(0);
        if let Some(required) = physical_claim {
            let root = dag.roots()[0];
            let node = dag.node_mut(root).unwrap();
            node.output_type.dims = vec![DimInfo::Lit(required)];
            if !token {
                node.shape_deps.clear();
            }
        }
        assert!(entry_extent_guards(&dag).is_empty());
        let sites = local_dim_guard_sites(&dag).unwrap();
        assert_eq!(sites.len(), 1);
        assert_eq!(sites[0].0, (dag.roots()[0].0, 0));
        assert_eq!(sites[0].1.op, "insert");
    }
    // A different physical requirement is independent of the literal token.
    let mut dag = graph(0);
    let root = dag.roots()[0];
    dag.node_mut(root).unwrap().output_type.dims = vec![DimInfo::Lit(4)];
    let sites = local_dim_guard_sites(&dag).unwrap();
    assert_eq!(sites.len(), 2);
    assert!(
        sites
            .iter()
            .all(|(site, claim)| *site == (root.0, 0) && claim.op == "insert")
    );
}

#[test]
fn missing_result_axis_is_a_checked_error() {
    for axis in [-1, 1] {
        let error = local_dim_guard_sites(&graph(axis)).unwrap_err();
        assert!(error.contains("producer axis"), "{error}");
        let runtime_error = chelis_ir::eval::eval_tensor_with(&graph(axis), |_| {
            panic!("a malformed producer relationship must reject before reading inputs")
        })
        .unwrap_err();
        assert!(runtime_error.contains("producer axis"), "{runtime_error}");
        let errors = chelis_ir::verify::verify(&graph(axis));
        assert!(
            errors
                .iter()
                .any(|error| error.contains("valid producing axis")),
            "{errors:?}"
        );
    }
}

#[test]
fn malformed_administrative_producer_is_a_checked_error() {
    let mut dag = graph(0);
    let root = dag.roots()[0];
    let node = dag.node_mut(root).unwrap();
    node.op = RiscOp::Copy;
    node.inputs = vec![root];
    let error = local_dim_guard_sites(&dag).unwrap_err();
    assert!(error.contains("producer axis"), "{error}");
}
