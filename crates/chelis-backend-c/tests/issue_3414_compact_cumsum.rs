//! [05-OP-33]: bounded scan emission and first-leaf prefix/suffix semantics.
mod support;
use chelis_dag_core::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_dag_core::grad::grad_dag_checked;
use chelis_dag_core::tier2_ordered::lower_cumsum;
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_numeric_core::prim::Prim;
use chelis_unord::UnordMap;
use std::time::Instant;

fn gradient(n: usize) -> (Dag, chelis_dag_core::dag::NodeId) {
    let mut dag = Dag::new();
    let owner = dag.declare("ordered scan");
    let ty = TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    };
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty.clone(),
        None,
    );
    let w = dag.add_node(
        owner,
        RiscOp::Load { name: "w".into() },
        vec![],
        ty.clone(),
        None,
    );
    let scan = lower_cumsum(owner.into(), &mut dag, x, &ty, 0, None)
        .unwrap_or_else(|_| panic!("float scan lowers"));
    let weighted = dag.add_node(owner, RiscOp::Mul, vec![scan, w], ty, None);
    let loss = dag.add_node(
        owner,
        RiscOp::sum_default(0, Prim::F32).unwrap(),
        vec![weighted],
        TensorType::scalar_f32(),
        None,
    );
    let differentiated = grad_dag_checked(&dag, loss, &[x]).unwrap();
    let root = differentiated.grad_nodes[&x];
    (differentiated.dag, root)
}

#[test]
fn cumsum_adjoint_last_suffix_preserves_negative_zero() {
    let (dag, root) = gradient(2);
    let inputs = UnordMap::from([
        ("x".into(), TensorValue::from_vec(vec![2], vec![0.0; 2])),
        ("w".into(), TensorValue::from_vec(vec![2], vec![-0.0; 2])),
    ]);
    let values = eval_tensor(&dag, &inputs).unwrap();
    let bits = values[&root]
        .to_f64_lossy_vec()
        .into_iter()
        .map(f64::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(bits, vec![(-0.0_f64).to_bits(); 2]);
    // Negative parity: a real positive-zero cotangent changes the suffix.
    let inputs = UnordMap::from([
        ("x".into(), TensorValue::from_vec(vec![2], vec![0.0; 2])),
        ("w".into(), TensorValue::from_vec(vec![2], vec![-0.0, 0.0])),
    ]);
    let values = eval_tensor(&dag, &inputs).unwrap();
    assert_eq!(
        values[&root]
            .to_f64_lossy_vec()
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>(),
        vec![0.0_f64.to_bits(); 2]
    );
}

#[test]
fn cumsum_grad_emission_is_independent_of_scan_length() {
    let mut sizes = Vec::new();
    for n in [16, 64, 4096] {
        let (dag, root) = gradient(n);
        let slots: usize = dag
            .nodes()
            .iter()
            .map(|node| {
                node.output_type
                    .dims
                    .iter()
                    .map(|dim| match dim {
                        DimInfo::Lit(n) => *n,
                        _ => panic!("static counterfactual"),
                    })
                    .product::<usize>()
            })
            .sum();
        let inputs: UnordMap<String, TensorValue> = UnordMap::from([
            ("x".into(), TensorValue::from_vec(vec![n], vec![0.1; n])),
            ("w".into(), TensorValue::from_vec(vec![n], vec![1.0; n])),
        ]);
        let eval_start = Instant::now();
        let values = chelis_ir::eval::eval_tensor_roots_with(&dag, &[root], |name| {
            inputs.get(name).cloned()
        })
        .unwrap();
        let eval_wall = eval_start.elapsed();
        assert_eq!(
            values[&root].to_f64_lossy_vec(),
            (0..n).map(|i| (n - i) as f64).collect::<Vec<_>>()
        );
        eprintln!("cumsum work n={n} logical_node_elements={slots} rooted_eval_wall={eval_wall:?}");
        let start = Instant::now();
        let source = support::codegen(&dag, "scan_gradient").unwrap().c_source;
        let elapsed = start.elapsed();
        let lines = source.lines().count();
        eprintln!(
            "cumsum candidate n={n} nodes={} lines={lines} bytes={} emit_wall={elapsed:?}",
            dag.len(),
            source.len()
        );
        sizes.push(lines);
    }
    assert!(
        sizes.iter().all(|&lines| lines <= sizes[0] + 1024),
        "scan emission scales with axis length: {sizes:?}"
    );
}
