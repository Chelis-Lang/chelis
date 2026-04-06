use chelis_e2e::pipeline::compile_surf;

#[test]
fn pipeline_smoke_test_relu() {
    let src = "def f(x: tensor[n, f32]): tensor[n, f32] = relu(x)";
    let result = compile_surf(src);
    assert!(result.is_ok(), "pipeline failed: {:?}", result.err());
    let dag = result.unwrap().dag;
    assert!(!dag.is_empty(), "DAG is empty");
}

#[test]
fn pipeline_mnist_model_lowers() {
    let src = include_str!("../../../examples/mnist.ch");
    let result = compile_surf(src).expect("full MNIST example should compile");
    let dag = result.dag;
    assert!(
        !dag.is_empty(),
        "MNIST example should lower to a non-empty DAG"
    );
    assert!(result.root_nodes.contains_key("logits"));
    assert!(result.root_nodes.contains_key("loss"));
}

#[test]
fn pipeline_tier2_relu_decomposes() {
    // Verify relu desugars through the pipeline and lowers to MaxElem + Const
    let src = "def f(x: tensor[n, f32]): tensor[n, f32] = relu(x)";
    let result = compile_surf(src).unwrap();
    let dag = result.dag;

    // Should contain MaxElem (from relu decomposition) and Const(0)
    let has_max_elem = dag
        .nodes()
        .iter()
        .any(|n| matches!(n.op, chelis_ir::dag::RiscOp::MaxElem));
    assert!(
        has_max_elem,
        "relu should decompose to MaxElem but DAG has no MaxElem node"
    );
}
