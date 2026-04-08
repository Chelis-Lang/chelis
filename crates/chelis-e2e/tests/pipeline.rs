use chelis_e2e::pipeline::compile_surf;

fn has_any_root(result: &chelis_e2e::pipeline::PipelineResult, names: &[&str]) -> bool {
    names
        .iter()
        .any(|name| result.root_nodes.contains_key(*name))
}

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
    assert!(
        !result.dag.is_empty(),
        "MNIST example should lower to a non-empty DAG"
    );
    assert!(result.root_nodes.contains_key("logits"));
    assert!(result.root_nodes.contains_key("loss"));
}

#[test]
fn pipeline_linreg_model_lowers() {
    let src = include_str!("../../../examples/linreg.ch");
    let result = compile_surf(src).expect("linear regression example should compile");
    assert!(
        !result.dag.is_empty(),
        "linear regression example should lower to a non-empty DAG"
    );
    assert!(has_any_root(&result, &["predict", "pred"]));
    assert!(result.root_nodes.contains_key("loss"));
}

#[test]
fn pipeline_transformer_model_lowers() {
    let src = include_str!("../../../examples/transformer_block.ch");
    let result = compile_surf(src).expect("transformer example should compile");
    assert!(
        !result.dag.is_empty(),
        "transformer example should lower to a non-empty DAG"
    );
    assert!(has_any_root(&result, &["forward", "out"]));
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

#[test]
fn pipeline_macro_composition_lowers() {
    let src = r#"
macro residual_relu(x) = add(copy(x), relu(x))
def f(x: tensor[n, f32]): tensor[n, f32] = residual_relu(x)
"#;
    let result = compile_surf(src).expect("macro program should compile");
    assert!(
        !result.dag.is_empty(),
        "macro pipeline DAG should not be empty"
    );
    assert!(result.deep_text.contains("source: (residual_relu"));
}
