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
fn pipeline_transformer_structural_smoke_lowers() {
    let src = r#"
def forward(
  x: tensor[seq, 16, f32],
  wq: tensor[16, 8, f32],
  wk: tensor[16, 8, f32],
  wv: tensor[16, 8, f32],
  wo: tensor[8, 16, f32],
  gamma: tensor[16, f32],
  beta: tensor[16, f32]
) -> tensor[seq, 16, f32] =
  {
    q = matmul(x, wq)
    k = matmul(x, wk)
    v = matmul(x, wv)
    scores = matmul(q, permute(k, 1, 0))
    probs = softmax(scores, 1)
    attn = matmul(matmul(probs, v), wo)
    out = layer_norm(add(x, attn), gamma, beta)
    _ = drop(q)
    _ = drop(k)
    _ = drop(v)
    _ = drop(scores)
    _ = drop(probs)
    _ = drop(attn)
    out
  }
"#;
    let result = compile_surf(src).expect("transformer structural smoke should compile");
    assert!(
        !result.dag.is_empty(),
        "transformer structural smoke should lower to a non-empty DAG"
    );
    assert!(has_any_root(&result, &["forward", "out"]));
}

#[test]
#[ignore = "manual long-running transformer gate"]
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

#[test]
fn pipeline_vmap_example_lowers() {
    let src = include_str!("../../../examples/vmap_relu.ch");
    let result = compile_surf(src).expect("vmap example should compile");
    assert!(
        !result.dag.is_empty(),
        "vmap example should lower to a non-empty DAG"
    );
    assert!(result.root_nodes.contains_key("batch_process"));
}

#[test]
fn pipeline_vmap_explicit_axis_lowers() {
    let src = r#"
def process(x: tensor[features, f32]): tensor[features, f32] = relu(x)
def batch_process(xs: tensor[features, batch, f32]): tensor[features, batch, f32] = vmap(process, axis=1)(xs)
"#;
    let result = compile_surf(src).expect("axis=1 vmap example should compile");
    assert!(
        !result.dag.is_empty(),
        "axis=1 vmap example should lower to a non-empty DAG"
    );
    assert!(result.root_nodes.contains_key("batch_process"));
}

#[test]
fn pipeline_grad_with_multiple_wrt_lowers_to_tuple_roots() {
    let src = r#"
def loss(
  x: tensor[features, f32],
  w: tensor[features, f32],
  v: tensor[features, f32]
) -> tensor[f32] =
  sum(mul(x, add(w, v)), 0)

def grads(x: tensor[features, f32], w: tensor[features, f32], v: tensor[features, f32])
    -> (tensor[features, f32], tensor[features, f32]) =
  grad(loss, wrt=(w, v))(x, w, v)
"#;
    let result = compile_surf(src).expect("multi-wrt grad program should compile");
    assert!(result.root_nodes.contains_key("loss"));
    assert!(result.root_nodes.contains_key("grads.0"));
    assert!(result.root_nodes.contains_key("grads.1"));
}

#[test]
fn pipeline_vmap_grad_with_multiple_wrt_lowers_to_tuple_roots() {
    let src = r#"
def loss(x: tensor[4, f32], w: tensor[4, f32], v: tensor[4, f32]) -> tensor[f32] =
  sum(mul(x, add(w, v)), 0)

def per_example_grads(
  xs: tensor[3, 4, f32],
  ws: tensor[3, 4, f32],
  vs: tensor[3, 4, f32]
) -> (tensor[3, 4, f32], tensor[3, 4, f32]) =
  vmap(grad(loss, wrt=(w, v)))(xs, ws, vs)
"#;
    let result = compile_surf(src).expect("vmapped multi-wrt grad program should compile");
    assert!(result.root_nodes.contains_key("loss"));
    assert!(result.root_nodes.contains_key("per_example_grads.0"));
    assert!(result.root_nodes.contains_key("per_example_grads.1"));
}
