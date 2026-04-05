use std::collections::HashMap;

use chelis_ir::dag::*;
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_ir::grad::GradResult;
use chelis_types::types::Prim;

pub struct TrainConfig {
    pub lr: f64,
    pub epochs: usize,
    pub batch_size: usize,
}

/// Build a simple forward DAG programmatically for an MLP.
/// x[batch,784] -> matmul(w1[784,128]) + b1[128] -> relu -> matmul(w2[128,10]) + b2[10]
/// Then: softmax -> log -> mul(labels) -> neg -> sum (cross-entropy loss)
pub fn build_mnist_dag() -> (Dag, NodeId, Vec<(String, NodeId)>) {
    let mut dag = Dag::new();
    let f32_ty = |dims: Vec<DimInfo>| TensorType {
        dims,
        precision: Prim::F32,
    };
    let named = |n: &str, s: usize| DimInfo::Named(n.to_string(), Some(s));
    let lit = |s: usize| DimInfo::Lit(s);

    // Inputs
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        f32_ty(vec![named("batch", 32), lit(784)]),
    );
    let labels = dag.add_node(
        RiscOp::Load {
            name: "labels".into(),
        },
        vec![],
        f32_ty(vec![named("batch", 32), lit(10)]),
    );

    // Parameters
    let w1 = dag.add_node(
        RiscOp::Load { name: "w1".into() },
        vec![],
        f32_ty(vec![lit(784), lit(128)]),
    );
    let b1 = dag.add_node(
        RiscOp::Load { name: "b1".into() },
        vec![],
        f32_ty(vec![lit(128)]),
    );
    let w2 = dag.add_node(
        RiscOp::Load { name: "w2".into() },
        vec![],
        f32_ty(vec![lit(128), lit(10)]),
    );
    let b2 = dag.add_node(
        RiscOp::Load { name: "b2".into() },
        vec![],
        f32_ty(vec![lit(10)]),
    );

    // Layer 1: matmul(x, w1) + b1
    let x_ty = dag.get(x).unwrap().output_type.clone();
    let w1_ty = dag.get(w1).unwrap().output_type.clone();
    let mm1 = chelis_ir::tier2::lower_matmul(&mut dag, x, w1, &x_ty, &w1_ty);

    let mm1_ty = f32_ty(vec![named("batch", 32), lit(128)]);
    let b1_exp = dag.add_node(
        RiscOp::Expand { axis: 0, size: 32 },
        vec![b1],
        mm1_ty.clone(),
    );
    let h1 = dag.add_node(RiscOp::Add, vec![mm1, b1_exp], mm1_ty.clone());

    // ReLU
    let relu_out = chelis_ir::tier2::lower_relu(&mut dag, h1, &mm1_ty);

    // Layer 2: matmul(relu_out, w2) + b2
    let relu_ty = dag.get(relu_out).unwrap().output_type.clone();
    let w2_ty = dag.get(w2).unwrap().output_type.clone();
    let mm2 = chelis_ir::tier2::lower_matmul(&mut dag, relu_out, w2, &relu_ty, &w2_ty);

    let mm2_ty = f32_ty(vec![named("batch", 32), lit(10)]);
    let b2_exp = dag.add_node(
        RiscOp::Expand { axis: 0, size: 32 },
        vec![b2],
        mm2_ty.clone(),
    );
    let logits = dag.add_node(RiscOp::Add, vec![mm2, b2_exp], mm2_ty.clone());

    // Softmax cross-entropy loss
    let softmax_out = chelis_ir::tier2::lower_softmax(&mut dag, logits, 1, &mm2_ty);

    // log(softmax)
    let log_probs = dag.add_node(RiscOp::Log, vec![softmax_out], mm2_ty.clone());

    // mul(log_probs, labels) -- select correct class
    let selected = dag.add_node(RiscOp::Mul, vec![log_probs, labels], mm2_ty.clone());

    // neg
    let neg_selected = dag.add_node(RiscOp::Neg, vec![selected], mm2_ty.clone());

    // sum over classes (axis=1)
    let per_sample_ty = f32_ty(vec![named("batch", 32)]);
    let per_sample = dag.add_node(
        RiscOp::Sum { axis: 1 },
        vec![neg_selected],
        per_sample_ty.clone(),
    );

    // mean over batch (sum / batch_size)
    let scalar_ty = f32_ty(vec![]);
    let batch_sum = dag.add_node(RiscOp::Sum { axis: 0 }, vec![per_sample], scalar_ty.clone());
    let batch_size_const = dag.add_node(RiscOp::Const { value: 32.0 }, vec![], scalar_ty.clone());
    let loss = chelis_ir::tier2::lower_div(&mut dag, batch_sum, batch_size_const, &scalar_ty);

    let params = vec![
        ("w1".to_string(), w1),
        ("b1".to_string(), b1),
        ("w2".to_string(), w2),
        ("b2".to_string(), b2),
    ];

    (dag, loss, params)
}

/// Initialize random parameters
pub fn init_params(rng_seed: u64) -> HashMap<String, TensorValue> {
    // Simple LCG for reproducibility
    let mut state = rng_seed;
    let mut next_f64 = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64) / (1u64 << 31) as f64 - 0.5
    };

    let mut params = HashMap::new();

    // Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
    let w1_scale = (2.0 / (784.0 + 128.0_f64)).sqrt();
    let w1_data: Vec<f64> = (0..784 * 128).map(|_| next_f64() * w1_scale).collect();
    params.insert(
        "w1".to_string(),
        TensorValue::from_vec(vec![784, 128], w1_data),
    );

    let b1_data: Vec<f64> = vec![0.0; 128];
    params.insert("b1".to_string(), TensorValue::from_vec(vec![128], b1_data));

    let w2_scale = (2.0 / (128.0 + 10.0_f64)).sqrt();
    let w2_data: Vec<f64> = (0..128 * 10).map(|_| next_f64() * w2_scale).collect();
    params.insert(
        "w2".to_string(),
        TensorValue::from_vec(vec![128, 10], w2_data),
    );

    let b2_data: Vec<f64> = vec![0.0; 10];
    params.insert("b2".to_string(), TensorValue::from_vec(vec![10], b2_data));

    params
}

/// Run one training step: forward + backward + SGD update
pub fn train_step(
    grad_result: &GradResult,
    loss_node: NodeId,
    param_nodes: &[(String, NodeId)],
    params: &mut HashMap<String, TensorValue>,
    x_batch: &TensorValue,
    y_batch: &TensorValue,
    lr: f64,
) -> Result<f64, String> {
    let mut inputs = params.clone();
    inputs.insert("x".to_string(), x_batch.clone());
    inputs.insert("labels".to_string(), y_batch.clone());

    let vals = eval_tensor(&grad_result.dag, &inputs).map_err(|e| format!("eval error: {e}"))?;

    let loss = vals
        .get(&loss_node)
        .ok_or("loss node not in eval results")?
        .data[0];

    // SGD update
    for (name, fwd_node_id) in param_nodes {
        if let Some(&grad_node_id) = grad_result.grad_nodes.get(fwd_node_id)
            && let Some(grad_val) = vals.get(&grad_node_id)
        {
            let param = params
                .get_mut(name)
                .ok_or(format!("param {name} not found"))?;
            for i in 0..param.data.len().min(grad_val.data.len()) {
                param.data[i] -= lr * grad_val.data[i];
            }
        }
    }

    Ok(loss)
}

/// Compute accuracy: fraction of samples where argmax(pred) == argmax(label)
pub fn accuracy(
    dag: &Dag,
    params: &HashMap<String, TensorValue>,
    data: &[(TensorValue, TensorValue)],
    logits_node: NodeId,
) -> Result<f64, String> {
    let mut correct = 0;
    let mut total = 0;

    for (x_batch, y_batch) in data {
        let mut inputs = params.clone();
        inputs.insert("x".to_string(), x_batch.clone());
        inputs.insert("labels".to_string(), y_batch.clone());

        let vals = eval_tensor(dag, &inputs).map_err(|e| format!("eval error: {e}"))?;

        let logits = vals.get(&logits_node).ok_or("logits node not found")?;

        let batch_size = y_batch.shape[0];
        let n_classes = y_batch.shape[1];

        for b in 0..batch_size {
            let pred_class = (0..n_classes)
                .max_by(|&i, &j| {
                    logits.data[b * n_classes + i]
                        .partial_cmp(&logits.data[b * n_classes + j])
                        .unwrap()
                })
                .unwrap();
            let true_class = (0..n_classes)
                .max_by(|&i, &j| {
                    y_batch.data[b * n_classes + i]
                        .partial_cmp(&y_batch.data[b * n_classes + j])
                        .unwrap()
                })
                .unwrap();
            if pred_class == true_class {
                correct += 1;
            }
            total += 1;
        }
    }

    Ok(correct as f64 / total as f64)
}
