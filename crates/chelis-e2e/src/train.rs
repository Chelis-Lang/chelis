use std::collections::{HashMap, HashSet};

use chelis_ir::dag::{Dag, NodeId, RiscOp};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict, eval_tensor_with_strict};
use chelis_ir::grad::GradResult;

use crate::pipeline::compile_surf;

pub struct TrainConfig {
    pub lr: f64,
    pub epochs: usize,
    pub batch_size: usize,
}

pub struct MnistProgram {
    pub dag: Dag,
    pub loss_node: NodeId,
    pub logits_node: NodeId,
    pub param_nodes: Vec<(String, NodeId)>,
}

const MNIST_PARAM_NAMES: &[&str] = &["w1", "b1", "w2", "b2"];

/// Compile the checked-in Surf MNIST example through the real frontend pipeline.
pub fn build_mnist_program() -> Result<MnistProgram, String> {
    let src = include_str!("../../../examples/mnist.ch");
    let compiled = compile_surf(src)?;

    let loss_node = *require_root(&compiled.root_nodes, "loss")?;
    let logits_node = *require_root(&compiled.root_nodes, "logits")?;

    let reachable = reachable_nodes(&compiled.dag, loss_node);
    let param_nodes = MNIST_PARAM_NAMES
        .iter()
        .map(|name| {
            let node = find_reachable_load(&compiled.dag, &reachable, name)
                .ok_or_else(|| format!("missing parameter load `{name}` in loss subgraph"))?;
            Ok(((*name).to_string(), node))
        })
        .collect::<Result<Vec<_>, String>>()?;

    Ok(MnistProgram {
        dag: compiled.dag,
        loss_node,
        logits_node,
        param_nodes,
    })
}

fn require_root<'a>(roots: &'a HashMap<String, NodeId>, name: &str) -> Result<&'a NodeId, String> {
    roots
        .get(name)
        .ok_or_else(|| format!("compiled MNIST program is missing `{name}` root"))
}

fn reachable_nodes(dag: &Dag, root: NodeId) -> HashSet<NodeId> {
    let mut seen = HashSet::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if let Some(node) = dag.get(id) {
            stack.extend(node.inputs.iter().copied());
        }
    }
    seen
}

fn find_reachable_load(dag: &Dag, reachable: &HashSet<NodeId>, name: &str) -> Option<NodeId> {
    dag.nodes().iter().find_map(|node| {
        if reachable.contains(&node.id)
            && matches!(&node.op, RiscOp::Load { name: load } if load == name)
        {
            Some(node.id)
        } else {
            None
        }
    })
}

/// Initialize random parameters
#[must_use]
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
    _loss_node: NodeId,
    param_nodes: &[(String, NodeId)],
    params: &mut HashMap<String, TensorValue>,
    x_batch: &TensorValue,
    y_batch: &TensorValue,
    lr: f64,
) -> Result<f64, String> {
    let vals = eval_tensor_with_strict(&grad_result.dag, |name| match name {
        "x" => Some(x_batch.clone()),
        "labels" => Some(y_batch.clone()),
        _ => params.get(name).cloned(),
    })
    .map_err(|e| format!("eval error: {e}"))?;

    let loss = vals
        .get(&grad_result.output_node)
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
    if data.is_empty() {
        return Err("accuracy requires non-empty evaluation data".to_string());
    }

    let mut correct = 0;
    let mut total = 0;

    for (x_batch, y_batch) in data {
        let vals = eval_tensor_roots_with_strict(dag, &[logits_node], |name| match name {
            "x" => Some(x_batch.clone()),
            "labels" => Some(y_batch.clone()),
            _ => params.get(name).cloned(),
        })
        .map_err(|e| format!("eval error: {e}"))?;

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

    Ok(f64::from(correct) / f64::from(total))
}
