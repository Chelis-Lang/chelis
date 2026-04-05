use chelis_e2e::data::*;
use chelis_e2e::train::*;
use chelis_ir::grad::grad_dag;
use std::path::PathBuf;

/// Sanity check: loss decreases on synthetic data
#[test]
fn mnist_synthetic_loss_decreases() {
    let (dag, loss_node, param_nodes) = build_mnist_dag();
    let grad_result = grad_dag(
        &dag,
        loss_node,
        &param_nodes.iter().map(|(_, id)| *id).collect::<Vec<_>>(),
    )
    .unwrap();

    let mut params = init_params(42);
    let data = synthetic_data(64, 32, 123);

    let mut losses = Vec::new();
    for _epoch in 0..3 {
        let mut epoch_loss = 0.0;
        for (x, y) in &data {
            let loss = train_step(
                &grad_result,
                loss_node,
                &param_nodes,
                &mut params,
                x,
                y,
                0.01,
            )
            .unwrap();
            epoch_loss += loss;
        }
        losses.push(epoch_loss / data.len() as f64);
    }

    // Loss should decrease
    assert!(
        losses[2] < losses[0],
        "loss did not decrease: epoch 0 = {}, epoch 2 = {}",
        losses[0],
        losses[2]
    );
}

/// The actual MNIST milestone: train on real data, >90% test accuracy
#[test]
#[ignore] // Run with: cargo test -p chelis-e2e -- --ignored mnist_real
fn mnist_real_over_90_percent() {
    let mnist_dir = std::env::var("MNIST_DIR").unwrap_or_else(|_| "data/mnist".to_string());
    let mnist_path = PathBuf::from(&mnist_dir);

    if !mnist_path.exists() {
        panic!(
            "MNIST data not found at {mnist_dir}. \
             Download from https://yann.lecun.com/exdb/mnist/ and set MNIST_DIR."
        );
    }

    let (train_data, test_data) = load_mnist(&mnist_path).unwrap();
    println!(
        "Loaded {} train batches, {} test batches",
        train_data.len(),
        test_data.len()
    );

    let (dag, loss_node, param_nodes) = build_mnist_dag();

    let wrt: Vec<_> = param_nodes.iter().map(|(_, id)| *id).collect();
    let grad_result = grad_dag(&dag, loss_node, &wrt).unwrap();

    let mut params = init_params(42);

    for epoch in 0..5 {
        let mut epoch_loss = 0.0;
        for (x, y) in &train_data {
            let loss = train_step(
                &grad_result,
                loss_node,
                &param_nodes,
                &mut params,
                x,
                y,
                0.01,
            )
            .unwrap();
            epoch_loss += loss;
        }
        let avg_loss = epoch_loss / train_data.len() as f64;

        let acc = accuracy(&dag, &params, &test_data, loss_node).unwrap_or(0.0);
        println!("Epoch {epoch}: loss={avg_loss:.4}, test_acc={acc:.4}");
    }

    let final_acc = accuracy(&dag, &params, &test_data, loss_node).unwrap_or(0.0);
    assert!(
        final_acc > 0.90,
        "MNIST test accuracy {final_acc:.4} < 0.90"
    );
}
