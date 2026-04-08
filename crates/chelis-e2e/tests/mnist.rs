use chelis_e2e::data::*;
use chelis_e2e::train::*;
use chelis_ir::grad_then_fuse;
use chelis_ir::verify;

#[test]
fn mnist_surf_pipeline_builds_trainable_program() {
    let program = build_mnist_program().expect("MNIST Surf example should compile");
    assert!(
        !program.dag.is_empty(),
        "compiled MNIST program should not be empty"
    );
    assert_eq!(
        program.param_nodes.len(),
        4,
        "expected 4 trainable parameters"
    );
    let verify_errors = verify::verify(&program.dag);
    assert!(
        verify_errors.is_empty(),
        "compiled MNIST program lowered an invalid DAG: {verify_errors:?}"
    );
}

/// Sanity check: the real Surf -> Deep -> typecheck -> lower -> grad -> eval path learns.
#[test]
fn mnist_synthetic_loss_decreases() {
    let program = build_mnist_program().expect("MNIST Surf example should compile");
    let grad_result = grad_then_fuse(
        &program.dag,
        program.loss_node,
        &program
            .param_nodes
            .iter()
            .map(|(_, id)| *id)
            .collect::<Vec<_>>(),
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
                program.loss_node,
                &program.param_nodes,
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

    assert!(
        losses[2] < losses[0],
        "loss did not decrease: epoch 0 = {}, epoch 2 = {}",
        losses[0],
        losses[2]
    );
}

/// Real-data smoke test on a small subset. Run manually when validating the phase.
#[test]
#[ignore]
fn mnist_subset_full_pipeline() {
    let mnist_path = default_mnist_dir();

    if !mnist_path.exists() {
        panic!(
            "MNIST data not found at {}. Set MNIST_DIR or populate data/mnist.",
            mnist_path.display()
        );
    }

    let (train_data, test_data) = load_mnist(&mnist_path).unwrap();
    let train_subset: Vec<_> = train_data.into_iter().take(31).collect();
    let test_subset: Vec<_> = test_data.into_iter().take(6).collect();

    let program = build_mnist_program().expect("MNIST Surf example should compile");
    let wrt: Vec<_> = program.param_nodes.iter().map(|(_, id)| *id).collect();
    let grad_result = grad_then_fuse(&program.dag, program.loss_node, &wrt).unwrap();
    let mut params = init_params(42);

    for epoch in 0..10 {
        let mut epoch_loss = 0.0;
        for (x, y) in &train_subset {
            let loss = train_step(
                &grad_result,
                program.loss_node,
                &program.param_nodes,
                &mut params,
                x,
                y,
                0.01,
            )
            .unwrap();
            epoch_loss += loss;
        }
        let avg_loss = epoch_loss / train_subset.len() as f64;
        let acc = accuracy(&program.dag, &params, &test_subset, program.logits_node).unwrap();
        println!("Epoch {epoch}: loss={avg_loss:.4}, test_acc={acc:.4}");
    }

    let final_acc = accuracy(&program.dag, &params, &test_subset, program.logits_node).unwrap();
    assert!(
        final_acc > 0.50,
        "MNIST subset accuracy {final_acc:.4} < 0.50"
    );
}

/// Full-data milestone check. Run manually in a long release validation.
#[test]
#[ignore]
fn mnist_real_over_90_percent() {
    let mnist_path = default_mnist_dir();

    if !mnist_path.exists() {
        panic!(
            "MNIST data not found at {}. Set MNIST_DIR or populate data/mnist.",
            mnist_path.display()
        );
    }

    let (train_data, test_data) = load_mnist(&mnist_path).unwrap();
    println!(
        "Loaded {} train batches, {} test batches",
        train_data.len(),
        test_data.len()
    );

    let program = build_mnist_program().expect("MNIST Surf example should compile");
    let wrt: Vec<_> = program.param_nodes.iter().map(|(_, id)| *id).collect();
    let grad_result = grad_then_fuse(&program.dag, program.loss_node, &wrt).unwrap();

    let mut params = init_params(42);

    for epoch in 0..5 {
        let mut epoch_loss = 0.0;
        for (x, y) in &train_data {
            let loss = train_step(
                &grad_result,
                program.loss_node,
                &program.param_nodes,
                &mut params,
                x,
                y,
                0.01,
            )
            .unwrap();
            epoch_loss += loss;
        }
        let avg_loss = epoch_loss / train_data.len() as f64;
        let acc = accuracy(&program.dag, &params, &test_data, program.logits_node).unwrap();
        println!("Epoch {epoch}: loss={avg_loss:.4}, test_acc={acc:.4}");
    }

    let final_acc = accuracy(&program.dag, &params, &test_data, program.logits_node).unwrap();
    assert!(
        final_acc > 0.90,
        "MNIST test accuracy {final_acc:.4} < 0.90"
    );
}
