use std::env;
use std::path::PathBuf;

use chelis_e2e::data::{default_mnist_dir, load_mnist};
use chelis_e2e::train::{accuracy, build_mnist_program, init_params, train_step};
use chelis_ir::grad_then_fuse;

struct Config {
    mnist_dir: PathBuf,
    epochs: usize,
    lr: f64,
    seed: u64,
    train_limit: Option<usize>,
    test_limit: Option<usize>,
    min_acc: Option<f64>,
}

fn parse_args() -> Result<Config, String> {
    let mut config = Config {
        mnist_dir: default_mnist_dir(),
        epochs: 5,
        lr: 0.01,
        seed: 42,
        train_limit: None,
        test_limit: None,
        min_acc: None,
    };

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for `{arg}`"))?;
        match arg.as_str() {
            "--mnist-dir" => config.mnist_dir = PathBuf::from(value),
            "--epochs" => {
                config.epochs = value.parse().map_err(|_| format!("bad epochs `{value}`"))?;
            }
            "--lr" => config.lr = value.parse().map_err(|_| format!("bad lr `{value}`"))?,
            "--seed" => config.seed = value.parse().map_err(|_| format!("bad seed `{value}`"))?,
            "--train-limit" => {
                config.train_limit = Some(
                    value
                        .parse()
                        .map_err(|_| format!("bad train limit `{value}`"))?,
                );
            }
            "--test-limit" => {
                config.test_limit = Some(
                    value
                        .parse()
                        .map_err(|_| format!("bad test limit `{value}`"))?,
                );
            }
            "--min-acc" => {
                config.min_acc = Some(
                    value
                        .parse()
                        .map_err(|_| format!("bad min acc `{value}`"))?,
                );
            }
            _ => return Err(format!("unknown arg `{arg}`")),
        }
    }

    Ok(config)
}

fn main() -> Result<(), String> {
    let config = parse_args()?;
    if matches!(config.train_limit, Some(0)) {
        return Err("--train-limit must be at least 1".to_string());
    }
    if matches!(config.test_limit, Some(0)) {
        return Err("--test-limit must be at least 1".to_string());
    }
    if !config.mnist_dir.exists() {
        return Err(format!(
            "MNIST data not found at {}",
            config.mnist_dir.display()
        ));
    }

    let (mut train_data, mut test_data) = load_mnist(&config.mnist_dir)?;
    if let Some(limit) = config.train_limit {
        train_data.truncate(limit);
    }
    if let Some(limit) = config.test_limit {
        test_data.truncate(limit);
    }
    if train_data.is_empty() {
        return Err("training data is empty after applying limits".to_string());
    }
    if test_data.is_empty() {
        return Err("test data is empty after applying limits".to_string());
    }

    println!(
        "Loaded {} train batches, {} test batches from {}",
        train_data.len(),
        test_data.len(),
        config.mnist_dir.display()
    );

    let program = build_mnist_program()?;
    let wrt: Vec<_> = program.param_nodes.iter().map(|(_, id)| *id).collect();
    let grad_result = grad_then_fuse(&program.dag, program.loss_node, &wrt)
        .ok_or_else(|| "failed to build gradient DAG".to_string())?;
    let mut params = init_params(config.seed);

    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0;
        for (x, y) in &train_data {
            let loss = train_step(
                &grad_result,
                program.loss_node,
                &program.param_nodes,
                &mut params,
                x,
                y,
                config.lr,
            )?;
            epoch_loss += loss;
        }

        let avg_loss = epoch_loss / train_data.len() as f64;
        let acc = accuracy(&program.dag, &params, &test_data, program.logits_node)?;
        println!("Epoch {epoch}: loss={avg_loss:.4}, test_acc={acc:.4}");
    }

    let final_acc = accuracy(&program.dag, &params, &test_data, program.logits_node)?;
    println!("Final test_acc={final_acc:.4}");

    if let Some(min_acc) = config.min_acc
        && final_acc < min_acc
    {
        return Err(format!(
            "final accuracy {final_acc:.4} below threshold {min_acc:.4}"
        ));
    }

    Ok(())
}
