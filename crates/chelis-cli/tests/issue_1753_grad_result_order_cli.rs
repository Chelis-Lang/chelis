//! spec/06 §2.2: real CLI and native C agree with independent derivative values.
use std::{path::Path, process::Command};

fn run(command: &mut Command) -> String {
    let output = command.output().unwrap();
    assert!(output.status.success(), "{command:?}\n{output:?}");
    String::from_utf8(output.stdout).unwrap()
}

fn cli(path: &Path, args: &[&str]) -> Command {
    let mut command = Command::new(assert_cmd::cargo_bin!("chelis"));
    command.current_dir(path).args(args);
    command
}

#[test]
fn original_example_checks_and_returns_true_in_eval_and_native_c() {
    native_control(
        include_str!("../../../examples/grad_wrt_order.ch"),
        "out = true",
    );
}

#[test]
fn native_scalar_subset_order_has_exact_independent_values() {
    native_control(
        "def triple(x: f32, w: f32, z: f32) -> f32 = mul(mul(x, w), z)\nout = grad(triple, wrt=(z, x))(2.0f32, 3.0f32, 5.0f32)\n",
        "out.0 = 6.0\nout.1 = 15.0",
    );
}

#[test]
fn native_repeated_selectors_preserve_values_and_finish_cleanup() {
    // spec/06 §2.2 returns one entry per listed parameter; the checker
    // already retains repeated indices. Exercise shared native result nodes.
    native_control(
        "def pair(x: f32, w: f32) -> f32 = mul(x, w)\nout = grad(pair, wrt=(w, w, x))(2.0f32, 3.0f32)\n",
        "out.0 = 2.0\nout.1 = 2.0\nout.2 = 3.0",
    );
    native_control(
        "def pair(x: tensor[f32], w: tensor[f32]) -> tensor[f32] = mul(x, w)\nout = grad(pair, wrt=(w, w, x))(scalar_to_tensor(2.0f32), scalar_to_tensor(3.0f32))\n",
        "out.0 = 2.0\nout.1 = 2.0\nout.2 = 3.0",
    );
}

fn native_control(source: &str, expected: &str) {
    let temp = tempfile::tempdir().unwrap();
    let source_path = temp.path().join("order.ch");
    std::fs::write(&source_path, source).unwrap();
    run(&mut cli(temp.path(), &["fmt", "--check", "order.ch"]));
    let check = run(&mut cli(temp.path(), &["check", "order.ch"]));
    let check: serde_json::Value = serde_json::from_str(&check).unwrap();
    assert_eq!(check["score"], 1);
    assert_eq!(check["errors"], serde_json::json!([]));
    assert_eq!(
        run(&mut cli(temp.path(), &["eval", "--file", "order.ch"])).trim(),
        expected
    );
    run(&mut cli(
        temp.path(),
        &["build", "order.ch", "--target", "c", "--output", "native"],
    ));
    let binary = temp.path().join("native").join("order");
    assert_eq!(run(&mut Command::new(&binary)).trim(), expected);
}
