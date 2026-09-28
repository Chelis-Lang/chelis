//! spec/06 §2.1 preserves scalar versus tensor cotangents at the C host boundary.
#[path = "common/mod.rs"]
mod common;
use assert_cmd::Command;

fn parity(source: &str, expected: &str) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("result.ch");
    common::write_file(&path, source);
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(eval.status.success(), "{eval:?}");
    assert_eq!(String::from_utf8(eval.stdout).unwrap().trim(), expected);
    assert_eq!(
        common::build_and_run(source, "host_results").trim(),
        expected
    );
}

#[test]
fn tensor_bodied_scalar_gradient_is_bound_and_consumed() {
    parity(
        "def loss(x: tensor[2, f32], scale: f32) -> f32 = tensor_to_scalar(sum(mul(x, insert(scalar_to_tensor(scale), 0i32, 2i64)), 0i32))\ng = grad(loss, wrt=scale)(to_tensor([2.0f32, 3.0f32]), 4.0f32)\nout = add(g, 1.0f32)\n",
        "g = 5.0\nout = 6.0",
    );
}

#[test]
fn explicit_key_gradient_result_is_consumed_and_discarded() {
    parity(
        "def loss(draw_key: key, x: tensor[2, f32], scale: f32) -> f32 = tensor_to_scalar(sum(mul(uniform_like(draw_key, x, 0.0f32, 1.0f32), insert(scalar_to_tensor(scale), 0i32, 2i64)), 0i32))\ndef run(scale: f32) -> bool = { g = grad(loss, wrt=scale)(key_from_seed(223i64), to_tensor([0.0f32, 0.0f32]), scale)\nunused = grad(loss, wrt=scale)(key_from_seed(224i64), to_tensor([0.0f32, 0.0f32]), scale)\neq(g, tensor_to_scalar(sum(uniform_like(key_from_seed(223i64), to_tensor([0.0f32, 0.0f32]), 0.0f32, 1.0f32), 0i32))) }\nout = run(2.0f32)\n",
        "out = true",
    );
}

#[test]
fn mixed_scalar_and_rank_zero_results_keep_tuple_positions() {
    parity(
        include_str!("../../../examples/grad_host_results.ch"),
        "g.0 = 3.0\ng.1 = 2.0\nout = 5.0",
    );
}

#[test]
fn retired_seed_handler_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("retired.ch");
    common::write_file(&path, "out = with seed(223i64) { 1.0f32 }\n");
    for args in [vec!["eval", "--file"], vec!["build"]] {
        let output = Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(args)
            .arg(&path)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("explicit key"),
            "{output:?}"
        );
    }
}

#[test]
fn ordinary_rank_zero_gradient_is_consumed_and_discarded() {
    parity(
        "def loss(x: tensor[f32], scale: tensor[f32]) -> f32 = tensor_to_scalar(mul(x, scale))\ndef run(scale: tensor[f32]) -> f32 = {\n  g = grad(loss, wrt=scale)(scalar_to_tensor(3.0f32), scale)\n  unused = grad(loss, wrt=scale)(scalar_to_tensor(7.0f32), scale)\n  tensor_to_scalar(g)\n}\nout = run(scalar_to_tensor(2.0f32))\n",
        "out = 3.0",
    );
}

#[test]
fn explicit_key_rank_zero_gradient_is_consumed_and_discarded() {
    parity(
        "def loss(draw_key: key, x: tensor[2, f32], scale: tensor[f32]) -> f32 = tensor_to_scalar(sum(mul(uniform_like(draw_key, x, 0.0f32, 1.0f32), insert(scale, 0i32, 2i64)), 0i32))\ndef run(scale: tensor[f32]) -> bool = {\n  g = grad(loss, wrt=scale)(key_from_seed(223i64), to_tensor([0.0f32, 0.0f32]), scale)\n  unused = grad(loss, wrt=scale)(key_from_seed(224i64), to_tensor([0.0f32, 0.0f32]), scale)\n  eq(tensor_to_scalar(g), tensor_to_scalar(sum(uniform_like(key_from_seed(223i64), to_tensor([0.0f32, 0.0f32]), 0.0f32, 1.0f32), 0i32)))\n}\nout = run(scalar_to_tensor(2.0f32))\n",
        "out = true",
    );
}
