//! Consolidated production-stdlib type-check coverage.
//!
//! Every production stdlib `.ch` file under `packages/chelis-std/src/`
//! that ships a generalized polymorphic-precision signature is checked
//! here exactly once. Pre-trim, the same handful of files were
//! independently `chelis check`'d (and a couple `chelis build`'d)
//! across `stdlib_precision_generalization_followups.rs`, `stdlib_precision_generalization.rs`,
//! `stdlib_generalization_adversarial.rs`, and `monomorphization_build.rs`. A single
//! `chelis check` on a stdlib file re-typechecks the entire chelis-std
//! transitive import graph (~4-9s), so running it N times across N
//! files was the dominant integration-suite runtime cost.
//!
//! This file owns that coverage. The synthetic single-file type-system
//! tests stay in the WS-* files; only the production-file checks moved
//! here. Each file is one `#[test]` so `cargo nextest` parallelizes the
//! per-file checks across its global pool.
//!
//! Spec authority: spec/04-type-system.md sections 5.4, 5.7.2, 5.8.
//! `monomorphization_build.rs` retains the `chelis build` path
//! coverage for `linear.ch` / `attention.ch`; this file is the `chelis
//! check` surface.

use assert_cmd::Command;
use serde_json::Value;
use std::path::{Path, PathBuf};

fn stdlib_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../packages/chelis-std")
        .join(rel)
        .canonicalize()
        .unwrap_or_else(|e| panic!("failed to canonicalize {rel}: {e}"))
}

fn run_check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    serde_json::from_slice(&output).expect("check output should be json")
}

/// Assert a production stdlib file type-checks clean: empty `errors`
/// array and `score` of exactly 1.0.
fn assert_stdlib_clean(rel: &str) {
    let path = stdlib_path(rel);
    let json = run_check(&path);
    let errs = json["errors"]
        .as_array()
        .expect("errors should be a json array");
    assert!(
        errs.is_empty(),
        "production {rel}: expected clean check, got {errs:?}"
    );
    let score = json["score"].as_f64().unwrap_or(0.0);
    assert!(
        (score - 1.0).abs() < f64::EPSILON,
        "production {rel}: expected score 1.0, got {score}"
    );
}

#[test]
fn production_stdlib_nn_linear_typechecks() {
    assert_stdlib_clean("src/nn/linear.ch");
}

#[test]
fn production_stdlib_nn_embedding_typechecks() {
    assert_stdlib_clean("src/nn/embedding.ch");
}

#[test]
fn production_stdlib_nn_rmsnorm_typechecks() {
    assert_stdlib_clean("src/nn/rmsnorm.ch");
}

#[test]
fn production_stdlib_nn_silu_typechecks() {
    assert_stdlib_clean("src/nn/silu.ch");
}

#[test]
fn production_stdlib_nn_gelu_typechecks() {
    assert_stdlib_clean("src/nn/gelu.ch");
}

#[test]
fn production_stdlib_nn_attention_typechecks() {
    assert_stdlib_clean("src/nn/attention.ch");
}

#[test]
fn production_stdlib_nn_generate_typechecks() {
    assert_stdlib_clean("src/nn/generate.ch");
}

#[test]
fn production_stdlib_nn_conv_typechecks() {
    assert_stdlib_clean("src/nn/conv.ch");
}

#[test]
fn production_stdlib_init_random_typechecks() {
    assert_stdlib_clean("src/init/random.ch");
}

#[test]
fn production_stdlib_init_kaiming_typechecks() {
    assert_stdlib_clean("src/init/kaiming.ch");
}

#[test]
fn production_stdlib_init_xavier_typechecks() {
    assert_stdlib_clean("src/init/xavier.ch");
}

#[test]
fn production_stdlib_init_xavierext_typechecks() {
    assert_stdlib_clean("src/init/xavierext.ch");
}

#[test]
fn production_stdlib_loss_bce_typechecks() {
    assert_stdlib_clean("src/loss/bce.ch");
}

#[test]
fn production_stdlib_loss_crossentropy_typechecks() {
    assert_stdlib_clean("src/loss/crossentropy.ch");
}

#[test]
fn production_stdlib_loss_kldiv_typechecks() {
    assert_stdlib_clean("src/loss/kldiv.ch");
}

#[test]
fn production_stdlib_loss_metrics_typechecks() {
    assert_stdlib_clean("src/loss/metrics.ch");
}

#[test]
fn production_stdlib_tensor_reduce_typechecks() {
    assert_stdlib_clean("src/tensor/reduce.ch");
}

#[test]
fn production_stdlib_optim_typechecks() {
    assert_stdlib_clean("src/optim.ch");
}

#[test]
fn production_stdlib_test_typechecks() {
    assert_stdlib_clean("src/test.ch");
}
