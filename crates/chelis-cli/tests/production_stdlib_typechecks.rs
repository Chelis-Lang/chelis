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
fn production_stdlib_init_random_typechecks() {
    assert_stdlib_clean("src/init/random.ch");
}

#[test]
fn production_stdlib_init_kaiming_typechecks() {
    assert_stdlib_clean("src/init/kaiming.ch");
}

#[test]
fn production_stdlib_init_xavierext_typechecks() {
    assert_stdlib_clean("src/init/xavierext.ch");
}

// chelis#333: src/tensor/reduce.ch (Std.Tensor.Reduce.{min,prod,argmax,
// argmin}) was removed — the four bodyless sigs took a runtime int32 axis
// but the *_reduce builtins they would forward to require a compile-time
// constant axis, so the module was unimplementable as declared and never
// had a runtime function. Consumers call the `*_reduce` builtins with a
// const axis directly. No `production_stdlib_tensor_reduce_typechecks`
// remains because there is no module to type-check.

#[test]
fn production_stdlib_test_typechecks() {
    assert_stdlib_clean("src/test.ch");
}

#[test]
fn production_stdlib_process_typechecks() {
    assert_stdlib_clean("src/process.ch");
}

#[test]
fn production_stdlib_contracts_typechecks() {
    assert_stdlib_clean("src/contracts.ch");
}
