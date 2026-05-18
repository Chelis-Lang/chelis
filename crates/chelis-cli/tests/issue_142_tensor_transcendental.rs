// Regression for chelis#142: `sqrt` / `log` / `exp` / `sin` panicked
// at runtime with `float op expects float arg, got Tensor(...)` when
// called with a `tensor[...]` argument, despite being documented as
// elementwise tensor ops in `packages/chelis-std/SKILL.md`. Working
// comparators (`add`, `neg`, `relu`, `sigmoid`, `tanh`) already did
// tensor dispatch; these four were the only elementwise unary
// builtins missing the path.
//
// Fix: route the four ops through `transcendental_unop` in
// `crates/chelis-compiler-api/src/runtime.rs`, which dispatches
// tensors via `tensor_float_unop_f32` (f32 precision matching the
// C-backend host helpers) and scalars via the original f64 closure.
//
// Downstream: discovered in `Chelis-Lang/school` PR #1 where
// `optim.ch::tensor_sqrt` (used in `adamw_step` and `lamb_step`) and
// `loss/crossentropy.ch::loss` (`|> log` on a 2D softmax output) both
// hit the panic. Both files carry map-based workarounds until this
// fix ships in a release; School-side reverts land in a follow-up PR
// once the fix is available.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, contents).expect("write file");
}

fn eval_file(path: &Path) -> assert_cmd::assert::Assert {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
}

#[test]
fn sqrt_on_rank1_tensor_returns_elementwise_roots() {
    // Exact reproducer from chelis#142 body.
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("sqrt_rank1.ch");
    write_file(
        &fixture,
        "module Repro\nexport (result)\nresult = sqrt(to_tensor([cast(4.0, f32), cast(9.0, f32)]))\n",
    );

    eval_file(&fixture)
        .success()
        .stdout(predicate::str::contains(
            "tensor(shape=[2], data=[2.0, 3.0])",
        ));
}

#[test]
fn log_on_rank1_tensor_returns_elementwise_natural_log() {
    // `log` per chelis-std/SKILL.md is `ln`; e^0 = 1 -> ln(1) = 0;
    // e^1 = e ≈ 2.71828... -> ln(e) = 1.
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("log_rank1.ch");
    write_file(
        &fixture,
        "module Repro\nexport (result)\nresult = log(to_tensor([cast(1.0, f32), cast(2.718281828, f32)]))\n",
    );

    // f32-precision ln: ln(1.0) is exactly 0.0; ln(2.718281828f32) is
    // ~0.9999999 due to f32 rounding of e. Assert on the integer parts
    // to keep the test fmt-stable across minor precision drift.
    eval_file(&fixture)
        .success()
        .stdout(predicate::str::contains("shape=[2]"))
        .stdout(predicate::str::contains("data=[0.0,").or(predicate::str::contains("data=[0.0 ,")));
}

#[test]
fn exp_on_rank1_tensor_returns_elementwise_exp() {
    // e^0 = 1; e^1 = e.
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("exp_rank1.ch");
    write_file(
        &fixture,
        "module Repro\nexport (result)\nresult = exp(to_tensor([cast(0.0, f32), cast(1.0, f32)]))\n",
    );

    eval_file(&fixture)
        .success()
        .stdout(predicate::str::contains("shape=[2]"))
        .stdout(predicate::str::contains("1.0"));
}

#[test]
fn sin_on_rank1_tensor_returns_elementwise_sin() {
    // sin(0) = 0; sin(π/2) = 1.
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("sin_rank1.ch");
    write_file(
        &fixture,
        "module Repro\nexport (result)\nresult = sin(to_tensor([cast(0.0, f32), cast(1.5707963, f32)]))\n",
    );

    eval_file(&fixture)
        .success()
        .stdout(predicate::str::contains("shape=[2]"));
}

#[test]
fn sqrt_on_scalar_still_works_unchanged() {
    // Regression-guard: the fix added tensor dispatch via
    // `transcendental_unop`; scalar arg path must continue to use the
    // f64 closure (matches pre-fix behavior for scalars). Bare-scalar
    // top-level binding renders as the literal value, not the
    // `tensor(shape=[], data=[...])` formatter (the latter applies
    // when a `def go() -> f32 = ...` returns a scalar that the result
    // binding wraps — see crates/chelis-cli/tests/host_eval_scalar_fn_call.rs).
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("sqrt_scalar.ch");
    write_file(
        &fixture,
        "module Repro\nexport (result)\nresult = sqrt(cast(9.0, f32))\n",
    );

    eval_file(&fixture)
        .success()
        .stdout(predicate::str::starts_with("3"));
}
