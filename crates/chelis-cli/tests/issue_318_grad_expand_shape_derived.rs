//! Issue #318: `grad` has no backward rule for the SHAPE-DERIVED
//! expand-of-scalar const-broadcast. Issue #288 fixed the LITERAL-size
//! form `expand(scalar_to_tensor(c), 0, cast(2, int64))`; the canonical
//! 0.7.x scalar-broadcast helper (`tensor_full_like` / `tensor_full_1d`)
//! instead emits the SHAPE-DERIVED form
//! `expand(scalar_to_tensor(c), 0, cast(shape(&x, 0), int64))`, whose
//! broadcast extent is the runtime dimension of `x` rather than a
//! literal. That form still failed to grad.
//!
//! Headline reproducer (the `tensor_full_like` idiom over a CONCRETE
//! `tensor[2, f32]` — the concrete form is what collapses the expand
//! output dim to `Lit(1)` and exposes the bug):
//! ```chelis
//! module Repro.GradExpandShape
//! def f(x: tensor[2, f32]) -> f32 = {
//!   k = expand(scalar_to_tensor(cast(3.0, f32)),
//!              cast(0, int32),
//!              cast(shape(&x, cast(0, int32)), int64))
//!   tensor_to_scalar(sum(mul(x, k), cast(0, int32)))
//! }
//! def df(x: tensor[2, f32]) -> tensor[2, f32] = grad(f)(x)
//! ```
//!
//! Before the fix:
//!   * `chelis check` -> clean (type-checks: the size-1 source broadcasts
//!     up to `tensor[2]`).
//!   * `chelis eval` -> `Lowering error: grad(...) lowering rejected:
//!     failed to construct backward DAG (... binary op ... mismatched
//!     dimension at axis 0: Lit(2) vs Lit(1))`.
//!
//! `f(x) = sum(x * 3.0) = 3.0 * (x0 + x1)`, so the gradient is the
//! constant `df(x) = [3.0, 3.0]` for any `x`. This file is the
//! end-to-end acceptance oracle: `check` clean and `eval` runs to the
//! correct gradient for the canonical rank-0 `scalar_to_tensor`
//! const-source (what `tensor_full_like` / `tensor_full_1d` build from);
//! `expand([], 0, shape(x, 0))` INSERTS axis 0 with the recovered extent
//! → `tensor[n]`. The IR-level siblings add the lowering-extent and
//! finite-difference checks.
//!
//! (A rank-1 `to_tensor([c])` source is intentionally NOT exercised: the
//! language's `expand` INSERTS rather than replaces, so
//! `expand([1], 0, n)` is `[n, 1]`, which does not broadcast against a
//! `tensor[n]` operand — that is not a valid const-broadcast and is not
//! what #318's downstream uses. See the lowering-rank-rule note in
//! `chelis-ir/src/lower.rs`'s
//! `issue_318_expand_shape_arg_recovers_extent_without_changing_rank_rule`.)

use std::fs;
use std::path::Path;

use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

/// rank-0 source (`scalar_to_tensor` / `tensor_full_like`), concrete
/// `tensor[2, f32]`, shape-derived `expand` size `shape(&x, 0)`.
const REPRO_RANK0: &str = "module Repro.GradExpandShape0\n\
def f(x: tensor[2, f32]) -> f32 = {\n\
  k = expand(scalar_to_tensor(cast(3.0, f32)), cast(0, int32), cast(shape(&x, cast(0, int32)), int64))\n\
  tensor_to_scalar(sum(mul(x, k), cast(0, int32)))\n\
}\n\
def df(x: tensor[2, f32]) -> tensor[2, f32] = grad(f)(x)\n\
out = df(to_tensor([3.0, 4.0]))\n";

/// The literal-size form (the #288 fix), kept for negative parity.
const REPRO_LITERAL: &str = "module Repro.GradExpandLiteral\n\
def f(x: tensor[2, f32]) -> f32 = {\n\
  k = expand(scalar_to_tensor(cast(3.0, f32)), cast(0, int32), cast(2, int64))\n\
  tensor_to_scalar(sum(mul(x, k), cast(0, int32)))\n\
}\n\
def df(x: tensor[2, f32]) -> tensor[2, f32] = grad(f)(x)\n\
out = df(to_tensor([3.0, 4.0]))\n";

fn run_check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout).expect("check output should be json")
}

fn error_messages(json: &Value) -> Vec<String> {
    json["errors"]
        .as_array()
        .expect("errors should be a json array")
        .iter()
        .map(|e| e["message"].as_str().unwrap_or("").to_string())
        .collect()
}

fn run_eval(source: &str, stem: &str) -> std::process::Output {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    fs::write(&path, source).expect("write source");
    Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("run chelis eval")
}

/// Assert `chelis eval` of `source` succeeds and prints the constant
/// gradient `[3, 3]` as a `tensor[2]`.
fn assert_eval_grad_is_3_3(source: &str, stem: &str, label: &str) {
    let output = run_eval(source, stem);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "{label}: `chelis eval` must succeed; stdout={stdout} stderr={stderr}",
    );
    assert!(
        stdout.contains("shape=[2]"),
        "{label}: gradient must be tensor[2]; got stdout={stdout}",
    );
    assert!(
        stdout.contains("data=[3, 3]") || stdout.contains("data=[3.0, 3.0]"),
        "{label}: df(x) must equal [3, 3]; got stdout={stdout}",
    );
}

/// `chelis check` of the shape-derived reproducer already passes today
/// (it passed before the fix too); pin it so a fix that accidentally
/// breaks type-checking is caught.
#[test]
fn issue_318_check_is_clean() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("repro.ch");
    fs::write(&path, REPRO_RANK0).expect("write source");
    let json = run_check(&path);
    let errs = error_messages(&json);
    assert!(
        errs.is_empty(),
        "issue #318 reproducer must type-check clean; got {errs:?}",
    );
}

/// The headline failure (acceptance oracle): `chelis eval` of the
/// rank-0-source shape-derived reproducer must succeed and print the
/// correct constant gradient. Before the fix this exits non-zero with
/// the "failed to construct backward DAG" lowering error / a `[2] vs
/// [1]` shape mismatch in the evaluator.
#[test]
fn issue_318_eval_gradient_is_correct() {
    assert_eval_grad_is_3_3(REPRO_RANK0, "repro0", "issue #318 rank-0 source");
}

/// Negative parity: the LITERAL form (the #288 fix) must STILL eval to
/// the same gradient. Running it here next to the shape-derived forms
/// pins that the #318 lowering fix did not regress the literal path.
#[test]
fn issue_318_literal_form_still_works() {
    assert_eval_grad_is_3_3(REPRO_LITERAL, "literal", "issue #288 literal form");
}
