//! Issue #318: `grad` has no backward rule for the SHAPE-DERIVED
//! expand-of-scalar const-broadcast. Issue #288 fixed the LITERAL-size
//! form `expand(scalar_to_tensor(c), 0, cast(2, int32))`; the canonical
//! 0.7.x scalar-broadcast helper (`tensor_full_like` / `tensor_full_1d`)
//! instead emits the SHAPE-DERIVED form
//! `expand(scalar_to_tensor(c), 0, shape(&x, cast(0, int32)))`, whose
//! broadcast extent is the runtime dimension of `x` rather than a
//! literal. That form still failed to grad.
//!
//! Headline reproducer (the `tensor_full_like` idiom):
//! ```chelis
//! module Repro.GradExpandShape
//! def f(x: tensor[n, f32]) -> f32 = {
//!   k = expand(scalar_to_tensor(cast(3.0, f32)),
//!              cast(0, int32),
//!              shape(&x, cast(0, int32)))
//!   tensor_to_scalar(sum(mul(x, k), cast(0, int32)))
//! }
//! def df(x: tensor[n, f32]) -> tensor[n, f32] = grad(f)(x)
//! ```
//!
//! Before the fix:
//!   * `chelis check` -> clean (type-checks: the size-1 source broadcasts
//!     up to `tensor[n]`).
//!   * `chelis eval` / `chelis build --target c` -> `Lowering error:
//!     grad(...) lowering rejected: failed to construct backward DAG
//!     (... binary op ... mismatched dimension at axis 0: Lit(2) vs
//!     Lit(1))`.
//!
//! `f(x) = sum(x * 3.0) = 3.0 * (x0 + x1)`, so the gradient is the
//! constant `df(x) = [3.0, 3.0]` for any `x`. This file is the
//! end-to-end acceptance oracle: `check` clean and `eval` runs to the
//! correct gradient. The IR-level sibling
//! (`chelis-ir/tests/issue_318_grad_expand_shape_derived.rs`) adds the
//! finite-difference cross-check and the literal/shape-derived parity.

use std::fs;
use std::path::Path;

use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

/// The shape-derived (`tensor_full_like`) reproducer: the `expand` size
/// is `shape(&x, 0)`, the runtime extent of `x`, not a literal.
const REPRO: &str = "module Repro.GradExpandShape\n\
def f(x: tensor[n, f32]) -> f32 = {\n\
  k = expand(scalar_to_tensor(cast(3.0, f32)), cast(0, int32), shape(&x, cast(0, int32)))\n\
  tensor_to_scalar(sum(mul(x, k), cast(0, int32)))\n\
}\n\
def df(x: tensor[n, f32]) -> tensor[n, f32] = grad(f)(x)\n\
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

/// `chelis check` already passes today (it passed before the fix too);
/// pin it so a fix that accidentally breaks type-checking is caught.
#[test]
fn issue_318_check_is_clean() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("repro.ch");
    fs::write(&path, REPRO).expect("write source");
    let json = run_check(&path);
    let errs = error_messages(&json);
    assert!(
        errs.is_empty(),
        "issue #318 reproducer must type-check clean; got {errs:?}",
    );
}

/// The headline failure: `chelis eval` must succeed and print the
/// correct constant gradient. Before the fix this exits non-zero with
/// the "failed to construct backward DAG" lowering error.
#[test]
fn issue_318_eval_gradient_is_correct() {
    let output = run_eval(REPRO, "repro");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "issue #318: `chelis eval` must succeed; stdout={stdout} stderr={stderr}",
    );
    assert!(
        stdout.contains("shape=[2]"),
        "gradient must be tensor[2]; got stdout={stdout}",
    );
    assert!(
        stdout.contains("data=[3, 3]") || stdout.contains("data=[3.0, 3.0]"),
        "df(x) must equal [3, 3]; got stdout={stdout}",
    );
}

/// Negative parity: the LITERAL form (the #288 fix) must STILL eval to
/// the same gradient. Running it here next to the shape-derived form
/// pins that the #318 lowering fix did not regress the literal path.
#[test]
fn issue_318_literal_form_still_works() {
    let literal = "module Repro.GradExpandLiteral\n\
def f(x: tensor[2, f32]) -> f32 = {\n\
  k = expand(scalar_to_tensor(cast(3.0, f32)), cast(0, int32), cast(2, int32))\n\
  tensor_to_scalar(sum(mul(x, k), cast(0, int32)))\n\
}\n\
def df(x: tensor[2, f32]) -> tensor[2, f32] = grad(f)(x)\n\
out = df(to_tensor([3.0, 4.0]))\n";
    let output = run_eval(literal, "literal");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "issue #288 literal form must still succeed; stdout={stdout} stderr={stderr}",
    );
    assert!(
        stdout.contains("shape=[2]")
            && (stdout.contains("data=[3, 3]") || stdout.contains("data=[3.0, 3.0]")),
        "literal df(x) must equal [3, 3]; got stdout={stdout}",
    );
}
