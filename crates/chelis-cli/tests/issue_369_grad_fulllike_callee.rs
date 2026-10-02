//! Issue #369: residue of the #318 shape-derived expand-of-scalar fix.
//!
//! PR #324 (#318) recovered the broadcast extent for the INLINE-DIRECT
//! form `expand(scalar_to_tensor(c), 0, cast(shape(&x, 0), i64))` — the
//! `shape(...)` application is DIRECTLY in the `expand` size argument. The
//! canonical `tensor_full_like` helper instead writes the shape read into
//! a `let` binding first:
//!
//! ```chelis
//! def tensor_full_like[n](x: &tensor[n, f32], value: f32) -> tensor[n, f32] = {
//!   len = shape(x, cast(0, i32))
//!   scalar_t = scalar_to_tensor(value)
//!   expand(scalar_t, cast(0, i32), cast(len, i64))
//! }
//! ```
//!
//! so the `expand` size argument is `cast(var len, i32)`, NOT a direct
//! `shape(...)` app. The pre-fix extent recognizer cannot see through the
//! `let`, so the size silently defaults to `Lit(1)`; the forward `mul(x,
//! twos)` then mixes `tensor[3]` with `tensor[1]`, the type checker
//! accepts it via size-1 broadcasting, but `grad` clones the forward into
//! the backward DAG and `verify` rejects:
//!
//! ```text
//! Lowering error: grad(...) lowering rejected: failed to construct
//! backward DAG (... binary op at node 4 has mismatched dimension at axis
//! 0: Lit(3) vs Lit(1); binary op at node 8 ... Lit(3) vs Lit(1))
//! ```
//!
//! ROOT-CAUSE REFINEMENT (verified on 0.10.0): the minimal trigger is the
//! `let len = shape(...)` indirection, not the `[n]`-callee boundary. The
//! `tensor_full_like` helper simply always writes that `let`; an inline
//! `let`-bound form fails identically without any callee, and the
//! INLINE-DIRECT form (shape inlined into the expand size) PASSES — it is
//! the discriminator and the #318 control here.
//!
//! `loss(x) = sum(2*x)`, so `df(x) = [2, 2, 2]` for any `x`. This file is
//! the end-to-end CLI acceptance oracle: `check` clean and `eval` runs to
//! the correct gradient through the full `[n]`-callee idiom. The lowering
//! and IR-level grad+eval coverage lives in `crates/chelis-ir/src/lower.rs`
//! (`issue_369_*`) and
//! `crates/chelis-ir/tests/...` — this asserts the user-facing surface.

use std::fs;
use std::path::Path;

use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

/// The headline reproducer: `tensor_full_like` behind the `[n]`-quantified
/// callee — the exact downstream `tests_blocked/grad/full_like_scalar_broadcast.ch`
/// idiom. `df(to_tensor([1,2,3])) = [2, 2, 2]`.
const REPRO_CALLEE: &str = "module Repro.GradFullLikeCallee\n\
def tensor_full_like[n](x: &tensor[n, f32], value: f32) -> tensor[n, f32] = {\n\
  len = shape(x, cast(0, i32))\n\
  scalar_t = scalar_to_tensor(value)\n\
  insert(scalar_t, cast(0, i32), cast(len, i64))\n\
}\n\
def loss_full_like(x: tensor[3, f32]) -> f32 = {\n\
  twos = tensor_full_like(&x, cast(2.0, f32))\n\
  sum(mul(x, twos), cast(0, i32)) |> tensor_to_scalar\n\
}\n\
def df(x: tensor[3, f32]) -> tensor[3, f32] = grad(loss_full_like)(x)\n\
out = df(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]))\n";

/// The minimal trigger: the same `let len = shape(...)` indirection
/// INLINE (no callee). Failed identically before the fix; pins that the
/// trigger is the `let` binding, not the callee boundary.
const REPRO_INLINE_LET: &str = "module Repro.GradInlineLet\n\
def loss_inline(x: tensor[3, f32]) -> f32 = {\n\
  len = shape(&x, cast(0, i32))\n\
  twos = insert(scalar_to_tensor(cast(2.0, f32)), cast(0, i32), cast(len, i64))\n\
  sum(mul(x, twos), cast(0, i32)) |> tensor_to_scalar\n\
}\n\
def df(x: tensor[3, f32]) -> tensor[3, f32] = grad(loss_inline)(x)\n\
out = df(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]))\n";

/// The #318 INLINE-DIRECT control: `shape(...)` directly in the expand
/// size (no `let`). PASSED before the fix and must keep passing — the
/// discriminator.
const REPRO_INLINE_DIRECT: &str = "module Repro.GradInlineDirect\n\
def loss_direct(x: tensor[3, f32]) -> f32 = {\n\
  twos = insert(scalar_to_tensor(cast(2.0, f32)), cast(0, i32), cast(shape(&x, cast(0, i32)), i64))\n\
  sum(mul(x, twos), cast(0, i32)) |> tensor_to_scalar\n\
}\n\
def df(x: tensor[3, f32]) -> tensor[3, f32] = grad(loss_direct)(x)\n\
out = df(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]))\n";

/// CROSS-DEF / callee-boundary shadow: the existing tests all keep the
/// `len = shape(...)` binding inside a SINGLE scope (the callee's body, or
/// an inline `let`). This fixture pins that the `shape_bindings` recovery
/// does NOT leak a stale entry ACROSS the def boundary. Both defs bind the
/// SAME name `len` to a `shape(...)` read, at DIFFERENT extents:
///
/// * caller `loss_cross(x: tensor[5])` binds `len = shape(&guide, 0)` where
///   `guide: tensor[3]` — caller's `len` is 3 — and uses it to build a
///   grad-neutral `tensor[3]` of zeros, so the caller's `shape_bindings`
///   genuinely carries `len -> 3`.
/// * the `tensor_full_like(&x, ...)` callee independently binds its OWN
///   `len = shape(x, 0)` where `x: tensor[5]` — callee's `len` is 5 — and
///   that inner `len` MUST win when its `expand` extent is recovered.
///
/// If the caller's `len -> 3` leaked into the callee, the callee's `expand`
/// would build a `tensor[3]` and `mul(x: tensor[5], twos: tensor[3])` would
/// fail the backward DAG with `Lit(5) vs Lit(3)` — the exact #369 symptom.
/// `loss = sum(2*x) + sum(zeros3)`, so `df(x) = [2, 2, 2, 2, 2]` (the
/// `tensor[5]` shape itself is the discriminator: a leaked caller extent
/// would have produced a tensor[3]-shaped failure, not a wrong gradient).
const REPRO_CROSS_DEF_SHADOW: &str = "module Repro.GradCrossDefShadow\n\
def tensor_full_like[n](y: &tensor[n, f32], value: f32) -> tensor[n, f32] = {\n\
  len = shape(y, cast(0, i32))\n\
  scalar_t = scalar_to_tensor(value)\n\
  insert(scalar_t, cast(0, i32), cast(len, i64))\n\
}\n\
def loss_cross(x: tensor[5, f32]) -> f32 = {\n\
  guide = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)])\n\
  len = shape(&guide, cast(0, i32))\n\
  zeros3 = insert(scalar_to_tensor(cast(0.0, f32)), cast(0, i32), cast(len, i64))\n\
  caller_contrib = sum(zeros3, cast(0, i32)) |> tensor_to_scalar\n\
  twos = tensor_full_like(&x, cast(2.0, f32))\n\
  main = sum(mul(x, twos), cast(0, i32)) |> tensor_to_scalar\n\
  add(main, caller_contrib)\n\
}\n\
def df(x: tensor[5, f32]) -> tensor[5, f32] = grad(loss_cross)(x)\n\
out = df(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32), cast(5.0, f32)]))\n";

/// NEGATIVE PARITY: a `tensor[k]` output whose `insert` size is a bare
/// `i32` SCALAR PARAMETER must STILL be rejected loudly, never silently
/// defaulted to extent 1. When this fixture was written the rejection was the
/// §4.7.2 provenance rule; chelis#469 removed that rule (an `i64` scalar
/// parameter is an admissible size), and the parameter here is `i32`, so the
/// rejection is now the size-dtype rule: section 4.7.2 accepts only an `i64`
/// size and no implicit promotion applies.
const REPRO_BARE_SCALAR_REJECTS: &str = "module Repro.BareScalarRejects\n\
def bad[k](x: tensor[3, f32], k: i32) -> tensor[k, f32] = {\n\
  scalar_t = scalar_to_tensor(cast(2.0, f32))\n\
  insert(scalar_t, cast(0, i32), k)\n\
}\n\
out = bad(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]), cast(4, i32))\n";

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
/// gradient `[2, 2, 2]` as a `tensor[3]`.
fn assert_eval_grad_is_2_2_2(source: &str, stem: &str, label: &str) {
    let output = run_eval(source, stem);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "{label}: `chelis eval` must succeed; stdout={stdout} stderr={stderr}",
    );
    assert!(
        stdout.contains("shape=[3]"),
        "{label}: gradient must be tensor[3]; got stdout={stdout}",
    );
    assert!(
        stdout.contains("data=[2, 2, 2]") || stdout.contains("data=[2.0, 2.0, 2.0]"),
        "{label}: df(x) must equal [2, 2, 2]; got stdout={stdout}",
    );
}

/// `chelis check` of the callee reproducer already passes today (the
/// forward type-checks via size-1 broadcasting); pin it so a fix that
/// accidentally breaks type-checking is caught.
#[test]
fn issue_369_check_is_clean() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("repro.ch");
    fs::write(&path, REPRO_CALLEE).expect("write source");
    let json = run_check(&path);
    let errs = error_messages(&json);
    assert!(
        errs.is_empty(),
        "issue #369 reproducer must type-check clean; got {errs:?}",
    );
}

/// The headline failure (acceptance oracle): `chelis eval` of the
/// `tensor_full_like` `[n]`-callee idiom must succeed and print the
/// correct constant gradient. Before the fix this exits non-zero with the
/// "failed to construct backward DAG (... Lit(3) vs Lit(1) ...)" lowering
/// error.
#[test]
fn issue_369_eval_callee_gradient_is_correct() {
    assert_eval_grad_is_2_2_2(REPRO_CALLEE, "callee", "issue #369 [n]-callee");
}

/// The minimal trigger inline (no callee): same `let len = shape(...)`
/// indirection. Also failed before the fix.
#[test]
fn issue_369_eval_inline_let_gradient_is_correct() {
    assert_eval_grad_is_2_2_2(REPRO_INLINE_LET, "inline_let", "issue #369 inline let");
}

/// Negative parity / discriminator: the #318 INLINE-DIRECT control must
/// STILL eval to the same gradient. Running it next to the `let`-bound
/// forms pins that the #369 fix did not regress the direct-shape path.
#[test]
fn issue_369_inline_direct_control_still_works() {
    assert_eval_grad_is_2_2_2(
        REPRO_INLINE_DIRECT,
        "inline_direct",
        "issue #318 inline-direct control",
    );
}

/// CROSS-DEF / callee-boundary shadow coverage: the caller and the
/// `tensor_full_like` callee both bind `len = shape(...)` at DIFFERENT
/// extents (caller 3, callee 5). The callee's inner `len` must win — its
/// `expand` recovers extent 5 (its own param), not the caller's 3 — so
/// `df(x)` is the `tensor[5]` gradient `[2, 2, 2, 2, 2]`. A stale
/// `shape_bindings` entry leaking across the def boundary would build a
/// `tensor[3]` inside the callee and fail the backward DAG with `Lit(5) vs
/// Lit(3)`; the `shape=[5]` + non-zero gradient assertion below is the
/// numeric oracle that the inner scope is honored. NOTE: this MUST run
/// through the full checked CLI pipeline — the unchecked unit lower path
/// has empty `program_defs`, so callee inlining does not resolve there.
#[test]
fn issue_369_eval_cross_def_callee_shadow_inner_len_wins() {
    let output = run_eval(REPRO_CROSS_DEF_SHADOW, "cross_def_shadow");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "cross-def shadow: `chelis eval` must succeed (callee's inner `len` \
         resolves to extent 5); a leaked caller `len -> 3` would fail with \
         `Lit(5) vs Lit(3)`. stdout={stdout} stderr={stderr}",
    );
    assert!(
        stdout.contains("shape=[5]"),
        "cross-def shadow: gradient must be tensor[5] (the callee's param \
         extent, not the caller's 3); got stdout={stdout}",
    );
    assert!(
        stdout.contains("data=[2, 2, 2, 2, 2]")
            || stdout.contains("data=[2.0, 2.0, 2.0, 2.0, 2.0]"),
        "cross-def shadow: df(x) must equal [2, 2, 2, 2, 2]; got stdout={stdout}",
    );
}

/// NEGATIVE PARITY: an `i32` scalar parameter as the `insert` size is still
/// rejected loudly, now by the size-dtype rule (chelis#469; see the fixture's
/// comment), and never defaulted to extent 1.
#[test]
fn issue_369_bare_scalar_expand_size_still_rejects() {
    let output = run_eval(REPRO_BARE_SCALAR_REJECTS, "bare_scalar");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "a bare-scalar `insert` size must be rejected, not silently \
         defaulted; stdout={stdout} stderr={stderr}",
    );
    let combined = format!("{stdout}{stderr}");
    assert!(
        combined.contains("insert expects an i64 size"),
        "rejection must name the `insert` size dtype (§4.7.2), not some \
         unrelated failure; got stdout={stdout} stderr={stderr}",
    );
}
