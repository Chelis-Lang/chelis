//! Issue #369: residue of the #318 shape-derived expand-of-scalar fix.
//!
//! PR #324 (#318) recovered the broadcast extent for the INLINE-DIRECT
//! form `expand(scalar_to_tensor(c), 0, cast(shape(&x, 0), int32))` — the
//! `shape(...)` application is DIRECTLY in the `expand` size argument. The
//! canonical `tensor_full_like` helper instead writes the shape read into
//! a `let` binding first:
//!
//! ```chelis
//! def tensor_full_like[n](x: &tensor[n, f32], value: f32) -> tensor[n, f32] = {
//!   len = shape(x, cast(0, int32))
//!   scalar_t = scalar_to_tensor(value)
//!   expand(scalar_t, cast(0, int32), cast(len, int32))
//! }
//! ```
//!
//! so the `expand` size argument is `cast(var len, int32)`, NOT a direct
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
  len = shape(x, cast(0, int32))\n\
  scalar_t = scalar_to_tensor(value)\n\
  expand(scalar_t, cast(0, int32), cast(len, int32))\n\
}\n\
def loss_full_like(x: tensor[3, f32]) -> f32 = {\n\
  twos = tensor_full_like(&x, cast(2.0, f32))\n\
  sum(mul(x, twos), cast(0, int32)) |> tensor_to_scalar\n\
}\n\
def df(x: tensor[3, f32]) -> tensor[3, f32] = grad(loss_full_like)(x)\n\
out = df(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]))\n";

/// The minimal trigger: the same `let len = shape(...)` indirection
/// INLINE (no callee). Failed identically before the fix; pins that the
/// trigger is the `let` binding, not the callee boundary.
const REPRO_INLINE_LET: &str = "module Repro.GradInlineLet\n\
def loss_inline(x: tensor[3, f32]) -> f32 = {\n\
  len = shape(&x, cast(0, int32))\n\
  twos = expand(scalar_to_tensor(cast(2.0, f32)), cast(0, int32), cast(len, int32))\n\
  sum(mul(x, twos), cast(0, int32)) |> tensor_to_scalar\n\
}\n\
def df(x: tensor[3, f32]) -> tensor[3, f32] = grad(loss_inline)(x)\n\
out = df(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]))\n";

/// The #318 INLINE-DIRECT control: `shape(...)` directly in the expand
/// size (no `let`). PASSED before the fix and must keep passing — the
/// discriminator.
const REPRO_INLINE_DIRECT: &str = "module Repro.GradInlineDirect\n\
def loss_direct(x: tensor[3, f32]) -> f32 = {\n\
  twos = expand(scalar_to_tensor(cast(2.0, f32)), cast(0, int32), cast(shape(&x, cast(0, int32)), int32))\n\
  sum(mul(x, twos), cast(0, int32)) |> tensor_to_scalar\n\
}\n\
def df(x: tensor[3, f32]) -> tensor[3, f32] = grad(loss_direct)(x)\n\
out = df(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]))\n";

/// NEGATIVE PARITY: a `tensor[k]` output whose `expand` size is a bare
/// `int32` SCALAR PARAMETER (no `shape(...)` read, no in-scope tensor dim)
/// must STILL be rejected loudly — the fix recovers extents only from
/// genuine `shape(...)` reads (direct or `let`-bound), never from an
/// arbitrary runtime scalar. Guards that the size-1-default removal did
/// not weaken the §4.7.2 Form-3 sourceless-size rejection (chelis#469).
const REPRO_BARE_SCALAR_REJECTS: &str = "module Repro.BareScalarRejects\n\
def bad(x: tensor[3, f32], k: int32) -> tensor[k, f32] = {\n\
  scalar_t = scalar_to_tensor(cast(2.0, f32))\n\
  expand(scalar_t, cast(0, int32), k)\n\
}\n\
out = bad(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]), cast(4, int32))\n";

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

/// NEGATIVE PARITY: a genuinely sourceless `expand` size (a bare `int32`
/// scalar parameter) must STILL be rejected loudly. The fix must not have
/// turned the size-1 default into a silent extent guess for arbitrary
/// runtime scalars — only `shape(...)` reads (direct or `let`-bound)
/// recover an extent.
#[test]
fn issue_369_bare_scalar_expand_size_still_rejects() {
    let output = run_eval(REPRO_BARE_SCALAR_REJECTS, "bare_scalar");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "a bare-scalar `expand` size must be rejected, not silently \
         defaulted; stdout={stdout} stderr={stderr}",
    );
    let combined = format!("{stdout}{stderr}");
    assert!(
        combined.contains("expand") && combined.contains("symbolic dimension"),
        "rejection must name the sourceless symbolic `expand` size \
         (§4.7.2 / chelis#469), not some unrelated failure; got \
         stdout={stdout} stderr={stderr}",
    );
}
