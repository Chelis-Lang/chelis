//! chelis#258 Tier-2 rank polymorphism acceptance corpus.
//!
//! A single `def` generic over tensor *rank* via the `..r` spread, in the
//! identity position (`&tensor[..r, p] -> tensor[..r, p]`), replaces the
//! per-rank verb-name family (`relu_forward` / `_2d` / `_3d` / `_4d`).
//!
//! Soundness boundary (`spec/design/rank_polymorphism.md` §Soundness Boundary,
//! spec §4.2): against an opaque rank there are no named axes left to catch a
//! transposition/reshape, so a `..r` body may call only shape-identity
//! (elementwise) builtins — the Body-Discipline check rejects everything else.
//!
//! Positive/negative parity per CLAUDE.md: every "checks clean" test has a
//! paired "rejected with the right reason" test.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use tempfile::tempdir;

fn check_json(src: &str) -> Value {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("m.ch");
    fs::write(&path, src).expect("write file");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", "--allow-style-violations", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout).expect("check output should be json")
}

fn assert_clean(json: &Value, label: &str) {
    let errors = json["errors"]
        .as_array()
        .unwrap_or_else(|| panic!("{label}: errors should be a json array, got {json}"));
    assert!(
        errors.is_empty(),
        "{label}: expected no check errors, got {errors:?}"
    );
    let score = json["score"].as_f64().unwrap_or(0.0);
    assert!(
        (score - 1.0).abs() < 1e-9,
        "{label}: expected score 1.0, got {score} ({json})"
    );
}

fn assert_body_discipline_rejected(json: &Value, op: &str, label: &str) {
    let errors = json["errors"]
        .as_array()
        .unwrap_or_else(|| panic!("{label}: errors should be a json array, got {json}"));
    assert!(
        !errors.is_empty(),
        "{label}: expected a Body-Discipline rejection, got a clean check ({json})"
    );
    let has = errors.iter().any(|e| {
        e["message"].as_str().is_some_and(|m| {
            m.contains("rank-polymorphic def") && m.contains(op) && m.contains("shape-rewriting")
        })
    });
    assert!(
        has,
        "{label}: expected a Body-Discipline error naming `{op}`, got {errors:?}"
    );
    let score = json["score"].as_f64().unwrap_or(1.0);
    assert!(
        score < 1.0,
        "{label}: a rejected body must score < 1.0 ({json})"
    );
}

fn assert_rejected(json: &Value, label: &str) {
    let errors = json["errors"]
        .as_array()
        .unwrap_or_else(|| panic!("{label}: errors should be a json array, got {json}"));
    assert!(
        !errors.is_empty(),
        "{label}: expected rejection, got a clean check ({json})"
    );
}

/// A Body-Discipline rejection citing the rank-polymorphic def (used for the
/// non-`app` bypass routes — transforms, computed callees, user-fn calls).
fn assert_rank_rejected(json: &Value, label: &str) {
    let errors = json["errors"]
        .as_array()
        .unwrap_or_else(|| panic!("{label}: errors should be a json array, got {json}"));
    let has = errors.iter().any(|e| {
        e["message"]
            .as_str()
            .is_some_and(|m| m.contains("rank-polymorphic def"))
    });
    assert!(
        has,
        "{label}: expected a rank-polymorphic Body-Discipline rejection, got {errors:?}"
    );
    let score = json["score"].as_f64().unwrap_or(1.0);
    assert!(
        score < 1.0,
        "{label}: a rejected body must score < 1.0 ({json})"
    );
}

// ── Positives: one rank-poly def, callable across ranks ─────────────────

/// The headline: a single identity-position `..r` def checks clean and is
/// callable at rank 1 AND rank 2 from concrete-rank callers (chelis#258 — the
/// per-rank verb proliferation collapses to one def).
#[test]
fn identity_rank_poly_def_callable_at_ranks_1_and_2() {
    let json = check_json(
        "def relu_forward(x: &tensor[..r, f32]) -> tensor[..r, f32] = relu(x)\n\
         def use_rank1(x: &tensor[n, f32]) -> tensor[n, f32] = relu_forward(x)\n\
         def use_rank2(x: &tensor[a, b, f32]) -> tensor[a, b, f32] = relu_forward(x)\n",
    );
    assert_clean(&json, "identity rank-poly callable at rank 1 and rank 2");
}

/// Also callable at rank 3 and rank 4 — the full activation-family range.
#[test]
fn identity_rank_poly_def_callable_at_ranks_3_and_4() {
    let json = check_json(
        "def relu_forward(x: &tensor[..r, f32]) -> tensor[..r, f32] = relu(x)\n\
         def r3(x: &tensor[a, b, c, f32]) -> tensor[a, b, c, f32] = relu_forward(x)\n\
         def r4(x: &tensor[a, b, c, d, f32]) -> tensor[a, b, c, d, f32] = relu_forward(x)\n",
    );
    assert_clean(&json, "identity rank-poly callable at rank 3 and rank 4");
}

/// A body composing several shape-identity builtins (silu = x * sigmoid(x))
/// stays clean — composition of elementwise ops preserves the shape.
#[test]
fn composed_identity_builtins_in_rank_poly_body_clean() {
    let json =
        check_json("def my_silu(x: &tensor[..r, f32]) -> tensor[..r, f32] = mul(x, sigmoid(x))\n");
    assert_clean(&json, "composed identity builtins (mul + sigmoid)");
}

// ── Negatives: Body Discipline rejects shape-rewriting ops ──────────────

/// The §4.2 trap: `permute` shares `&tv -> tv` with `relu` but transposes.
/// Under an opaque `..r` no named axis catches it, so Body Discipline must
/// reject it at the definition site.
#[test]
fn permute_in_rank_poly_body_rejected() {
    let json =
        check_json("def evil(x: &tensor[..r, f32]) -> tensor[..r, f32] = permute(x, 1, 0)\n");
    assert_body_discipline_rejected(&json, "permute", "permute in ..r body");
}

/// A rank-reducing op (`sum`) is likewise shape-rewriting and rejected.
#[test]
fn reduce_in_rank_poly_body_rejected() {
    let json =
        check_json("def bad(x: &tensor[..r, f32]) -> tensor[..r, f32] = sum(x, cast(0, int32))\n");
    assert_body_discipline_rejected(&json, "sum", "sum in ..r body");
}

/// `reshape` (the other op that shares `&tv -> tv` with `relu`) is rejected.
#[test]
fn reshape_in_rank_poly_body_rejected() {
    let json =
        check_json("def bad(x: &tensor[..r, f32]) -> tensor[..r, f32] = reshape(x, [2, 3])\n");
    assert_body_discipline_rejected(&json, "reshape", "reshape in ..r body");
}

// ── Negatives: the non-`app` bypass routes (transforms / computed callees) ──
// A shape-rewriting op must not sneak into a `..r` body by routing through a
// transform node or a non-builtin/computed callee — the §4.2 hole a pure
// "reject shape-rewriting `app` builtins" walker would leave open.

/// `vmap` applies a *referenced* user function (which here transposes) across
/// the opaque rank — rejected outright. Without this the body checks clean and
/// the build later panics on the surviving rank var.
#[test]
fn vmap_transform_in_rank_poly_body_rejected() {
    let json = check_json(
        "def inner(x: &tensor[a, b, f32]) -> tensor[b, a, f32] = permute(x, 1, 0)\n\
         def evil(x: &tensor[..r, f32]) -> tensor[..r, f32] = x |> vmap(inner, axis=0)\n",
    );
    assert_rank_rejected(&json, "vmap transform in ..r body");
}

/// `grad(loss)(x)` routes through a transform node and a computed callee — rejected.
#[test]
fn grad_transform_in_rank_poly_body_rejected() {
    let json = check_json(
        "def loss(x: &tensor[a, f32]) -> tensor[f32] = sum(x, cast(0, int32))\n\
         def evil(x: &tensor[..r, f32]) -> tensor[..r, f32] = grad(loss)(x)\n",
    );
    assert_rank_rejected(&json, "grad transform in ..r body");
}

/// A direct call to a user-defined function (not proven rank-safe) is rejected.
#[test]
fn user_fn_call_in_rank_poly_body_rejected() {
    let json = check_json(
        "def helper(x: &tensor[a, f32]) -> tensor[a, f32] = relu(x)\n\
         def evil(x: &tensor[..r, f32]) -> tensor[..r, f32] = helper(x)\n",
    );
    assert_rank_rejected(&json, "user-fn call in ..r body");
}

// ── Negative: the Tier-2/Tier-3 parse boundary ──────────────────────────

/// `..r` adjacent to a concrete dim is Tier-3 rank arithmetic — a parse error,
/// enforcing the boundary syntactically (not discovered at unification).
#[test]
fn rank_var_adjacent_to_concrete_dim_rejected() {
    let json = check_json("def f(x: &tensor[..r, k, f32]) -> tensor[..r, k, f32] = relu(x)\n");
    assert_rejected(&json, "tensor[..r, k] adjacency");
}

/// Control: the same activation written WITHOUT `..r` (concrete rank) still
/// checks clean — the feature does not regress ordinary tensor defs.
#[test]
fn concrete_rank_activation_still_clean() {
    let json = check_json("def relu2d(x: &tensor[a, b, f32]) -> tensor[a, b, f32] = relu(x)\n");
    assert_clean(&json, "concrete-rank activation control");
}
