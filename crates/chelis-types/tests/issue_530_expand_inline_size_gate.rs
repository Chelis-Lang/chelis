//! Issue #530 regression: an inline `expand` size that is a **tuple
//! projection** (`t.0`), an inline **`match`/`if`**, or any `cast` /
//! integer arithmetic **containing** one must be rejected at check by the
//! §4.7.2 Form-3 size gate, exactly like the bare-runtime-scalar
//! (`k`) and function-call (`ident(k)`) spellings #397/#469 already
//! reject.
//!
//! Root cause (verified on HEAD by instrumented bisection): the size
//! sub-expression `t.0` infers to `Type::Error` (the projected tuple
//! resolves to `Error` in the arg-inference context), so `infer_app`'s
//! generic error-propagation gate returned `Type::Error` WITHOUT pushing
//! a diagnostic and WITHOUT ever reaching the per-builtin `"expand"`
//! Form-3 gate. The sourceless size was silently accepted: `chelis check`
//! scored 1.0 / 0 errors, `chelis build` emitted C that hardcodes the
//! inserted axis to extent 1, while `chelis eval` computes the real
//! extent — an eval-vs-C divergence (`[3i64, 2i64]` vs `[1i64, 2i64]`). A second
//! latent layer: even when the size typed cleanly, `classify_expand_size`
//! returned `Unknown` (not `Sourceless`) for an unmodeled List tag, so
//! `check_expand_signature` accepted it via the Form-2 named-dim
//! fallback.
//!
//! Fix: the Form-3 size gate now runs for EVERY 2-/3-arg `expand` on the
//! raw size AST (before the error-propagation gate can swallow it), and
//! `classify_expand_size`'s unmodeled-List catch-all is `Sourceless`
//! (fail-closed), so a sourceless inline size of ANY spelling rejects at
//! check.
//!
//! Spec: spec/04-type-system.md §4.7.2 (Form-3), §4.5.3 (named-axis).

use chelis_deep::Expr;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::{InferResult, check_ir_program};

fn surf_to_deep(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

fn messages(rep: &InferResult) -> Vec<String> {
    rep.errors.iter().map(|e| e.message.clone()).collect()
}

/// The reject-path helper: `check_ir_program` returns `Err(InferResult)`
/// on any check error. Confirm both that it errs AND that the error is the
/// Form-3 sourceless diagnostic (FAIL-CLOSED: a generic propagated error
/// with an empty message vector would be a swallowed reject in a mask).
fn assert_form3_reject_message(source: &str, label: &str) {
    let deep = surf_to_deep(source);
    let rep = check_ir_program(&deep)
        .err()
        .unwrap_or_else(|| panic!("{label}: sourceless inline `expand` size must reject at check"));
    let msgs = messages(&rep);
    assert!(
        msgs.iter().any(|m| m.contains("expand")
            && m.contains("no tensor in scope carries it")
            && m.contains("chelis#469")),
        "{label}: expected the Form-3 sourceless-size reject diagnostic citing #469, got {msgs:?}"
    );
    assert!(
        !msgs.iter().any(|m| m.contains("internal compiler error")),
        "{label}: the reject must be a clean diagnostic, never an ICE; got {msgs:?}"
    );
}

// ---------------------------------------------------------------------------
// Reject set: every inline-size spelling #530 found bypassing the gate.
// ---------------------------------------------------------------------------

/// The issue's exact reproducer: a tuple projection `t.0` as the expand
/// size has no tensor shape source and must reject.
#[test]
fn issue530_tuple_get_size_rejected() {
    assert_form3_reject_message(
        "def g[a, n](b: tensor[n, f32], t: (int64, int64)) -> tensor[a, n, f32] = expand(b, 0, t.0)\n",
        "tuple-get size",
    );
}

/// `cast(t.0, int64)`: the `cast` wrapper must not launder the sourceless
/// tuple-get into an accept (the provenance walk follows through `cast`).
#[test]
fn issue530_cast_wrapped_tuple_get_size_rejected() {
    assert_form3_reject_message(
        "def g[a, n](b: tensor[n, f32], t: (int64, int64)) -> tensor[a, n, f32] = expand(b, 0, cast(t.0, int64))\n",
        "cast(tuple-get) size",
    );
}

/// `add(t.0, cast(0, int64))`: integer arithmetic CONTAINING a sourceless
/// operand is itself sourceless (`Sourceless` is absorbing) and rejects.
#[test]
fn issue530_arith_over_tuple_get_size_rejected() {
    assert_form3_reject_message(
        "def g[a, n](b: tensor[n, f32], t: (int64, int64)) -> tensor[a, n, f32] = expand(b, 0, add(t.0, cast(0, int64)))\n",
        "add(tuple-get, ...) size",
    );
}

/// An inline `match` size produces a runtime int with no shape source and
/// must reject (the #494 re-review escape).
#[test]
fn issue530_inline_match_size_rejected() {
    assert_form3_reject_message(
        "def g[a, n](b: tensor[n, f32], k: int64) -> tensor[a, n, f32] = {\n\
         \x20 expand(b, 0, match k with {\n\
         \x20   | 0 => 1i64\n\
         \x20   | _ => 2i64\n\
         \x20 })\n\
         }\n",
        "inline match size",
    );
}

/// An inline `if` size is the same unmodeled-List class as `match` and
/// rejects.
#[test]
fn issue530_inline_if_size_rejected() {
    assert_form3_reject_message(
        "def g[a, n](b: tensor[n, f32], c: bool) -> tensor[a, n, f32] = expand(b, 0, if c then 3i64 else 4i64)\n",
        "inline if size",
    );
}

// ---------------------------------------------------------------------------
// Positive control (reject): the `ident(k)` function-call spelling that
// #397/#469 already reject must reject IDENTICALLY (same diagnostic). This
// proves the inline forms now reach the same gate, not a parallel one.
// ---------------------------------------------------------------------------

#[test]
fn issue530_ident_callee_size_rejects_identically() {
    assert_form3_reject_message(
        "def ident(x: int64) -> int64 = x\n\
         def g[a, n](b: tensor[n, f32], k: int64) -> tensor[a, n, f32] = expand(b, 0, ident(k))\n",
        "ident(k) callee size",
    );
}

#[test]
fn issue530_bare_scalar_size_rejects_identically() {
    assert_form3_reject_message(
        "def g[a, n](b: tensor[n, f32], k: int64) -> tensor[a, n, f32] = expand(b, 0, k)\n",
        "bare runtime scalar size",
    );
}

// ---------------------------------------------------------------------------
// Positive control (accept): a genuinely SOURCED size must still PASS.
// FAIL-CLOSED must not become reject-everything: the materializable forms
// stay accepted.
// ---------------------------------------------------------------------------

/// A `shape(c, 0)` read of an in-scope tensor is the canonical Form-3
/// shape source and must check clean.
#[test]
fn issue530_shape_sourced_size_still_accepted() {
    let source = "def g[a, n](b: tensor[n, f32], c: tensor[a, f32]) -> tensor[a, n, f32] = expand(b, 0, shape(c, 0))\n";
    let deep = surf_to_deep(source);
    let rep = check_ir_program(&deep);
    assert!(
        rep.is_ok(),
        "a shape(c, 0)-sourced expand size must still check clean; got {:?}",
        rep.err().map(|r| messages(&r))
    );
}

/// A bare in-scope dimension name (`a`, carried by `c: tensor[a, f32]`) is
/// a §4.7.2 Form-2 symbolic dim and must check clean.
#[test]
fn issue530_named_dim_size_still_accepted() {
    let source = "def g[a, n](b: tensor[n, f32], c: tensor[a, f32]) -> tensor[a, n, f32] = expand(b, 0, a)\n";
    let deep = surf_to_deep(source);
    let rep = check_ir_program(&deep);
    assert!(
        rep.is_ok(),
        "a Form-2 named-dim expand size must still check clean; got {:?}",
        rep.err().map(|r| messages(&r))
    );
}

/// A static literal size must still check clean.
#[test]
fn issue530_static_literal_size_still_accepted() {
    let source = "def g[n](b: tensor[n, f32]) -> tensor[3, n, f32] = expand(b, 0, 3i64)\n";
    let deep = surf_to_deep(source);
    let rep = check_ir_program(&deep);
    assert!(
        rep.is_ok(),
        "a static literal expand size must still check clean; got {:?}",
        rep.err().map(|r| messages(&r))
    );
}

/// A static `cast`-wrapped literal size (`cast(2, int64)`) must still
/// check clean — the provenance walk classifies it `Static`.
#[test]
fn issue530_cast_literal_size_still_accepted() {
    let source =
        "def g[n](b: tensor[n, f32]) -> tensor[2, n, f32] = expand(b, 0, cast(2, int64))\n";
    let deep = surf_to_deep(source);
    let rep = check_ir_program(&deep);
    assert!(
        rep.is_ok(),
        "a cast(2, int64) static expand size must still check clean; got {:?}",
        rep.err().map(|r| messages(&r))
    );
}
