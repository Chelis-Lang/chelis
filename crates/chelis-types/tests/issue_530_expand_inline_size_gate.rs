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
        &desugar_program(&decls).expect("Surf fixture must desugar"),
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
        msgs.iter().any(|m| m.contains("insert")
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
        "def g[a, n](b: tensor[n, f32], t: (i64, i64)) -> tensor[a, n, f32] = insert(b, 0, t.0)\n",
        "tuple-get size",
    );
}

/// `cast(t.0, i64)`: the `cast` wrapper must not launder the sourceless
/// tuple-get into an accept (the provenance walk follows through `cast`).
#[test]
fn issue530_cast_wrapped_tuple_get_size_rejected() {
    assert_form3_reject_message(
        "def g[a, n](b: tensor[n, f32], t: (i64, i64)) -> tensor[a, n, f32] = insert(b, 0, cast(t.0, i64))\n",
        "cast(tuple-get) size",
    );
}

/// `add(t.0, cast(0, i64))`: integer arithmetic CONTAINING a sourceless
/// operand is itself sourceless (`Sourceless` is absorbing) and rejects.
#[test]
fn issue530_arith_over_tuple_get_size_rejected() {
    assert_form3_reject_message(
        "def g[a, n](b: tensor[n, f32], t: (i64, i64)) -> tensor[a, n, f32] = insert(b, 0, add(t.0, cast(0, i64)))\n",
        "add(tuple-get, ...) size",
    );
}

/// An inline `match` size produces a runtime int with no shape source and
/// must reject (the #494 re-review escape).
#[test]
fn issue530_inline_match_size_rejected() {
    assert_form3_reject_message(
        "def g[a, n](b: tensor[n, f32], k: i64) -> tensor[a, n, f32] = insert(b, 0, match k with {\n\
         \x20   | 0 => 1i64\n\
         \x20   | _ => 2i64\n\
         \x20 })\n",
        "inline match size",
    );
}

/// An inline `if` size is the same unmodeled-List class as `match` and
/// rejects.
#[test]
fn issue530_inline_if_size_rejected() {
    assert_form3_reject_message(
        "def g[a, n](b: tensor[n, f32], c: bool) -> tensor[a, n, f32] = insert(b, 0, if c then 3i64 else 4i64)\n",
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
        "def ident(x: i64) -> i64 = x\n\
         def g[a, n](b: tensor[n, f32], k: i64) -> tensor[a, n, f32] = insert(b, 0, ident(k))\n",
        "ident(k) callee size",
    );
}

#[test]
fn issue530_bare_scalar_size_rejects_identically() {
    assert_form3_reject_message(
        "def g[a, n](b: tensor[n, f32], k: i64) -> tensor[a, n, f32] = insert(b, 0, k)\n",
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
    let source = "def g[a, n](b: tensor[n, f32], c: tensor[a, f32]) -> tensor[a, n, f32] = insert(b, 0, shape(c, 0))\n";
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
    let source = "def g[a, n](b: tensor[n, f32], c: tensor[a, f32]) -> tensor[a, n, f32] = insert(b, 0, a)\n";
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
    let source = "def g[n](b: tensor[n, f32]) -> tensor[3, n, f32] = insert(b, 0, 3i64)\n";
    let deep = surf_to_deep(source);
    let rep = check_ir_program(&deep);
    assert!(
        rep.is_ok(),
        "a static literal expand size must still check clean; got {:?}",
        rep.err().map(|r| messages(&r))
    );
}

/// A static `cast`-wrapped literal size (`cast(2, i64)`) must still
/// check clean — the provenance walk classifies it `Static`.
#[test]
fn issue530_cast_literal_size_still_accepted() {
    let source = "def g[n](b: tensor[n, f32]) -> tensor[2, n, f32] = insert(b, 0, cast(2, i64))\n";
    let deep = surf_to_deep(source);
    let rep = check_ir_program(&deep);
    assert!(
        rep.is_ok(),
        "a cast(2, i64) static expand size must still check clean; got {:?}",
        rep.err().map(|r| messages(&r))
    );
}

// ---------------------------------------------------------------------------
// chelis#1791 half B and chelis#1923: the same gate in PIPE position.
//
// `spec/02-surf-syntax.md` section 0.1 says `x |> f(y)` MEANS `f(x, y)`, and
// `chelis_deep::pipe::fold_pipe` now states that once, over the checker's
// input. Before it, a pipe stage was typed from the callee's function type
// rather than as the application it denotes, so the operand reached every
// application-arm rule as an unresolved `Type::Var`: `check_expand_signature`
// matches the operand's type before applying its size rule and its
// `Type::Var(_) | Type::Error(_)` arm returns early, dropping the rule. One
// program was therefore rejected written directly and accepted written as a
// pipe stage, and a genuinely sourceless size reached the lowerer where
// section 4.7.2 says the checker owes the rejection.
//
// The rows below are receipts on the fold, not on a rule of their own. No
// size rule was hoisted, reordered or widened: the direct spelling's
// renderings are untouched by construction, because after the fold the two
// spellings are the same tree. That is the whole reason to state the pipe's
// meaning once rather than teach each arm about pipes.
//
// chelis#1909 moved the base under these rows and is why their evidentiary
// statuses name two commits. On `6abca2406` the pipe spellings checked clean;
// since #1909 most of them reject with the declaration-boundary obligation
// diagnostic instead, which is a rejection but not the one section 4.7.2 asks
// for, so the positions still disagreed. The four-argument anchored form is
// the one that still checked clean on `08e46ebe6`.
// ---------------------------------------------------------------------------

/// The issue's reproducer B, whose size `a_dim` is a cast over a bare `i32`
/// parameter and therefore has no tensor shape source.
const SOURCELESS_IN_PIPE_POSITION: &str = "sig f[a]: tensor[a, f32] -> i32 -> tensor[a, f32]\n\
def f(x: tensor[a, f32], k: i32) = {\n  \
a_dim = k |> cast(i64)\n  \
[0.25f32] |> to_tensor |> expand(0i32, a_dim)\n\
}\n";

/// The same program with the `expand` written directly, which is the spelling
/// that already rejected.
const SOURCELESS_IN_DIRECT_POSITION: &str = "sig f[a]: tensor[a, f32] -> i32 -> tensor[a, f32]\n\
def f(x: tensor[a, f32], k: i32) = {\n  \
a_dim = k |> cast(i64)\n  \
expand(to_tensor([0.25f32]), 0i32, a_dim)\n\
}\n";

/// The same pipe chain whose size IS shape-sourced, so nothing should reject.
const SHAPE_SOURCED_IN_PIPE_POSITION: &str = "sig f[a]: tensor[a, f32] -> tensor[a, f32]\n\
def f(x: tensor[a, f32]) = {\n  \
a_dim = cast(shape(x, cast(0, i32)), i64)\n  \
[0.25f32] |> to_tensor |> expand(0i32, a_dim)\n\
}\n";

/// chelis#1791's pipe half: the two spellings are one program, so they reject
/// with the same bytes.
///
/// Comparing the whole message list rather than a substring is the point of
/// this row. A pipe stage that rejected with some other diagnostic would still
/// be a check/build disagreement fixed by accident, and section 4.7.2 requires
/// the positions to agree, not merely both to fail. That is exactly what the
/// base does.
///
/// EVIDENTIARY STATUS: regression test on the pipe spelling, disposition lock
/// on the direct one. On `08e46ebe6` the pipe spelling rejected at score
/// 0.96 carrying the declaration-boundary obligation diagnostic while the
/// direct spelling rejected at 0.9142857142857143 carrying this one, so the
/// message lists were unequal and this assertion failed. On `6abca2406`,
/// before chelis#1909, the pipe spelling checked clean at score 1 instead.
#[test]
fn issue1791_a_sourceless_size_rejects_in_pipe_position_too() {
    let piped = check_ir_program(&surf_to_deep(SOURCELESS_IN_PIPE_POSITION))
        .expect_err("a sourceless size must reject in pipe position");
    let direct = check_ir_program(&surf_to_deep(SOURCELESS_IN_DIRECT_POSITION))
        .expect_err("the direct spelling already rejected");
    assert_eq!(
        messages(&piped),
        messages(&direct),
        "one program, one diagnostic, whichever position it is written in"
    );
    assert!(
        messages(&piped)
            .iter()
            .any(|m| m.contains("no tensor in scope carries it") && m.contains("chelis#469")),
        "and it is the section 4.7.2 sourceless-size diagnostic: {:?}",
        messages(&piped)
    );
}

/// The direct spelling's own rendering, asserted independently so the row above
/// cannot pass by making BOTH positions wrong in the same way.
///
/// EVIDENTIARY STATUS: disposition lock. Byte-identical on `08e46ebe6`. The
/// reported check SCORE does move for a program containing any pipe, because
/// the fold replaces a `pipe` node and its synthesized stage lambda with one
/// `app`; the score is a ratio over node counts and this program's fell from
/// 0.9142857142857143 to 0.9. The diagnostic text, which is what a user acts
/// on, does not move.
#[test]
fn issue1791_the_direct_spelling_keeps_its_exact_rendering() {
    let direct = check_ir_program(&surf_to_deep(SOURCELESS_IN_DIRECT_POSITION))
        .expect_err("the direct spelling rejects");
    assert_eq!(
        messages(&direct),
        vec![
            "`expand` size resolves to the symbolic dimension `a_dim`, but no tensor in scope \
             carries it. Runtime extents use exact `i64`; source the value from an in-scope \
             tensor dimension or a `shape(tensor, i32-axis)` read. A bare runtime scalar has \
             no shape identity to attach to the result yet. Tracked by Chelis-Lang/chelis#469 \
             (spec/04-type-system.md \u{00a7}4.7.2)"
                .to_string()
        ]
    );
}

/// The negative twin of the pipe row: the repair must reject on PROVENANCE,
/// not on the pipe spelling. This program is chelis#1923's reproducer, so the
/// same assertion is also that issue's checker-level receipt.
///
/// EVIDENTIARY STATUS: regression test. On `08e46ebe6` this scored
/// 0.9647058823529412 and carried the declaration-boundary obligation
/// diagnostic: a materializable size was rejected for being written as a pipe
/// stage, which is the accepting direction inverted. It checked clean on
/// `6abca2406`, before chelis#1909.
#[test]
fn issue1791_a_shape_sourced_size_in_pipe_position_still_checks_clean() {
    let rep = check_ir_program(&surf_to_deep(SHAPE_SOURCED_IN_PIPE_POSITION));
    assert!(
        rep.is_ok(),
        "a shape-sourced size is materializable in either position; got {:?}",
        rep.err().map(|r| messages(&r))
    );
}

/// The named-axis form keeps its own section 4.5.3 diagnostic, in BOTH
/// positions.
///
/// Which rule answers is decided by the form, and the fold does not touch that
/// decision: after it, the pipe spelling IS the direct spelling, so the
/// named-axis form's compile-time-literal rule reaches it for the same reason
/// it reaches the direct one. The alternative repair this replaces, hoisting
/// the section 4.7.2 provenance rule above the operand-type match, was
/// measured to replace the direct-position rendering for
/// `insert(b, m, cast(k, i64), n)` with the sourceless one: a silent change
/// to an established diagnostic that nothing asked for. Stating the pipe's
/// meaning once cannot have that effect, because it changes no rule.
///
/// EVIDENTIARY STATUS: disposition lock on the direct spelling, regression
/// test on the two pipe spellings. On `08e46ebe6` the direct four-argument
/// form carried this exact message at score 0.88, unchanged here. The
/// three-argument pipe spelling rejected at 0.9571428571428572 with the
/// declaration-boundary obligation diagnostic rather than this one, and the
/// four-argument anchored pipe spelling checked CLEAN at score 1, so a
/// sourceless named-axis size reached the lowerer from that spelling.
#[test]
fn issue1791_the_named_axis_form_keeps_its_own_literal_size_diagnostic() {
    let needle = "the named-axis insert form requires a compile-time literal size";

    // Direct position, four-argument anchored form: the lock.
    let anchored = check_ir_program(&surf_to_deep(
        "def g[n, m](b: tensor[n, f32], k: i32) -> tensor[m, n, f32] = \
         insert(b, m, cast(k, i64), n)\n",
    ))
    .expect_err("the named-axis form requires a literal size");
    assert!(
        messages(&anchored).iter().any(|m| m.contains(needle)),
        "the form's own diagnostic is unchanged in direct position: {:?}",
        messages(&anchored)
    );

    // Pipe position, three-argument named form: the regression.
    let named_in_pipe = check_ir_program(&surf_to_deep(
        "sig f[m]: i32 -> tensor[m, 1, f32]\n\
         def f(k: i32) = {\n  \
         a_dim = k |> cast(i64)\n  \
         [0.25f32] |> to_tensor |> insert(m, a_dim)\n\
         }\n",
    ))
    .expect_err("a sourceless named-axis size must reject in pipe position too");
    assert!(
        messages(&named_in_pipe).iter().any(|m| m.contains(needle)),
        "and it is the SAME diagnostic there, not the provenance one: {:?}",
        messages(&named_in_pipe)
    );

    // Pipe position, four-argument anchored form: the spelling that checked
    // clean on the base.
    let anchored_in_pipe = check_ir_program(&surf_to_deep(
        "sig f[m]: i32 -> tensor[m, 1, f32]\n\
         def f(k: i32) = {\n  \
         a_dim = k |> cast(i64)\n  \
         [0.25f32] |> to_tensor |> insert(m, a_dim, n)\n\
         }\n",
    ))
    .expect_err("the anchored form rejects in pipe position too");
    assert!(
        messages(&anchored_in_pipe)
            .iter()
            .any(|m| m.contains(needle)),
        "{:?}",
        messages(&anchored_in_pipe)
    );
}
