//! Issue #530 regression: an inline `expand` size that is a **tuple
//! projection** (`t.0`), an inline **`match`/`if`**, or any `cast` /
//! integer arithmetic **containing** one must reach the `expand` size rule
//! with its real type rather than being swallowed by error propagation.
//!
//! Root cause (verified by instrumented bisection when #530 was fixed): the
//! size sub-expression `t.0` inferred to `Type::Error`, so `infer_app`'s
//! generic error-propagation gate returned `Type::Error` WITHOUT pushing a
//! diagnostic and WITHOUT ever reaching the per-builtin `"expand"` rule.
//! `chelis check` scored 1.0, `chelis build` emitted C that hardcoded the
//! inserted axis to extent 1, and `chelis eval` computed the real extent: an
//! eval-vs-C divergence (`[3i64, 2i64]` vs `[1i64, 2i64]`).
//!
//! #530's fix routed the size slot past the error-propagation gate. At the
//! time, the rule it reached rejected every size with no tensor shape source.
//! chelis#469 removed that rule: `spec/04-type-system.md` section 4.7.2 admits
//! any `i64` size and forbids rejecting an extent because of its provenance.
//! These spellings therefore now check clean, and their executed agreement on
//! eval and compiled C is `crates/chelis-cli/tests/issue_469_runtime_scalar_extent.rs`
//! (`a_tuple_projection_size_executes_on_both_lanes` among them). The silent
//! hardcoded extent #530 found is not reachable: lowering gives every such size
//! a runtime extent node.
//!
//! Spec: spec/04-type-system.md §4.7.2, §4.5.3 (named-axis).

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

/// The accept-path helper: an `i64` size of any provenance checks clean
/// (section 4.7.2, chelis#469). The declared result names the inserted extent
/// `a`, so the size reaches the claim rather than a defaulted slot.
fn assert_runtime_size_checks_clean(source: &str, label: &str) {
    let rep = check_ir_program(&surf_to_deep(source));
    assert!(
        rep.is_ok(),
        "{label}: an inline i64 `insert` size is admissible under section 4.7.2; got {:?}",
        rep.err().map(|r| messages(&r))
    );
}

/// A size that is not `i64` is still a type error, carried by the size rule
/// itself: the error-propagation gate must not swallow it into an accept.
fn assert_size_dtype_rejected(source: &str, label: &str) {
    let rep = check_ir_program(&surf_to_deep(source))
        .err()
        .unwrap_or_else(|| panic!("{label}: a non-i64 size must reject at check"));
    assert!(
        rep.errors.iter().any(|error| {
            error.kind.diagnostic_name() == "TypeMismatch"
                && error.expected.as_deref() == Some("i64")
                && error.got.as_deref() == Some("i32")
                && error.span_offset == source.find("insert(")
        }),
        "{label}: expected the rejected size's dtype and own call site, got {:?}",
        rep.errors
    );
}

// ---------------------------------------------------------------------------
// Every inline-size spelling #530 found bypassing the size rule. Each now
// reaches it and is admitted.
// ---------------------------------------------------------------------------

/// The issue's exact reproducer: a tuple projection `t.0` as the expand
/// size reaches the size rule and is admitted.
#[test]
fn issue530_tuple_get_size_checks_clean() {
    assert_runtime_size_checks_clean(
        "def g[a, n](b: tensor[n, f32], t: (i64, i64)) -> tensor[a, n, f32] = insert(b, 0, t.0)\n",
        "tuple-get size",
    );
}

/// `cast(t.0, i64)`: the `cast` wrapper over the projection is admitted too.
#[test]
fn issue530_cast_wrapped_tuple_get_size_checks_clean() {
    assert_runtime_size_checks_clean(
        "def g[a, n](b: tensor[n, f32], t: (i64, i64)) -> tensor[a, n, f32] = insert(b, 0, cast(t.0, i64))\n",
        "cast(tuple-get) size",
    );
}

/// `add(t.0, cast(0, i64))`: checked integer arithmetic over the projection
/// is admitted.
#[test]
fn issue530_arith_over_tuple_get_size_checks_clean() {
    assert_runtime_size_checks_clean(
        "def g[a, n](b: tensor[n, f32], t: (i64, i64)) -> tensor[a, n, f32] = insert(b, 0, add(t.0, cast(0, i64)))\n",
        "add(tuple-get, ...) size",
    );
}

/// An inline `match` size (the #494 re-review escape) is admitted.
#[test]
fn issue530_inline_match_size_checks_clean() {
    assert_runtime_size_checks_clean(
        "def g[a, n](b: tensor[n, f32], k: i64) -> tensor[a, n, f32] = insert(b, 0, match k with {\n\
         \x20   | 0 => 1i64\n\
         \x20   | _ => 2i64\n\
         \x20 })\n",
        "inline match size",
    );
}

/// An inline `if` size is admitted like the `match` form.
#[test]
fn issue530_inline_if_size_checks_clean() {
    assert_runtime_size_checks_clean(
        "def g[a, n](b: tensor[n, f32], c: bool) -> tensor[a, n, f32] = insert(b, 0, if c then 3i64 else 4i64)\n",
        "inline if size",
    );
}

// ---------------------------------------------------------------------------
// The function-call and bare-parameter spellings #397 used to reject are
// admitted by the same rule as the inline forms above.
// ---------------------------------------------------------------------------

#[test]
fn issue530_ident_callee_size_checks_clean() {
    assert_runtime_size_checks_clean(
        "def ident(x: i64) -> i64 = x\n\
         def g[a, n](b: tensor[n, f32], k: i64) -> tensor[a, n, f32] = insert(b, 0, ident(k))\n",
        "ident(k) callee size",
    );
}

#[test]
fn issue530_bare_scalar_size_checks_clean() {
    assert_runtime_size_checks_clean(
        "def g[a, n](b: tensor[n, f32], k: i64) -> tensor[a, n, f32] = insert(b, 0, k)\n",
        "bare runtime scalar size",
    );
}

// ---------------------------------------------------------------------------
// Negative parity: the size rule the inline forms now reach still rejects a
// size whose type is not `i64`, in the same inline spellings.
// ---------------------------------------------------------------------------

#[test]
fn issue530_inline_i32_tuple_get_size_rejected() {
    assert_size_dtype_rejected(
        "def g[a, n](b: tensor[n, f32], t: (i32, i32)) -> tensor[a, n, f32] = insert(b, 0, t.0)\n",
        "i32 tuple-get size",
    );
}

#[test]
fn issue530_inline_i32_if_size_rejected() {
    assert_size_dtype_rejected(
        "def g[a, n](b: tensor[n, f32], c: bool) -> tensor[a, n, f32] = insert(b, 0, if c then 3i32 else 4i32)\n",
        "i32 inline if size",
    );
}

#[test]
fn issue530_bare_i32_scalar_size_rejected() {
    assert_size_dtype_rejected(
        "def g[a, n](b: tensor[n, f32], k: i32) -> tensor[a, n, f32] = insert(b, 0, k)\n",
        "bare i32 scalar size",
    );
}

// ---------------------------------------------------------------------------
// The shape-sourced, named-dimension and static forms stay accepted.
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
/// check clean: it folds to a literal extent.
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
// chelis#1791 half B and chelis#1923: the same rule in PIPE position.
//
// `spec/02-surf-syntax.md` section 0.1 says `x |> f(y)` MEANS `f(x, y)`, and
// `chelis_deep::pipe::fold_pipe` states that once, over the checker's input.
// Before it, a pipe stage was typed from the callee's function type rather
// than as the application it denotes, so the operand reached every
// application-arm rule as an unresolved `Type::Var`: `check_expand_signature`
// matches the operand's type before applying its size rule and its
// `Type::Var(_) | Type::Error(_)` arm returns early, dropping the rule. One
// program was therefore rejected written directly and accepted written as a
// pipe stage.
//
// The rows below are receipts on the fold, not on a rule of their own: after
// the fold the two spellings are the same tree, so they get the same verdict.
// When these rows were written the verdict for a size with no tensor shape
// source was a rejection. chelis#469 removed that rule (section 4.7.2 forbids
// rejecting an extent for its provenance), so the shared verdict is now
// acceptance, and the size-dtype rule is the rejection both positions must
// still agree on. The executed eval/C agreement of the pipe spelling is
// `issue_469_runtime_scalar_extent::a_pipe_position_runtime_size_executes_on_both_lanes`,
// the receipt of the `expand.runtime_size.pipe_position` rows in
// `scripts/runtime_extent_oracle.py`.
// ---------------------------------------------------------------------------

/// The issue's reproducer B, whose size `a_dim` is a cast over a bare `i32`
/// parameter.
const RUNTIME_SIZE_IN_PIPE_POSITION: &str = "sig f[a]: tensor[a, f32] -> i32 -> tensor[a, f32]\n\
def f(x: tensor[a, f32], k: i32) = {\n  \
a_dim = k |> cast(i64)\n  \
[0.25f32] |> to_tensor |> expand(0i32, a_dim)\n\
}\n";

/// The same program with the `expand` written directly.
const RUNTIME_SIZE_IN_DIRECT_POSITION: &str = "sig f[a]: tensor[a, f32] -> i32 -> tensor[a, f32]\n\
def f(x: tensor[a, f32], k: i32) = {\n  \
a_dim = k |> cast(i64)\n  \
expand(to_tensor([0.25f32]), 0i32, a_dim)\n\
}\n";

/// The pipe spelling with an `i32` size, which the size rule rejects.
const I32_SIZE_IN_PIPE_POSITION: &str = "sig f[a]: tensor[a, f32] -> i32 -> tensor[a, f32]\n\
def f(x: tensor[a, f32], k: i32) = [0.25f32] |> to_tensor |> expand(0i32, k)\n";

/// The direct spelling with an `i32` size.
const I32_SIZE_IN_DIRECT_POSITION: &str = "sig f[a]: tensor[a, f32] -> i32 -> tensor[a, f32]\n\
def f(x: tensor[a, f32], k: i32) = expand(to_tensor([0.25f32]), 0i32, k)\n";

/// The same pipe chain whose size IS shape-sourced, so nothing should reject.
const SHAPE_SOURCED_IN_PIPE_POSITION: &str = "sig f[a]: tensor[a, f32] -> tensor[a, f32]\n\
def f(x: tensor[a, f32]) = {\n  \
a_dim = cast(shape(x, cast(0, i32)), i64)\n  \
[0.25f32] |> to_tensor |> expand(0i32, a_dim)\n\
}\n";

/// chelis#1791's pipe half: the two spellings are one program, so they get one
/// verdict, which under section 4.7.2 is acceptance.
///
/// EVIDENTIARY STATUS: regression test. Before chelis#469 both positions
/// rejected with the provenance diagnostic; on `08e46ebe6` the pipe spelling
/// rejected with the declaration-boundary obligation diagnostic instead, and
/// on `6abca2406` it checked clean while the direct spelling rejected.
#[test]
fn issue1791_a_runtime_size_checks_clean_in_pipe_position_too() {
    for (position, source) in [
        ("pipe", RUNTIME_SIZE_IN_PIPE_POSITION),
        ("direct", RUNTIME_SIZE_IN_DIRECT_POSITION),
    ] {
        let rep = check_ir_program(&surf_to_deep(source));
        assert!(
            rep.is_ok(),
            "{position}: a runtime i64 size is admissible in either position; got {:?}",
            rep.err().map(|r| messages(&r))
        );
    }
}

/// The rejecting twin: an `i32` size is refused with the same bytes in both
/// positions, so the row above cannot pass by the pipe spelling skipping the
/// size rule.
///
/// EVIDENTIARY STATUS: disposition lock.
#[test]
fn issue1791_a_non_i64_size_rejects_identically_in_both_positions() {
    let piped = check_ir_program(&surf_to_deep(I32_SIZE_IN_PIPE_POSITION))
        .expect_err("an i32 size rejects in pipe position");
    let direct = check_ir_program(&surf_to_deep(I32_SIZE_IN_DIRECT_POSITION))
        .expect_err("an i32 size rejects written directly");
    assert_eq!(
        messages(&piped),
        messages(&direct),
        "one program, one diagnostic, whichever position it is written in"
    );
    assert!(
        direct.errors.iter().any(|error| {
            error.kind.diagnostic_name() == "TypeMismatch"
                && error.expected.as_deref() == Some("i64")
                && error.got.as_deref() == Some("i32")
                && error.span_offset == I32_SIZE_IN_DIRECT_POSITION.find("expand(")
        }),
        "the direct spelling must reject its size dtype: {:?}",
        direct.errors
    );
}

/// The shape-sourced pipe spelling. This program is chelis#1923's reproducer,
/// so the assertion is also that issue's checker-level receipt.
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
    let assert_literal_size = |source: &str| {
        let report = check_ir_program(&surf_to_deep(source))
            .expect_err("a sourceless named-axis size must reject");
        assert!(
            report.errors.iter().any(|error| {
                error.kind.diagnostic_name() == "DimensionMismatch"
                    && error.message.contains("insert")
                    && error.message.contains("argument 3")
                    && error.message.contains("stampable onto the new named dim")
                    && error.expected.as_deref() == Some("compile-time literal size")
                    && error.got.is_some()
                    && error
                        .span_offset
                        .is_some_and(|offset| offset < source.len())
            }),
            "named-axis insertion must reject its non-literal size: {:?}",
            report.errors
        );
    };

    // Direct position, four-argument anchored form.
    assert_literal_size(
        "def g[n, m](b: tensor[n, f32], k: i32) -> tensor[m, n, f32] = \
         insert(b, m, cast(k, i64), n)\n",
    );

    // Pipe position, three-argument named form.
    assert_literal_size(
        "sig f[m]: i32 -> tensor[m, 1, f32]\n\
         def f(k: i32) = {\n  \
         a_dim = k |> cast(i64)\n  \
         [0.25f32] |> to_tensor |> insert(m, a_dim)\n\
         }\n",
    );

    // Pipe position, four-argument anchored form.
    assert_literal_size(
        "sig f[m]: i32 -> tensor[m, 1, f32]\n\
         def f(k: i32) = {\n  \
         a_dim = k |> cast(i64)\n  \
         [0.25f32] |> to_tensor |> insert(m, a_dim, n)\n\
         }\n",
    );
}
