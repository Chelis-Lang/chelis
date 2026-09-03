//! Issue #1355: `diagonal` declares the smaller selected extent, not a wildcard.
//!
//! `[05-OP-33]` (`spec/05-risc-primitives.md`) states the extent exactly:
//! `diagonal` "keeps source axis order with the second axis removed, and
//! replaces the retained first axis extent with **the smaller selected
//! extent**". When both selected extents are literal, that minimum is
//! statically known, so the checker must declare it. `infer_diagonal_result_type`
//! previously widened every unequal literal pair to `Dim::Wildcard`, which
//! unifies with any declared extent, so a declared return type the runtime
//! cannot produce still type-checked.
//!
//! Every case here uses NON-SQUARE extents. A square `tensor[n, n]` cannot
//! distinguish the old behaviour from the correct one, which is why the defect
//! survived: every checked-in `diagonal`/`trace` case was square.
//!
//! This suite covers the rank-2 and rank-3 f32 literal-extent forms named in
//! each test, the identical-named-extent form, and the distinct-name and
//! mixed literal/symbolic forms that keep the wildcard; it is not evidence for
//! the full per-dtype [05-OP-33] runtime contract.

use chelis_deep::Expr;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_ir_program;
use chelis_types::errors::CheckErrorKind;

fn surf_to_deep(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

/// The single diagnostic a wrong declared `diagonal`/`trace` extent must
/// produce, so a negative cannot pass on an unrelated or extra error.
fn sole_error(source: &str, what: &str) -> (CheckErrorKind, String) {
    let deep = surf_to_deep(source);
    let Err(report) = check_ir_program(&deep) else {
        panic!("{what} must be rejected");
    };
    let [error] = report.errors.as_slice() else {
        panic!(
            "expected exactly one error for {what}, got {:?}",
            report.errors
        );
    };
    (error.kind.clone(), error.message.clone())
}

/// The rejection message for a declared extent the checker refuses, pinned to
/// the `DimensionMismatch` kind so a negative cannot pass on some other error.
fn sole_dimension_mismatch(source: &str, what: &str) -> String {
    let (kind, message) = sole_error(source, what);
    assert!(
        matches!(kind, CheckErrorKind::DimensionMismatch),
        "{what} must reject as a DimensionMismatch, got {kind:?}"
    );
    message
}

fn accepts(source: &str, what: &str) {
    let deep = surf_to_deep(source);
    if let Err(report) = check_ir_program(&deep) {
        panic!("{what} must type-check, got {:?}", report.errors);
    }
}

// ---------------------------------------------------------------------------
// Regression tests (red before the fix, green after).
// ---------------------------------------------------------------------------

/// REGRESSION TEST. The exact program from chelis#1355: `tensor[3, 4]` over
/// axes (0, 1) has diagonal extent min(3, 4) = 3, so a declared `tensor[4]`
/// is a type error. Before the fix the result extent was `Dim::Wildcard`,
/// which unified with the declared 4 and scored a clean check.
#[test]
fn wider_declared_extent_is_rejected() {
    let message = sole_dimension_mismatch(
        "def f(x: tensor[3, 4, f32]) -> tensor[4, f32] = diagonal(x, 0, 1)\n",
        "diagonal on tensor[3, 4] declared as tensor[4]",
    );
    assert!(
        message.contains("tensor[3, f32]") && message.contains("tensor[4, f32]"),
        "the rejection must name the inferred tensor[3, f32] against the \
         declared tensor[4, f32], got {message}"
    );
}

/// REGRESSION TEST. Mirror of the issue program: with the extents swapped the
/// minimum is the SECOND selected extent, so `tensor[4, 3]` also diagonalises
/// to `tensor[3]` and a declared `tensor[4]` is a type error. This is the case
/// a `min` that silently returned the first (retained) extent would pass.
#[test]
fn mirror_operand_rejects_the_retained_axis_extent() {
    let message = sole_dimension_mismatch(
        "def f(x: tensor[4, 3, f32]) -> tensor[4, f32] = diagonal(x, 0, 1)\n",
        "diagonal on tensor[4, 3] declared as tensor[4]",
    );
    assert!(
        message.contains("tensor[3, f32]") && message.contains("tensor[4, f32]"),
        "the mirror rejection must name the inferred tensor[3, f32], got {message}"
    );
}

/// REGRESSION TEST. Rank 3 with a retained axis: `diagonal(x, 1, 2)` on
/// `tensor[2, 3, 5]` removes axis 2 and replaces axis 1 with min(3, 5) = 3,
/// so the result is `tensor[2, 3]` and a declared `tensor[2, 5]` is a type
/// error. Proves the retained non-selected axis is untouched while the
/// selected one is narrowed.
#[test]
fn rank_three_retained_axis_rejects_the_larger_selected_extent() {
    let message = sole_dimension_mismatch(
        "def f(x: tensor[2, 3, 5, f32]) -> tensor[2, 5, f32] = diagonal(x, 1, 2)\n",
        "diagonal on tensor[2, 3, 5] over axes (1, 2) declared as tensor[2, 5]",
    );
    assert!(
        message.contains("tensor[2, 3, f32]") && message.contains("tensor[2, 5, f32]"),
        "the rank-3 rejection must name the inferred tensor[2, 3, f32], got {message}"
    );
}

/// REGRESSION TEST. `[05-OP-33]`'s smaller selected extent for two occurrences
/// of ONE named extent is that name: `spec/04-type-system.md` §4.1 makes two
/// `d-name` unify only when equal, so both axes of `tensor[hidden, hidden]`
/// denote a single runtime value. Declaring a DIFFERENT rigid name is therefore
/// a type error. Before the fix the result extent was `Dim::Wildcard`, which
/// unified with `width` and scored a clean check.
#[test]
fn identical_named_extents_reject_a_different_declared_name() {
    let message = sole_dimension_mismatch(
        "def f(x: tensor[hidden, hidden, f32]) -> tensor[width, f32] = diagonal(x, 0, 1)\n",
        "diagonal on tensor[hidden, hidden] declared as tensor[width]",
    );
    assert!(
        message.contains("tensor[hidden, f32]") && message.contains("tensor[width, f32]"),
        "the rejection must name the inferred tensor[hidden, f32] against the \
         declared tensor[width, f32], got {message}"
    );
}

// ---------------------------------------------------------------------------
// Positive controls (green before and after; they lock that the narrowing is
// not over-applied).
// ---------------------------------------------------------------------------

/// DISPOSITION LOCK. The correct declaration for the issue program keeps
/// type-checking. Green before the fix too (a wildcard unified with 3), so
/// this proves only that the narrowing did not turn a correct program into a
/// rejection.
#[test]
fn correct_smaller_extent_is_accepted() {
    accepts(
        "def f(x: tensor[3, 4, f32]) -> tensor[3, f32] = diagonal(x, 0, 1)\n",
        "diagonal on tensor[3, 4] declared as tensor[3]",
    );
}

/// DISPOSITION LOCK. Mirror of the above: `tensor[4, 3]` declared `tensor[3]`.
#[test]
fn correct_mirror_extent_is_accepted() {
    accepts(
        "def f(x: tensor[4, 3, f32]) -> tensor[3, f32] = diagonal(x, 0, 1)\n",
        "diagonal on tensor[4, 3] declared as tensor[3]",
    );
}

/// DISPOSITION LOCK. Rank 3 with the correct declaration.
#[test]
fn correct_rank_three_extent_is_accepted() {
    accepts(
        "def f(x: tensor[2, 3, 5, f32]) -> tensor[2, 3, f32] = diagonal(x, 1, 2)\n",
        "diagonal on tensor[2, 3, 5] over axes (1, 2) declared as tensor[2, 3]",
    );
}

/// DISPOSITION LOCK. The square case the old equal-literals arm already
/// handled is subsumed by `min`, so `tensor[4, 4]` still declares `tensor[4]`
/// and still rejects `tensor[3]`. Green in both states.
#[test]
fn square_operand_keeps_its_exact_extent() {
    accepts(
        "def f(x: tensor[4, 4, f32]) -> tensor[4, f32] = diagonal(x, 0, 1)\n",
        "diagonal on the square tensor[4, 4]",
    );
    let message = sole_dimension_mismatch(
        "def f(x: tensor[4, 4, f32]) -> tensor[3, f32] = diagonal(x, 0, 1)\n",
        "diagonal on tensor[4, 4] declared as tensor[3]",
    );
    assert!(
        message.contains("tensor[4, f32]") && message.contains("tensor[3, f32]"),
        "the square rejection must name the inferred tensor[4, f32], got {message}"
    );
}

/// DISPOSITION LOCK. A symbolic selected extent keeps the wildcard: the
/// minimum of `n` and 4 is genuinely unknown at check time, so both the
/// symbolic and the literal declaration must keep type-checking. Green in
/// both states; it is the control that the fix narrowed only the
/// both-literal arm.
#[test]
fn symbolic_selected_extent_stays_wildcard_compatible() {
    accepts(
        "def f(x: tensor[n, 4, f32]) -> tensor[n, f32] = diagonal(x, 0, 1)\n",
        "diagonal on tensor[n, 4] declared as tensor[n]",
    );
    accepts(
        "def f(x: tensor[n, 4, f32]) -> tensor[4, f32] = diagonal(x, 0, 1)\n",
        "diagonal on tensor[n, 4] declared as tensor[4]",
    );
}

/// DISPOSITION LOCK. The positive twin of the two named-extent rejections:
/// `tensor[hidden, hidden]` declared `tensor[hidden]` is the correct program
/// and keeps type-checking. Green before the fix too (a wildcard unified with
/// `hidden`), so this proves only that the narrowing did not turn the correct
/// program into a rejection.
#[test]
fn identical_named_extents_accept_that_name() {
    accepts(
        "def f(x: tensor[hidden, hidden, f32]) -> tensor[hidden, f32] = diagonal(x, 0, 1)\n",
        "diagonal on tensor[hidden, hidden] declared as tensor[hidden]",
    );
}

/// DISPOSITION LOCK, recording a boundary this change does NOT move. The named
/// arm does not make a declared LITERAL extent reject: `unify_dim` accepts
/// `Name` against `Lit` without binding a substitution, by the chelis#219
/// Option A rule that lets `def f(x: tensor[batch, hidden, f32])` be called with
/// concrete-shaped inputs. So `tensor[hidden, hidden]` declared `tensor[3, f32]`
/// is still admitted, before and after. That is dimension-unification policy,
/// not diagonal's extent rule, and #1355 does not touch it.
#[test]
fn identical_named_extents_still_admit_a_declared_literal() {
    accepts(
        "def f(x: tensor[hidden, hidden, f32]) -> tensor[3, f32] = diagonal(x, 0, 1)\n",
        "diagonal on tensor[hidden, hidden] declared as tensor[3]",
    );
}

/// DISPOSITION LOCK. Two DISTINCT named extents keep the wildcard: the minimum
/// of `hidden` and `width` is genuinely unknown at check time, so every
/// declaration must keep type-checking, including a literal one. Green in both
/// states; it is the control that the named arm fires only on equal names.
#[test]
fn distinct_named_extents_stay_wildcard_compatible() {
    accepts(
        "def f(x: tensor[hidden, width, f32]) -> tensor[hidden, f32] = diagonal(x, 0, 1)\n",
        "diagonal on tensor[hidden, width] declared as tensor[hidden]",
    );
    accepts(
        "def f(x: tensor[hidden, width, f32]) -> tensor[width, f32] = diagonal(x, 0, 1)\n",
        "diagonal on tensor[hidden, width] declared as tensor[width]",
    );
    accepts(
        "def f(x: tensor[hidden, width, f32]) -> tensor[3, f32] = diagonal(x, 0, 1)\n",
        "diagonal on tensor[hidden, width] declared as tensor[3]",
    );
}

/// DISPOSITION LOCK. `trace` on identical named extents removes both axes and
/// is unaffected by the named arm, exactly as it is by the literal one.
#[test]
fn trace_on_identical_named_extents_is_rank_zero() {
    accepts(
        "def f(x: tensor[hidden, hidden, f32]) -> tensor[f32] = trace(x, 0, 1)\n",
        "trace on tensor[hidden, hidden]",
    );
}

// ---------------------------------------------------------------------------
// `trace` controls: it routes through `infer_trace_result_type`, which removes
// both selected axes and never consults their extents. Green in both states.
// ---------------------------------------------------------------------------

/// DISPOSITION LOCK. `trace` on a NON-SQUARE operand removes both axes and
/// yields the rank-zero tensor, unaffected by the diagonal extent rule.
#[test]
fn trace_on_non_square_operand_is_rank_zero() {
    accepts(
        "def f(x: tensor[3, 4, f32]) -> tensor[f32] = trace(x, 0, 1)\n",
        "trace on tensor[3, 4]",
    );
}

/// DISPOSITION LOCK. The failure twin of the above: `trace` removes BOTH
/// axes, so a declared `tensor[3]` (the diagonal's shape) is still a type
/// error. Green in both states; it proves the trace path did not inherit the
/// diagonal's retained axis.
#[test]
fn trace_does_not_retain_the_diagonal_extent() {
    let message = sole_dimension_mismatch(
        "def f(x: tensor[3, 4, f32]) -> tensor[3, f32] = trace(x, 0, 1)\n",
        "trace on tensor[3, 4] declared as tensor[3]",
    );
    assert!(
        message.contains("tensor[, f32]") && message.contains("tensor[3, f32]"),
        "trace must infer the rank-zero tensor[, f32], got {message}"
    );
}

/// DISPOSITION LOCK. Rank 3 `trace` keeps only the unselected axis.
#[test]
fn trace_keeps_only_the_unselected_axis() {
    accepts(
        "def f(x: tensor[2, 3, 5, f32]) -> tensor[2, f32] = trace(x, 1, 2)\n",
        "trace on tensor[2, 3, 5] over axes (1, 2)",
    );
}
