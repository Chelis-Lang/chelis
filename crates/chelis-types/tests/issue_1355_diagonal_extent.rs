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
        &desugar_program(&decls).expect("Surf fixture must desugar"),
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
        "def f[n](x: tensor[n, 4, f32]) -> tensor[n, f32] = diagonal(x, 0, 1)\n",
        "diagonal on tensor[n, 4] declared as tensor[n]",
    );
    accepts(
        "def f[n](x: tensor[n, 4, f32]) -> tensor[4, f32] = diagonal(x, 0, 1)\n",
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

// ---------------------------------------------------------------------------
// chelis#1739: a literal selected axis is an UPPER BOUND on the result extent.
//
// #1355 left the mixed (symbolic, literal) pair at `Dim::Wildcard`, which
// unifies with every declared extent. `[05-OP-33]` takes the SMALLER selected
// extent, so `min(n, 4) <= 4` holds for every runtime `n`: a declared extent
// strictly greater than the literal axis is unreachable and is now a type
// error. At or below the literal stays accepted, because the checker cannot
// decide which side of the minimum wins and the runtime guard owns the rest.
//
// The result dim itself is unchanged (still `Dim::Wildcard`); only the
// declared literal is compared against the bound, so the three #1355
// disposition locks above keep their exact dispositions.
// ---------------------------------------------------------------------------

/// The upper-bound rejection retains the admissible bound, attempted
/// declaration and authored call location at either declaration ingress.
fn assert_bound_rejection(
    source: &str,
    what: &str,
    literal_axis: usize,
    bound: usize,
    declared: usize,
) {
    let deep = surf_to_deep(source);
    let Err(report) = check_ir_program(&deep) else {
        panic!("{what} must be rejected");
    };
    let [error] = report.errors.as_slice() else {
        panic!("expected one error for {what}, got {:?}", report.errors);
    };
    assert!(matches!(error.kind, CheckErrorKind::DimensionMismatch), "{error:?}");
    assert_eq!(error.expected.as_deref(), Some(format!("extent at most {bound}").as_str()));
    assert_eq!(error.got.as_deref(), Some(format!("declared extent {declared}").as_str()));
    assert_eq!(error.span_offset, source.find("diagonal("), "{error:?}");
    assert!(error.message.contains(&format!("axis {literal_axis}")), "{error:?}");
}

/// REGRESSION TEST. The exact program from chelis#1739: `tensor[n, 4]` over
/// axes (0, 1) has diagonal extent min(n, 4), which is at most 4 for every
/// runtime `n`, so a declared `tensor[9]` is unreachable. Before the fix the
/// result extent was `Dim::Wildcard` and the program scored a clean 1.0.
#[test]
fn a_declared_literal_wider_than_the_literal_axis_is_rejected() {
    assert_bound_rejection(
        "def f[n](x: tensor[n, 4, f32]) -> tensor[9, f32] = diagonal(x, 0, 1)\n",
        "diagonal on tensor[n, 4] declared as tensor[9]",
        1,
        4,
        9,
    );
}

/// REGRESSION TEST. The mirrored pair: the literal is the FIRST selected axis
/// and the symbolic one is second. `min(4, n) <= 4` is the same bound, so
/// `tensor[4, n]` declared `tensor[9]` is the same type error. This is the
/// case a bound that only read the second selected axis would admit.
#[test]
fn the_mirrored_literal_axis_bounds_the_declared_extent_too() {
    assert_bound_rejection(
        "def f[n](x: tensor[4, n, f32]) -> tensor[9, f32] = diagonal(x, 0, 1)\n",
        "diagonal on tensor[4, n] declared as tensor[9]",
        0,
        4,
        9,
    );
}

/// REGRESSION TEST. A RIGID named extent beside the literal is bounded exactly
/// like a dimension variable. Surf desugars a multi-letter dim to `d-name` and
/// a single-letter one to `d-var`, so the two spellings reach
/// `infer_diagonal_result_type` as different `Dim` variants; both are
/// non-literal and both are bounded by the literal axis.
#[test]
fn a_rigid_named_extent_beside_a_literal_is_bounded_the_same_way() {
    assert_bound_rejection(
        "def f(x: tensor[hidden, 4, f32]) -> tensor[9, f32] = diagonal(x, 0, 1)\n",
        "diagonal on tensor[hidden, 4] declared as tensor[9]",
        1,
        4,
        9,
    );
}

/// REGRESSION TEST. Rank 3 with a retained axis: `diagonal(x, 1, 2)` on
/// `tensor[2, n, 5]` removes axis 2 and bounds the retained axis 1 by 5, so a
/// declared `tensor[2, 9]` is rejected while the untouched leading 2 is
/// compared as usual. Proves the bound is applied at the RESULT index of the
/// retained axis, not at its source index.
#[test]
fn a_rank_three_retained_axis_is_bounded_at_its_result_index() {
    assert_bound_rejection(
        "def f[n](x: tensor[2, n, 5, f32]) -> tensor[2, 9, f32] = diagonal(x, 1, 2)\n",
        "diagonal on tensor[2, n, 5] over axes (1, 2) declared as tensor[2, 9]",
        2,
        5,
        9,
    );
}

/// REGRESSION TEST. The second ingress. A block-scoped ascription
/// (`y: tensor[9, f32] = diagonal(...)`) carries its declared type as `"type"`
/// metadata on the application node itself rather than through the enclosing
/// signature, so a rejection that only consulted the expected result type
/// would admit this spelling. Before the fix it scored a clean 1.0.
#[test]
fn the_block_ascription_ingress_is_bounded_too() {
    assert_bound_rejection(
        "def f[n](x: tensor[n, 4, f32]) -> tensor[9, f32] = {\n  \
         y: tensor[9, f32] = diagonal(x, 0, 1)\n  y\n}\n",
        "a block ascription of tensor[9] over tensor[n, 4]",
        1,
        4,
        9,
    );
}

/// DISPOSITION LOCK. At the bound is accepted: `min(n, 4)` reaches 4 whenever
/// `n >= 4`, so the declaration is satisfiable and only the runtime can decide.
/// Green before the fix too (a wildcard unified with 4); it is the control that
/// the bound rejects STRICTLY greater and not greater-or-equal.
#[test]
fn a_declared_literal_at_the_literal_axis_is_accepted() {
    accepts(
        "def f[n](x: tensor[n, 4, f32]) -> tensor[4, f32] = diagonal(x, 0, 1)\n",
        "diagonal on tensor[n, 4] declared as tensor[4]",
    );
}

/// DISPOSITION LOCK. Below the bound is accepted for the same reason:
/// `min(n, 4) = 3` whenever `n = 3`. Green before the fix too.
#[test]
fn a_declared_literal_below_the_literal_axis_is_accepted() {
    accepts(
        "def f[n](x: tensor[n, 4, f32]) -> tensor[3, f32] = diagonal(x, 0, 1)\n",
        "diagonal on tensor[n, 4] declared as tensor[3]",
    );
}

/// DISPOSITION LOCK recording a case this change does NOT claim. Two
/// occurrences of ONE dimension VARIABLE (`tensor[n, n]`, which Surf desugars
/// to two `d-var` nodes rather than the `d-name` pair of `tensor[hidden,
/// hidden]`) keep the wildcard, so a declared literal is still admitted. chelis
/// #1517 left that disposition explicitly unclaimed and chelis#1739 does not
/// claim it either; this test records the behaviour, it does not assert that
/// the behaviour is right.
#[test]
fn the_two_dimension_variable_pair_stays_unclaimed() {
    accepts(
        "def f[n](x: tensor[n, n, f32]) -> tensor[9, f32] = diagonal(x, 0, 1)\n",
        "diagonal on tensor[n, n] declared as tensor[9] (unclaimed disposition)",
    );
}

/// DISPOSITION LOCK. `trace` removes BOTH selected axes, so no result axis
/// carries the diagonal's extent and the literal selected axis bounds nothing.
/// The rank-zero declaration is accepted before and after.
#[test]
fn trace_over_a_symbolic_literal_pair_is_unaffected_by_the_bound() {
    accepts(
        "def f[n](x: tensor[n, 4, f32]) -> tensor[f32] = trace(x, 0, 1)\n",
        "trace on tensor[n, 4]",
    );
}

/// DISPOSITION LOCK, the failure twin of the above. `trace` on the same
/// operand declared `tensor[9]` is still rejected for the RANK, naming the
/// inferred rank-zero tensor rather than the diagonal's bound. Green in both
/// states; it proves the bound did not leak into the trace route.
#[test]
fn trace_over_a_symbolic_literal_pair_rejects_for_rank_not_for_the_bound() {
    let message = sole_dimension_mismatch(
        "def f[n](x: tensor[n, 4, f32]) -> tensor[9, f32] = trace(x, 0, 1)\n",
        "trace on tensor[n, 4] declared as tensor[9]",
    );
    assert!(
        message.contains("tensor[, f32]"),
        "trace must infer the rank-zero tensor[, f32], got {message}"
    );
    assert!(
        !message.contains("at most 4"),
        "the diagonal bound must not appear on the trace route, got {message}"
    );
}

/// REGRESSION TEST. A declared type ALIAS carries the same verdict as its
/// expansion at both ingresses. The bound compares a resolved `Type`, not the
/// shape of the annotation's Deep metadata, so `type Row = tensor[9, f32]` is
/// rejected exactly as the spelled-out tensor type is.
#[test]
fn a_type_alias_carries_the_same_bound_verdict() {
    assert_bound_rejection(
        "type Row = tensor[9, f32]\n\
         def f[n](x: tensor[n, 4, f32]) -> Row = diagonal(x, 0, 1)\n",
        "diagonal on tensor[n, 4] declared as the alias Row",
        1,
        4,
        9,
    );
    assert_bound_rejection(
        "type Row = tensor[9, f32]\n\
         def f[n](x: tensor[n, 4, f32]) -> Row = {\n  y: Row = diagonal(x, 0, 1)\n  y\n}\n",
        "a block ascription of the alias Row over tensor[n, 4]",
        1,
        4,
        9,
    );
}

/// DISPOSITION LOCK. The positive twin: an alias at the bound is accepted, so
/// the alias route is not simply rejecting every aliased declaration.
#[test]
fn a_type_alias_at_the_bound_is_accepted() {
    accepts(
        "type Row = tensor[4, f32]\n\
         def f[n](x: tensor[n, 4, f32]) -> Row = diagonal(x, 0, 1)\n",
        "diagonal on tensor[n, 4] declared as the alias Row = tensor[4, f32]",
    );
}

/// REGRESSION TEST (round 1 P2). A declaration whose RANK disagrees is a
/// signature mismatch, not a bound violation. Reading an axis out of it anyway
/// reported "the result extent is at most 4" for `tensor[n, 4, 5] ->
/// tensor[7]`, naming a repair that would not fix the program, and the error
/// type that path returns then suppressed the accurate diagnostic. The bound
/// now stands down on a rank disagreement and unification reports it.
#[test]
fn a_rank_disagreement_reports_the_signature_mismatch_not_the_bound() {
    let message = sole_dimension_mismatch(
        "def f[n](x: tensor[n, 4, 5, f32]) -> tensor[7, f32] = diagonal(x, 0, 1)\n",
        "diagonal on tensor[n, 4, 5] over axes (0, 1) declared as the rank-1 tensor[7]",
    );
    assert!(
        message.contains("body doesn't match declared signature")
            && message.contains("tensor[7, f32]"),
        "a rank disagreement must reject as the signature mismatch it is, got {message}"
    );
    assert!(
        !message.contains("at most"),
        "the bound must not speak for a rank error, got {message}"
    );
}

/// DISPOSITION LOCK, the control for the row above. With the RANK agreeing, the
/// same operand and axes keep the bound: the retained axis is bounded by the
/// literal 4 and a declared `tensor[9, 5]` is rejected as unreachable. Proves
/// the precondition narrowed the bound to rank agreement and did not disable it
/// for rank-2 results.
#[test]
fn a_rank_agreeing_declaration_still_carries_the_bound() {
    assert_bound_rejection(
        "def f[n](x: tensor[n, 4, 5, f32]) -> tensor[9, 5, f32] = diagonal(x, 0, 1)\n",
        "diagonal on tensor[n, 4, 5] over axes (0, 1) declared as tensor[9, 5]",
        1,
        4,
        9,
    );
}
