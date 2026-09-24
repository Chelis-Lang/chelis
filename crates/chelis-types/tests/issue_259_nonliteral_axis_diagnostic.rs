//! Issue #259 regression: reduce/expand builtins with a non-literal axis
//! must emit a precise diagnostic at the reduction/expand call site
//! naming the axis-must-be-a-compile-time-constant root cause, rather
//! than leaving the output type variable unresolved and surfacing a
//! misleading borrow/linearity diagnostic at a downstream `&` site.
//!
//! Root cause (see the issue body): `tensor_reduce_to_out` and
//! `tensor_expand_to_out` in `crates/chelis-types/src/builtins.rs`
//! register the reduction/expand family with an unconstrained fresh
//! output type variable `out`. All shape resolution comes from the
//! infer-time post-check (`check_reduction_signature` /
//! `check_expand_signature`), which reads the axis literal. When the
//! axis is a literal (`mean(&x, cast(0, i32))` or `mean(&x, 0)`) the
//! post-check fires and resolves `out` to the reduced tensor shape.
//! When the axis is a non-literal (`mean(&x, ax)` where `ax: i32` is a
//! function parameter), `extract_int_for_dim` returns `None`, the
//! post-check pre-fix short-circuited with `subst.apply(result_ty)`, and
//! `out` stayed a fresh `Type::Var`. A subsequent `&` borrow of the
//! reduction's result then saw `?N` and the deferred-borrow validation
//! reported `borrow requires tensor or tensor-carrying input, got ?N`,
//! which does not point at the reduction call as the root cause.
//!
//! Fix (issue option 2, the targeted-diagnostic short term): when the
//! input is a concrete tensor (so the only obstacle to a resolved output
//! shape is the axis) and the axis arg does not resolve to a
//! compile-time-constant int, emit a `DimensionMismatch` at the
//! reduction/expand site naming the builtin, the axis arg, and the
//! compile-time-constant requirement. Return `Type::Error` so the
//! unresolved output variable never leaks to a borrow site.
//!
//! Spec sources of truth:
//!   - spec/05-risc-primitives.md lines 112-125 (reductions remove the
//!     dimension at `axis`; the axis is a zero-indexed int, negative axes
//!     index from the end). The output dims are "the dimension at
//!     position `axis` is removed" — undeterminable without a concrete
//!     axis.
//!   - spec/04-type-system.md §5.7.1 (reduction result precision).

use chelis_deep::Expr;
use chelis_deep::parser::parse_str;
use chelis_types::{InferResult, check_ir_program};

fn core_fixture(source: &str) -> Vec<Expr> {
    parse_str(source).expect("core fixture")
}

fn messages(rep: &InferResult) -> Vec<String> {
    rep.errors.iter().map(|e| e.message.clone()).collect()
}

// ---------------------------------------------------------------------------
// Clarity: the issue's exact reproducer. A non-literal reduction axis must
// produce a targeted diagnostic naming the axis cause, NOT a borrow error.
// ---------------------------------------------------------------------------

/// EXPECT (issue #259 reproducer): `mean(&x, ax)` with `ax: i32` a
/// function parameter is rejected with a diagnostic at the `mean` call
/// site naming the compile-time-constant-axis requirement. The
/// misleading `borrow requires tensor or tensor-carrying input, got ?N`
/// borrow diagnostic must NOT appear.
#[test]
fn issue259_mean_nonliteral_axis_reports_axis_cause_not_borrow() {
    let src = include_str!(
        "fixtures/core/tests/issue_259_nonliteral_axis_diagnostic/issue259_mean_nonliteral_axis_reports_axis_cause_not_borrow.deep"
    );
    let deep = core_fixture(src);
    let rep = check_ir_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .expect_err("non-literal reduction axis must be rejected at infer");
    let msgs = messages(&rep);

    // The root-cause diagnostic fires at the reduction site, exactly
    // once: returning `Type::Error` must not re-fire the post-check or
    // cascade a second copy through the deferred-borrow re-check pass.
    let axis_cause_count = msgs
        .iter()
        .filter(|m| m.contains("mean") && m.contains("axis") && m.contains("compile-time constant"))
        .count();
    assert_eq!(
        axis_cause_count, 1,
        "expected exactly one `mean` compile-time-constant-axis diagnostic, got {msgs:?}"
    );
    // It names the runtime axis binding so the user can see the cause.
    assert!(
        msgs.iter().any(|m| m.contains("ax")),
        "diagnostic should name the runtime axis binding `ax`, got {msgs:?}"
    );
    // The misleading borrow/linearity diagnostic must NOT leak.
    assert!(
        !msgs
            .iter()
            .any(|m| m.contains("borrow requires tensor or tensor-carrying input")),
        "the misleading borrow `got ?N` diagnostic must not surface; got {msgs:?}"
    );
}

/// EXPECT: the reduction family is covered uniformly, not just `mean`.
/// `sum`, `max_reduce`, `min_reduce`, `prod_reduce`, `argmax_reduce`,
/// `argmin_reduce` all route through `check_reduction_signature`, so a
/// non-literal axis on each one emits the same targeted diagnostic.
#[test]
fn issue259_reduction_family_nonliteral_axis_all_report_axis_cause() {
    for name in [
        "sum",
        "max_reduce",
        "min_reduce",
        "prod_reduce",
        "argmax_reduce",
        "argmin_reduce",
    ] {
        let source = include_str!(
            "fixtures/core/tests/issue_259_nonliteral_axis_diagnostic/reduction_family.deep"
        )
        .replace(" sum)", &format!(" {name})"));
        let deep = core_fixture(&source);
        let rep = check_ir_program(
            &deep,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        )
        .err()
        .unwrap_or_else(|| panic!("{name} non-literal axis must be rejected"));
        let msgs = messages(&rep);
        assert!(
            msgs.iter().any(|m| m.contains(name)
                && m.contains("axis")
                && m.contains("compile-time constant")),
            "expected a `{name}` compile-time-constant-axis diagnostic, got {msgs:?}"
        );
        assert!(
            !msgs
                .iter()
                .any(|m| m.contains("borrow requires tensor or tensor-carrying input")),
            "`{name}`: the misleading borrow diagnostic must not surface; got {msgs:?}"
        );
    }
}

/// EXPECT: `expand` with a non-literal axis is also rejected at the
/// expand site naming the compile-time-constant-axis requirement, not a
/// downstream borrow error. `expand` registers through
/// `tensor_expand_to_out` and shares the same unconstrained-`out` shape.
#[test]
fn issue259_expand_nonliteral_axis_reports_axis_cause_not_borrow() {
    let src = include_str!(
        "fixtures/core/tests/issue_259_nonliteral_axis_diagnostic/issue259_expand_nonliteral_axis_reports_axis_cause_not_borrow.deep"
    );
    let deep = core_fixture(src);
    let rep = check_ir_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .expect_err("non-literal expand axis must be rejected at infer");
    let msgs = messages(&rep);
    assert!(
        msgs.iter().any(|m| m.contains("insert")
            && m.contains("axis")
            && m.contains("compile-time constant")),
        "expected an `insert` compile-time-constant-axis diagnostic, got {msgs:?}"
    );
    assert!(
        !msgs
            .iter()
            .any(|m| m.contains("borrow requires tensor or tensor-carrying input")),
        "the misleading borrow diagnostic must not surface; got {msgs:?}"
    );
}

// ---------------------------------------------------------------------------
// Control: the literal-axis form still type-checks (no regression).
// ---------------------------------------------------------------------------

/// Positive control: the literal-axis reduction still resolves the
/// output shape and type-checks cleanly. `mean(&x, 1)` on `tensor[m, n,
/// f32]` removes axis 1, yielding `tensor[m, f32]`. The reduced axis
/// (axis 1) carries a concrete extent `4` so the separate
/// `concrete reduced axis extent` IR validator (spec/05 mean rule) is
/// satisfied; that validator is orthogonal to the #259 non-literal-axis
/// path. This guards the fix against over-rejecting the legitimate
/// literal-axis path.
#[test]
fn issue259_mean_literal_axis_still_typechecks() {
    let src = include_str!(
        "fixtures/core/tests/issue_259_nonliteral_axis_diagnostic/issue259_mean_literal_axis_still_typechecks.deep"
    );
    let deep = core_fixture(src);
    if let Err(rep) = check_ir_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    ) {
        panic!(
            "literal-axis reduction must still type-check, got {:?}",
            messages(&rep)
        );
    }
}

/// Positive control: the `cast(N, i32)`-wrapped literal axis form also
/// still type-checks. Issue #216 made `extract_int_for_dim` cast-aware;
/// this confirms the #259 non-literal arm does not swallow the
/// cast-wrapped literal that #216 deliberately admits.
#[test]
fn issue259_mean_cast_wrapped_literal_axis_still_typechecks() {
    let src = include_str!(
        "fixtures/core/tests/issue_259_nonliteral_axis_diagnostic/issue259_mean_cast_wrapped_literal_axis_still_typechecks.deep"
    );
    let deep = core_fixture(src);
    if let Err(rep) = check_ir_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    ) {
        panic!(
            "cast-wrapped literal-axis reduction must still type-check, got {:?}",
            messages(&rep)
        );
    }
}

/// Positive control: `expand` with a literal axis still type-checks,
/// including the symbolic-*size* form the existing checker admits
/// (`expand(x, 0, n)` inserts a leading dim of symbolic size `n`). The
/// fix touches only the axis arm, not the size arm.
#[test]
fn issue259_expand_literal_axis_symbolic_size_still_typechecks() {
    let src = include_str!(
        "fixtures/core/tests/issue_259_nonliteral_axis_diagnostic/issue259_expand_literal_axis_symbolic_size_still_typechecks.deep"
    );
    let deep = core_fixture(src);
    if let Err(rep) = check_ir_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    ) {
        panic!(
            "literal-axis expand must still type-check, got {:?}",
            messages(&rep)
        );
    }
}

// ---------------------------------------------------------------------------
// Negative parity: a genuinely wrong program still fails for the right
// reason — the fix does not mask unrelated, legitimate errors.
// ---------------------------------------------------------------------------

/// Negative parity: an out-of-bounds *literal* axis still fails with the
/// out-of-bounds diagnostic, NOT the new non-literal-axis diagnostic.
/// The two arms are distinct: a resolvable-but-out-of-range axis is a
/// different error than an unresolvable axis.
#[test]
fn issue259_oob_literal_axis_still_reports_out_of_bounds() {
    let src = include_str!(
        "fixtures/core/tests/issue_259_nonliteral_axis_diagnostic/issue259_oob_literal_axis_still_reports_out_of_bounds.deep"
    );
    let deep = core_fixture(src);
    let rep = check_ir_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .expect_err("out-of-bounds literal axis must be rejected");
    let msgs = messages(&rep);
    assert!(
        msgs.iter()
            .any(|m| m.contains("mean") && m.contains("out of bounds")),
        "expected the out-of-bounds-axis diagnostic, got {msgs:?}"
    );
    assert!(
        !msgs.iter().any(|m| m.contains("compile-time constant")),
        "an in-range-but-OOB literal axis is not the non-literal case; got {msgs:?}"
    );
}

/// Negative parity: a genuinely non-tensor borrow (a stack `i32` bound
/// against a `&tensor[..]` parameter) still fails for the right reason.
/// This is unrelated to the reduction axis path and must keep its own
/// diagnostic; the #259 fix must not suppress it.
#[test]
fn issue259_non_tensor_borrow_still_rejected() {
    let src = include_str!(
        "fixtures/core/tests/issue_259_nonliteral_axis_diagnostic/issue259_non_tensor_borrow_still_rejected.deep"
    );
    let deep = core_fixture(src);
    let rep = check_ir_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .expect_err("borrow of i32 against &tensor must be rejected");
    assert!(
        !rep.errors.is_empty(),
        "borrow of a non-tensor must surface at least one error"
    );
}
