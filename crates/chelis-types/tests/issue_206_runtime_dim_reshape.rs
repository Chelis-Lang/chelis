//! Probes for chelis#206: `reshape(x, [cast(shape(x, axis), i64), ...])`
//! is syntactically accepted but the type checker does not propagate the
//! corresponding symbolic dim of `x` into the reshape result type.
//!
//! Background
//! ----------
//! From the issue's reproducer, a function declared as
//!
//! ```text
//! sig flatten_batch[n]: &tensor[n, 4, f32] -> tensor[n, 4, f32]
//! def flatten_batch(x) = reshape(x, [cast(shape(x, cast(0, i32)), i64), cast(4, i64)])
//! ```
//!
//! reports a `TypeMismatch` because the body is inferred as
//! `tensor[Wildcard, f32]`. The neighbor pattern
//! `expand(b, 0, shape(x, cast(0, i32)))` type-checks cleanly because
//! `expand`'s output dim is forced by the declared signature -- but the
//! issue is that `reshape` collapses the entire output rank to a
//! `vec![Wildcard]` (the input precision case in `infer_reshape_app`
//! recovers only the precision, not the dims).
//!
//! Expected fix: when an element of `reshape`'s shape list is the
//! syntactic form `cast(shape(x, lit_axis), i64)` and `x` is the same
//! input tensor being reshaped AND `lit_axis` resolves to a known axis
//! of `x`, the typer must propagate `x`'s dim at that axis into the
//! corresponding output dim. Plain literal dims like `cast(4, i64)`
//! must continue to produce `Dim::Lit(4)`. Any other shape-list element
//! shape (arbitrary expression that the recognizer doesn't know how to
//! interpret) must fall back to `Dim::Wildcard` -- no regression in
//! diagnostic quality.

use chelis_types::CheckedProgram;
use chelis_types::check_typed_program;
use chelis_types::errors::CheckError;

fn typecheck_core(source: &str) -> Vec<CheckError> {
    let deep: Vec<chelis_deep::Expr> =
        chelis_deep::parse_and_stamp_file(source).expect("valid core fixture");
    match check_typed_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    ) {
        Ok(_) => Vec::new(),
        Err(prog_errors) => prog_errors.errors,
    }
}

fn typecheck_core_program(source: &str) -> Result<CheckedProgram, Vec<CheckError>> {
    let deep: Vec<chelis_deep::Expr> =
        chelis_deep::parse_and_stamp_file(source).expect("valid core fixture");
    check_typed_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .map_err(|prog_errors| prog_errors.errors)
}

fn errors_summary(errors: &[CheckError]) -> String {
    if errors.is_empty() {
        "(no errors)".to_string()
    } else {
        errors
            .iter()
            .map(|e| format!("[{:?}] {}", e.kind, e.message))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn type_env_string(checked: &CheckedProgram, name: &str) -> String {
    checked
        .type_env()
        .get(name)
        .map(|e| format!("{e:?}"))
        .unwrap_or_else(|| format!("(no type_env entry for {name})"))
}

/// The issue's exact reproducer (`flatten_batch`) MUST type-check. Before
/// the fix this is the failing case and emits a `TypeMismatch`.
#[test]
fn issue_206_flatten_batch_typechecks() {
    let errors = typecheck_core(include_str!(
        "fixtures/core/tests/issue_206_runtime_dim_reshape/issue_206_flatten_batch_typechecks.0.deep"
    ));
    assert!(
        errors.is_empty(),
        "flatten_batch from issue 206 should type-check; got errors:\n{}",
        errors_summary(&errors)
    );
}

/// The neighbor case (`bias_broadcast`) MUST still type-check -- this
/// pins the no-regression for the existing working pattern.
#[test]
fn issue_206_bias_broadcast_typechecks() {
    let errors = typecheck_core(include_str!(
        "fixtures/core/tests/issue_206_runtime_dim_reshape/issue_206_bias_broadcast_typechecks.0.deep"
    ));
    assert!(
        errors.is_empty(),
        "bias_broadcast should keep working after the fix; got errors:\n{}",
        errors_summary(&errors)
    );
}

/// The issue's full module exactly as written -- both functions in one
/// module must pass `chelis check`.
#[test]
fn issue_206_full_module_typechecks() {
    let errors = typecheck_core(include_str!(
        "fixtures/core/tests/issue_206_runtime_dim_reshape/issue_206_full_module_typechecks.0.deep"
    ));
    assert!(
        errors.is_empty(),
        "full issue 206 module should type-check; got errors:\n{}",
        errors_summary(&errors)
    );
}

/// Two symbolic dims sourced from different axes of the same input. Both
/// axis-0 and axis-1 dims should propagate.
#[test]
fn issue_206_two_symbolic_dims_propagate() {
    let errors = typecheck_core(include_str!(
        "fixtures/core/tests/issue_206_runtime_dim_reshape/issue_206_two_symbolic_dims_propagate.0.deep"
    ));
    assert!(
        errors.is_empty(),
        "two symbolic dims from axis-0 and axis-1 of the same input \
         should both propagate; got errors:\n{}",
        errors_summary(&errors)
    );
}

/// Mixed: symbolic dim from `shape(x, 0)` plus a plain literal. The
/// symbolic dim propagates; the literal stays a `Dim::Lit`.
#[test]
fn issue_206_mixed_symbolic_and_literal_dim() {
    let errors = typecheck_core(include_str!(
        "fixtures/core/tests/issue_206_runtime_dim_reshape/issue_206_mixed_symbolic_and_literal_dim.0.deep"
    ));
    assert!(
        errors.is_empty(),
        "mixed symbolic+literal dim list should preserve the symbolic dim; \
         got errors:\n{}",
        errors_summary(&errors)
    );
}

/// Regression: all-literal dim list still type-checks. This was the
/// RT-A1W1 CRIT case (chelis#35) and should not regress.
#[test]
fn issue_206_all_literal_dims_still_work() {
    let errors = typecheck_core(include_str!(
        "fixtures/core/tests/issue_206_runtime_dim_reshape/issue_206_all_literal_dims_still_work.0.deep"
    ));
    assert!(
        errors.is_empty(),
        "all-literal dim list should keep type-checking; got errors:\n{}",
        errors_summary(&errors)
    );
}

/// Negative: `cast(shape(y, ...), i64)` where `y` is a DIFFERENT
/// tensor (not the input being reshaped). The recognizer requires the
/// shape source to match the reshape input expression's bound name, so
/// the cross-tensor pattern must fall back to `Dim::Wildcard` for that
/// element.
///
/// `Dim::Wildcard` is intentionally permissive (`unify_dim(Wildcard,
/// _)` always succeeds without binding, see `unify.rs`), so the
/// observable effect is "no error" -- which is also what we want.
/// The risk this test guards against is the recognizer mis-firing and
/// stamping `y`'s dim (`Var(d_m)`) into `x`'s reshape result while the
/// sig declares `Var(d_n)`. That would unify `d_n := d_m` and trip the
/// post-body rigidity check (two distinct declared dim vars resolving
/// together), producing a `DimensionMismatch`. So the assertion is:
/// no `DimensionMismatch` -- recognizer correctly fell back.
#[test]
fn issue_206_shape_of_other_tensor_does_not_propagate() {
    let errors = typecheck_core(include_str!(
        "fixtures/core/tests/issue_206_runtime_dim_reshape/issue_206_shape_of_other_tensor_does_not_propagate.0.deep"
    ));
    let has_dim_mismatch = errors.iter().any(|e| {
        matches!(
            e.kind,
            chelis_types::errors::CheckErrorKind::DimensionMismatch
        )
    });
    assert!(
        !has_dim_mismatch,
        "cross_dim must not collapse the sig's `n` and `m` dim params -- \
         the recognizer must require the shape source to be the same \
         input tensor being reshaped (var name match), otherwise it must \
         fall back to Dim::Wildcard. Got DimensionMismatch errors:\n{}",
        errors_summary(&errors)
    );
}

/// Negative: arbitrary expression in the dim list (e.g. `cast(add(...),
/// i64)`) is not a recognized symbolic-dim source -- the inner shape
/// call is wrapped in `add`, so the outer cast peel does not find a
/// direct `shape(input, lit_axis)`. The recognizer falls back to
/// `Dim::Wildcard`.
///
/// Similar to the cross-tensor case, `Wildcard` is permissive so the
/// observable effect is no error. The risk guarded against is the
/// recognizer firing too eagerly through arithmetic wrappers and
/// returning a dim that does not match what `add(shape(x, 0), 1)`
/// actually computes at runtime (which would be a soundness bug).
#[test]
fn issue_206_unrecognized_dim_expression_falls_back_to_wildcard() {
    let errors = typecheck_core(include_str!(
        "fixtures/core/tests/issue_206_runtime_dim_reshape/issue_206_unrecognized_dim_expression_falls_back_to_wildcard.0.deep"
    ));
    // The arithmetic-wrapped shape source is NOT a recognized symbolic
    // dim. Recognizer must fall back to Wildcard, which means no
    // dim-related diagnostic surfaces. This locks the recognizer's
    // syntactic-pattern-only contract.
    let has_dim_mismatch = errors.iter().any(|e| {
        matches!(
            e.kind,
            chelis_types::errors::CheckErrorKind::DimensionMismatch
        )
    });
    assert!(
        !has_dim_mismatch,
        "arithmetic-wrapped shape source must fall back to Wildcard, \
         not invent a propagated dim; got DimensionMismatch errors:\n{}",
        errors_summary(&errors)
    );
}

/// Recognizer specificity: when the reshape input is itself a non-`var`
/// expression (e.g. inlined function call), there is no input var name
/// to match against the inner `shape(...)`'s tensor arg, so the
/// recognizer must fall back. Body still typechecks via the polymorphic
/// fresh ret-var unifying with the declared sig.
#[test]
fn issue_206_non_var_reshape_input_falls_back_safely() {
    let errors = typecheck_core(include_str!(
        "fixtures/core/tests/issue_206_runtime_dim_reshape/issue_206_non_var_reshape_input_falls_back_safely.0.deep"
    ));
    // Inner reshape's input is a `var x` -- recognizer fires, body type
    // `tensor[n, 4, f32]`. Outer reshape's input is the inner reshape's
    // result (an `(app reshape ...)`, not a `var`), but the inner
    // `shape(x, ...)` argument is still `x`, which is NOT bound to the
    // outer reshape input's name. The recognizer must NOT match across
    // that boundary, and falls back to Wildcard for the outer reshape
    // -- the polymorphic ret-var unifies with declared `tensor[n, 4,
    // f32]` and the def type-checks. This locks the no-misfire path.
    assert!(
        errors.is_empty(),
        "nested reshape with non-`var` outer input should still \
         type-check (recognizer falls back to Wildcard safely); got \
         errors:\n{}",
        errors_summary(&errors)
    );
}

/// Recognizer doesn't accidentally propagate when reshape input has the
/// same name as something else in scope. With the only `x` in scope
/// being the reshape input, the var-name guard succeeds. This positive
/// case anchors what the negatives are guarding against -- the very
/// common "shape source is the input" pattern from the issue.
#[test]
fn issue_206_var_name_match_is_load_path_for_propagation() {
    let errors = typecheck_core(include_str!(
        "fixtures/core/tests/issue_206_runtime_dim_reshape/issue_206_var_name_match_is_load_path_for_propagation.0.deep"
    ));
    assert!(
        errors.is_empty(),
        "named symbolic dim `batch` should propagate via the recognizer; \
         got errors:\n{}",
        errors_summary(&errors)
    );
}

/// Cross-program inspection: when the recognizer fires correctly, the
/// CheckedProgram should record a sensible type_env entry for the def
/// (this also locks that fix-time changes don't silently drop the def
/// from the type_env -- the API surface downstream passes rely on).
#[test]
fn issue_206_propagated_def_appears_in_type_env() {
    let checked = typecheck_core_program(
        include_str!("fixtures/core/tests/issue_206_runtime_dim_reshape/issue_206_propagated_def_appears_in_type_env.0.deep"),
    )
    .expect("flatten_batch must type-check");
    let env = checked.type_env();
    assert!(
        env.contains_key("flatten_batch"),
        "type_env must surface the def name after the fix; type_env_string \
         debug = {}",
        type_env_string(&checked, "flatten_batch")
    );
}
