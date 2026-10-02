//! Declared-result extent checks for the diagonal operation after application unification.

use super::*;

/// The declared extent at one axis of an application's result, read from the
/// two ingresses a declaration can reach a builtin call through.
///
/// The `def`/`sig` route hands the declared return type down as
/// `expected_result`. A block-scoped ascription (`y: tensor[9, f32] = ...`)
/// does not: `infer_let` infers the right-hand side with no expectation and
/// unifies afterwards, so the ascription arrives as `"type"` metadata on this
/// very application node (`crates/chelis-surf/src/desugar.rs`'s
/// `inject_type_metadata`). Resolving it here costs one speculative
/// `resolve_deep_type`, whose diagnostics are rolled back because `infer_let`
/// owns reporting for that same annotation and would otherwise report twice.
/// Going through the resolver rather than reading the metadata shape directly
/// is what makes a declared type alias (`type Row = tensor[9, f32]`) carry the
/// same verdict as its expansion.
#[allow(clippy::too_many_arguments)]
fn declared_result_literal(
    result_axis: usize,
    result_rank: usize,
    expected_result: Option<&Type>,
    node: &DeepNode,
    env: &Env,
    vg: &mut VarGen,
    adt_reg: &AdtRegistry,
    subst: &Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Option<i64> {
    // The rank-equality precondition matters as much as the literal does. A
    // declaration whose RANK disagrees is already a signature mismatch, and the
    // checker reports it with the two full tensor types. Reading an axis out of
    // it anyway turns `tensor[n, 4, 5] -> tensor[7]` into "the result extent is
    // at most 4", which names a repair that would not fix the program, and the
    // `Type::Error` this path returns then suppresses the accurate diagnostic
    // (round 1 P2). Rank belongs to unification; the bound stands down.
    let literal_at = |ty: &Type| -> Option<i64> {
        let Type::Tensor(dims, _) = ty else {
            return None;
        };
        if dims.len() != result_rank {
            return None;
        }
        subst.observe_dim(dims.get(result_axis)?).literal_extent()
    };

    if let Some(found) = expected_result
        .map(|expected| subst.apply(expected))
        .as_ref()
        .and_then(&literal_at)
    {
        return Some(found);
    }

    let declared = node.meta().ty().map(|value| value.expression())?;
    let checkpoint = errors.checkpoint();
    let resolved = resolve_deep_type(
        declared,
        vg,
        adt_reg,
        TypeUseSite::Annotation,
        annotation_binder_mode(env),
        errors,
    );
    errors.retain_since(checkpoint, |_| false);
    literal_at(&resolved.ok()?)
}

/// chelis#1739. `[05-OP-33]` replaces the retained axis extent with the smaller
/// selected extent, so a literal selected axis bounds the result from above for
/// every runtime value of the other axis. A declared extent strictly greater
/// than that bound is unreachable and is a `DimensionMismatch`; at or below it
/// is satisfiable and the runtime guard owns the verdict.
///
/// Returns the error type when it rejects, so the caller propagates it instead
/// of the inferred result. `Type::Error` unifies with anything, which is what
/// keeps the enclosing signature or ascription from reporting a second time.
#[allow(clippy::too_many_arguments)]
pub(super) fn reject_unreachable_diagonal_extent(
    operand: &Type,
    inferred: &Type,
    axis1: usize,
    axis2: usize,
    node: &DeepNode,
    site: CheckSite<'_>,
    env: &Env,
    vg: &mut VarGen,
    adt_reg: &AdtRegistry,
    subst: &Subst,
    errors: &mut DiagnosticSink<'_>,
    expected_result: Option<&Type>,
) -> Option<Type> {
    let (result_axis, source_axis, bound) = diagonal_result_bound(operand, axis1, axis2, subst)?;
    let Type::Tensor(inferred_dims, _) = inferred else {
        return None;
    };
    let declared = declared_result_literal(
        result_axis,
        inferred_dims.len(),
        expected_result,
        node,
        env,
        vg,
        adt_reg,
        subst,
        errors,
    )?;
    if declared <= bound {
        return None;
    }
    let expected = format!("extent at most {bound}");
    let got = format!("declared extent {declared}");
    Some(report_at_check_site(
        errors,
        CheckError::with_types(
            CheckErrorKind::DimensionMismatch,
            with_node_provenance(
                node,
                format!(
                    "diagonal argument 1, axis {source_axis}: expected {expected}, got {got}; \
                     the result declares the smaller selected extent ([05-OP-33])"
                ),
            ),
            expected,
            got,
            vec![format!(
                "Declare an extent at or below {bound}, or select an axis pair \
                 whose literal extent is at least {declared}"
            )],
        ),
        site,
    ))
}
