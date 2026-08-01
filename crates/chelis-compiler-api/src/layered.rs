//! Layered `chelis check` / `chelis build` front-ends that consume the
//! cross-process chelis-std typecheck cache.
//!
//! The monolithic path runs full Hindley-Milner inference over the
//! entire merged program (chelis-std + the user package + the entry
//! file) on every invocation. These functions instead:
//!
//! 1. resolve the bundled chelis-std sub-context once via
//!    [`crate::stdlib_cache::load_or_build_stdlib_context`] (a
//!    cross-process disk-cache hit on the warm path), then
//! 2. check / lower only the non-chelis-std decls `_with_context`
//!    against that cached sub-context.
//!
//! ## Byte-identical contract
//!
//! The acceptance oracle
//! (`crates/chelis-cli/tests/stdlib_typecheck_cache_oracle.rs`) requires
//! that the layered path produce byte-identical CLI output to the
//! monolithic path. Two design choices keep that true:
//!
//! - **Whole-program report reconstitution.** A `chelis check` JSON
//!   report carries whole-program `total_nodes` / `typed_nodes` /
//!   `structure`. The layered path reconstitutes the checker-visit counters
//!   from the two checked products, and computes structure from the cached
//!   chelis-std structural stats plus the fresh non-chelis-std stats. Both
//!   metrics are additive across the partition (chelis#973).
//! - **Monolithic fallback on the error path.** When the non-chelis-std
//!   decls do not type-check clean, [`check_layered`] returns
//!   `Ok(None)` and the caller falls back to the monolithic checker.
//!   Error-path reports carry whole-program node accounting and
//!   error-specific spans/scores that are not worth reconstructing
//!   piecewise; the monolithic path already produces them byte-for-byte.
//!   The cache optimizes the clean hot path, which is the ~200-worker
//!   workload; the error path stays correct by deferring to the
//!   existing checker.

use chelis_effects::EffectError;
use chelis_types::errors::CheckError;
use chelis_types::{CheckedProgram, FitnessReport, InferStats, StructuralStats};

use crate::compiler::CompilerError;
use crate::stdlib_cache::{StdLibContext, load_or_build_stdlib_context};

/// The layered-check result for a clean or semantically rejected program.
///
/// One result cannot contain both effect and linearity errors:
///
/// ```compile_fail
/// use chelis_compiler_api::LayeredCheck;
///
/// let _ = LayeredCheck::EffectRejected {
///     fitness: todo!(),
///     effect_errors: Vec::new(),
///     linearity_errors: Vec::new(),
///     typed_program: todo!(),
/// };
/// ```
pub enum LayeredCheck {
    Clean {
        fitness: FitnessReport,
        typed_program: CheckedProgram,
    },
    EffectRejected {
        fitness: FitnessReport,
        effect_errors: Vec<EffectError>,
        typed_program: CheckedProgram,
    },
    LinearityRejected {
        fitness: FitnessReport,
        linearity_errors: Vec<CheckError>,
        typed_program: CheckedProgram,
    },
}

/// Run the layered `chelis check` front-end.
///
/// Returns:
/// - `Ok(Some(LayeredCheck))` — the program type-checks clean; the
///   caller emits the reconstituted whole-program report.
/// - `Ok(None)` — the non-chelis-std decls do NOT type-check clean. The
///   caller must fall back to the monolithic checker so the error-path
///   report stays byte-identical. This is not an error condition; it is
///   the documented error-path handoff.
/// - `Err(CompilerError)` — building the chelis-std sub-context itself
///   failed (a chelis-std regression). Surfaced, never swallowed.
///
/// `stdlib_decls` / `non_stdlib_decls` are the partitioned, linked,
/// internal-name-rewritten decl lists from
/// `chelis_reef::PreparedProgram`.
pub fn check_layered(
    stdlib_decls: &[chelis_surf::ast::Decl],
    non_stdlib_decls: &[chelis_surf::ast::Decl],
) -> Result<Option<LayeredCheck>, CompilerError> {
    // RFC v5 (RT-1 F2 bypass): both `stdlib_decls` and `non_stdlib_decls`
    // are reef-linker output (internal-name-mangled), so the reserved
    // linker-name rejection must be off for this check.
    let _linked = chelis_types::install_linked_program_guard();
    let stdlib_ctx = load_or_build_stdlib_context(stdlib_decls)?;

    // Desugar + macro-expand the non-chelis-std decls. A macro-expansion
    // failure here is a real front-end error, not a clean miss — but the
    // monolithic path would surface it too, so hand back `Ok(None)` and
    // let the monolithic path produce the byte-identical diagnostic.
    let prepared = match crate::pipeline::prepare_surf_decls(non_stdlib_decls, None) {
        Ok(prepared) => prepared,
        Err(_) => return Ok(None),
    };

    // Type-check the non-chelis-std decls `_with_context` against the
    // cached chelis-std sub-context. A type error => `Ok(None)` =>
    // monolithic fallback for the byte-identical error report.
    let analysis = match crate::pipeline::analyze_prepared_with_context(
        prepared,
        &stdlib_ctx.type_env,
        stdlib_ctx.library_checked.signature_inference(),
    ) {
        Ok(analysis) => analysis,
        Err(_) => return Ok(None),
    };
    let fitness = reconstitute_clean_fitness(
        &stdlib_ctx,
        analysis.prepared().expanded_deep(),
        analysis.program(),
    );
    let typed_program = analysis.program().clone();

    // Effects + linearity use the canonical context-aware stage order.
    match crate::pipeline::complete_checks(
        analysis,
        crate::pipeline::SemanticContext::Library(&stdlib_ctx.library_checked),
    ) {
        Ok(_) => Ok(Some(LayeredCheck::Clean {
            fitness,
            typed_program: pick_typed_program(&stdlib_ctx, typed_program),
        })),
        Err(crate::pipeline::SemanticRejection::Effects {
            errors: effect_errors,
        }) => Ok(Some(LayeredCheck::EffectRejected {
            fitness,
            effect_errors,
            typed_program: pick_typed_program(&stdlib_ctx, typed_program),
        })),
        Err(crate::pipeline::SemanticRejection::Linearity {
            errors: linearity_errors,
        }) => Ok(Some(LayeredCheck::LinearityRejected {
            fitness,
            linearity_errors,
            typed_program: pick_typed_program(&stdlib_ctx, typed_program),
        })),
    }
}

/// Pick the `CheckedProgram` the CLI reads for `--show-inferred`: the
/// `_with_context`-checked non-chelis-std program, unless there are no
/// non-chelis-std decls (a chelis-std file), in which case the cached
/// chelis-std library `CheckedProgram` is the whole program.
fn pick_typed_program(
    stdlib_ctx: &StdLibContext,
    non_stdlib_checked: CheckedProgram,
) -> CheckedProgram {
    if non_stdlib_checked.exprs().is_empty() {
        stdlib_ctx.library_checked.clone()
    } else {
        non_stdlib_checked
    }
}

/// Reconstitute the whole-program fitness report for a clean program by
/// adding the cached chelis-std structural stats to the freshly-computed
/// non-chelis-std stats.
///
/// On the clean path the monolithic `check_ir_fitness` returns the inference
/// product's honest checker-visit counters. Those counters are stored on each
/// `CheckedProgram` and add across the stdlib / non-stdlib partition. The
/// `structure` component remains a distinct structural-AST metric computed
/// from the cached and fresh structural stats (chelis#973).
fn reconstitute_clean_fitness(
    stdlib_ctx: &StdLibContext,
    non_stdlib_deep: &[chelis_deep::Expr],
    non_stdlib_checked: &CheckedProgram,
) -> FitnessReport {
    let stdlib_structural = stdlib_ctx.structural_stats;
    let non_stdlib_structural = chelis_types::structural_stats(non_stdlib_deep);
    let structural = StructuralStats {
        total_nodes: stdlib_structural.total_nodes + non_stdlib_structural.total_nodes,
        invalid_nodes: stdlib_structural.invalid_nodes + non_stdlib_structural.invalid_nodes,
    };

    let stdlib_infer = stdlib_ctx.library_checked.infer_stats();
    let non_stdlib_infer = non_stdlib_checked.infer_stats();
    let infer = InferStats {
        typed_nodes: stdlib_infer.typed_nodes + non_stdlib_infer.typed_nodes,
        total_nodes: stdlib_infer.total_nodes + non_stdlib_infer.total_nodes,
    };

    chelis_types::clean_fitness_from_stats(structural, infer)
}

/// The cached chelis-std structural stats — exposed so the CLI's `build`
/// path can reconstitute whole-program accounting the same way.
pub fn stdlib_structural_stats(
    stdlib_decls: &[chelis_surf::ast::Decl],
) -> Result<StructuralStats, CompilerError> {
    // RFC v5: chelis-std decls are reef-linker output.
    let _linked = chelis_types::install_linked_program_guard();
    Ok(load_or_build_stdlib_context(stdlib_decls)?.structural_stats)
}

/// Run the layered `chelis build` type-check stage.
///
/// `chelis build` consumes ONE whole-program `CheckedProgram` (with
/// effects + linearity applied) for its lowering pipeline. This function
/// produces that program without re-inferring chelis-std: it checks the
/// non-chelis-std decls + entry `_with_context` against the cached
/// chelis-std sub-context, then [`CheckedProgram::compose`]s the result
/// with the cached chelis-std `library_checked`.
///
/// Returns:
/// - `Ok(Some(checked))` — the whole program passes all checks. `checked`
///   is the composed whole-program state that the caller lowers.
/// - `Ok(None)` — the non-chelis-std decls do NOT type-check clean (type,
///   effect, linearity, or macro error). The caller falls back to the
///   monolithic `checked_program_with_effects` so the error path stays
///   byte-identical.
/// - `Err(CompilerError)` — building the chelis-std sub-context failed.
///
/// `merged_non_stdlib_decls` is `non_stdlib_decls` from the partitioned
/// `PreparedProgram` — the linked, internal-name-rewritten decls for the
/// user package's own modules plus path-deps, which already includes the
/// entry module's decls.
pub fn check_layered_for_build(
    stdlib_decls: &[chelis_surf::ast::Decl],
    merged_non_stdlib_decls: &[chelis_surf::ast::Decl],
) -> Result<Option<crate::pipeline::CheckedCompilation>, CompilerError> {
    // RFC v5 (RT-1 F2 bypass): linked decls; accept the linker name format.
    let _linked = chelis_types::install_linked_program_guard();
    let stdlib_ctx = load_or_build_stdlib_context(stdlib_decls)?;

    let prepared = match crate::pipeline::prepare_surf_decls(merged_non_stdlib_decls, None) {
        Ok(prepared) => prepared,
        Err(_) => return Ok(None),
    };

    let analysis = match crate::pipeline::analyze_prepared_with_context(
        prepared,
        &stdlib_ctx.type_env,
        stdlib_ctx.library_checked.signature_inference(),
    ) {
        Ok(analysis) => analysis,
        Err(_) => return Ok(None),
    };

    let checked = match crate::pipeline::complete_checks(
        analysis,
        crate::pipeline::SemanticContext::Library(&stdlib_ctx.library_checked),
    ) {
        Ok(checked) => checked,
        Err(crate::pipeline::SemanticRejection::Effects { .. })
        | Err(crate::pipeline::SemanticRejection::Linearity { .. }) => return Ok(None),
    };

    // Compose the cached chelis-std half with the checked non-chelis-std
    // half. The result remains a typed pipeline state for the build lower step.
    Ok(Some(crate::pipeline::compose_checked(
        &stdlib_ctx.library_checked,
        checked,
    )))
}

#[cfg(test)]
mod artifact_outcome_tests {
    use super::*;

    fn check(source: &str) -> LayeredCheck {
        let decls = chelis_surf::parser::parse_str(source).expect("Surf parse");
        check_layered(&[], &decls)
            .expect("empty library context")
            .expect("the fixture must pass type analysis")
    }

    #[test]
    fn layered_outcomes_are_exclusive_for_each_semantic_stage() {
        let clean = check("def identity(x: tensor[n, f32]) -> tensor[n, f32] = x\n");
        assert!(matches!(clean, LayeredCheck::Clean { .. }));

        let effect =
            check("def noisy(x: tensor[4, f32]) -> tensor[4, f32] ! { } = dropout(x, 0.5)\n");
        assert!(matches!(
            effect,
            LayeredCheck::EffectRejected { effect_errors, .. } if !effect_errors.is_empty()
        ));

        let linearity = check(
            "def broken(x: tensor[4, f32]) -> tensor[4, f32] = { y = realize(x); add(x, y) }\n",
        );
        assert!(matches!(
            linearity,
            LayeredCheck::LinearityRejected {
                linearity_errors,
                ..
            } if !linearity_errors.is_empty()
        ));
    }
}
