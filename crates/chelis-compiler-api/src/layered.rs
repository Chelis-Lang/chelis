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
//!   `structure`. The layered path reconstitutes these by adding the
//!   cached chelis-std structural stats to the freshly-computed
//!   non-chelis-std stats. `count_nodes` and the Deep tag validator are
//!   pure per-expr structural walks, so for a clean program the sum is
//!   exact.
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
use chelis_types::fitness::FitnessComponents;
use chelis_types::{CheckedProgram, FitnessReport, StructuralStats};

use crate::compiler::CompilerError;
use crate::stdlib_cache::{StdLibContext, load_or_build_stdlib_context};

/// The layered-check result for a clean program: a whole-program
/// fitness report plus the checked non-chelis-std program (used by the
/// CLI for `--show-inferred` signature output).
pub struct LayeredCheck {
    /// Whole-program fitness report, reconstituted to be byte-identical
    /// to the monolithic `check_ir_fitness` output for a clean program.
    pub fitness: FitnessReport,
    /// Effect-checker errors over the non-chelis-std decls
    /// (`_with_context` against the cached chelis-std sub-context).
    /// Empty on the clean path.
    pub effect_errors: Vec<EffectError>,
    /// Linearity-checker errors over the non-chelis-std decls. Empty on
    /// the clean path.
    pub linearity_errors: Vec<CheckError>,
    /// The `CheckedProgram` whose `signature_inference()` the CLI reads
    /// for `--show-inferred`. For a non-chelis-std fixture this is the
    /// `_with_context`-checked non-chelis-std program; for a chelis-std
    /// file (no non-chelis-std decls) it is the cached chelis-std
    /// library `CheckedProgram`.
    pub typed_program: CheckedProgram,
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
    let stdlib_ctx = load_or_build_stdlib_context(stdlib_decls)?;

    // Desugar + macro-expand the non-chelis-std decls. A macro-expansion
    // failure here is a real front-end error, not a clean miss — but the
    // monolithic path would surface it too, so hand back `Ok(None)` and
    // let the monolithic path produce the byte-identical diagnostic.
    let non_stdlib_deep = match expand(non_stdlib_decls) {
        Ok(deep) => deep,
        Err(_) => return Ok(None),
    };

    // Type-check the non-chelis-std decls `_with_context` against the
    // cached chelis-std sub-context. A type error => `Ok(None)` =>
    // monolithic fallback for the byte-identical error report.
    let non_stdlib_checked = match chelis_types::check_ir_with_signature_context(
        &stdlib_ctx.type_env,
        stdlib_ctx.library_checked.signature_inference(),
        &non_stdlib_deep,
    ) {
        Ok(checked) => checked,
        Err(_) => return Ok(None),
    };

    // Effects + linearity, `_with_context` against the cached chelis-std
    // library `CheckedProgram`. chelis-std's own effects + linearity
    // were already checked when the sub-context was built.
    let non_stdlib_effects_checked = match chelis_effects::check_effects_with_context(
        &stdlib_ctx.library_checked,
        &non_stdlib_checked,
    ) {
        Ok(checked) => checked,
        Err(effect_errors) => {
            // Effect errors are reported by the CLI directly (they do
            // not block the JSON emission), so surface them rather than
            // falling back. The fitness report itself is still clean
            // (type inference succeeded); the CLI folds effect errors
            // into the score exactly as the monolithic path does.
            let fitness =
                reconstitute_clean_fitness(&stdlib_ctx, &non_stdlib_deep, &non_stdlib_checked);
            return Ok(Some(LayeredCheck {
                fitness,
                effect_errors,
                linearity_errors: Vec::new(),
                typed_program: pick_typed_program(&stdlib_ctx, non_stdlib_checked),
            }));
        }
    };

    let linearity_errors = match chelis_types::check_linearity_with_context(
        &stdlib_ctx.library_checked,
        &non_stdlib_effects_checked,
    ) {
        Ok(_) => Vec::new(),
        Err(errors) => errors,
    };

    let fitness = reconstitute_clean_fitness(&stdlib_ctx, &non_stdlib_deep, &non_stdlib_checked);
    Ok(Some(LayeredCheck {
        fitness,
        effect_errors: Vec::new(),
        linearity_errors,
        typed_program: pick_typed_program(&stdlib_ctx, non_stdlib_checked),
    }))
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
/// On the clean path the monolithic `check_ir_fitness` returns
/// `score: 1.0`, every component `1.0` except `structure`, and
/// `typed_nodes == total_nodes == count_nodes(whole_program)`. The
/// `structure` component is `valid_nodes / total_nodes` over the whole
/// program. Both `count_nodes` and the Deep validator are pure per-expr
/// walks, so the whole-program counts are the partition sums.
fn reconstitute_clean_fitness(
    stdlib_ctx: &StdLibContext,
    non_stdlib_deep: &[chelis_deep::Expr],
    non_stdlib_checked: &CheckedProgram,
) -> FitnessReport {
    let stdlib_stats = stdlib_ctx.structural_stats;
    let non_stdlib_stats = chelis_types::structural_stats(non_stdlib_deep);
    let total_nodes = stdlib_stats.total_nodes + non_stdlib_stats.total_nodes;
    let invalid_nodes = stdlib_stats.invalid_nodes + non_stdlib_stats.invalid_nodes;

    let structure = if total_nodes == 0 {
        1.0
    } else {
        let valid = total_nodes.saturating_sub(invalid_nodes);
        valid as f64 / total_nodes as f64
    };

    // Mirror `chelis_types::check_ir_fitness`'s clean-path return: parse
    // / names / types are 1.0, structure is the computed fraction, score
    // is the weighted sum. For a clean program with structure 1.0 this
    // is exactly `score: 1.0`, which the all-clean corpus exercises.
    const W_PARSE: f64 = 0.1;
    const W_STRUCTURE: f64 = 0.1;
    const W_NAMES: f64 = 0.2;
    const W_TYPES: f64 = 0.6;
    let score = W_PARSE * 1.0 + W_STRUCTURE * structure + W_NAMES * 1.0 + W_TYPES * 1.0;

    // `non_stdlib_checked` is not needed for the structural counts (they
    // come from the partition sums); the parameter keeps call sites
    // uniform with a possible future error-path variant.
    let _ = non_stdlib_checked;

    FitnessReport {
        score,
        components: FitnessComponents {
            parse: 1.0,
            structure,
            names: 1.0,
            types: 1.0,
        },
        errors: Vec::new(),
        typed_nodes: total_nodes,
        untyped_nodes: 0,
        total_nodes,
        unresolved_names: Vec::new(),
    }
}

fn expand(decls: &[chelis_surf::ast::Decl]) -> Result<Vec<chelis_deep::Expr>, String> {
    let desugared = chelis_surf::desugar::desugar_program(decls);
    chelis_macros::expand_program(&desugared, &chelis_macros::ExpansionOptions::default())
        .map(|expanded| expanded.into_exprs())
        .map_err(|err| err.to_string())
}

/// The cached chelis-std structural stats — exposed so the CLI's `build`
/// path can reconstitute whole-program accounting the same way.
pub fn stdlib_structural_stats(
    stdlib_decls: &[chelis_surf::ast::Decl],
) -> Result<StructuralStats, CompilerError> {
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
/// - `Ok(Some(checked))` — the whole program type-checks clean; `checked`
///   is the composed whole-program `CheckedProgram` the caller lowers.
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
) -> Result<Option<CheckedProgram>, CompilerError> {
    let stdlib_ctx = load_or_build_stdlib_context(stdlib_decls)?;

    let non_stdlib_deep = match expand(merged_non_stdlib_decls) {
        Ok(deep) => deep,
        Err(_) => return Ok(None),
    };

    let non_stdlib_checked = match chelis_types::check_ir_with_signature_context(
        &stdlib_ctx.type_env,
        stdlib_ctx.library_checked.signature_inference(),
        &non_stdlib_deep,
    ) {
        Ok(checked) => checked,
        Err(_) => return Ok(None),
    };

    let non_stdlib_effects_checked = match chelis_effects::check_effects_with_context(
        &stdlib_ctx.library_checked,
        &non_stdlib_checked,
    ) {
        Ok(checked) => checked,
        Err(_) => return Ok(None),
    };

    let non_stdlib_linearity_checked = match chelis_types::check_linearity_with_context(
        &stdlib_ctx.library_checked,
        &non_stdlib_effects_checked,
    ) {
        Ok(checked) => checked,
        Err(_) => return Ok(None),
    };

    // Compose the cached chelis-std half with the freshly-checked
    // non-chelis-std half into the one whole-program CheckedProgram the
    // `build` lowering pipeline expects.
    Ok(Some(CheckedProgram::compose(
        &stdlib_ctx.library_checked,
        &non_stdlib_linearity_checked,
    )))
}
