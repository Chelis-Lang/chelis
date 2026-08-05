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

/// Emit a one-line reason when the layered build cache bails to the monolithic
/// path, under `CHELIS_PROFILE_COMPILE_CONTEXT=1`. Some bail conditions are
/// silent by default and leave an otherwise-VALID build permanently slower
/// (a dependency prefix that does not check standalone, or a macro cross-talk
/// digest mismatch), so this makes "why did my build get slow" diagnosable
/// (chelis#1176 review F3).
fn profile_context_bail(reason: &str) {
    if std::env::var_os("CHELIS_PROFILE_COMPILE_CONTEXT").map(|v| v == "1") == Some(true) {
        eprintln!("chelis: build-lane library cache bailed to monolithic ({reason})");
    }
}

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

/// Run the layered `chelis build` type-check stage over THREE cache
/// layers (chelis#1168).
///
/// `chelis build` consumes ONE whole-program `CheckedProgram` (with
/// effects + linearity applied) for its lowering pipeline. This function
/// produces that program without re-inferring chelis-std OR the dependency
/// (shell) library on every call:
///
/// 1. resolve the cached chelis-std sub-context ([`StdLibContext`],
///    Layer 1);
/// 2. resolve the cached dependency sub-context
///    ([`crate::library_cache::LibraryContext`], Layer 2) — the composed
///    `chelis-std ++ dependencies` state, content-addressed on the
///    dependency decls folded with the stdlib key, so an entry-only edit
///    warm-hits it;
/// 3. analyze the entry `_with_context` against that composed library and
///    [`CheckedProgram::compose`] the result — the only re-inferred layer.
///
/// The three decl slices are the caller's build-lane partition (see
/// `chelis_reef::PreparedProgram::dependency_entry_partition`):
/// `dependency_decls` is the stable prefix of the non-chelis-std decls and
/// `entry_decls` is the volatile suffix (the entry module onward). Their
/// concatenation equals the old `non_stdlib_decls`, so the composed
/// `stdlib ++ dependency ++ entry` reconstructs the same whole program the
/// two-layer path composed — byte-identical to the monolithic path, per
/// the acceptance oracle. A dep-free package passes an empty
/// `dependency_decls` and takes the exact two-layer path unchanged.
///
/// Returns:
/// - `Ok(Some(checked))` — the whole program passes all checks. `checked`
///   is the composed whole-program state that the caller lowers.
/// - `Ok(None)` — a dependency or entry decl does NOT type-check clean
///   (type, effect, linearity, or macro error). The caller falls back to
///   the monolithic `checked_program_with_effects` so the error path stays
///   byte-identical.
/// - `Err(CompilerError)` — building the chelis-std sub-context failed.
pub fn check_layered_for_build(
    stdlib_decls: &[chelis_surf::ast::Decl],
    dependency_decls: &[chelis_surf::ast::Decl],
    entry_decls: &[chelis_surf::ast::Decl],
) -> Result<Option<crate::pipeline::CheckedCompilation>, CompilerError> {
    // RFC v5 (RT-1 F2 bypass): linked decls; accept the linker name format.
    let _linked = chelis_types::install_linked_program_guard();
    let stdlib_ctx = load_or_build_stdlib_context(stdlib_decls)?;

    // Dep-free package: no middle layer to amortize. Expand the whole
    // non-chelis-std program (which is exactly `entry_decls` here) and
    // check it against the chelis-std sub-context — the pre-chelis#1168
    // two-layer path, byte-for-byte.
    if dependency_decls.is_empty() {
        let prepared = match crate::pipeline::prepare_surf_decls(entry_decls, None) {
            Ok(prepared) => prepared,
            Err(_) => return Ok(None),
        };
        return check_entry_against_library(
            &stdlib_ctx.type_env,
            &stdlib_ctx.library_checked,
            prepared,
        );
    }

    // Layer 2: the cached dependency sub-context. `Ok(None)` (the deps did
    // not compose cleanly) falls back to the monolithic path for the
    // byte-identical error report.
    let stdlib_key = crate::stdlib_cache::stdlib_cache_key(stdlib_decls);
    let library_ctx = match crate::library_cache::load_or_build_library_context(
        &stdlib_ctx,
        stdlib_key,
        dependency_decls,
    )? {
        Some(ctx) => ctx,
        None => {
            // The dependency prefix did not check standalone (e.g. a non-entry
            // module references the entry module's own defs). The cache cannot
            // engage; the caller re-checks monolithically. This is silent by
            // default and leaves a valid build permanently slower, so surface
            // the reason under the compile-context profile.
            profile_context_bail("dependency prefix did not check standalone");
            return Ok(None);
        }
    };

    // CRITICAL (chelis#1168 macro hygiene): macro expansion is stateful —
    // the hygiene counter and the expansion budget are per-`expand_program`
    // call. Expanding the dependency prefix and the entry separately would
    // restart both, shifting the entry's hygienic binder names relative to
    // the monolithic path (which expands the whole program once) and
    // silently changing the emitted C. So expand `dependencies ++ entry` as
    // ONE unit — identical to the monolithic non-chelis-std expansion,
    // since chelis-std mints no macros — then split the expanded Deep at
    // the dependency boundary. Only the entry suffix is re-analyzed.
    let mut combined_decls = Vec::with_capacity(dependency_decls.len() + entry_decls.len());
    combined_decls.extend_from_slice(dependency_decls);
    combined_decls.extend_from_slice(entry_decls);
    let combined_deep = match crate::pipeline::prepare_surf_decls(&combined_decls, None) {
        Ok(prepared) => prepared.into_expanded_deep(),
        Err(_) => return Ok(None),
    };

    let split = library_ctx.dependency_expanded_len;
    if split > combined_deep.len() {
        // The cached boundary is inconsistent with this expansion (should
        // not happen for a matching key); fall back rather than mis-split.
        // Same class as the digest bail below: a cache-consistency fall-back on
        // an otherwise-fine program, silent by default, so surface it under the
        // profile (chelis#1176 review G2).
        profile_context_bail("cached dependency expansion boundary is out of range");
        return Ok(None);
    }
    // Defence-in-depth guard: the dependency prefix of THIS combined expansion
    // must be byte-identical to the expansion the cached context was
    // type-checked from; if not, the cached context is invalid for this program
    // and we fall back.
    //
    // The modeled cross-talk case — the entry redefining a macro a dependency
    // invokes — cannot actually arise on the build lane: reef's `rewrite_decl`
    // module-qualifies every `MacroDef` name (`internal_name(package, module,
    // name)`), so no two modules can bind the same macro name and the entry
    // cannot shadow a dependency's macro. No input has been constructed that
    // trips this digest from `check_layered_for_build`. The guard is kept as
    // defence-in-depth because `check_layered_for_build` is `pub`: a future
    // non-linker caller could feed un-mangled decls and reintroduce the hazard
    // (chelis#1176 review G1).
    if crate::library_cache::expanded_deep_digest(&combined_deep[..split])
        != library_ctx.dependency_deep_digest
    {
        profile_context_bail(
            "dependency expansion digest changed (cached context invalid for this program)",
        );
        return Ok(None);
    }

    // The entry suffix carries the continued hygiene counter, so it matches
    // the monolithic path's entry annotations exactly.
    let entry_deep = combined_deep[split..].to_vec();
    check_entry_against_library(
        &library_ctx.type_env,
        &library_ctx.library_checked,
        crate::pipeline::prepare_deep(entry_deep, None),
    )
}

/// Analyze an already-expanded entry program against an outer library
/// context (`type_env` + `library_checked`) and compose the whole-program
/// checked state. Any type / effect / linearity rejection returns
/// `Ok(None)` for the monolithic fallback. Shared by the dep-free two-layer
/// path and the three-layer path so both compose identically.
fn check_entry_against_library(
    outer_type_env: &chelis_types::TypeEnv,
    outer_library_checked: &CheckedProgram,
    entry: crate::pipeline::PreparedProgram,
) -> Result<Option<crate::pipeline::CheckedCompilation>, CompilerError> {
    let analysis = match crate::pipeline::analyze_prepared_with_context(
        entry,
        outer_type_env,
        outer_library_checked.signature_inference(),
    ) {
        Ok(analysis) => analysis,
        Err(_) => return Ok(None),
    };

    let checked = match crate::pipeline::complete_checks(
        analysis,
        crate::pipeline::SemanticContext::Library(outer_library_checked),
    ) {
        Ok(checked) => checked,
        Err(crate::pipeline::SemanticRejection::Effects { .. })
        | Err(crate::pipeline::SemanticRejection::Linearity { .. }) => return Ok(None),
    };

    // Compose the library half with the checked entry half. The result
    // remains a typed pipeline state for the build lower step.
    Ok(Some(crate::pipeline::compose_checked(
        outer_library_checked,
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
            "def broken(x: tensor[4, f32]) -> tensor[4, f32] = {\n  y = realize(x)\n  add(x, y)\n}\n",
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

#[cfg(test)]
mod build_layering_tests {
    //! Byte-identity of the three-layer build composition (chelis#1168),
    //! pinned hermetically at the composition seam (no disk). The
    //! end-to-end disk-cached cold/warm build byte-identity lives in the
    //! CLI acceptance oracle; these tests pin the pure composition the
    //! oracle's `monolithic_vs_incontext_build` property depends on, at the
    //! newly-added dependency boundary.

    use crate::library_cache::{build_library_context, expanded_deep_digest};
    use crate::stdlib_cache::build_stdlib_context;
    use chelis_types::CheckedProgram;

    fn parse(src: &str) -> Vec<chelis_surf::ast::Decl> {
        chelis_surf::parser::parse_str(src).expect("Surf parse")
    }

    /// Order-independent semantic equality of two checked programs.
    ///
    /// `CheckedProgram` carries `HashMap`/`HashSet`-backed state
    /// (`type_env`, the ADT registry, `library_def_names`) whose bincode
    /// serialization order is nondeterministic, so a raw byte compare is
    /// meaningless. This compares the substantive typed program: the
    /// annotated exprs (an ordered `Vec`, and the lowering input), the type
    /// environment (a `HashMap`, compared as a set by `PartialEq`), and the
    /// inferred signatures. The definitive whole-output byte-identity is
    /// pinned end-to-end by the CLI acceptance oracle over real C build
    /// stdout, which is invariant to the benign map ordering.
    fn checked_semantically_eq(a: &CheckedProgram, b: &CheckedProgram) -> bool {
        a.annotated_exprs() == b.annotated_exprs()
            && a.type_env() == b.type_env()
            && a.signature_inference() == b.signature_inference()
            && a.linearity() == b.linearity()
    }

    /// Compose `stdlib | deps | entry` the way `check_layered_for_build`
    /// does (without the disk cache): expand `deps ++ entry` as ONE unit,
    /// split the expanded Deep at the dependency boundary, and check the
    /// entry suffix against the composed library. This mirrors the real
    /// hygiene-safe path — a manual per-slice expansion here would hide the
    /// exact macro-hygiene bug these tests must catch.
    fn three_layer_program(
        deps: &[chelis_surf::ast::Decl],
        entry: &[chelis_surf::ast::Decl],
    ) -> chelis_types::CheckedProgram {
        let stdlib_ctx = build_stdlib_context(&[]).expect("empty stdlib context");

        if deps.is_empty() {
            let prepared = crate::pipeline::prepare_surf_decls(entry, None).expect("prepare entry");
            return check_entry(&stdlib_ctx.type_env, &stdlib_ctx.library_checked, prepared);
        }

        let library_ctx = build_library_context(&stdlib_ctx, deps)
            .expect("build library context")
            .expect("deps compose cleanly");

        let mut combined = deps.to_vec();
        combined.extend(entry.iter().cloned());
        let combined_deep = crate::pipeline::prepare_surf_decls(&combined, None)
            .expect("prepare combined")
            .into_expanded_deep();
        let split = library_ctx.dependency_expanded_len;
        // The dependency prefix of the combined expansion must match the
        // cached dependency expansion (digest guard in the real path).
        assert_eq!(
            expanded_deep_digest(&combined_deep[..split]),
            library_ctx.dependency_deep_digest,
            "dependency prefix digest must match the cached context"
        );
        let entry_deep = combined_deep[split..].to_vec();
        check_entry(
            &library_ctx.type_env,
            &library_ctx.library_checked,
            crate::pipeline::prepare_deep(entry_deep, None),
        )
    }

    fn check_entry(
        outer_type_env: &chelis_types::TypeEnv,
        outer_library: &CheckedProgram,
        entry: crate::pipeline::PreparedProgram,
    ) -> CheckedProgram {
        let analysis = crate::pipeline::analyze_prepared_with_context(
            entry,
            outer_type_env,
            outer_library.signature_inference(),
        )
        .expect("entry analyzes against composed library");
        let checked = crate::pipeline::complete_checks(
            analysis,
            crate::pipeline::SemanticContext::Library(outer_library),
        )
        .expect("entry completes checks");
        crate::pipeline::compose_checked(outer_library, checked)
            .program()
            .clone()
    }

    /// The monolithic isolated check over `deps ++ entry`, matching the
    /// CLI's monolithic build fallback.
    fn monolithic_program(
        deps: &[chelis_surf::ast::Decl],
        entry: &[chelis_surf::ast::Decl],
    ) -> chelis_types::CheckedProgram {
        let mut merged = deps.to_vec();
        merged.extend(entry.iter().cloned());
        let prepared = crate::pipeline::prepare_surf_decls(&merged, None).expect("prepare merged");
        let analysis = match crate::pipeline::analyze_prepared(prepared) {
            crate::pipeline::PreparedTypeAnalysisOutcome::Accepted(analysis) => *analysis,
            crate::pipeline::PreparedTypeAnalysisOutcome::Rejected { .. } => {
                panic!("monolithic analysis rejected the fixture")
            }
        };
        crate::pipeline::complete_checks(analysis, crate::pipeline::SemanticContext::Isolated)
            .expect("monolithic completes checks")
            .program()
            .clone()
    }

    /// PARITY: the composed (layered) whole-program checked state is
    /// byte-identical to the monolithic checked state for the same program
    /// that imports a dependency. A divergence here is a compiler
    /// correctness bug in the dependency-layer split, not a cache bug.
    #[test]
    fn three_layer_compose_matches_monolithic_with_dependency() {
        // `dep_double` stands in for a dependency-library def; `main_value`
        // (the entry) references it. Flat top-level defs so no cross-module
        // linking is required for the raw (unlinked) test decls.
        let deps = parse("def dep_double(x: int32) -> int32 = add(x, x)\n");
        let entry = parse("def main_value -> int32 = dep_double(cast(21, int32))\n");

        let layered = three_layer_program(&deps, &entry);
        let monolithic = monolithic_program(&deps, &entry);
        assert!(
            checked_semantically_eq(&layered, &monolithic),
            "the stdlib|deps|entry composition must match the monolithic check \
             over deps ++ entry (annotated program + type env + signatures)"
        );
    }

    /// PARITY (entry-only change reflected): with the dependency layer
    /// fixed, changing ONLY the entry still composes to exactly the
    /// monolithic result for the changed program — so a warm dependency
    /// cache hit never masks an entry edit.
    #[test]
    fn entry_change_still_matches_monolithic() {
        let deps = parse("def dep_double(x: int32) -> int32 = add(x, x)\n");
        let entry_a = parse("def main_value -> int32 = dep_double(cast(21, int32))\n");
        let entry_b = parse("def main_value -> int32 = dep_double(cast(100, int32))\n");

        assert!(checked_semantically_eq(
            &three_layer_program(&deps, &entry_a),
            &monolithic_program(&deps, &entry_a),
        ));
        assert!(checked_semantically_eq(
            &three_layer_program(&deps, &entry_b),
            &monolithic_program(&deps, &entry_b),
        ));
        // And the two entry variants really are distinct programs.
        assert!(
            !checked_semantically_eq(
                &three_layer_program(&deps, &entry_a),
                &three_layer_program(&deps, &entry_b),
            ),
            "distinct entries must produce distinct checked programs"
        );
    }

    /// The empty-dependency branch (a dep-free package) composes to exactly
    /// the monolithic result too — the two-layer path is preserved
    /// unchanged when there is no dependency layer.
    #[test]
    fn empty_dependency_branch_matches_monolithic() {
        let deps: Vec<chelis_surf::ast::Decl> = Vec::new();
        let entry = parse("def main_value -> int32 = add(cast(1, int32), cast(2, int32))\n");
        assert!(checked_semantically_eq(
            &three_layer_program(&deps, &entry),
            &monolithic_program(&deps, &entry),
        ));
    }

    /// PARITY (macro hygiene, chelis#1168 regression): a binder-minting
    /// macro in the dependency advances the shared hygiene counter, so the
    /// entry's hygienic binder names must continue from where the
    /// dependency left off — exactly as the monolithic single-pass
    /// expansion produces them. A per-slice expansion would restart the
    /// counter and silently rename the entry's binders (miscompile). The
    /// combined-expand-and-split path must stay byte-identical here.
    #[test]
    fn macro_hygiene_across_dependency_boundary_matches_monolithic() {
        let deps = parse(
            "macro dmk(a) = {\n  q = a\n  add(q, q)\n}\n\
             def dep_val(x: int32) -> int32 = dmk(x)\n",
        );
        // The entry both invokes its own binder-minting macro AND binds a
        // name (`v_macro_0`) that a restarted hygiene counter would collide
        // with — the exact capture the reviewer's fixture exhibited.
        let entry = parse(
            "macro emk(a) = {\n  v = cast(7, int32)\n  add(v, a)\n}\n\
             def main_value -> int32 = {\n  v_macro_0 = dep_val(cast(5, int32))\n  emk(v_macro_0)\n}\n",
        );

        let layered = three_layer_program(&deps, &entry);
        let monolithic = monolithic_program(&deps, &entry);
        assert!(
            checked_semantically_eq(&layered, &monolithic),
            "macro hygiene must be preserved across the dependency/entry split"
        );
    }

    /// NEGATIVE: dependency decls that do not type-check compose to
    /// `Ok(None)` (the monolithic-fallback handoff), never a partial or
    /// wrong context.
    #[test]
    fn rejected_dependency_falls_back() {
        let stdlib_ctx = build_stdlib_context(&[]).expect("empty stdlib context");
        // `no_such_builtin` is unbound: the dependency does not compose.
        let bad_deps = parse("def broken(x: int32) -> int32 = no_such_builtin(x)\n");
        let built = build_library_context(&stdlib_ctx, &bad_deps).expect("build returns Ok");
        assert!(
            built.is_none(),
            "a rejected dependency must fold to Ok(None), not a partial context"
        );
    }
}
