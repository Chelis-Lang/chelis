//! Fragment-scoped validation of a single function-body replacement.
//!
//! The Deep authoring loop replaces one function body at a time. Re-running
//! the whole-program `chelis check` pipeline on every keystroke would be
//! wasteful, so this module checks the spliced body against a held context
//! built once from the rest of the module. The keystone invariant is that the
//! fragment verdict MUST AGREE with full `chelis check` of the rewritten
//! module: a fragment check that accepts a body full check would reject is the
//! cardinal failure. The agreement is locked by the differential gate at
//! `tests/fragment_parity.rs`.
//!
//! ## Fragment-scoped vs whole-module checks
//!
//! Three passes are fragment-scoped, checking only the spliced body against the
//! held context: the type pass, the effect pass, and the linearity pass. One
//! check is whole-module and cannot be fragment-scoped: the structural fitness
//! pass `chelis_types::check_ir_fitness`, which runs the cross-def detectors
//! `detect_trivial_non_terminating_fns` (a base-case-free recursion group) and
//! `detect_top_level_binding_cycles` (a value-binding cycle). A recursion group
//! spans defs the 2-decl fragment never sees, so a sibling holding the sole
//! base case could be spliced away undetected by the fragment-scoped passes
//! while full `chelis check` rejects the rewritten module. To keep the verdict
//! in agreement, [`check_body_replacement`] runs `check_ir_fitness` over the
//! full rewritten module after the fragment-scoped passes accept. This re-does
//! whole-module type/structural work; incremental whole-module fitness is a
//! future concern, not an L0 correctness requirement.
//!
//! ## The seam
//!
//! For a parsed Deep module and a target function:
//!
//! - the *held context* is the module with the target's `(def ...)` node
//!   removed and its `(defsig ...)` RETAINED (see
//!   [`chelis_deep::module_excluding_function_def`]), so a sibling that calls
//!   the target still resolves against the target's declared signature;
//! - the *fragment* is a two-decl mini-program `[ target (defsig ...),
//!   (def <name> <new body>) ]`: the target's signature is carried INTO the
//!   fragment, adjacent to the spliced def (see [`chelis_deep::function_defsig`]
//!   and [`chelis_deep::spliced_function_def`]), and is checked as new code
//!   against the held context.
//!
//! Carrying the target's `defsig` into the fragment mirrors the whole-program
//! analysis: the expected return type derives from the defsig, the effect
//! declared-vs-inferred mismatch FIRES against the new body (catching a
//! declared-pure body that performs `Random`/`Io`), a recursive new body
//! resolves against the signature, and linearity is unchanged. The held context
//! keeps the same `defsig` so sibling calls into the target still resolve.
//! Together these match what whole-program `chelis check` does on the flattened
//! rewritten module, where every def is checked against its own defsig over one
//! flat decl set while every sibling call also resolves against that defsig.
//!
//! The held context is built ONCE per replacement
//! ([`build_compiled_library_context`] runs the per-decl inference and
//! annotation a single time and returns both the [`TypeEnv`] for the type
//! pass and the held library [`CheckedProgram`] for the effect and linearity
//! passes). The fragment then runs the three context-scoped passes in the
//! pinned order TYPE -> EFFECTS -> LINEARITY, mirroring
//! `compile_new_source_in_context`'s Phase C/D/E block minus the
//! reef-rewrite and lower stages. The held library is never re-type-checked
//! per pass.

use chelis_deep::Expr;
use chelis_effects::check_effects_with_context;
use chelis_types::{
    CheckedProgram, TypeEnv, build_compiled_library_context, check_ir_with_signature_context,
    check_linearity_with_context,
};

use crate::schema::Span;

/// The pass that rejected a fragment, with its human-readable diagnostics.
///
/// Each variant carries the concatenated `message` text the underlying pass
/// produced and an optional source `location`. The check-error variants also
/// carry a forward-compatible `deep_path` slot that L2 will populate with the
/// Deep address of the offending node; it is always `None` today and threading
/// provenance into it does not change the variant's other fields.
#[derive(Debug, Clone)]
pub enum ReplacementError {
    /// The target function could not be resolved by name in the module
    /// (no module, wrong module prefix, missing/ambiguous name, or a value
    /// binding rather than a function).
    NameResolution {
        message: String,
        location: Option<Span>,
    },
    /// The target function has no `(defsig ...)` declaration, so its
    /// signature is inferred from its body rather than declared. The fragment
    /// carries the target's `(defsig ...)` alongside the new `(def ...)` so a
    /// recursive new body resolves against the declared signature; an
    /// inferred-signature target has no `(defsig ...)` to move into the
    /// fragment, so a recursive new body would be checked against an absent
    /// signature, diverging from a full check that re-infers the signature
    /// from the new body. Rather than risk that silent fragment-vs-full
    /// divergence, the fragment check rejects inferred-signature targets up
    /// front. Rendered Deep (from `def f(x: T) -> U = ...`) always emits a
    /// `(defsig ...)`, so only hand-authored defsig-less defs reach this.
    UndeclaredSignature {
        message: String,
        location: Option<Span>,
    },
    /// The type pass (`check_ir_with_signature_context`) rejected the spliced
    /// body.
    Type {
        message: String,
        location: Option<Span>,
        /// Reserved for the L2 Deep-address of the offending node. Always
        /// `None` in L0; populating it later adds no new field.
        deep_path: Option<DeepErrorPath>,
    },
    /// The effect pass (`check_effects_with_context`) rejected the spliced
    /// body.
    Effect {
        message: String,
        location: Option<Span>,
        /// Reserved for the L2 Deep-address of the offending node. Always
        /// `None` in L0; populating it later adds no new field.
        deep_path: Option<DeepErrorPath>,
    },
    /// The linearity pass (`check_linearity_with_context`) rejected the
    /// spliced body.
    Linearity {
        message: String,
        location: Option<Span>,
        /// Reserved for the L2 Deep-address of the offending node. Always
        /// `None` in L0; populating it later adds no new field.
        deep_path: Option<DeepErrorPath>,
    },
}

impl ReplacementError {
    /// The pass that produced this error, as a stable lowercase tag matching
    /// the `stage` strings used elsewhere in the compiler API (`check`,
    /// `effects`, `linearity`) plus `name-resolution` for the resolve miss.
    pub fn stage(&self) -> &'static str {
        match self {
            ReplacementError::NameResolution { .. } => "name-resolution",
            ReplacementError::UndeclaredSignature { .. } => "undeclared-signature",
            ReplacementError::Type { .. } => "check",
            ReplacementError::Effect { .. } => "effects",
            ReplacementError::Linearity { .. } => "linearity",
        }
    }

    /// The diagnostic text the underlying pass produced.
    pub fn message(&self) -> &str {
        match self {
            ReplacementError::NameResolution { message, .. }
            | ReplacementError::UndeclaredSignature { message, .. }
            | ReplacementError::Type { message, .. }
            | ReplacementError::Effect { message, .. }
            | ReplacementError::Linearity { message, .. } => message,
        }
    }
}

/// Forward-compatible Deep-address slot for L2 provenance. L0 never
/// constructs one; the type exists so adding provenance threading later does
/// not change [`ReplacementError`]'s shape. The inner address form is left
/// open (a [`chelis_deep::DeepPath`] plus the owning def name is the expected
/// payload) and is intentionally not wired today.
#[derive(Debug, Clone)]
pub struct DeepErrorPath {
    /// The qualified name of the def the address is relative to.
    pub def_qualified_name: String,
    /// The path from the def node to the offending subtree.
    pub path: chelis_deep::DeepPath,
}

/// A clean fragment check.
///
/// `rewritten_module` is the full module with the target body replaced (the
/// same program full `chelis check` would be run on). `checks_clean` is a
/// marker that all three context-scoped passes accepted the spliced body; it
/// is always `true` on a returned `Ok` and exists so a consumer can assert
/// the success path without inspecting the absence of an error.
#[derive(Debug, Clone)]
pub struct ReplacementReport {
    /// The full rewritten module (held decls plus the spliced def), in
    /// canonical declaration order.
    pub rewritten_module: Vec<Expr>,
    /// Always `true`: the type, effect, and linearity passes all accepted the
    /// spliced body against the held context.
    pub checks_clean: bool,
}

/// The held context for a single function-body replacement, built once.
///
/// Holds the [`TypeEnv`] used by the type pass and the held library
/// [`CheckedProgram`] used by the effect and linearity passes. Both come from
/// a single [`build_compiled_library_context`] call over the held decls (the
/// module with the target `(def ...)` excluded, its `(defsig ...)` retained),
/// so the per-decl inference and annotation run exactly once.
pub struct HeldContext {
    type_env: TypeEnv,
    library_checked: CheckedProgram,
}

impl HeldContext {
    /// Build the held context from a held decl list. The held decls must be
    /// the module with the target function's `(def ...)` already removed; this
    /// runs the library half of the check pipeline once over them.
    pub fn build(held_decls: &[Expr]) -> Result<HeldContext, ReplacementError> {
        let (type_env, library_checked) = build_compiled_library_context(held_decls)
            .map_err(|report| infer_result_to_type_error(&report))?;
        Ok(HeldContext {
            type_env,
            library_checked,
        })
    }
}

/// Check a spliced fragment against an already-built held context, running the
/// three context-scoped passes in the pinned order TYPE -> EFFECTS ->
/// LINEARITY.
///
/// `frag_deep` is the fragment mini-program `[ target (defsig ...),
/// (def <name> <new body>) ]`: the target's signature travels with the new def
/// so the effect declared-vs-inferred check fires against the new body and a
/// recursive new body resolves against the signature, mirroring whole-program
/// `chelis check`, which checks every def against its own defsig over one flat
/// decl set.
///
/// This mirrors `compile_new_source_in_context`'s Phase C/D/E block: the type
/// pass uses the signature-context form (matching that path's borrow-arg
/// handling), and the effect and linearity passes run against the held library
/// `CheckedProgram`. The held context is consumed by reference only and is
/// never re-type-checked here.
pub fn check_fragment_def(
    context: &HeldContext,
    frag_deep: &[Expr],
) -> Result<(), ReplacementError> {
    // Phase C: type-check the spliced body against the held type env, using
    // the held library's signature inference for borrow-arg parity (the same
    // signature-context form `compile_new_source_in_context` uses).
    let new_typed = check_ir_with_signature_context(
        &context.type_env,
        context.library_checked.signature_inference(),
        frag_deep,
    )
    .map_err(|report| infer_result_to_type_error(&report))?;

    // Phase D: effects, held library + the spliced body.
    let new_effects =
        check_effects_with_context(&context.library_checked, &new_typed).map_err(|errors| {
            ReplacementError::Effect {
                message: join_messages(errors.iter().map(|error| error.message.as_str())),
                location: None,
                deep_path: None,
            }
        })?;

    // Phase E: linearity, held library + the spliced body.
    check_linearity_with_context(&context.library_checked, &new_effects).map_err(|errors| {
        ReplacementError::Linearity {
            message: join_messages(errors.iter().map(|error| error.message.as_str())),
            location: None,
            deep_path: None,
        }
    })?;

    Ok(())
}

/// Check replacing the body of `target_qualified_name` in `module` with
/// `new_body`, running TYPE + EFFECTS + LINEARITY scoped to the spliced body
/// against a held context built once from the rest of the module, then a
/// whole-module structural fitness gate over the full rewritten module.
///
/// This is the primary fragment-check surface. It resolves the target,
/// excludes its `(def ...)` node (retaining its `(defsig ...)`), builds the
/// held context once, splices `new_body`, and runs the three context-scoped
/// passes in order. The three context-scoped passes are fragment-scoped; the
/// final fitness gate (`chelis_types::check_ir_fitness`) is whole-module and
/// catches cross-def structural violations (base-case-free recursion groups,
/// value-binding cycles) the fragment-scoped passes cannot see. On success it
/// returns the full rewritten module; on rejection it returns a tagged
/// [`ReplacementError`] naming the first failing pass (a fitness rejection is
/// tagged `Type`).
///
/// The verdict MUST AGREE with full `chelis check` of the returned
/// `rewritten_module`. A disagreement is a bug in this function, not a case to
/// be papered over by weakening a pass.
pub fn check_body_replacement(
    module: &[Expr],
    target_qualified_name: &str,
    new_body: &Expr,
) -> Result<ReplacementReport, ReplacementError> {
    // Resolve first so a name miss is reported before any context work.
    chelis_deep::resolve_function(module, target_qualified_name)
        .map_err(resolve_error_to_replacement_error)?;

    // The fragment carries the target's `(defsig ...)` alongside the new
    // `(def ...)` so a recursive new body resolves against the declared
    // signature and the effect declared-vs-inferred check fires against it. A
    // target with no `(defsig ...)` has its signature inferred from the body,
    // so there is no signature to move into the fragment and a recursive new
    // body would be checked against an absent signature, diverging from a full
    // check that re-infers from the new body. Reject up front rather than risk
    // that silent fragment-vs-full divergence.
    if !chelis_deep::module_has_defsig_for(module, target_qualified_name) {
        return Err(ReplacementError::UndeclaredSignature {
            message: format!(
                "target `{target_qualified_name}` has no defsig declaration; fragment-scoped \
                 body replacement requires a declared signature for the target so a recursive \
                 body resolves against it, and inferred-signature targets are unsupported in \
                 this run"
            ),
            location: None,
        });
    }

    // Held decls: the module with the target's `(def ...)` removed and its
    // `(defsig ...)` RETAINED, so a sibling that calls the target still
    // resolves against the target's declared signature in the held context.
    let held_decls = chelis_deep::module_excluding_function_def(module, target_qualified_name)
        .map_err(resolve_error_to_replacement_error)?;

    // Build the held context ONCE (one inference + annotation pass over the
    // held decls); reused for all three fragment passes below.
    let context = HeldContext::build(&held_decls)?;

    // The fragment mini-program: the target's defsig followed by the spliced
    // def, checked as new code against the held context. Carrying the defsig
    // into the fragment is what makes the fragment's declared-vs-inferred
    // effect check fire against the new body (catching a declared-pure body
    // that performs Random/Io); the held context retains the same defsig so
    // sibling calls into the target still resolve. The defsig is present
    // because `module_has_defsig_for` returned true above.
    let frag_defsig =
        chelis_deep::function_defsig(module, target_qualified_name).ok_or_else(|| {
            ReplacementError::UndeclaredSignature {
                message: format!(
                    "target `{target_qualified_name}` has no defsig declaration to carry into the \
                 fragment"
                ),
                location: None,
            }
        })?;
    let frag_def =
        chelis_deep::spliced_function_def(module, target_qualified_name, new_body.clone())
            .map_err(resolve_error_to_replacement_error)?;
    let frag_deep = [frag_defsig, frag_def];

    check_fragment_def(&context, &frag_deep)?;

    // The full rewritten module the verdict is defined to agree with.
    let rewritten_module =
        chelis_deep::splice_function_body(module, target_qualified_name, new_body.clone())
            .map_err(resolve_error_to_replacement_error)?;

    // Whole-module structural fitness gate. The three passes above are
    // fragment-scoped (local type, effects, and linearity of the new body
    // against the held context), but `chelis check` also runs whole-module
    // structural detectors that are recursion-group properties, chiefly
    // `detect_trivial_non_terminating_fns` (a base-case-free self- or
    // mutual-recursion group) and `detect_top_level_binding_cycles` (a
    // value-binding cycle). A base case the held context still holds in a
    // SIBLING def cannot be seen by the 2-decl fragment, so splicing it away
    // would slip past the fragment-scoped passes while full `chelis check`
    // rejects the rewritten module. To keep the verdict in agreement, run the
    // same whole-module fitness pass `cmd_check_one_deep` runs first
    // (`chelis_types::check_ir_fitness`) over the rewritten module and reject
    // if it reports any error. This re-does whole-module type/structural work
    // rather than reusing the held context; that is acceptable for L0
    // correctness (incremental whole-module fitness is a future concern). The
    // fitness pass rejects a base-case-free recursion group promptly, before
    // the separate whole-module `check_typed_program` inference that can wedge
    // on such a module, so the tool path does not hang.
    let fitness = chelis_types::check_ir_fitness(&rewritten_module);
    if !fitness.errors.is_empty() {
        return Err(ReplacementError::Type {
            message: join_messages(fitness.errors.iter().map(|error| error.message.as_str())),
            location: None,
            deep_path: None,
        });
    }

    Ok(ReplacementReport {
        rewritten_module,
        checks_clean: true,
    })
}

/// Map a [`chelis_deep::ResolveError`] to a [`ReplacementError::NameResolution`].
fn resolve_error_to_replacement_error(error: chelis_deep::ResolveError) -> ReplacementError {
    ReplacementError::NameResolution {
        message: error.to_string(),
        location: None,
    }
}

/// Map a type-check [`chelis_types::InferResult`] failure to a
/// [`ReplacementError::Type`]. The underlying `CheckError`s carry no span
/// today, so `location` is `None`.
fn infer_result_to_type_error(report: &chelis_types::InferResult) -> ReplacementError {
    ReplacementError::Type {
        message: join_messages(report.errors.iter().map(|error| error.message.as_str())),
        location: None,
        deep_path: None,
    }
}

/// Join one-or-more diagnostic messages into a single string, one per line, so
/// a multi-error pass surfaces every message rather than only the first.
fn join_messages<'a>(messages: impl Iterator<Item = &'a str>) -> String {
    messages.collect::<Vec<_>>().join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_deep::Atom;

    /// Render Surf source to canonical Deep, as `chelis deep` does for `.ch`.
    fn render_deep(surf: &str) -> Vec<Expr> {
        let decls = chelis_surf::parser::parse_str(surf).expect("surf parse");
        let deep = chelis_surf::desugar::desugar_program(&decls);
        chelis_macros::expand_program(&deep, &chelis_macros::ExpansionOptions::default())
            .expect("macro expand")
            .into_exprs()
    }

    /// The body subtree of `function_name` in a single-function module.
    fn render_body(surf: &str, function_name: &str) -> Expr {
        let module = render_deep(surf);
        let resolved = chelis_deep::resolve_function(&module, function_name).expect("resolve");
        let def = module
            .iter()
            .find_map(|expr| {
                let Expr::List(list, _) = expr else {
                    return None;
                };
                let is_module = matches!(
                    list.elements.first(),
                    Some(Expr::Atom(Atom::Symbol(t), _)) if t == "module"
                );
                if !is_module {
                    return None;
                }
                list.elements.get(3 + resolved.decl_index)
            })
            .expect("def node")
            .clone();
        chelis_deep::function_body(&def).expect("body slot").clone()
    }

    const TWO_FN: &str = "module Frag.Two\nexport (f, g)\ndef f(x: f32) -> f32 = add(x, x)\ndef g(y: f32) -> f32 = mul(y, y)\n";

    /// The `(tag, name)` of each top-level decl inside the single module node.
    fn held_decl_tags(program: &[Expr]) -> Vec<(String, Option<String>)> {
        program
            .iter()
            .find_map(|expr| {
                let Expr::List(list, _) = expr else {
                    return None;
                };
                let is_module = matches!(
                    list.elements.first(),
                    Some(Expr::Atom(Atom::Symbol(t), _)) if t == "module"
                );
                is_module.then_some(&list.elements[3..])
            })
            .expect("module node")
            .iter()
            .filter_map(|decl| {
                let Expr::List(list, _) = decl else {
                    return None;
                };
                let tag = match list.elements.first() {
                    Some(Expr::Atom(Atom::Symbol(t), _)) => t.clone(),
                    _ => return None,
                };
                let name = match list.elements.get(2) {
                    Some(Expr::Atom(Atom::Symbol(s), _)) => Some(s.clone()),
                    _ => None,
                };
                Some((tag, name))
            })
            .collect()
    }

    #[test]
    fn held_context_excludes_target_def_keeps_defsig() {
        // The held decls produced for `f` must drop `f`'s def but RETAIN its
        // defsig, so a sibling that calls `f` still resolves against the
        // declared signature. `g` (defsig + def) is intact.
        let module = render_deep(TWO_FN);
        let held = chelis_deep::module_excluding_function_def(&module, "f").expect("exclude f");
        let tags = held_decl_tags(&held);
        // f's defsig is retained; f's def is gone; g (defsig + def) is intact.
        assert!(
            tags.contains(&("defsig".to_string(), Some("f".to_string()))),
            "f's defsig retained: {tags:?}"
        );
        assert!(
            !tags.contains(&("def".to_string(), Some("f".to_string()))),
            "f's def excluded: {tags:?}"
        );
        assert!(
            tags.contains(&("def".to_string(), Some("g".to_string()))),
            "g's def intact: {tags:?}"
        );
        // The held context builds cleanly (one inference pass over held decls).
        HeldContext::build(&held).expect("held context builds");
    }

    #[test]
    fn check_fragment_def_against_prebuilt_context_accepts_well_typed() {
        let module = render_deep(TWO_FN);
        let held = chelis_deep::module_excluding_function_def(&module, "f").expect("exclude f");
        let context = HeldContext::build(&held).expect("held context");
        let new_body = render_body("module M\ndef h(x: f32) -> f32 = sub(x, x)\n", "h");
        let frag_defsig = chelis_deep::function_defsig(&module, "f").expect("f defsig");
        let frag_def = chelis_deep::spliced_function_def(&module, "f", new_body).expect("splice");
        check_fragment_def(&context, &[frag_defsig, frag_def]).expect("well-typed body accepts");
    }

    #[test]
    fn check_body_replacement_returns_rewritten_module_on_accept() {
        let module = render_deep(TWO_FN);
        let new_body = render_body("module M\ndef h(x: f32) -> f32 = mul(x, x)\n", "h");
        let report = check_body_replacement(&module, "f", &new_body).expect("accept");
        assert!(report.checks_clean);
        // The rewritten module still resolves `f` and `g`.
        chelis_deep::resolve_function(&report.rewritten_module, "f").expect("f present");
        chelis_deep::resolve_function(&report.rewritten_module, "g").expect("g present");
    }

    #[test]
    fn type_error_body_is_tagged_type() {
        let module = render_deep(TWO_FN);
        // f64 result returned from an f32-declared function: precision mismatch.
        let new_body = render_body("module M\ndef h(x: f32) -> f32 = cast(x, f64)\n", "h");
        let err = check_body_replacement(&module, "f", &new_body).expect_err("type error");
        assert!(
            matches!(err, ReplacementError::Type { .. }),
            "expected Type, got {err:?}",
        );
        assert_eq!(err.stage(), "check");
    }

    #[test]
    fn unknown_target_is_tagged_name_resolution() {
        let module = render_deep(TWO_FN);
        let new_body = render_body("module M\ndef h(x: f32) -> f32 = x\n", "h");
        let err = check_body_replacement(&module, "absent", &new_body).expect_err("unknown target");
        assert!(
            matches!(err, ReplacementError::NameResolution { .. }),
            "expected NameResolution, got {err:?}",
        );
        assert_eq!(err.stage(), "name-resolution");
    }

    #[test]
    fn defsig_less_target_is_rejected_not_diverged() {
        // A hand-authored Deep module whose target `f` has a `(def ...)` but no
        // `(defsig ...)`: its signature is inferred from the body. The held
        // context retains only defsigs for the excluded target, so it would
        // carry no binding for `f` and a recursive new body would be checked
        // differently than a full re-infer-from-body check. The fragment check
        // must reject this with a clear UndeclaredSignature error rather than
        // silently accept or reject (the cardinal fragment-vs-full divergence).
        let module = chelis_deep::parser::parse_str(
            "(module {} m \
               (def {} f (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x))))",
        )
        .expect("deep module parse");
        // A well-typed identity body that would pass a full check; the point is
        // it never reaches a pass, because the missing defsig is rejected first.
        let new_body = render_body("module M\ndef h(x: f32) -> f32 = x\n", "h");
        let err = check_body_replacement(&module, "f", &new_body)
            .expect_err("defsig-less target rejected");
        assert!(
            matches!(err, ReplacementError::UndeclaredSignature { .. }),
            "expected UndeclaredSignature, got {err:?}",
        );
        assert_eq!(err.stage(), "undeclared-signature");
    }

    #[test]
    fn rendered_deep_target_always_has_defsig_and_accepts() {
        // Rendered Deep (from `def f(x: T) -> U = ...`) always emits a defsig,
        // so the defsig-less guard never fires on rendered modules: the same
        // identity replacement that the hand-authored defsig-less module
        // rejects is accepted here.
        let module = render_deep(TWO_FN);
        assert!(chelis_deep::module_has_defsig_for(&module, "f"));
        let new_body = render_body("module M\ndef h(x: f32) -> f32 = x\n", "h");
        let report = check_body_replacement(&module, "f", &new_body).expect("accept");
        assert!(report.checks_clean);
    }

    /// A `ping`/`pong` mutually-recursive pair where `ping` holds the sole base
    /// case. Splicing `ping`'s body to call `pong` unconditionally removes the
    /// only base case, closing a base-case-free recursion group that the
    /// whole-module `detect_trivial_non_terminating_fns` detector flags.
    const PINGPONG: &str = "module Frag.PingPong\nexport (ping, pong)\ndef ping(n: int32) -> int32 = if eq(n, 0) then 0 else pong(sub(n, 1))\ndef pong(n: int32) -> int32 = ping(sub(n, 1))\n";

    #[test]
    fn cross_def_base_case_drop_is_rejected_promptly() {
        // The fragment-scoped passes see only `[ping defsig, ping def]` and
        // would accept the base-case-free body; the whole-module fitness gate
        // catches it, matching full `chelis check`. The fitness pass rejects a
        // base-case-free recursion group fast (it precedes the whole-module
        // inference that can wedge on such a module), so this returns promptly.
        let module = render_deep(PINGPONG);
        let new_body = render_body(
            "module M\ndef f(n: int32) -> int32 = pong(sub(n, 1))\n",
            "f",
        );
        let start = std::time::Instant::now();
        let err = check_body_replacement(&module, "ping", &new_body)
            .expect_err("base-case drop must be rejected to agree with full check");
        let elapsed = start.elapsed();
        assert!(
            matches!(err, ReplacementError::Type { .. }),
            "expected Type (structural fitness) rejection, got {err:?}",
        );
        assert_eq!(err.stage(), "check");
        assert!(
            err.message().contains("trivially non-terminating"),
            "fitness diagnostic should name the non-termination cause: {}",
            err.message(),
        );
        // The path returns without wedging; the fitness pass precedes the
        // whole-module inference that can hang on base-case-free recursion.
        assert!(
            elapsed < std::time::Duration::from_secs(30),
            "fragment path should reject promptly, took {elapsed:?}",
        );
    }

    #[test]
    fn check_error_variants_have_none_deep_path_in_l0() {
        // L0 never populates the forward-compatible deep_path slot.
        let module = render_deep(TWO_FN);
        let new_body = render_body("module M\ndef h(x: f32) -> f32 = cast(x, f64)\n", "h");
        let err = check_body_replacement(&module, "f", &new_body).expect_err("type error");
        if let ReplacementError::Type { deep_path, .. } = err {
            assert!(deep_path.is_none(), "deep_path is None in L0");
        } else {
            panic!("expected Type variant, got {err:?}");
        }
    }
}
