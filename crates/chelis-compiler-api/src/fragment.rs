//! Whole-module validation of a single function-body replacement.
//!
//! The Deep authoring loop replaces one function body at a time. The
//! authoritative question is: does the module obtained by splicing the new body
//! in still pass full `chelis check`? [`check_body_replacement`] answers it by
//! construction: it resolves the target, splices the new body, and then runs the
//! exact whole-module check pipeline `chelis check` runs on a `.dp` module. The
//! verdict therefore EQUALS full `chelis check` of the rewritten module; there
//! is no separate fragment-scoped analysis that could disagree with it.
//!
//! ## Why whole-module, and what the future optimization is
//!
//! Single-def-scoped validation is unsound for cross-def properties. Two of
//! them bite:
//!
//! - effect propagation to held callers: splicing a `Random`/`Io`-performing
//!   body into a function declared pure must be rejected because a sibling that
//!   calls it and is itself declared pure now violates its own declared purity.
//!   A def-local effect check never sees the sibling.
//! - recursion-group termination: a base-case-free recursion group is a property
//!   of the whole strongly-connected group, not of any single def. A sibling
//!   holding the sole base case can be spliced away while a def-local view of
//!   the rewritten def still looks fine.
//!
//! The deferred optimization is NOT "incremental validation is unsound" and NOT
//! "whole-module forever." Single-def scoping is unsound; CLOSURE-scoped
//! validation is sound and is the real later optimization: validate effects over
//! the caller-ward effect closure of the target (every def that transitively
//! holds the target), validate termination over the target's
//! strongly-connected component, and validate type per-def against its declared
//! signature. Until that closure machinery exists, the L0 verdict is the full
//! whole-module check of the rewritten module.
//!
//! ## The pipeline
//!
//! `check_body_replacement` runs the same four passes `cmd_check_one_deep` runs,
//! in the same pinned order, over the rewritten module, returning the first
//! failing pass as a tagged [`ReplacementError`]:
//!
//! 1. `chelis_types::check_ir_fitness` (whole-module structural/type fitness,
//!    including the cross-def `detect_trivial_non_terminating_fns` and
//!    `detect_top_level_binding_cycles` detectors) -> [`ReplacementError::Type`]
//! 2. `chelis_types::check_typed_program` (whole-module HM inference) ->
//!    [`ReplacementError::Type`]
//! 3. `chelis_effects::check_program` (effects; descends into the `(module ...)`
//!    wrapper the rewritten module carries) -> [`ReplacementError::Effect`]
//! 4. `chelis_types::check_linearity` -> [`ReplacementError::Linearity`]
//!
//! The rewritten module is `(module ...)`-wrapped (it comes from
//! [`chelis_deep::splice_function_body`]), so the effect pass MUST descend into
//! that wrapper; the whole-module effect validators do.
//!
//! `check_ir_fitness` runs first for the same reason `cmd_check_one_deep` runs
//! it first: it rejects a base-case-free recursion group promptly, before the
//! whole-module inference (`check_typed_program`) that can wedge on such a
//! module, so the tool path does not hang.

use chelis_deep::Expr;

use crate::schema::Span;

/// The pass that rejected a body replacement, with its human-readable
/// diagnostics.
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
    /// The fitness or type pass (`check_ir_fitness` or `check_typed_program`)
    /// rejected the rewritten module. This covers cross-def structural
    /// violations (a base-case-free recursion group, a value-binding cycle)
    /// the fitness pass detects, and any type error whole-module inference
    /// reports.
    Type {
        message: String,
        location: Option<Span>,
        /// Reserved for the L2 Deep-address of the offending node. Always
        /// `None` in L0; populating it later adds no new field.
        deep_path: Option<DeepErrorPath>,
    },
    /// The effect pass (`check_program`) rejected the rewritten module. This
    /// fires when the new body performs an effect (`Random`/`Io`) the target's
    /// declared signature does not permit, or when that effect propagates to a
    /// held caller declared not to perform it.
    Effect {
        message: String,
        location: Option<Span>,
        /// Reserved for the L2 Deep-address of the offending node. Always
        /// `None` in L0; populating it later adds no new field.
        deep_path: Option<DeepErrorPath>,
    },
    /// The linearity pass (`check_linearity`) rejected the rewritten module.
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
            ReplacementError::Type { .. } => "check",
            ReplacementError::Effect { .. } => "effects",
            ReplacementError::Linearity { .. } => "linearity",
        }
    }

    /// The diagnostic text the underlying pass produced.
    pub fn message(&self) -> &str {
        match self {
            ReplacementError::NameResolution { message, .. }
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

/// A clean body replacement.
///
/// `rewritten_module` is the full module with the target body replaced (the
/// same program full `chelis check` would be run on, and was run on, here).
/// `checks_clean` is a marker that the whole-module pipeline accepted; it is
/// always `true` on a returned `Ok` and exists so a consumer can assert the
/// success path without inspecting the absence of an error.
#[derive(Debug, Clone)]
pub struct ReplacementReport {
    /// The full rewritten module (held decls plus the spliced def), in
    /// canonical declaration order.
    pub rewritten_module: Vec<Expr>,
    /// Always `true`: full `chelis check` of `rewritten_module` accepted.
    pub checks_clean: bool,
}

/// Check replacing the body of `target_qualified_name` in `module` with
/// `new_body` by running full `chelis check` over the rewritten module.
///
/// This is the primary body-replacement surface. It resolves the target,
/// splices `new_body` into the module, and runs the same whole-module pipeline
/// `cmd_check_one_deep` runs, in the same pinned order:
/// `check_ir_fitness` -> `check_typed_program` -> `check_program` (effects) ->
/// `check_linearity`. On success it returns the full rewritten module; on
/// rejection it returns a tagged [`ReplacementError`] naming the first failing
/// pass (a fitness or type rejection is tagged `Type`).
///
/// The verdict EQUALS full `chelis check` of the returned `rewritten_module`
/// BY CONSTRUCTION: there is no separate scoped analysis. Closure-scoped
/// validation is the future optimization; see the module docs.
pub fn check_body_replacement(
    module: &[Expr],
    target_qualified_name: &str,
    new_body: &Expr,
) -> Result<ReplacementReport, ReplacementError> {
    // Resolve first so a name miss is reported before any check work.
    chelis_deep::resolve_function(module, target_qualified_name)
        .map_err(resolve_error_to_replacement_error)?;

    // The full rewritten module the verdict is defined to equal. It is
    // `(module ...)`-wrapped, so the whole-module passes below (the effect
    // pass in particular) must descend into the wrapper.
    let rewritten_module =
        chelis_deep::splice_function_body(module, target_qualified_name, new_body.clone())
            .map_err(resolve_error_to_replacement_error)?;

    // Whole-module check pipeline, in the same order `cmd_check_one_deep` runs
    // it. Return the FIRST failing pass as a tagged error.
    //
    // Pass 1: structural/type fitness. Runs first because it rejects a
    // base-case-free recursion group promptly, before the whole-module
    // inference that can wedge on such a module.
    let fitness = chelis_types::check_ir_fitness(&rewritten_module);
    if !fitness.errors.is_empty() {
        return Err(ReplacementError::Type {
            message: join_messages(fitness.errors.iter().map(|error| error.message.as_str())),
            location: None,
            deep_path: None,
        });
    }

    // Pass 2: whole-module HM type inference.
    let typed = chelis_types::check_typed_program(&rewritten_module)
        .map_err(|report| infer_result_to_type_error(&report))?;

    // Pass 3: effects. The validators descend into the `(module ...)` wrapper,
    // so a declared-pure body that performs `Random`/`Io` is rejected, and so
    // is an effect that propagates to a held caller declared not to perform it.
    let effected =
        chelis_effects::check_program(&typed).map_err(|errors| ReplacementError::Effect {
            message: join_messages(errors.iter().map(|error| error.message.as_str())),
            location: None,
            deep_path: None,
        })?;

    // Pass 4: linearity.
    chelis_types::check_linearity(&effected).map_err(|errors| ReplacementError::Linearity {
        message: join_messages(errors.iter().map(|error| error.message.as_str())),
        location: None,
        deep_path: None,
    })?;

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
    fn defsig_less_target_is_accepted_under_whole_module_check() {
        // A hand-authored Deep module whose target `f` has a `(def ...)` but no
        // `(defsig ...)`: its signature is inferred from the body. Under
        // whole-module routing this is no longer a divergence case: full
        // `chelis check` re-infers `f`'s signature from the rewritten body, and
        // a well-typed identity body checks clean. No special rejection is
        // needed; the tool's verdict equals full check by construction.
        let module = chelis_deep::parser::parse_str(
            "(module {} m \
               (def {} f (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x))))",
        )
        .expect("deep module parse");
        let new_body = render_body("module M\ndef h(x: f32) -> f32 = x\n", "h");
        let report = check_body_replacement(&module, "f", &new_body)
            .expect("defsig-less target checks clean under whole-module routing");
        assert!(report.checks_clean);
    }

    #[test]
    fn rendered_deep_target_accepts() {
        // Rendered Deep (from `def f(x: T) -> U = ...`) always emits a defsig;
        // an identity replacement checks clean.
        let module = render_deep(TWO_FN);
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
        // Whole-module fitness over the rewritten module catches the
        // base-case-free recursion group the 2-decl-local view never sees,
        // matching full `chelis check`. The fitness pass runs first, before the
        // whole-module inference that can wedge on such a module, so this
        // returns promptly.
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
            "body-replacement path should reject promptly, took {elapsed:?}",
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
