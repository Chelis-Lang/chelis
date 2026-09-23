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
//! Single-def-scoped validation is unsound for cross-def properties. The
//! clearest example is:
//!
//! - effect propagation to held callers: splicing a `Random`/`Io`-performing
//!   body into a function declared pure must be rejected because a sibling that
//!   calls it and is itself declared pure now violates its own declared purity.
//!   A def-local effect check never sees the sibling.
//!
//! The deferred optimization is NOT "incremental validation is unsound" and NOT
//! "whole-module forever." Single-def scoping is unsound; CLOSURE-scoped
//! validation is sound and is the real later optimization: validate effects over
//! the caller-ward effect closure of the target (every def that transitively
//! holds the target), retain the declarations needed to resolve sibling calls,
//! and validate type per-def against its declared signature. Until that closure
//! machinery exists, the L0 verdict is the full whole-module check of the
//! rewritten module.
//!
//! Uniform recursion without a syntactic base case is checker-legal under
//! [04-INF-2]/[04-INF-3]. Whether a backend can represent or lower a particular
//! recursive program is a separate capability boundary owned by chelis#730.
//!
//! ## The pipeline
//!
//! `check_body_replacement` delegates to the canonical compiler-API pipeline.
//! The pipeline runs these stages in a fixed order:
//!
//! 1. Combined type analysis produces fitness and one typed program.
//! 2. The effect check returns [`ReplacementError::Effect`] on rejection.
//! 3. The linearity check returns [`ReplacementError::Linearity`] on rejection.
//!
//! Type analysis returns [`ReplacementError::Type`] on rejection. It also
//! rejects top-level binding cycles before later stages start.
//!
//! The rewritten module contains a `(module ...)` wrapper. The shared effect
//! transition checks declarations inside that wrapper.
//!
//! Callers cannot construct a false validation proof:
//!
//! ```compile_fail
//! use chelis_compiler_api::ValidatedModule;
//!
//! let _ = ValidatedModule(Vec::new());
//! ```

use chelis_deep::Expr;

use crate::schema::DiagnosticSpan;

/// A whole-module edit rejected by the compiler-owned validation pipeline.
///
/// Edit tools may do structural prechecks (parse the request, resolve an
/// insertion target, locate a body slot), but `ok:true` is reserved for this
/// pipeline accepting the full rewritten module.
#[derive(Debug, Clone)]
pub enum EditValidationError {
    /// The fitness or type pass (`check_ir_fitness` or `check_typed_program`)
    /// rejected the rewritten module.
    Type {
        message: String,
        location: Option<DiagnosticSpan>,
        /// Reserved for the L2 Deep-address of the offending node. Always
        /// `None` in L0.
        deep_path: Option<DeepErrorPath>,
    },
    /// The effect pass (`check_program`) rejected the rewritten module.
    Effect {
        message: String,
        location: Option<DiagnosticSpan>,
        /// Reserved for the L2 Deep-address of the offending node. Always
        /// `None` in L0.
        deep_path: Option<DeepErrorPath>,
    },
    /// The linearity pass (`check_linearity`) rejected the rewritten module.
    Linearity {
        message: String,
        location: Option<DiagnosticSpan>,
        /// Reserved for the L2 Deep-address of the offending node. Always
        /// `None` in L0.
        deep_path: Option<DeepErrorPath>,
    },
}

impl EditValidationError {
    /// The pass that produced this error, as a stable lowercase stage tag.
    pub fn stage(&self) -> &'static str {
        match self {
            EditValidationError::Type { .. } => "check",
            EditValidationError::Effect { .. } => "effects",
            EditValidationError::Linearity { .. } => "linearity",
        }
    }

    /// The diagnostic text the underlying pass produced.
    pub fn message(&self) -> &str {
        match self {
            EditValidationError::Type { message, .. }
            | EditValidationError::Effect { message, .. }
            | EditValidationError::Linearity { message, .. } => message,
        }
    }
}

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
        location: Option<DiagnosticSpan>,
    },
    /// The fitness or type pass (`check_ir_fitness` or `check_typed_program`)
    /// rejected the rewritten module. This covers structural violations such
    /// as a value-binding cycle that the fitness pass detects, and any type
    /// error whole-module inference reports. Base-case-free uniform recursion
    /// is not a type error; unsupported lowering remains a chelis#730 boundary.
    Type {
        message: String,
        location: Option<DiagnosticSpan>,
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
        location: Option<DiagnosticSpan>,
        /// Reserved for the L2 Deep-address of the offending node. Always
        /// `None` in L0; populating it later adds no new field.
        deep_path: Option<DeepErrorPath>,
    },
    /// The linearity pass (`check_linearity`) rejected the rewritten module.
    Linearity {
        message: String,
        location: Option<DiagnosticSpan>,
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

/// A Deep module that passed the complete compiler-owned edit check.
///
/// Only [`check_whole_module_edit`] constructs this proof. Callers can inspect
/// or consume the checked expressions, but cannot attach proof to unchecked expressions.
#[derive(Debug, Clone)]
pub struct ValidatedModule(Vec<Expr>);

impl ValidatedModule {
    pub fn as_exprs(&self) -> &[Expr] {
        &self.0
    }

    pub fn into_exprs(self) -> Vec<Expr> {
        self.0
    }
}

/// A clean body replacement with proof for the complete rewritten module.
#[derive(Debug, Clone)]
pub struct ReplacementReport {
    pub validated_module: ValidatedModule,
}

/// Run the compiler-owned whole-module validation pipeline over an edited Deep
/// module.
///
/// The shared transition runs type analysis, effects, and linearity in order.
/// The first failed stage returns a tagged error.
pub fn check_whole_module_edit(
    rewritten_module: Vec<Expr>,
) -> Result<ValidatedModule, EditValidationError> {
    let prepared = crate::pipeline::prepare_deep(rewritten_module.clone(), None);
    let analysis = match crate::pipeline::analyze_prepared(prepared) {
        crate::pipeline::PreparedTypeAnalysisOutcome::Accepted(analysis) => *analysis,
        crate::pipeline::PreparedTypeAnalysisOutcome::Rejected { fitness } => {
            return Err(EditValidationError::Type {
                message: join_messages(fitness.errors.iter().map(|error| error.message.as_str())),
                location: None,
                deep_path: None,
            });
        }
    };

    crate::pipeline::complete_checks(analysis, crate::pipeline::SemanticContext::Isolated)
        .map_err(|rejection| match rejection {
            crate::pipeline::SemanticRejection::Effects { errors } => EditValidationError::Effect {
                message: join_messages(errors.iter().map(|error| error.message.as_str())),
                location: None,
                deep_path: None,
            },
            crate::pipeline::SemanticRejection::Linearity { errors } => {
                EditValidationError::Linearity {
                    message: join_messages(errors.iter().map(|error| error.message.as_str())),
                    location: None,
                    deep_path: None,
                }
            }
        })?;

    Ok(ValidatedModule(rewritten_module))
}

/// Check replacing the body of `target_qualified_name` in `module` with
/// `new_body` by running full `chelis check` over the rewritten module.
///
/// This is the primary body-replacement surface. It resolves the target,
/// splices `new_body` into the module, and runs the same whole-module pipeline
/// `cmd_check_one_deep` runs. On success, it returns the full rewritten module.
/// On rejection, it returns a tagged [`ReplacementError`] for the first failed
/// stage. A type-analysis rejection uses the `Type` tag.
///
/// The verdict EQUALS full `chelis check` of the returned validation proof
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

    let validated_module =
        check_whole_module_edit(rewritten_module).map_err(edit_error_to_replacement_error)?;

    Ok(ReplacementReport { validated_module })
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
fn edit_error_to_replacement_error(error: EditValidationError) -> ReplacementError {
    match error {
        EditValidationError::Type {
            message,
            location,
            deep_path,
        } => ReplacementError::Type {
            message,
            location,
            deep_path,
        },
        EditValidationError::Effect {
            message,
            location,
            deep_path,
        } => ReplacementError::Effect {
            message,
            location,
            deep_path,
        },
        EditValidationError::Linearity {
            message,
            location,
            deep_path,
        } => ReplacementError::Linearity {
            message,
            location,
            deep_path,
        },
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
    use chelis_deep::DeepTag;

    /// Render Surf source to canonical Deep, as `chelis deep` does for `.ch`.
    fn render_deep(surf: &str) -> Vec<Expr> {
        let decls = chelis_surf::parser::parse_str(surf).expect("surf parse");
        let deep =
            chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");
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
                let Expr::Node(node, _) = expr else {
                    return None;
                };
                if node.tag() != DeepTag::Module {
                    return None;
                }
                // Child 0 is the module name; declarations follow it.
                node.children_slice().get(1 + resolved.decl_index)
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
        let rewritten_module = report.validated_module.as_exprs();
        // The rewritten module still resolves `f` and `g`.
        chelis_deep::resolve_function(rewritten_module, "f").expect("f present");
        chelis_deep::resolve_function(rewritten_module, "g").expect("g present");
    }

    #[test]
    fn whole_module_accept_returns_the_checked_expressions() {
        let rewritten_module = render_deep(TWO_FN);
        let validated = check_whole_module_edit(rewritten_module.clone()).expect("accept");
        assert_eq!(validated.as_exprs(), rewritten_module.as_slice());
        assert_eq!(validated.into_exprs(), rewritten_module);
    }

    #[test]
    fn whole_module_rejections_return_no_validation_proof() {
        let fixtures = [
            ("module M\ndef broken() -> i32 = missing\n", "check"),
            (
                "module M\ndef noisy(x: tensor[4, f32]) -> tensor[4, f32] ! { } = dropout(x, 0.5)\n",
                "effects",
            ),
            (
                "module M\ndef broken(x: tensor[4, f32]) -> tensor[4, f32] = {\n  y = realize(x)\n  add(x, y)\n}\n",
                "linearity",
            ),
        ];

        for (source, expected_stage) in fixtures {
            let error = check_whole_module_edit(render_deep(source))
                .expect_err("the rejected module must not return a validation proof");
            assert_eq!(error.stage(), expected_stage);
        }
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
        let module = chelis_deep::parse_and_stamp_file(
            "(module {} m \
               (def {} f (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x))))",
        )
        .expect("deep module parse");
        let new_body = render_body("module M\ndef h(x: f32) -> f32 = x\n", "h");
        let report = check_body_replacement(&module, "f", &new_body)
            .expect("defsig-less target checks clean under whole-module routing");
        assert!(!report.validated_module.as_exprs().is_empty());
    }

    #[test]
    fn rendered_deep_target_accepts() {
        // Rendered Deep (from `def f(x: T) -> U = ...`) always emits a defsig;
        // an identity replacement checks clean.
        let module = render_deep(TWO_FN);
        let new_body = render_body("module M\ndef h(x: f32) -> f32 = x\n", "h");
        let report = check_body_replacement(&module, "f", &new_body).expect("accept");
        assert!(!report.validated_module.as_exprs().is_empty());
    }

    /// A `ping`/`pong` mutually-recursive pair where `ping` holds the sole base
    /// case. Splicing `ping`'s body to call `pong` unconditionally removes that
    /// base case while preserving one uniform recursive instantiation.
    const PINGPONG: &str = "module Frag.PingPong\nexport (ping, pong)\ndef ping(n: i32) -> i32 = if eq(n, 0) then 0 else pong(sub(n, 1))\ndef pong(n: i32) -> i32 = ping(sub(n, 1))\n";

    #[test]
    fn cross_def_base_case_drop_is_accepted_promptly() {
        // [04-INF-2]/[04-INF-3] admit this uniform recursive group. PP9 removed
        // the old syntactic-base-case checker restriction, so fragment
        // replacement must agree with full check by accepting it promptly.
        let module = render_deep(PINGPONG);
        let new_body = render_body("module M\ndef f(n: i32) -> i32 = pong(sub(n, 1))\n", "f");
        let start = std::time::Instant::now();
        let report = check_body_replacement(&module, "ping", &new_body)
            .expect("uniform base-case-free recursion is checker-legal");
        let elapsed = start.elapsed();
        assert!(
            !report.validated_module.as_exprs().is_empty(),
            "accepted replacement returns its validated module",
        );
        assert!(
            elapsed < std::time::Duration::from_secs(30),
            "body-replacement acceptance should return promptly, took {elapsed:?}",
        );
    }

    #[test]
    fn base_case_free_replacement_reaches_loud_lowering_boundary() {
        // The checker accepts uniform recursion, but that does not promise that
        // every lowering strategy supports it. chelis#730 owns this capability
        // boundary; today's static DAG lowering refuses the unbounded expansion
        // at the chelis#620 unroll cap instead of hanging or silently changing
        // the program.
        let module = render_deep(
            "module Frag.TensorLoop\nexport (loop_self)\ndef loop_self(x: tensor[3, f32]) -> tensor[3, f32] = x\n",
        );
        let new_body = render_body(
            "module M\ndef replacement(x: tensor[3, f32]) -> tensor[3, f32] = loop_self(x)\n",
            "replacement",
        );
        let report = check_body_replacement(&module, "loop_self", &new_body)
            .expect("uniform tensor recursion is checker-legal");
        let error = crate::compiler::lower(crate::schema::LowerRequest {
            source_kind: crate::schema::SourceKind::Deep,
            source: chelis_deep::printer::print_canonical(report.validated_module.as_exprs()),
            entry: Some("loop_self".to_string()),
        })
        .expect_err("static DAG lowering must refuse unbounded recursive expansion");
        assert_eq!(error.stage, "lower", "{error:?}");
        assert_eq!(error.errors.len(), 1, "{error:?}");
        let message = &error.errors[0].message;
        assert!(
            message.contains("static unroll limit"),
            "lowering refusal must name the unroll boundary: {message}",
        );
        assert!(
            message.contains("loop_self"),
            "lowering refusal must name the recursive callee: {message}",
        );
        assert!(
            message.contains("chelis#620"),
            "lowering refusal must retain its implementation receipt: {message}",
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
