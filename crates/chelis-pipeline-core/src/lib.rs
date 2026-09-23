#![forbid(unsafe_code)]

//! Typed semantic transitions for an owned expanded Deep program.
//!
//! Successful state cannot be constructed through private fields:
//!
//! ```compile_fail
//! use chelis_pipeline_core::PreparedProgram;
//!
//! let _prepared = PreparedProgram {
//!     expanded_deep: Vec::new(),
//! };
//! ```
//!
//! A semantic rejection exposes no successful checked product:
//!
//! ```compile_fail
//! use chelis_pipeline_core::SemanticRejection;
//!
//! fn expose_checked(rejection: SemanticRejection) {
//!     let _ = rejection.checked();
//! }
//! ```
//!
//! The core exports no unchecked composition escape hatch:
//!
//! ```compile_fail
//! use chelis_pipeline_core::compose_checked;
//!
//! let _ = compose_checked;
//! ```
//!
//! The core exports no unchecked type-product adoption function:
//!
//! ```compile_fail
//! use chelis_pipeline_core::prepared_analysis_from_checked;
//!
//! let _ = prepared_analysis_from_checked;
//! ```
//!
//! Context-bound analysis also requires a checked library proof:
//!
//! ```compile_fail
//! use chelis_pipeline_core::analyze_prepared_with_context;
//!
//! let _ = analyze_prepared_with_context;
//! ```
//!
//! A caller also cannot bind an arbitrary context to checked state:
//!
//! ```compile_fail
//! use chelis_pipeline_core::bind_checked_library;
//!
//! let _ = bind_checked_library;
//! ```
//!
//! Contextual success cannot discard its library binding:
//!
//! ```compile_fail
//! use chelis_pipeline_core::ContextCheckedCompilation;
//!
//! let checked: ContextCheckedCompilation<'static> = todo!();
//! let _ = checked.into_extension();
//! ```
//!
//! Contextual composition accepts no replacement library:
//!
//! ```compile_fail
//! use chelis_pipeline_core::{CheckedLibrary, ContextCheckedCompilation};
//!
//! fn replace_library(
//!     checked: ContextCheckedCompilation<'_>,
//!     other: &CheckedLibrary,
//! ) {
//!     let _ = checked.compose(other);
//! }
//! ```
//!
//! Declared roots and the forward node index have different types:
//!
//! ```compile_fail
//! use chelis_pipeline_core::{LoweredParts, NamedRoots};
//!
//! fn consume_declared_roots(_: NamedRoots) {}
//!
//! let parts: LoweredParts = todo!();
//! consume_declared_roots(parts.forward_node_index);
//! ```

mod artifacts;
mod lower;
mod roots;
mod semantic;

pub use artifacts::{
    AllRootNames, CheckedCompilation, CheckedLibrary, ContextCheckedCompilation,
    ContextualLibraryTypeAnalysis, ContextualTypeAnalysis, CoreLowerError, ForwardNodeIndex,
    IrName, LibraryRejection, LoweredCompilation, LoweredParts, LoweringMode, NamedRoots,
    PreparedLibraryAnalysis, PreparedProgram, PreparedTypeAnalysis, PreparedTypeAnalysisOutcome,
    RootCountContext, RootMetadata, SemanticContext, SemanticRejection, TensorRootNames,
};
#[cfg(feature = "lowering-trace")]
pub use lower::lower_checked_for_c_execution_with_trace;
pub use lower::{
    LoweredLibrary, lower_checked, lower_checked_for_c_execution, lower_checked_for_evaluation,
    lower_checked_with_context, lower_checked_with_evaluation_context, lower_library,
};
pub use semantic::{
    analyze_prepared, analyze_prepared_library, analyze_prepared_library_with_base,
    analyze_prepared_with_library, check_prepared_library, complete_checks,
    complete_context_checks, complete_context_library_checks, complete_library_checks,
    validate_cached_library,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn prepared(source: &str) -> PreparedProgram {
        PreparedProgram::from_expanded_deep(
            chelis_deep::parser::parse_str(source).expect("the test Deep must parse"),
        )
    }

    fn accepted_analysis(source: &str) -> PreparedTypeAnalysis {
        match analyze_prepared(prepared(source)) {
            PreparedTypeAnalysisOutcome::Accepted(analysis) => *analysis,
            PreparedTypeAnalysisOutcome::Rejected { fitness } => {
                panic!("the fixture must type-check: {:?}", fitness.errors)
            }
        }
    }

    fn checked(source: &str) -> CheckedCompilation {
        complete_checks(accepted_analysis(source), SemanticContext::Isolated)
            .expect("the fixture must pass semantic checks")
    }

    #[test]
    fn contextual_products_reject_a_replacement_library() {
        let first =
            check_prepared_library(prepared("(def {} one (lit {type: (t-prim {} i32)} 1))"))
                .expect("the first library must pass all checks");
        let replacement =
            check_prepared_library(prepared("(def {} one (lit {type: (t-prim {} i32)} 2))"))
                .expect("the replacement library must pass all checks");
        let analysis = analyze_prepared_with_library(prepared(""), &first)
            .expect("the empty extension must type-check");
        let checked = complete_context_checks(analysis)
            .expect("the empty extension must pass semantic checks");

        assert!(
            chelis_types::CheckedProgram::compose(
                replacement.program(),
                checked.extension().program(),
            )
            .is_none(),
            "a context-checked program must reject another library"
        );
        let replacement_dag =
            lower_library(&replacement).expect("the replacement library must lower");
        let first_evaluation =
            chelis_ir::lower::try_lower_program_to_evaluation_library(first.program()).unwrap();
        lower_checked_with_evaluation_context(&checked, &first_evaluation)
            .expect("the opaque evaluation library matches the sealed context");
        let replacement_evaluation =
            chelis_ir::lower::try_lower_program_to_evaluation_library(replacement.program())
                .unwrap();
        assert!(matches!(
            lower_checked_with_evaluation_context(&checked, &replacement_evaluation),
            Err(CoreLowerError::Lower(_))
        ));
        assert!(matches!(
            lower_checked_with_context(checked, &replacement_dag, LoweringMode::Strict),
            Err(CoreLowerError::Lower(_))
        ));
    }

    #[test]
    fn cached_library_parser_rejects_a_same_shape_foreign_context() {
        let exported = check_prepared_library(prepared(
            "(module {} m (export {} value) \
             (def {} value (lit {type: (t-prim {} i32)} 1)))",
        ))
        .expect("the exported library must pass all checks");
        let private = check_prepared_library(prepared(
            "(module {} m \
             (def {} value (lit {type: (t-prim {} i32)} 1)))",
        ))
        .expect("the private library must pass all checks");
        assert_eq!(private.program().type_env(), exported.program().type_env());
        assert!(
            !private
                .type_env()
                .matches_checked_program(exported.program())
        );

        let rejection =
            validate_cached_library(private.type_env().clone(), exported.program().clone())
                .expect_err("a same-shape foreign context must fail at the cache boundary");

        assert!(matches!(rejection, LibraryRejection::ContextMismatch));
    }

    #[test]
    fn cached_library_parser_rejects_a_mismatched_type_environment() {
        let library =
            check_prepared_library(prepared("(def {} one (lit {type: (t-prim {} i32)} 1))"))
                .expect("the library must pass all checks");

        let rejection =
            validate_cached_library(chelis_types::TypeEnv::empty(), library.program().clone())
                .expect_err("a foreign type environment must fail at the cache boundary");

        assert!(matches!(rejection, LibraryRejection::ContextMismatch));
    }

    #[test]
    fn lowering_preserves_canonical_tuple_roots() {
        let lowered = lower_checked(
            checked(include_str!("../tests/fixtures/pair.dp")),
            LoweringMode::Strict,
        )
        .expect("the tuple must lower");
        assert_eq!(
            lowered
                .checked()
                .root_metadata()
                .tensor_names()
                .iter()
                .map(IrName::as_str)
                .collect::<Vec<_>>(),
            ["pair.0", "pair.1"]
        );
        assert_eq!(lowered.named_roots().len(), 2);
        assert_eq!(lowered.dag().roots().len(), 2);
    }

    #[test]
    fn declared_roots_and_forward_load_aliases_have_distinct_lookups() {
        let lowered = lower_checked(
            checked(include_str!("../tests/fixtures/lookup.dp")),
            LoweringMode::Strict,
        )
        .expect("the function must lower");
        let output = IrName::new("out");
        let input = IrName::new("input");
        assert!(lowered.named_roots().get(&output).is_some());
        assert!(lowered.named_roots().get(&input).is_none());
        assert!(lowered.forward_node_index().get(&output).is_some());
        assert!(lowered.forward_node_index().get(&input).is_some());
    }

    #[test]
    fn host_only_output_keeps_the_explicit_empty_root_product() {
        let lowered = lower_checked(
            checked(include_str!("../tests/fixtures/host.dp")),
            LoweringMode::AllowHostOnly,
        )
        .expect("the host-only value must lower without tensor roots");
        assert!(lowered.dag().roots().is_empty());
        assert!(lowered.named_roots().is_empty());
    }

    #[test]
    fn type_rejection_has_no_checked_product() {
        assert!(matches!(
            analyze_prepared(prepared(include_str!(
                "../tests/fixtures/type_rejection.dp"
            ))),
            PreparedTypeAnalysisOutcome::Rejected { .. }
        ));
    }

    #[test]
    fn effect_rejection_uses_the_narrow_semantic_error() {
        assert!(matches!(
            complete_checks(
                accepted_analysis(include_str!("../tests/fixtures/effect_rejection.dp")),
                SemanticContext::Isolated,
            ),
            Err(SemanticRejection::Effects { .. })
        ));
    }

    #[test]
    fn linearity_rejection_uses_the_narrow_semantic_error() {
        assert!(matches!(
            complete_checks(
                accepted_analysis(include_str!("../tests/fixtures/linearity_rejection.dp")),
                SemanticContext::Isolated,
            ),
            Err(SemanticRejection::Linearity { .. })
        ));
    }

    #[test]
    fn strict_lowering_rejection_retains_the_lower_diagnostic() {
        assert!(matches!(
            lower_checked(
                checked(include_str!("../tests/fixtures/lower_rejection.dp")),
                LoweringMode::Strict,
            ),
            Err(CoreLowerError::Lower(_))
        ));
    }
}
