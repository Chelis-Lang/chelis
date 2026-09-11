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
pub use lower::{
    LoweredLibrary, lower_checked, lower_checked_for_evaluation, lower_checked_with_context,
    lower_checked_with_evaluation_context, lower_library,
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

    #[test]
    fn carrier_owns_the_expanded_deep_vector() {
        let expressions =
            chelis_deep::parser::parse_str("(lit {} 1)").expect("the fixture Deep must parse");
        let expected = expressions.clone();

        let prepared = PreparedProgram::from_expanded_deep(expressions);

        assert_eq!(prepared.expanded_deep(), expected.as_slice());
        assert_eq!(prepared.into_expanded_deep(), expected);
    }

    #[test]
    fn empty_expanded_deep_is_an_explicit_owned_program() {
        let prepared = PreparedProgram::from_expanded_deep(Vec::new());

        assert!(prepared.expanded_deep().is_empty());
        assert!(prepared.into_expanded_deep().is_empty());
    }

    fn prepared(source: &str) -> PreparedProgram {
        PreparedProgram::from_expanded_deep(
            chelis_deep::parser::parse_str(source).expect("the test Deep must parse"),
        )
    }

    #[test]
    fn contextual_success_composes_its_bound_library() {
        let library =
            check_prepared_library(prepared("(def {} one (lit {type: (t-prim {} int32)} 1))"))
                .expect("the library must pass all checks");
        let analysis =
            analyze_prepared_with_library(prepared("(def {} two (var {} one))"), &library)
                .expect("the extension must type-check against the library");
        let checked =
            complete_context_checks(analysis).expect("the extension must pass semantic checks");

        assert!(std::ptr::eq(checked.library(), &library));
        let combined = checked.compose();
        assert_eq!(combined.program().exprs().len(), 2);
    }

    #[test]
    fn contextual_products_reject_a_replacement_library() {
        let first =
            check_prepared_library(prepared("(def {} one (lit {type: (t-prim {} int32)} 1))"))
                .expect("the first library must pass all checks");
        let replacement =
            check_prepared_library(prepared("(def {} one (lit {type: (t-prim {} int32)} 2))"))
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
        let error = lower_checked_with_evaluation_context(&checked, &replacement_evaluation)
            .expect_err("plan lowering must reject a different checked library too");
        assert!(error.to_string().contains("does not match"));
        let error = lower_checked_with_context(checked, &replacement_dag, LoweringMode::Strict)
            .expect_err("contextual lowering must reject another library DAG");
        assert!(error.to_string().contains("does not match"));
    }

    #[test]
    fn library_extension_composes_with_its_bound_context_and_environment() {
        let library =
            check_prepared_library(prepared("(def {} one (lit {type: (t-prim {} int32)} 1))"))
                .expect("the library must pass all checks");
        let analysis =
            analyze_prepared_library_with_base(prepared("(def {} two (var {} one))"), &library)
                .expect("the library extension must type-check");

        assert!(std::ptr::eq(analysis.library(), &library));
        let combined = complete_context_library_checks(analysis)
            .expect("the library extension must pass semantic checks");
        assert_eq!(combined.program().exprs().len(), 2);
        assert!(
            combined
                .type_env()
                .matches_checked_program(combined.program())
        );
    }

    #[test]
    fn cached_library_parser_rejects_a_same_shape_foreign_context() {
        let exported = check_prepared_library(prepared(
            "(module {} m (export {} value) \
             (def {} value (lit {type: (t-prim {} int32)} 1)))",
        ))
        .expect("the exported library must pass all checks");
        let private = check_prepared_library(prepared(
            "(module {} m \
             (def {} value (lit {type: (t-prim {} int32)} 1)))",
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
            check_prepared_library(prepared("(def {} one (lit {type: (t-prim {} int32)} 1))"))
                .expect("the library must pass all checks");

        let rejection =
            validate_cached_library(chelis_types::TypeEnv::empty(), library.program().clone())
                .expect_err("a foreign type environment must fail at the cache boundary");

        assert!(matches!(rejection, LibraryRejection::ContextMismatch));
    }
}
