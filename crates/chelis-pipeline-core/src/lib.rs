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
//! Checked-state composition requires an unsafe adapter boundary:
//!
//! ```compile_fail
//! use chelis_pipeline_core::{CheckedCompilation, compose_checked};
//! use chelis_types::CheckedProgram;
//!
//! let library: CheckedProgram = todo!();
//! let extension: CheckedCompilation = todo!();
//! let _ = compose_checked(&library, extension);
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
    AllRootNames, CheckedCompilation, CoreLowerError, ForwardNodeIndex, IrName, LoweredCompilation,
    LoweredParts, LoweringMode, NamedRoots, PreparedProgram, PreparedTypeAnalysis,
    PreparedTypeAnalysisOutcome, RootCountContext, RootMetadata, SemanticContext,
    SemanticRejection, TensorRootNames,
};
pub use lower::{lower_checked, lower_checked_with_context, lower_library};
pub use semantic::{
    analyze_prepared, analyze_prepared_with_context, complete_checks, compose_checked,
    prepared_analysis_from_checked,
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
}
