//! Authoritative checker diagnostic-session boundary.
//!
//! `DiagnosticSink` deliberately exposes no constructor, owned storage,
//! conversion, clone, default, or mutable dereference outside this module.
//! Lower checker layers can only receive an already-open sink, so a fresh
//! [`crate::errors::ErrorWitness`] can never be minted into a throwaway
//! `Vec<CheckError>`.

use crate::context::TypeEnv;
use crate::errors::CheckError;
use crate::infer::{CheckedProgram, InferResult, InferStats, SignatureInferenceMetadata};

#[cfg(test)]
thread_local! {
    static TYPE_ANALYSIS_SESSION_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_type_analysis_session_count() {
    TYPE_ANALYSIS_SESSION_COUNT.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn type_analysis_session_count() -> usize {
    TYPE_ANALYSIS_SESSION_COUNT.with(std::cell::Cell::get)
}

#[cfg(test)]
fn record_type_analysis_session() {
    TYPE_ANALYSIS_SESSION_COUNT.with(|count| count.set(count.get() + 1));
}

/// A sink-issued boundary for later diagnostic iteration.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DiagnosticCheckpoint {
    offset: usize,
}

/// The sole destination accepted by witness-minting checker code.
///
/// Its storage and constructor are private to this module.  The narrow
/// mutation surface is enough for ordinary diagnostics while deliberately
/// preventing early extraction or replacement of the canonical vector.
pub struct DiagnosticSink<'session> {
    errors: &'session mut Vec<CheckError>,
}

impl DiagnosticSink<'_> {
    pub(crate) fn push(&mut self, error: CheckError) {
        self.errors.push(error);
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    pub(crate) fn checkpoint(&self) -> DiagnosticCheckpoint {
        DiagnosticCheckpoint {
            offset: self.errors.len(),
        }
    }

    pub(crate) fn iter(&self) -> std::slice::Iter<'_, CheckError> {
        self.errors.iter()
    }

    pub(crate) fn iter_since(
        &self,
        checkpoint: DiagnosticCheckpoint,
    ) -> std::slice::Iter<'_, CheckError> {
        self.errors[checkpoint.offset..].iter()
    }

    /// Retain only diagnostics accepted by `keep` after an earlier checkpoint.
    ///
    /// This deliberately cannot inspect, replace, or discard diagnostics that
    /// precede the checkpoint. It exists for an exact compiler-owned source
    /// boundary whose structural validator accepts a closed wrapper graph while
    /// ordinary HM inference still records useful child stamps for that graph.
    pub(crate) fn retain_since(
        &mut self,
        checkpoint: DiagnosticCheckpoint,
        mut keep: impl FnMut(&CheckError) -> bool,
    ) {
        let retained = self
            .errors
            .drain(checkpoint.offset..)
            .filter(|error| keep(error))
            .collect::<Vec<_>>();
        self.errors.extend(retained);
    }
}

#[cfg(test)]
mod diagnostic_checkpoint_tests {
    use super::*;
    use crate::errors::CheckErrorKind;

    fn diagnostic(message: &str) -> CheckError {
        CheckError::new(CheckErrorKind::Other, message.to_string(), vec![])
    }

    #[test]
    fn iter_since_includes_diagnostics_after_the_checkpoint() {
        let mut errors = Vec::new();
        let mut sink = DiagnosticSink {
            errors: &mut errors,
        };
        let checkpoint = sink.checkpoint();
        sink.push(diagnostic("first new diagnostic"));
        sink.push(diagnostic("second new diagnostic"));

        let messages = sink
            .iter_since(checkpoint)
            .map(|error| error.message.as_str())
            .collect::<Vec<_>>();
        assert_eq!(messages, ["first new diagnostic", "second new diagnostic"]);
    }

    #[test]
    fn iter_since_excludes_diagnostics_before_the_checkpoint() {
        let mut errors = Vec::new();
        let mut sink = DiagnosticSink {
            errors: &mut errors,
        };
        sink.push(diagnostic("earlier diagnostic"));
        let checkpoint = sink.checkpoint();
        sink.push(diagnostic("new diagnostic"));

        let messages = sink
            .iter_since(checkpoint)
            .map(|error| error.message.as_str())
            .collect::<Vec<_>>();
        assert_eq!(messages, ["new diagnostic"]);
    }
}

pub(crate) fn infer_program(exprs: &[chelis_deep::Expr]) -> InferResult {
    crate::infer::run_on_grown_stack(|| {
        let mut errors = Vec::new();
        let stats = {
            let mut sink = DiagnosticSink {
                errors: &mut errors,
            };
            match admit_metadata(exprs, &mut sink) {
                Ok(()) => crate::infer::infer_program_in_session(exprs, &mut sink),
                Err(stats) => stats,
            }
        };
        InferResult {
            errors,
            typed_nodes: stats.typed_nodes,
            total_nodes: stats.total_nodes,
        }
    })
}

#[cfg(test)]
mod authoritative_type_stamp_tests {
    use super::*;
    use crate::infer::{TypeStampMutationCase, run_type_stamp_mutation_case};

    fn run(case: TypeStampMutationCase) -> (bool, Vec<CheckError>) {
        let mut errors = Vec::new();
        let result = {
            let mut sink = DiagnosticSink {
                errors: &mut errors,
            };
            run_type_stamp_mutation_case(case, &mut sink)
        };
        (result, errors)
    }

    #[test]
    fn missing_registered_owner_stamp_is_loud_and_has_no_fallback() {
        let (missing, errors) = run(TypeStampMutationCase::Missing);
        assert!(missing);
        assert_eq!(errors.len(), 1, "missing lookup must report exactly once");
        assert!(
            errors[0]
                .message
                .contains("missing authoritative type stamp")
        );
    }

    #[test]
    fn compatible_repeated_owner_write_is_accepted() {
        let (resolved, errors) = run(TypeStampMutationCase::CompatibleRepeat);
        assert!(resolved);
        assert!(
            errors.is_empty(),
            "compatible writes must agree: {errors:?}"
        );
    }

    #[test]
    fn incompatible_repeated_owner_write_is_loud() {
        let (_, errors) = run(TypeStampMutationCase::IncompatibleRepeat);
        assert_eq!(errors.len(), 1, "conflicting writes must report once");
        assert!(
            errors[0]
                .message
                .contains("conflicting authoritative type writes")
        );
    }

    #[test]
    fn unregistered_synthesized_node_cannot_enter_owner_registry() {
        let (missing, errors) = run(TypeStampMutationCase::UnregisteredSynthesized);
        assert!(missing);
        assert_eq!(
            errors.len(),
            1,
            "unregistered lookup must report exactly once"
        );
        assert!(
            errors[0]
                .message
                .contains("missing authoritative type stamp")
        );
    }

    #[test]
    fn inferred_runtime_non_stamp_owner_is_available_to_contextual_consumers() {
        let (resolved, errors) = run(TypeStampMutationCase::RuntimeNonStampOwnerLookup);
        assert!(resolved);
        assert!(
            errors.is_empty(),
            "contextual consumers must reuse the canonical runtime child type: {errors:?}"
        );
    }
}

#[cfg(test)]
mod annotated_totality_finalization_tests {
    use super::*;
    use crate::infer::{FinalizationMutationCase, run_finalization_mutation_case};

    fn errors_for(case: FinalizationMutationCase) -> Vec<CheckError> {
        let mut errors = Vec::new();
        {
            let mut sink = DiagnosticSink {
                errors: &mut errors,
            };
            run_finalization_mutation_case(case, &mut sink);
        }
        errors
    }

    #[test]
    fn missing_runtime_stamp_is_rejected_at_shared_finalization() {
        let errors = errors_for(FinalizationMutationCase::MissingRuntimeStamp);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("missing its type stamp"));
    }

    #[test]
    fn silent_error_owner_is_rejected_at_shared_finalization() {
        let errors = errors_for(FinalizationMutationCase::SilentErrorOwner);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("missing its type stamp"));
    }

    #[test]
    fn silent_error_signature_is_rejected_at_private_finalization() {
        let errors = errors_for(FinalizationMutationCase::SilentErrorSignature);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("totality invariant"));
    }
}

#[cfg(test)]
mod result_boundary_tests {
    use super::*;
    use crate::errors::CheckErrorKind;

    #[test]
    fn run_result_rejects_ok_after_an_authoritative_diagnostic() {
        let result = run_result(|sink| {
            sink.push(CheckError::new(
                CheckErrorKind::Other,
                "planted authoritative diagnostic".to_string(),
                vec![],
            ));
            Ok(())
        })
        .expect_err("a non-empty authoritative sink vetoes Ok");
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.typed_nodes, 0);
        assert_eq!(result.total_nodes, 0);
    }
}

fn run_result<T>(
    run: impl FnOnce(&mut DiagnosticSink<'_>) -> Result<T, InferStats>,
) -> Result<T, InferResult> {
    let mut errors = Vec::new();
    let result = {
        let mut sink = DiagnosticSink {
            errors: &mut errors,
        };
        run(&mut sink)
    };
    match result {
        Ok(value) if errors.is_empty() => Ok(value),
        Ok(_) if !errors.is_empty() => Err(InferResult {
            errors,
            typed_nodes: 0,
            total_nodes: 0,
        }),
        Ok(_) => unreachable!("the empty/non-empty diagnostic cases are exhaustive"),
        Err(stats) => Err(InferResult {
            errors,
            typed_nodes: stats.typed_nodes,
            total_nodes: stats.total_nodes,
        }),
    }
}

// Text ingress and Node construction have already checked these rules.
// Legacy programmatic carriers still enter here: reject before inference so
// shape errors cannot be reinterpreted or reported again by semantic owners.
fn admit_metadata(
    exprs: &[chelis_deep::Expr],
    sink: &mut DiagnosticSink<'_>,
) -> Result<(), InferStats> {
    if let Err(error) = chelis_deep::metadata::validate_metadata(exprs) {
        sink.push(
            CheckError::new(
                crate::errors::CheckErrorKind::MalformedForm,
                error.to_string(),
                vec![],
            )
            .at_offset(error.span.offset),
        );
        return Err(InferStats {
            typed_nodes: 0,
            total_nodes: 0,
        });
    }
    Ok(())
}

fn run_with_metadata<T>(
    exprs: &[chelis_deep::Expr],
    run: impl FnOnce(&mut DiagnosticSink<'_>) -> Result<T, InferStats>,
) -> Result<T, InferResult> {
    run_result(|sink| {
        admit_metadata(exprs, sink)?;
        run(sink)
    })
}

pub(crate) fn build_type_env_from_library(
    exprs: &[chelis_deep::Expr],
) -> Result<TypeEnv, InferResult> {
    crate::infer::run_on_grown_stack(|| {
        run_with_metadata(exprs, |sink| {
            crate::infer::build_type_env_from_library_in_session(exprs, sink)
        })
    })
}

pub(crate) fn build_compiled_library_context(
    exprs: &[chelis_deep::Expr],
) -> Result<(TypeEnv, CheckedProgram), InferResult> {
    run_with_metadata(exprs, |sink| {
        crate::infer::build_compiled_library_context_in_session(exprs, sink)
    })
}

pub(crate) fn build_compiled_library_context_with_base(
    base: &TypeEnv,
    exprs: &[chelis_deep::Expr],
) -> Result<(TypeEnv, CheckedProgram), InferResult> {
    run_with_metadata(exprs, |sink| {
        crate::infer::build_compiled_library_context_with_base_in_session(base, exprs, sink)
    })
}

pub(crate) fn check_ir_with_signature_context(
    context: &TypeEnv,
    signature_context: &SignatureInferenceMetadata,
    exprs: &[chelis_deep::Expr],
) -> Result<CheckedProgram, InferResult> {
    #[cfg(test)]
    record_type_analysis_session();
    crate::infer::run_on_grown_stack(|| {
        run_with_metadata(exprs, |sink| {
            crate::infer::check_ir_with_signature_context_in_session(
                context,
                signature_context,
                exprs,
                sink,
            )
        })
    })
}

pub(crate) fn check_typed_program(
    exprs: &[chelis_deep::Expr],
) -> Result<CheckedProgram, InferResult> {
    crate::infer::run_on_grown_stack(|| {
        run_with_metadata(exprs, |sink| {
            crate::infer::check_typed_program_in_session(exprs, sink)
        })
    })
}

pub(crate) fn infer_ir_program(exprs: &[chelis_deep::Expr]) -> InferResult {
    crate::infer::run_on_grown_stack(|| {
        let mut errors = Vec::new();
        let stats = {
            let mut sink = DiagnosticSink {
                errors: &mut errors,
            };
            match admit_metadata(exprs, &mut sink) {
                Ok(()) => crate::infer::infer_ir_program_in_session(exprs, &mut sink),
                Err(stats) => stats,
            }
        };
        InferResult {
            errors,
            typed_nodes: stats.typed_nodes,
            total_nodes: stats.total_nodes,
        }
    })
}

pub(crate) fn try_checked_program_with_effect_annotations(
    original: &CheckedProgram,
    annotated_exprs: Vec<chelis_deep::Expr>,
) -> Result<CheckedProgram, InferResult> {
    run_result(|sink| {
        admit_metadata(&annotated_exprs, sink)?;
        Ok(
            crate::infer::checked_program_with_effect_annotations_in_session(
                original,
                annotated_exprs,
                sink,
            ),
        )
    })
}

pub(crate) fn param_has_consuming_use(
    expr: &chelis_deep::Expr,
    param: &str,
    available_signatures: &chelis_unord::UnordMap<String, crate::types::Type>,
    type_env: &std::collections::BTreeMap<String, chelis_deep::Expr>,
    type_headers: &crate::deep_type::TypeResolutionEnv,
) -> Result<bool, InferResult> {
    run_result(|sink| {
        Ok(crate::infer::param_has_consuming_use_in_session(
            expr,
            param,
            available_signatures,
            type_env,
            type_headers,
            sink,
        ))
    })
}
