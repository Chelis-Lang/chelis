#![forbid(unsafe_code)]

//! Typed semantic transitions for an owned expanded Deep program.
//!
//! Successful state cannot be constructed through private fields:
//!
//! ```compile_fail
//! use chelis_pipeline_core::PreparedProgram;
//! use chelis_types::ProgramOrigin;
//!
//! let _prepared = PreparedProgram {
//!     expanded_deep: Vec::new(),
//!     origin: ProgramOrigin::Unlinked,
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
//! Contextual success has no consuming extension accessor. Borrowed inspection
//! and clones retain provenance and are rejected by isolated lowering:
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
//! use chelis_ir::lowering_runtime::LoweringRuntime;
//! fn replace_library(
//!     checked: ContextCheckedCompilation<'_>,
//!     other: &CheckedLibrary,
//!     runtime: &LoweringRuntime,
//! ) {
//!     let _ = checked.compose(other, runtime);
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
    use chelis_types::ProgramOrigin;
    use chelis_types::errors::CheckErrorKind;

    use super::*;

    fn prepared(source: &str) -> PreparedProgram {
        prepared_with_origin(source, ProgramOrigin::Unlinked)
    }

    fn prepared_with_origin(source: &str, origin: ProgramOrigin) -> PreparedProgram {
        PreparedProgram::from_expanded_deep(
            chelis_deep::parser::parse_str(source).expect("the test Deep must parse"),
            origin,
        )
    }

    fn accepted_analysis(source: &str) -> PreparedTypeAnalysis {
        match analyze_prepared(
            prepared(source),
            &mut chelis_types::NativeServices::new(None),
        ) {
            PreparedTypeAnalysisOutcome::Accepted(analysis) => *analysis,
            PreparedTypeAnalysisOutcome::Rejected { fitness } => {
                panic!("the fixture must type-check: {:?}", fitness.errors)
            }
        }
    }

    fn checked(source: &str) -> CheckedCompilation {
        chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            complete_checks(
                accepted_analysis(source),
                SemanticContext::Isolated,
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )
        })
        .expect("the fixture must pass semantic checks")
    }

    const CONTEXT_LIBRARY: &str =
        "(def {} lib_const (lit {type: (t-tensor {} (t-prim {} f32))} 7.0))";
    const CONTEXT_EXTENSION: &str = "
        (def {} y
          (app {type: (t-tensor {} (t-prim {} f32))}
            (var {} mul)
            (var {type: (t-tensor {} (t-prim {} f32))} lib_const)
            (lit {type: (t-tensor {} (t-prim {} f32))} 3.0)))";

    fn context_checked(library: &CheckedLibrary) -> ContextCheckedCompilation<'_> {
        chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            complete_context_checks(
                analyze_prepared_with_library(
                    prepared(CONTEXT_EXTENSION),
                    library,
                    &mut chelis_types::NativeServices::new(None),
                )
                .unwrap(),
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )
        })
        .unwrap()
    }

    const LINKED_INTERFACE: &str =
        "(defsig {} pkg__dep__Api__external (t-fn {} (t-prim {} string)))";
    const LINKED_LIBRARY: &str = "(def {} pkg__dep__Api__value (lit {type: (t-prim {} i32)} 7))";
    const LINKED_EXTENSION: &str = "(def {} pkg__app__Api__value (var {} pkg__dep__Api__value))";
    const UNLINKED_EXTENSION: &str = "(def {} answer (var {} pkg__dep__Api__value))";

    #[test]
    fn monolithic_origin_distinguishes_linked_interfaces_from_forged_source() {
        assert!(matches!(
            analyze_prepared(
                prepared_with_origin(LINKED_INTERFACE, ProgramOrigin::Linked),
                &mut chelis_types::NativeServices::new(None)
            ),
            PreparedTypeAnalysisOutcome::Accepted(_)
        ));
        let PreparedTypeAnalysisOutcome::Rejected { fitness } = analyze_prepared(
            prepared(LINKED_INTERFACE),
            &mut chelis_types::NativeServices::new(None),
        ) else {
            panic!("identical unlinked input must not gain dependency-interface authority");
        };
        assert!(
            fitness
                .errors
                .iter()
                .any(|error| matches!(error.kind, CheckErrorKind::ReservedLinkerName))
        );
        assert!(fitness.errors.iter().any(|error| matches!(
            &error.kind,
            CheckErrorKind::UnboundVariable { identifier }
                if identifier == "pkg__dep__Api__external"
        )));
    }

    #[test]
    fn library_origin_distinguishes_linked_declarations_from_forged_source() {
        chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            check_prepared_library(
                prepared_with_origin(LINKED_LIBRARY, ProgramOrigin::Linked),
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )
        })
        .expect("trusted linked declarations must pass the library semantic entry point");
        let LibraryRejection::Type { report } =
            chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
                check_prepared_library(
                    prepared(LINKED_LIBRARY),
                    runtime,
                    &mut chelis_types::NativeServices::new(None),
                )
            })
            .expect_err("identical unlinked declarations must be rejected")
        else {
            panic!("forged linker names must be rejected during type analysis");
        };
        assert!(
            report
                .errors
                .iter()
                .any(|error| matches!(error.kind, CheckErrorKind::ReservedLinkerName))
        );
    }

    #[test]
    fn contextual_origin_is_not_inherited_from_a_linked_library() {
        let library = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            check_prepared_library(
                prepared_with_origin(LINKED_LIBRARY, ProgramOrigin::Linked),
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )
        })
        .unwrap();
        let linked = analyze_prepared_with_library(
            prepared_with_origin(LINKED_EXTENSION, ProgramOrigin::Linked),
            &library,
            &mut chelis_types::NativeServices::new(None),
        )
        .expect("a separately trusted linked extension must type-check");
        chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            complete_context_checks(
                linked,
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )
        })
        .expect("linked extension must pass semantic checks");

        let rejection = analyze_prepared_with_library(
            prepared(LINKED_EXTENSION),
            &library,
            &mut chelis_types::NativeServices::new(None),
        )
        .expect_err("a linked base must not authorize new unlinked declarations");
        assert!(
            rejection
                .errors
                .iter()
                .any(|error| matches!(error.kind, CheckErrorKind::ReservedLinkerName))
        );

        let unlinked = analyze_prepared_with_library(
            prepared(UNLINKED_EXTENSION),
            &library,
            &mut chelis_types::NativeServices::new(None),
        )
        .expect("unlinked code may reference already-checked linked declarations");
        chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            complete_context_checks(
                unlinked,
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )
        })
        .expect("ordinary unlinked extension must remain valid");
    }

    #[test]
    fn layered_origin_is_not_inherited_from_a_linked_library() {
        let library = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            check_prepared_library(
                prepared_with_origin(LINKED_LIBRARY, ProgramOrigin::Linked),
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )
        })
        .unwrap();
        let linked = analyze_prepared_library_with_base(
            prepared_with_origin(LINKED_EXTENSION, ProgramOrigin::Linked),
            &library,
            &mut chelis_types::NativeServices::new(None),
        )
        .expect("a separately trusted linked library layer must type-check");
        chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            complete_context_library_checks(
                linked,
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )
        })
        .expect("linked library layer must compose with its checked base");

        let rejection = analyze_prepared_library_with_base(
            prepared(LINKED_EXTENSION),
            &library,
            &mut chelis_types::NativeServices::new(None),
        )
        .expect_err("a linked base must not authorize an unlinked library layer");
        assert!(
            rejection
                .errors
                .iter()
                .any(|error| matches!(error.kind, CheckErrorKind::ReservedLinkerName))
        );

        let unlinked = analyze_prepared_library_with_base(
            prepared(UNLINKED_EXTENSION),
            &library,
            &mut chelis_types::NativeServices::new(None),
        )
        .expect("an unlinked layer may reference already-checked linked declarations");
        chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            complete_context_library_checks(
                unlinked,
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )
        })
        .expect("ordinary unlinked layer must compose with its checked base");
    }

    fn manifest(checked: &CheckedCompilation) -> chelis_types::manifest::RootManifest {
        let realizability = chelis_effects::realizability::infer_realizability(
            checked.program(),
            &[chelis_types::types::Prim::F32],
        );
        chelis_effects::realizability::compute_root_manifest(checked.program(), &realizability)
    }

    fn assert_isolated_lowering_rejects(checked: &CheckedCompilation) {
        assert!(matches!(
            chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
                lower_checked_for_evaluation(checked, runtime)
            }),
            Err(CoreLowerError::ContextRequired)
        ));
        let manifest = manifest(checked);
        for mode in [
            LoweringMode::Strict,
            LoweringMode::AllowHostOnly,
            LoweringMode::AllowHostBackend,
        ] {
            assert!(matches!(
                chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| lower_checked(
                    checked.clone(),
                    mode,
                    runtime
                )),
                Err(CoreLowerError::ContextRequired)
            ));
            assert!(matches!(
                chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
                    lower_checked_for_c_execution(checked.clone(), &manifest, mode, runtime)
                }),
                Err(CoreLowerError::ContextRequired)
            ));
            #[cfg(feature = "lowering-trace")]
            assert!(matches!(
                chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
                    lower_checked_for_c_execution_with_trace(
                        checked.clone(),
                        &manifest,
                        mode,
                        runtime,
                    )
                }),
                Err(CoreLowerError::ContextRequired)
            ));
        }
    }

    fn injected_library_input(name: &str) -> Option<chelis_ir::eval::TensorValue> {
        (name == "lib_const").then(|| chelis_ir::eval::TensorValue::scalar(100.0))
    }

    fn assert_sealed_dag_result(lowered: &LoweredCompilation) {
        let root = *lowered.named_roots().get(&IrName::new("y")).unwrap();
        let values = chelis_ir::eval::eval_tensor_roots_with_strict(
            lowered.dag(),
            &[root],
            injected_library_input,
            &mut chelis_types::NativeServices::new(None),
        )
        .unwrap();
        assert_eq!(values[&root].to_f64_lossy_vec(), vec![21.0]);
    }

    fn assert_sealed_evaluation_result(program: chelis_ir::lower::EvaluationProgram) {
        let selected = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            program.select_roots(&["y".to_owned()], runtime)
        })
        .unwrap();
        let mut context = chelis_ir::evaluation::RandomExecutionContext::new(
            chelis_ir::host::RandomLoweringState {
                seed: None,
                counter: 0,
            },
        );
        let values = chelis_ir::eval::eval_tensor_plan_with_strict(
            selected.plan(),
            &mut context,
            injected_library_input,
            &mut chelis_types::NativeServices::new(None),
        )
        .unwrap();
        assert_eq!(
            values[&selected.roots()["y"]].to_f64_lossy_vec(),
            vec![21.0]
        );
    }

    #[test]
    fn detached_context_extensions_reject_every_isolated_lowering_entrypoint() {
        let library = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            check_prepared_library(
                prepared(CONTEXT_LIBRARY),
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )
        })
        .unwrap();
        let contextual = context_checked(&library);
        assert_isolated_lowering_rejects(contextual.extension());
        let detached = contextual.extension().clone();
        drop(contextual);
        drop(library);
        assert_isolated_lowering_rejects(&detached);
    }

    #[test]
    fn contextual_lowered_products_cannot_be_relowered_without_their_library() {
        let library = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            check_prepared_library(
                prepared(CONTEXT_LIBRARY),
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )
        })
        .unwrap();
        let lowered_library =
            chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
                lower_library(&library, runtime)
            })
            .unwrap();
        let lowered = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            lower_checked_with_context(
                context_checked(&library),
                &lowered_library,
                LoweringMode::Strict,
                runtime,
            )
        })
        .unwrap();
        assert_sealed_dag_result(&lowered);
        assert_isolated_lowering_rejects(lowered.checked());
        let parts = lowered.into_parts();
        assert_isolated_lowering_rejects(&parts.checked);
    }

    #[test]
    fn bound_and_composed_programs_keep_library_values_sealed() {
        let library = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            check_prepared_library(
                prepared(CONTEXT_LIBRARY),
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )
        })
        .unwrap();
        let evaluation_library =
            chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
                chelis_ir::lower::try_lower_program_to_evaluation_library(
                    library.program(),
                    runtime,
                )
            })
            .unwrap();
        assert_sealed_evaluation_result(
            chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
                lower_checked_with_evaluation_context(
                    &context_checked(&library),
                    &evaluation_library,
                    runtime,
                )
            })
            .unwrap(),
        );

        let composed = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            context_checked(&library).compose(runtime)
        });
        let isolated = checked(&format!("{CONTEXT_LIBRARY}\n{CONTEXT_EXTENSION}"));
        for checked in [composed, isolated] {
            assert_sealed_evaluation_result(
                chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
                    lower_checked_for_evaluation(&checked, runtime)
                })
                .unwrap(),
            );
            let manifest = manifest(&checked);
            assert_sealed_dag_result(
                &chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
                    lower_checked(checked.clone(), LoweringMode::Strict, runtime)
                })
                .unwrap(),
            );
            let (lowered, _, _) =
                chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
                    lower_checked_for_c_execution(
                        checked.clone(),
                        &manifest,
                        LoweringMode::Strict,
                        runtime,
                    )
                })
                .unwrap();
            assert_sealed_dag_result(&lowered);
            #[cfg(feature = "lowering-trace")]
            {
                let (lowered, _, _) =
                    chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
                        lower_checked_for_c_execution_with_trace(
                            checked.clone(),
                            &manifest,
                            LoweringMode::Strict,
                            runtime,
                        )
                    })
                    .unwrap();
                assert_sealed_dag_result(&lowered);
            }
        }
    }

    #[test]
    fn library_extensions_require_composition_before_library_lowering() {
        let library = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            check_prepared_library(
                prepared(CONTEXT_LIBRARY),
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )
        })
        .unwrap();
        let extension = analyze_prepared_library_with_base(
            prepared(CONTEXT_EXTENSION),
            &library,
            &mut chelis_types::NativeServices::new(None),
        )
        .unwrap();
        assert!(matches!(
            validate_cached_library(
                extension.type_env().clone(),
                extension.analysis().program().clone(),
                &mut chelis_types::NativeServices::new(None),
            ),
            Err(LibraryRejection::ContextMismatch)
        ));

        let composed = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            complete_context_library_checks(
                extension,
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )
        })
        .unwrap();
        let composed = validate_cached_library(
            composed.type_env().clone(),
            composed.program().clone(),
            &mut chelis_types::NativeServices::new(None),
        )
        .unwrap();
        let lowered = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            lower_library(&composed, runtime)
        })
        .unwrap();
        let values = chelis_ir::eval::eval_tensor_roots_with_strict(
            lowered.dag(),
            lowered.dag().roots(),
            injected_library_input,
            &mut chelis_types::NativeServices::new(None),
        )
        .unwrap();
        let root = lowered.dag().roots().last().unwrap();
        assert_eq!(values[root].to_f64_lossy_vec(), vec![21.0]);
    }

    #[test]
    fn shadowed_host_declaration_retains_its_new_tensor_root() {
        let library = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            check_prepared_library(
                prepared(r#"(def {} value (lit {} "host"))"#),
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )
        })
        .unwrap();
        let lowered_library =
            chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
                lower_library(&library, runtime)
            })
            .unwrap();
        let analysis = analyze_prepared_with_library(
            prepared(
                "(defsig {} value (t-prim {} i32))
                 (def {} value (lit {type: (t-prim {} i32)} 2))",
            ),
            &library,
            &mut chelis_types::NativeServices::new(None),
        )
        .unwrap();
        let lowered = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            lower_checked_with_context(
                chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
                    complete_context_checks(
                        analysis,
                        runtime,
                        &mut chelis_types::NativeServices::new(None),
                    )
                })
                .unwrap(),
                &lowered_library,
                LoweringMode::Strict,
                runtime,
            )
        })
        .unwrap();
        let root = *lowered.named_roots().get(&IrName::new("value")).unwrap();
        assert_eq!(lowered.dag().roots(), &[root]);
        let values = chelis_ir::eval::eval_tensor_roots_with_strict(
            lowered.dag(),
            &[root],
            |_| None,
            &mut chelis_types::NativeServices::new(None),
        )
        .unwrap();
        assert_eq!(values[&root].to_f64_lossy_vec(), vec![2.0]);
    }

    #[test]
    fn contextual_products_reject_a_replacement_library() {
        let first = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            check_prepared_library(
                prepared("(def {} one (lit {type: (t-prim {} i32)} 1))"),
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )
        })
        .expect("the first library must pass all checks");
        let replacement = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            check_prepared_library(
                prepared("(def {} one (lit {type: (t-prim {} i32)} 2))"),
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )
        })
        .expect("the replacement library must pass all checks");
        let analysis = analyze_prepared_with_library(
            prepared(""),
            &first,
            &mut chelis_types::NativeServices::new(None),
        )
        .expect("the empty extension must type-check");
        let checked = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            complete_context_checks(
                analysis,
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )
        })
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
            chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
                lower_library(&replacement, runtime)
            })
            .expect("the replacement library must lower");
        let first_evaluation =
            chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
                chelis_ir::lower::try_lower_program_to_evaluation_library(first.program(), runtime)
            })
            .unwrap();
        chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            lower_checked_with_evaluation_context(&checked, &first_evaluation, runtime)
        })
        .expect("the opaque evaluation library matches the sealed context");
        let replacement_evaluation =
            chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
                chelis_ir::lower::try_lower_program_to_evaluation_library(
                    replacement.program(),
                    runtime,
                )
            })
            .unwrap();
        assert!(matches!(
            chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
                lower_checked_with_evaluation_context(&checked, &replacement_evaluation, runtime)
            }),
            Err(CoreLowerError::Lower(_))
        ));
        assert!(matches!(
            chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
                lower_checked_with_context(checked, &replacement_dag, LoweringMode::Strict, runtime)
            }),
            Err(CoreLowerError::Lower(_))
        ));
    }

    #[test]
    fn cached_library_parser_rejects_a_same_shape_foreign_context() {
        let exported = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            check_prepared_library(
                prepared(
                    "(module {} m (export {} value) \
             (def {} value (lit {type: (t-prim {} i32)} 1)))",
                ),
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )
        })
        .expect("the exported library must pass all checks");
        let private = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            check_prepared_library(
                prepared(
                    "(module {} m \
             (def {} value (lit {type: (t-prim {} i32)} 1)))",
                ),
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )
        })
        .expect("the private library must pass all checks");
        assert_eq!(private.program().type_env(), exported.program().type_env());
        assert!(
            !private
                .type_env()
                .matches_checked_program(exported.program())
        );

        let rejection = validate_cached_library(
            private.type_env().clone(),
            exported.program().clone(),
            &mut chelis_types::NativeServices::new(None),
        )
        .expect_err("a same-shape foreign context must fail at the cache boundary");

        assert!(matches!(rejection, LibraryRejection::ContextMismatch));
    }

    #[test]
    fn cached_library_parser_rejects_a_mismatched_type_environment() {
        let library = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            check_prepared_library(
                prepared("(def {} one (lit {type: (t-prim {} i32)} 1))"),
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )
        })
        .expect("the library must pass all checks");

        let rejection = validate_cached_library(
            chelis_types::TypeEnv::empty(),
            library.program().clone(),
            &mut chelis_types::NativeServices::new(None),
        )
        .expect_err("a foreign type environment must fail at the cache boundary");

        assert!(matches!(rejection, LibraryRejection::ContextMismatch));
    }

    #[test]
    fn lowering_preserves_canonical_tuple_roots() {
        let lowered = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            lower_checked(
                checked(include_str!("../tests/fixtures/pair.dp")),
                LoweringMode::Strict,
                runtime,
            )
        })
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
        let lowered = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            lower_checked(
                checked(include_str!("../tests/fixtures/lookup.dp")),
                LoweringMode::Strict,
                runtime,
            )
        })
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
        let lowered = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            lower_checked(
                checked(include_str!("../tests/fixtures/host.dp")),
                LoweringMode::AllowHostOnly,
                runtime,
            )
        })
        .expect("the host-only value must lower without tensor roots");
        assert!(lowered.dag().roots().is_empty());
        assert!(lowered.named_roots().is_empty());
    }

    #[test]
    fn type_rejection_has_no_checked_product() {
        assert!(matches!(
            analyze_prepared(
                prepared(include_str!("../tests/fixtures/type_rejection.dp")),
                &mut chelis_types::NativeServices::new(None)
            ),
            PreparedTypeAnalysisOutcome::Rejected { .. }
        ));
    }

    #[test]
    fn effect_rejection_uses_the_narrow_semantic_error() {
        assert!(matches!(
            chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| complete_checks(
                accepted_analysis(include_str!("../tests/fixtures/effect_rejection.dp")),
                SemanticContext::Isolated,
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )),
            Err(SemanticRejection::Effects { .. })
        ));
    }

    #[test]
    fn linearity_rejection_uses_the_narrow_semantic_error() {
        assert!(matches!(
            chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| complete_checks(
                accepted_analysis(include_str!("../tests/fixtures/linearity_rejection.dp")),
                SemanticContext::Isolated,
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )),
            Err(SemanticRejection::Linearity { .. })
        ));
    }

    #[test]
    fn strict_lowering_rejection_retains_the_lower_diagnostic() {
        assert!(matches!(
            chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| lower_checked(
                checked(include_str!("../tests/fixtures/lower_rejection.dp")),
                LoweringMode::Strict,
                runtime,
            )),
            Err(CoreLowerError::Lower(_))
        ));
    }
}
