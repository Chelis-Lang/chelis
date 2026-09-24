//! Authoritative checker diagnostic-session boundary.
//!
//! `DiagnosticSink` deliberately exposes no constructor, owned storage,
//! conversion, clone, default, or mutable dereference outside this module.
//! Lower checker layers can only receive an already-open sink, so a fresh
//! [`crate::errors::ErrorWitness`] can never be minted into a throwaway
//! `Vec<CheckError>`.

use crate::context::TypeEnv;
use crate::errors::{CheckError, ErrorWitness};
use crate::infer::recursion::{Caller, RecursiveState};
use crate::infer::{
    CheckedProgram, InferResult, InferStats, ProgramOrigin, SignatureInferenceMetadata,
};
use crate::opacity::OpacityContextData;
use chelis_unord::UnordMap;

/// A sink-issued boundary for later diagnostic iteration.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DiagnosticCheckpoint {
    offset: usize,
}

/// Stable identity shared by one standalone signature and its matching
/// definition.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct DeclarationDiagnosticOwner {
    lexical_module: Option<String>,
    lexical_scope_occurrence: usize,
    declaration_name: String,
    declaration_occurrence: usize,
}

impl DeclarationDiagnosticOwner {
    pub(crate) fn new(
        lexical_module: Option<&str>,
        lexical_scope_occurrence: usize,
        declaration_name: &str,
        declaration_occurrence: usize,
    ) -> Self {
        Self {
            lexical_module: lexical_module.map(str::to_string),
            lexical_scope_occurrence,
            declaration_name: declaration_name.to_string(),
            declaration_occurrence,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum DeclarationTypeDiagnosticClass {
    UnknownPrimitive,
    RejectedPrimitive,
    UndeclaredTypeVariable,
    UndeclaredDimensionVariable,
    UndeclaredRankVariable,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum TypeResolutionDiagnosticOwner {
    Declaration {
        declaration: DeclarationDiagnosticOwner,
        class: DeclarationTypeDiagnosticClass,
    },
    UnknownPrimitiveLocation {
        span_offset: Option<usize>,
        span_id: Option<String>,
    },
}

/// The sole destination accepted by witness-minting checker code.
///
/// Its storage and constructor are private to this module.  The narrow
/// mutation surface is enough for ordinary diagnostics while deliberately
/// preventing early extraction or replacement of the canonical vector.
pub struct DiagnosticSink<'session> {
    errors: &'session mut Vec<CheckError>,
    type_resolution_witnesses: UnordMap<(TypeResolutionDiagnosticOwner, String), ErrorWitness>,
    opacity: Option<OpacityContextData>,
    recursive: RecursiveState,
}

impl<'session> DiagnosticSink<'session> {
    /// A checking pass starts with no active group, including declaration
    /// collection. Reentrant passes move the enclosing frame into the scope.
    pub(crate) fn enter_recursive_inference(&mut self) -> RecursiveInferenceScope<'_, 'session> {
        let previous = std::mem::take(&mut self.recursive);
        RecursiveInferenceScope {
            sink: self,
            previous,
        }
    }

    pub(crate) fn recursive_state(&self) -> &RecursiveState {
        &self.recursive
    }

    pub(crate) fn recursive_state_mut(&mut self) -> &mut RecursiveState {
        &mut self.recursive
    }

    pub(crate) fn enter_recursive_caller(
        &mut self,
        caller: Option<Caller>,
    ) -> RecursiveCallerScope<'_, 'session> {
        let active = caller.is_some();
        let previous = if active {
            self.recursive.replace_caller(caller)
        } else {
            None
        };
        RecursiveCallerScope {
            sink: self,
            previous,
            active,
        }
    }

    pub(crate) fn finish_recursive_group(&mut self, subst: &mut crate::unify::Subst) {
        if let Some(group) = self.recursive.take_group() {
            group.finish(subst, self);
        }
    }

    /// Install this inference pass's policy without opening another diagnostic
    /// destination. The scope restores the enclosing pass on return or unwind.
    pub(crate) fn install_opacity_context(
        &mut self,
        context: OpacityContextData,
    ) -> OpacityScope<'_, 'session> {
        let previous = self.opacity.replace(context);
        OpacityScope {
            sink: self,
            previous,
        }
    }

    pub(crate) fn opacity_context(&self) -> Option<&OpacityContextData> {
        self.opacity.as_ref()
    }

    pub(crate) fn set_opacity_current_item(
        &mut self,
        module: Option<String>,
        declaration: Option<String>,
    ) {
        if let Some(context) = self.opacity.as_mut() {
            context.current_module = module;
            context.current_decl = declaration;
        }
    }

    pub(crate) fn push(&mut self, error: CheckError) {
        self.errors.push(error);
    }

    pub(crate) fn declaration_type_witness(
        &self,
        owner: &DeclarationDiagnosticOwner,
        spelling: &str,
        class: DeclarationTypeDiagnosticClass,
    ) -> Option<ErrorWitness> {
        self.type_resolution_witnesses
            .get(&(
                TypeResolutionDiagnosticOwner::Declaration {
                    declaration: owner.clone(),
                    class,
                },
                spelling.to_string(),
            ))
            .copied()
    }

    pub(crate) fn record_declaration_type_witness(
        &mut self,
        owner: DeclarationDiagnosticOwner,
        spelling: String,
        class: DeclarationTypeDiagnosticClass,
        witness: ErrorWitness,
    ) {
        self.type_resolution_witnesses.insert(
            (
                TypeResolutionDiagnosticOwner::Declaration {
                    declaration: owner,
                    class,
                },
                spelling,
            ),
            witness,
        );
    }

    pub(crate) fn unknown_primitive_site_witness(
        &self,
        span_offset: Option<usize>,
        span_id: Option<&str>,
        primitive_name: &str,
    ) -> Option<ErrorWitness> {
        self.type_resolution_witnesses
            .get(&(
                TypeResolutionDiagnosticOwner::UnknownPrimitiveLocation {
                    span_offset,
                    span_id: span_id.map(str::to_string),
                },
                primitive_name.to_string(),
            ))
            .copied()
    }

    pub(crate) fn record_unknown_primitive_site_witness(
        &mut self,
        span_offset: Option<usize>,
        span_id: Option<String>,
        primitive_name: String,
        witness: ErrorWitness,
    ) {
        self.type_resolution_witnesses.insert(
            (
                TypeResolutionDiagnosticOwner::UnknownPrimitiveLocation {
                    span_offset,
                    span_id,
                },
                primitive_name,
            ),
            witness,
        );
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

/// A scoped policy change borrowing the existing authoritative sink. No
/// diagnostic vector or witness table is replaced, copied, or discarded.
#[must_use = "opacity enforcement lasts only while this scope is held"]
pub(crate) struct OpacityScope<'sink, 'session> {
    sink: &'sink mut DiagnosticSink<'session>,
    previous: Option<OpacityContextData>,
}

impl<'session> OpacityScope<'_, 'session> {
    pub(crate) fn sink(&mut self) -> &mut DiagnosticSink<'session> {
        self.sink
    }
}

impl Drop for OpacityScope<'_, '_> {
    fn drop(&mut self) {
        self.sink.opacity = self.previous.take();
    }
}

#[must_use = "the enclosing recursive frame is restored when this scope drops"]
pub(crate) struct RecursiveInferenceScope<'sink, 'session> {
    sink: &'sink mut DiagnosticSink<'session>,
    previous: RecursiveState,
}

impl<'session> RecursiveInferenceScope<'_, 'session> {
    pub(crate) fn sink(&mut self) -> &mut DiagnosticSink<'session> {
        self.sink
    }
}

impl Drop for RecursiveInferenceScope<'_, '_> {
    fn drop(&mut self) {
        std::mem::swap(&mut self.sink.recursive, &mut self.previous);
    }
}

#[must_use = "the enclosing recursive caller is restored when this scope drops"]
pub(crate) struct RecursiveCallerScope<'sink, 'session> {
    sink: &'sink mut DiagnosticSink<'session>,
    previous: Option<Caller>,
    active: bool,
}

impl<'session> RecursiveCallerScope<'_, 'session> {
    pub(crate) fn sink(&mut self) -> &mut DiagnosticSink<'session> {
        self.sink
    }
}

impl Drop for RecursiveCallerScope<'_, '_> {
    fn drop(&mut self) {
        if self.active {
            let _finished_caller = self.sink.recursive.replace_caller(self.previous.take());
        }
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
            type_resolution_witnesses: UnordMap::new(),
            opacity: None,
            recursive: RecursiveState::default(),
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
            type_resolution_witnesses: UnordMap::new(),
            opacity: None,
            recursive: RecursiveState::default(),
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

#[cfg(test)]
mod recursive_scope_tests {
    use super::*;
    use crate::env::Env;
    use crate::errors::CheckErrorKind;
    use crate::types::{Prim, Type, VarGen};
    use crate::unify::{Subst, unify};
    use chelis_unord::UnordSet;

    struct NestedAbort;

    fn fixture() -> (Env, VarGen, Subst) {
        let mut env = Env::new();
        let mut variables = VarGen::default();
        let mut subst = Subst::new();
        for name in ["left", "right"] {
            let level = subst.enter_level(&variables);
            let parameter = variables.fresh_tvar();
            let ty = Type::Fn(vec![Type::Var(parameter)], Box::new(Type::Var(parameter)));
            subst.leave_level(level, &variables);
            let scheme = env.generalize(&ty, &mut subst, &UnordSet::new());
            env.bind(name.to_string(), scheme);
        }
        (env, variables, subst)
    }

    fn begin_group(sink: &mut DiagnosticSink<'_>, env: &Env) {
        sink.recursive_state_mut()
            .begin_group([("left", true), ("right", true)].into_iter(), env);
    }

    fn caller(
        sink: &DiagnosticSink<'_>,
        env: &Env,
        variables: &mut VarGen,
        subst: &mut Subst,
        name: &str,
        binder: &str,
    ) -> Caller {
        let own = env.instantiate_scheme(env.lookup(name).unwrap(), variables, subst);
        let binders = [binder.to_string()].into_iter().collect();
        sink.recursive_state()
            .prepare_caller(name, Some(&binders), &own.tvars)
            .expect("the caller belongs to the active group")
    }

    fn reference(
        sink: &mut DiagnosticSink<'_>,
        env: &Env,
        variables: &mut VarGen,
        subst: &mut Subst,
        name: &str,
        offset: usize,
    ) -> Type {
        let scheme = env.lookup(name).unwrap();
        let level = subst.enter_level(variables);
        let instantiated = env.instantiate_scheme(scheme, variables, subst);
        assert!(
            sink.recursive_state()
                .should_record_occurrence(name, scheme)
        );
        sink.recursive_state_mut().record_occurrence(
            name,
            &instantiated.tvars,
            Some(format!("surf:{offset}..{}", offset + 1)),
            Some(offset),
        );
        subst.leave_level(level, variables);
        instantiated.ty
    }

    fn identity_type(primitive: Prim) -> Type {
        Type::Fn(vec![Type::Prim(primitive)], Box::new(Type::Prim(primitive)))
    }

    fn admits_independent_uses(
        sink: &DiagnosticSink<'_>,
        env: &Env,
        variables: &mut VarGen,
        subst: &mut Subst,
        ty: &Type,
    ) -> bool {
        let alias = env.generalize(ty, subst, sink.recursive_state().pinned_vars());
        let mut probe = subst.clone();
        let first = env.instantiate(&alias, variables, &mut probe);
        let second = env.instantiate(&alias, variables, &mut probe);
        unify(&first, &identity_type(Prim::Int32), &mut probe).expect("first alias use");
        unify(&second, &identity_type(Prim::Bool), &mut probe).is_ok()
    }

    #[test]
    fn recursive_pins_are_inputs_of_one_session_not_ambient_generalization() {
        let (env, mut variables, mut subst) = fixture();
        run_result(|outer| {
            begin_group(outer, &env);
            let current = caller(outer, &env, &mut variables, &mut subst, "left", "a");
            let mut outer_scope = outer.enter_recursive_caller(Some(current));
            let outer = outer_scope.sink();
            let outer_reference = reference(outer, &env, &mut variables, &mut subst, "right", 10);
            assert!(!admits_independent_uses(
                outer,
                &env,
                &mut variables,
                &mut subst,
                &outer_reference,
            ));

            run_result(|inner| {
                begin_group(inner, &env);
                let current = caller(inner, &env, &mut variables, &mut subst, "right", "b");
                let mut inner_scope = inner.enter_recursive_caller(Some(current));
                let inner = inner_scope.sink();
                let inner_reference =
                    reference(inner, &env, &mut variables, &mut subst, "left", 20);
                assert!(admits_independent_uses(
                    inner,
                    &env,
                    &mut variables,
                    &mut subst,
                    &outer_reference,
                ));
                assert!(admits_independent_uses(
                    outer,
                    &env,
                    &mut variables,
                    &mut subst,
                    &inner_reference,
                ));
                assert!(!admits_independent_uses(
                    inner,
                    &env,
                    &mut variables,
                    &mut subst,
                    &inner_reference,
                ));
                Ok(())
            })
            .expect("the independent session has no rejected occurrence");
            assert!(!admits_independent_uses(
                outer,
                &env,
                &mut variables,
                &mut subst,
                &outer_reference,
            ));
            Ok(())
        })
        .expect("unbound recursive instantiations have no diagnostic");
    }

    #[test]
    fn recursive_invocations_restore_pending_evidence_on_return_and_unwind() {
        let nested: Vec<chelis_deep::Expr> = serde_json::from_str(include_str!(
            "../tests/fixtures/core/src/recursion_uniformity/direct_polymorphic_recursion_is_rejected_with_the_atom.json"
        ))
        .unwrap();
        for driver in [
            crate::infer::infer_program_in_session,
            crate::infer::infer_ir_program_in_session,
        ] {
            let (env, mut variables, mut subst) = fixture();
            let report = run_result(|sink| {
                begin_group(sink, &env);
                let current = caller(sink, &env, &mut variables, &mut subst, "left", "a");
                {
                    let mut outer_scope = sink.enter_recursive_caller(Some(current));
                    let outer = outer_scope.sink();
                    let first =
                        reference(outer, &env, &mut variables, &mut subst, "right", 1_000_000);
                    driver(&nested, ProgramOrigin::Unlinked, outer);
                    assert!(!admits_independent_uses(
                        outer,
                        &env,
                        &mut variables,
                        &mut subst,
                        &first,
                    ));

                    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let mut invocation = outer.enter_recursive_inference();
                        let inner = invocation.sink();
                        begin_group(inner, &env);
                        let current = caller(inner, &env, &mut variables, &mut subst, "right", "b");
                        let mut member = inner.enter_recursive_caller(Some(current));
                        let abandoned = reference(
                            member.sink(),
                            &env,
                            &mut variables,
                            &mut subst,
                            "left",
                            1_000_001,
                        );
                        unify(&abandoned, &identity_type(Prim::Int32), &mut subst).unwrap();
                        std::panic::panic_any(NestedAbort);
                    }));
                    assert!(
                        unwind
                            .expect_err("nested check unwinds")
                            .is::<NestedAbort>()
                    );
                    assert!(!admits_independent_uses(
                        outer,
                        &env,
                        &mut variables,
                        &mut subst,
                        &first,
                    ));
                    let last =
                        reference(outer, &env, &mut variables, &mut subst, "right", 1_000_002);
                    unify(&first, &identity_type(Prim::Int32), &mut subst).unwrap();
                    unify(&last, &identity_type(Prim::Int32), &mut subst).unwrap();
                }
                sink.finish_recursive_group(&mut subst);
                Ok(())
            })
            .expect_err("completed groups report their nonuniform instantiations");

            assert_eq!(report.errors.len(), 3, "{:?}", report.errors);
            assert!(
                report
                    .errors
                    .iter()
                    .all(|error| matches!(error.kind, CheckErrorKind::TypeMismatch,))
            );
            assert_eq!(report.errors[0].got.as_deref(), Some("[Box[a]]"));
            assert_eq!(
                report.errors[1..]
                    .iter()
                    .map(|error| error.span_offset)
                    .collect::<Vec<_>>(),
                [Some(1_000_000), Some(1_000_002)],
            );
        }
    }

    #[test]
    fn recursive_caller_scopes_restore_attribution_and_retain_recorded_occurrences() {
        let (env, mut variables, mut subst) = fixture();
        let report = run_result(|sink| {
            begin_group(sink, &env);
            let current = caller(sink, &env, &mut variables, &mut subst, "left", "a");
            {
                let mut outer_scope = sink.enter_recursive_caller(Some(current));
                let outer = outer_scope.sink();
                let first = reference(outer, &env, &mut variables, &mut subst, "right", 10);
                unify(&first, &identity_type(Prim::Int32), &mut subst).unwrap();
                {
                    let current = caller(outer, &env, &mut variables, &mut subst, "right", "b");
                    let mut inner = outer.enter_recursive_caller(Some(current));
                    let second =
                        reference(inner.sink(), &env, &mut variables, &mut subst, "left", 20);
                    unify(&second, &identity_type(Prim::Int32), &mut subst).unwrap();
                }
                let third = reference(outer, &env, &mut variables, &mut subst, "right", 30);
                unify(&third, &identity_type(Prim::Int32), &mut subst).unwrap();
                let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let current = caller(outer, &env, &mut variables, &mut subst, "right", "b");
                    let mut inner = outer.enter_recursive_caller(Some(current));
                    let fourth =
                        reference(inner.sink(), &env, &mut variables, &mut subst, "left", 40);
                    unify(&fourth, &identity_type(Prim::Int32), &mut subst).unwrap();
                    std::panic::panic_any(NestedAbort);
                }));
                assert!(
                    unwind
                        .expect_err("nested caller unwinds")
                        .is::<NestedAbort>()
                );
                let fifth = reference(outer, &env, &mut variables, &mut subst, "right", 50);
                unify(&fifth, &identity_type(Prim::Int32), &mut subst).unwrap();
            }
            sink.finish_recursive_group(&mut subst);
            Ok(())
        })
        .expect_err("every completed group's recorded occurrence remains authoritative");

        assert_eq!(
            report
                .errors
                .iter()
                .map(|error| (error.span_offset, error.expected.as_deref()))
                .collect::<Vec<_>>(),
            [
                (Some(10), Some("[a]")),
                (Some(20), Some("[b]")),
                (Some(30), Some("[a]")),
                (Some(40), Some("[b]")),
                (Some(50), Some("[a]")),
            ],
        );
    }
}

#[cfg(test)]
mod opacity_scope_tests {
    use super::*;
    use crate::errors::CheckErrorKind;
    use crate::opacity::{OpaqueAction, check_opaque_use};

    fn library() -> TypeEnv {
        let program = chelis_deep::parse_and_stamp_file(include_str!(
            "../tests/fixtures/core/tests/opaque_types/shared_104.deep"
        ))
        .expect("opaque library parses");
        crate::build_type_env_from_library(&program, crate::ProgramOrigin::Unlinked)
            .expect("opaque library checks")
    }

    fn policy(library: &TypeEnv, module: &str) -> OpacityContextData {
        let mut context = OpacityContextData::from_meta(library.inner().opacity.clone());
        context.current_module = Some(module.to_string());
        context
    }

    fn rejects_construction(library: &TypeEnv, sink: &mut DiagnosticSink<'_>) -> bool {
        check_opaque_use(
            OpaqueAction::RecordConstruction,
            "Probability",
            &library.inner().adt_reg,
            sink,
        )
    }

    #[test]
    fn opacity_policy_belongs_to_the_diagnostic_session_not_the_thread() {
        let library = library();
        let report = run_result(|outer| {
            let mut outer_scope = outer.install_opacity_context(policy(&library, "agent.strategy"));
            let outer = outer_scope.sink();

            // Annotation-only sessions do not inherit another check's policy.
            run_result(|annotation| {
                assert!(!rejects_construction(&library, annotation));
                Ok(())
            })
            .expect("an inactive session does not enforce the outer policy");

            run_result(|inner| {
                let mut inner_scope = inner.install_opacity_context(policy(&library, "stats.prob"));
                assert!(rejects_construction(&library, outer));
                assert!(!rejects_construction(&library, inner_scope.sink()));
                Ok(())
            })
            .expect("the defining module remains authorized");
            Ok(())
        })
        .expect_err("the outer session still rejects construction outside the defining module");

        assert_eq!(report.errors.len(), 1);
        assert!(matches!(
            report.errors[0].kind,
            CheckErrorKind::OpaqueTypeViolation
        ));
    }

    #[test]
    fn opacity_scopes_restore_policy_on_return_and_unwind_without_losing_errors() {
        let library = library();
        let report = run_result(|sink| {
            {
                let mut outer_scope = sink.install_opacity_context(policy(&library, "stats.prob"));
                let outer = outer_scope.sink();
                {
                    let mut inner_scope =
                        outer.install_opacity_context(policy(&library, "agent.strategy"));
                    assert!(rejects_construction(&library, inner_scope.sink()));
                }
                assert!(!rejects_construction(&library, outer));

                let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut inner_scope =
                        outer.install_opacity_context(policy(&library, "agent.strategy"));
                    assert!(rejects_construction(&library, inner_scope.sink()));
                    std::panic::panic_any("abort nested opacity check");
                }));
                assert!(unwind.is_err());
                assert!(!rejects_construction(&library, outer));

                outer.set_opacity_current_item(Some("agent.strategy".into()), None);
                assert!(rejects_construction(&library, outer));
            }
            assert!(!rejects_construction(&library, sink));
            Ok(())
        })
        .expect_err("scope restoration must retain every authoritative violation");

        assert_eq!(report.errors.len(), 3);
        assert!(
            report
                .errors
                .iter()
                .all(|error| matches!(error.kind, CheckErrorKind::OpaqueTypeViolation))
        );
    }
}

#[cfg(test)]
mod rejected_signature_resolution_tests {
    use super::*;
    use crate::deep_type::{BinderMode, DeepTypeResolver, TypeResolutionEnv, TypeUseSite};
    use crate::types::{Prim, Type, VarGen};

    fn resolve<T>(
        source: &str,
        operation: impl FnOnce(&mut DeepTypeResolver<'_, '_, '_>, &chelis_deep::Expr) -> T,
    ) -> (T, Vec<CheckError>) {
        resolve_with_binders(source, &[], operation)
    }

    fn resolve_with_binders<T>(
        source: &str,
        binder_names: &[&str],
        operation: impl FnOnce(&mut DeepTypeResolver<'_, '_, '_>, &chelis_deep::Expr) -> T,
    ) -> (T, Vec<CheckError>) {
        let program = chelis_deep::parse_and_stamp_file(&format!("(defsig {{}} fixture {source})"))
            .expect("parse declaration type");
        let chelis_deep::Expr::Node(declaration, _) = &program[0] else {
            panic!("a declaration is a stamped node");
        };
        let ty = declaration.children_slice().last().expect("defsig type");
        let headers = TypeResolutionEnv::default();
        let binders = binder_names
            .iter()
            .map(|name| (*name).to_string())
            .collect();
        let mut variables = VarGen::default();
        let mut errors = Vec::new();
        let result = {
            let mut sink = DiagnosticSink {
                errors: &mut errors,
                type_resolution_witnesses: UnordMap::new(),
                opacity: None,
                recursive: RecursiveState::default(),
            };
            let mut resolver = DeepTypeResolver::new(
                TypeUseSite::Defsig,
                BinderMode::ExplicitGeneric(&binders),
                &headers,
                &mut variables,
                &mut sink,
            );
            operation(&mut resolver, ty)
        };
        (result, errors)
    }

    #[test]
    fn ordinary_resolution_never_returns_partial_structure_as_success() {
        let (result, errors) = resolve(
            "(t-fn {} (t-prim {} f8e4m3) (t-prim {} i32) (t-prim {} f8e5m2))",
            |resolver, ty| resolver.resolve(ty),
        );
        assert!(result.is_err());
        assert_eq!(errors.len(), 2, "{errors:?}");
        assert!(errors[0].message.contains("f8e4m3"));
        assert!(errors[1].message.contains("f8e5m2"));
    }

    #[test]
    fn only_a_rejected_outcome_carries_partial_signature_structure() {
        let (result, errors) = resolve(
            "(t-fn {} (t-prim {} f8e4m3) (t-prim {} i32) (t-prim {} f8e5m2))",
            |resolver, ty| resolver.resolve_signature(ty),
        );
        let rejected = result.expect_err("an invalid signature cannot resolve");
        let Type::Fn(arguments, result) = rejected.ty else {
            panic!("retain the callable structure");
        };
        assert_eq!(arguments.len(), 2);
        assert!(matches!(arguments[0], Type::Error(_)));
        assert_eq!(arguments[1], Type::Prim(Prim::Int32));
        assert!(matches!(*result, Type::Error(_)));
        assert_eq!(errors.len(), 2, "{errors:?}");
    }

    #[test]
    fn valid_structure_resolves_without_an_error_witness() {
        let (result, errors) = resolve(
            "(t-fn {} (t-prim {} i32) (t-prim {} bool))",
            |resolver, ty| resolver.resolve_signature(ty),
        );
        assert_eq!(
            result.expect("valid signature").into_type(),
            Type::Fn(
                vec![Type::Prim(Prim::Int32)],
                Box::new(Type::Prim(Prim::Bool))
            )
        );
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn parameter_holes_are_distinct_from_named_binders() {
        let (hole, errors) = resolve("(t-var {} _)", |resolver, ty| {
            resolver.resolve_parameter(ty)
        });
        assert!(hole.expect("a parameter admits a hole").is_none());
        assert!(errors.is_empty());
        let (named, errors) = resolve_with_binders("(t-var {} a)", &["a"], |resolver, ty| {
            resolver.resolve_parameter(ty)
        });
        assert!(matches!(
            named
                .expect("named binder")
                .expect("a binder is a constraint")
                .into_type(),
            Type::Var(_)
        ));
        assert!(errors.is_empty());
    }
}

pub(crate) fn infer_program(exprs: &[chelis_deep::Expr], origin: ProgramOrigin) -> InferResult {
    crate::infer::run_on_grown_stack(|| {
        let mut errors = Vec::new();
        let stats = {
            let mut sink = DiagnosticSink {
                errors: &mut errors,
                type_resolution_witnesses: UnordMap::new(),
                opacity: None,
                recursive: RecursiveState::default(),
            };
            match admit_metadata(exprs, &mut sink) {
                Ok(()) => crate::infer::infer_program_in_session(exprs, origin, &mut sink),
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
                type_resolution_witnesses: UnordMap::new(),
                opacity: None,
                recursive: RecursiveState::default(),
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
mod unresolved_operand_reconcile_tests {
    use super::*;
    use crate::errors::CheckErrorKind;
    use crate::infer::{ReconcileMutationCase, run_reconcile_mutation_case};

    /// This module lives here, beside the finalization cases below, for the
    /// same reason they do: `DiagnosticSink`'s constructor is private to this
    /// file on purpose, and adding a second production constructor to hand
    /// one to a test would be exactly the seam
    /// `rt800_checker_session_source_contract.rs` pins shut. The case runs in
    /// `infer`; only the sink is built here.
    fn run(case: ReconcileMutationCase) -> (String, bool, Vec<CheckError>) {
        let mut errors = Vec::new();
        let (produced, bound) = {
            let mut sink = DiagnosticSink {
                errors: &mut errors,
                type_resolution_witnesses: UnordMap::new(),
                opacity: None,
                recursive: RecursiveState::default(),
            };
            run_reconcile_mutation_case(case, &mut sink)
        };
        (produced, bound, errors)
    }

    /// REGRESSION TEST for chelis#1512's disagreement branch. Reconciling a
    /// replayed answer against the type the suspended call published must
    /// push exactly one typed error AND hand back the PRODUCED type.
    /// Returning the published type instead would be a silent override: the
    /// program would disagree with itself and still check clean.
    #[test]
    fn a_replayed_answer_that_disagrees_is_reported_and_never_silently_overridden() {
        let (produced, _, errors) = run(ReconcileMutationCase::Disagrees);
        assert_eq!(
            produced, "tensor[4, f32]",
            "the replayed rule's own answer is what the call produces"
        );
        assert_eq!(
            errors.len(),
            1,
            "a disagreement is one diagnostic, got {errors:?}"
        );
        assert!(matches!(errors[0].kind, CheckErrorKind::TypeMismatch));
        assert!(
            errors[0].message.contains(
                "`permute` result does not match the type this call produces once its operand is known"
            ),
            "the diagnostic must say which call disagreed, got {:?}",
            errors[0].message
        );
    }

    /// NEGATIVE TWIN. Agreement is silent, and the published variable is
    /// bound to the produced type rather than left free: a suspended call
    /// publishes a fresh variable, so leaving it unbound would strand every
    /// consumer of the call.
    #[test]
    fn a_replayed_answer_that_agrees_reports_nothing_and_binds_the_published_variable() {
        let (produced, bound, errors) = run(ReconcileMutationCase::Agrees);
        assert_eq!(produced, "tensor[3, f32]");
        assert!(
            bound,
            "the published variable must be bound to the rule's answer"
        );
        assert!(
            errors.is_empty(),
            "agreement must be silent, got {errors:?}"
        );
    }

    #[test]
    fn replay_preserves_the_consumers_dtype_family_contract() {
        let (produced, _, errors) = run(ReconcileMutationCase::FamilyDisagrees);
        assert_eq!(produced, "tensor[3, i32]");
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(matches!(errors[0].kind, CheckErrorKind::PrecisionMismatch));
        assert!(errors[0].message.contains("Float") && errors[0].message.contains("i32"));
        let (produced, bound, errors) = run(ReconcileMutationCase::FamilyAgrees);
        assert_eq!(produced, "tensor[3, f32]");
        assert!(bound);
        assert!(errors.is_empty(), "{errors:?}");
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
                type_resolution_witnesses: UnordMap::new(),
                opacity: None,
                recursive: RecursiveState::default(),
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
            type_resolution_witnesses: UnordMap::new(),
            opacity: None,
            recursive: RecursiveState::default(),
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
    origin: ProgramOrigin,
) -> Result<TypeEnv, InferResult> {
    crate::infer::run_on_grown_stack(|| {
        run_with_metadata(exprs, |sink| {
            crate::infer::build_type_env_from_library_in_session(exprs, origin, sink)
        })
    })
}

pub(crate) fn resolve_declared_surface(
    exprs: &[chelis_deep::Expr],
) -> Result<crate::infer::DeclaredTypeSurface, InferResult> {
    crate::infer::run_on_grown_stack(|| {
        run_with_metadata(exprs, |sink| {
            crate::infer::resolve_declared_surface_in_session(exprs, sink)
        })
    })
}

pub(crate) fn build_compiled_library_context(
    exprs: &[chelis_deep::Expr],
    origin: ProgramOrigin,
) -> Result<(TypeEnv, CheckedProgram), InferResult> {
    run_with_metadata(exprs, |sink| {
        crate::infer::build_compiled_library_context_in_session(exprs, origin, sink)
    })
}

pub(crate) fn build_compiled_library_context_with_base(
    base: &TypeEnv,
    exprs: &[chelis_deep::Expr],
    origin: ProgramOrigin,
) -> Result<(TypeEnv, CheckedProgram), InferResult> {
    run_with_metadata(exprs, |sink| {
        crate::infer::build_compiled_library_context_with_base_in_session(base, exprs, origin, sink)
    })
}

pub(crate) fn check_ir_with_signature_context(
    context: &TypeEnv,
    signature_context: &SignatureInferenceMetadata,
    exprs: &[chelis_deep::Expr],
    origin: ProgramOrigin,
) -> Result<CheckedProgram, InferResult> {
    crate::infer::run_on_grown_stack(|| {
        run_with_metadata(exprs, |sink| {
            crate::infer::check_ir_with_signature_context_in_session(
                context,
                signature_context,
                exprs,
                origin,
                sink,
            )
        })
    })
}

pub(crate) fn check_typed_program(
    exprs: &[chelis_deep::Expr],
    origin: ProgramOrigin,
) -> Result<CheckedProgram, InferResult> {
    crate::infer::run_on_grown_stack(|| {
        run_with_metadata(exprs, |sink| {
            crate::infer::check_typed_program_in_session(exprs, origin, sink)
        })
    })
}

pub(crate) fn infer_ir_program(exprs: &[chelis_deep::Expr], origin: ProgramOrigin) -> InferResult {
    crate::infer::run_on_grown_stack(|| {
        let mut errors = Vec::new();
        let stats = {
            let mut sink = DiagnosticSink {
                errors: &mut errors,
                type_resolution_witnesses: UnordMap::new(),
                opacity: None,
                recursive: RecursiveState::default(),
            };
            match admit_metadata(exprs, &mut sink) {
                Ok(()) => crate::infer::infer_ir_program_in_session(exprs, origin, &mut sink),
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

#[cfg(test)]
mod builtin_selection_tests {
    use super::*;
    use crate::builtin_discovery::BuiltinCaseSelection;

    fn selections(source: &str) -> Result<Vec<BuiltinCaseSelection>, InferResult> {
        let deep = chelis_deep::parser::parse_str(source).expect("core discovery fixture");
        run_with_metadata(&deep, |sink| {
            Ok(crate::infer::builtin_selection_probe(&deep, sink))
        })
    }

    #[test]
    fn builtin_atom_discovery_inference_resolves_enclosing_operands_and_preserves_shadows() {
        let selected = selections(include_str!(
            "../tests/fixtures/core/src/session/map_to_string.deep"
        ))
        .unwrap();
        assert!(selected.contains(&BuiltinCaseSelection::Resolved(
            "Boundary:to_string:ToStringScalar".into()
        )));
        assert!(
            selected
                .iter()
                .all(|s| matches!(s, BuiltinCaseSelection::Resolved(_)))
        );
        for bound in ["float", "int", "numeric"] {
            let selected = selections(
                &include_str!("../tests/fixtures/core/src/session/equal_bound.deep")
                    .replace("BOUND", bound),
            )
            .unwrap();
            assert!(selected.contains(&BuiltinCaseSelection::Resolved("Numeric:eq:TableA".into())));
            assert!(!selected.contains(&BuiltinCaseSelection::Resolved(
                "Container:eq:EqRecursive".into()
            )));
        }
        let selected = selections(include_str!(
            "../tests/fixtures/core/src/session/shadow_to_string.deep"
        ))
        .unwrap();
        assert!(
            selected.is_empty(),
            "lexical shadow acquired builtin metadata: {selected:?}"
        );
        assert!(
            selections(include_str!(
                "../tests/fixtures/core/src/session/invalid_len.deep"
            ))
            .is_err(),
            "concrete invalid case must not skip deferred resolution"
        );
    }
}
