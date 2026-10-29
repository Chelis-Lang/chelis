//! Native checker entry points and private checked-product admission.

use crate::context::{LibraryProofId, TypeEnv};
use crate::native_stack::NativeStack;
use chelis_checker_core::adt::AdtRegistry;
use chelis_checker_core::deep_type::TypeResolutionEnv;
use chelis_checker_core::infer::{
    CheckedLocalAscription, DeclaredTypeSurface, SignatureInferenceMetadata,
};
use chelis_checker_core::{CheckFacts, CompletedCheck, InferResult, InferStats, LinearityInfo};
use chelis_deep::ast as deep;
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::time::Instant;

pub use crate::native_stack::{
    reset_grow_segment_bytes_for_test, run_on_grown_stack, run_on_grown_stack_of,
    set_grow_segment_bytes_for_test,
};

struct ProfileObserver {
    phase_start: Cell<Instant>,
    declaration_start: Cell<Option<Instant>>,
}
impl chelis_checker_core::CheckObserver for ProfileObserver {
    fn phase(&self, label: &str) {
        let (prefix, label) = label.split_once(':').unwrap_or(("check_ir_sub", label));
        if label != "start" {
            eprintln!(
                "{prefix}: {:>8.4}s {}",
                self.phase_start.get().elapsed().as_secs_f64(),
                label
            );
        }
        self.phase_start.set(Instant::now());
    }
    fn declaration_begin(&self, _name: &str) {
        self.declaration_start.set(Some(Instant::now()));
    }
    fn declaration_end(&self, name: &str) {
        if let Some(start) = self.declaration_start.take() {
            eprintln!(
                "infer_ir_decl: {:>8.4}s {}",
                start.elapsed().as_secs_f64(),
                name
            );
        }
    }
}

pub(crate) fn with_resources<R>(
    run: impl FnOnce(&chelis_checker_core::Resources<'_, NativeStack>) -> R,
) -> R {
    with_resources_for(false, run)
}

fn with_profile_resources<R>(
    run: impl FnOnce(&chelis_checker_core::Resources<'_, NativeStack>) -> R,
) -> R {
    with_resources_for(true, run)
}

fn with_resources_for<R>(
    profile: bool,
    run: impl FnOnce(&chelis_checker_core::Resources<'_, NativeStack>) -> R,
) -> R {
    let cancellation = crate::cancel::current_cancel_token();
    let observer = profile
        .then(|| std::env::var_os("CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL"))
        .flatten()
        .filter(|value| value == "1")
        .map(|_| ProfileObserver {
            phase_start: Cell::new(Instant::now()),
            declaration_start: Cell::new(None),
        });
    let mut resources = chelis_checker_core::Resources::new(&NativeStack);
    resources.cancellation = cancellation
        .as_ref()
        .map(|token| token as &dyn chelis_checker_core::Cancellation);
    resources.policy.linked_program = crate::opacity::linked_program();
    resources.stack_bytes = crate::native_stack::grow_segment_bytes();
    resources.generalize_sweep_oracle =
        if chelis_checker_core::env::generalize_sweep_oracle_available() {
            crate::generalize::enabled()
        } else {
            true
        };
    resources.observer = observer
        .as_ref()
        .map(|value| value as &dyn chelis_checker_core::CheckObserver);
    run(&resources)
}

#[derive(Debug, Clone)]
pub struct CheckedProgram {
    facts: CheckFacts,
    library_proof_id: Option<LibraryProofId>,
    context_library_proof_id: Option<LibraryProofId>,
}

#[derive(Serialize)]
#[serde(rename = "CheckedProgram")]
struct CheckedProgramWireRef<'a> {
    annotated_exprs: &'a Vec<deep::Expr>,
    type_env: &'a BTreeMap<String, deep::Expr>,
    local_ascriptions: &'a Vec<CheckedLocalAscription>,
    linearity: &'a LinearityInfo,
    signature_inference: &'a SignatureInferenceMetadata,
    type_headers: &'a TypeResolutionEnv,
    adt_registry: &'a AdtRegistry,
    infer_stats: InferStats,
    library_proof_id: Option<LibraryProofId>,
    context_library_proof_id: Option<LibraryProofId>,
}
#[derive(Deserialize)]
#[serde(rename = "CheckedProgram")]
struct CheckedProgramWire {
    annotated_exprs: Vec<deep::Expr>,
    type_env: BTreeMap<String, deep::Expr>,
    local_ascriptions: Vec<CheckedLocalAscription>,
    linearity: LinearityInfo,
    signature_inference: SignatureInferenceMetadata,
    type_headers: TypeResolutionEnv,
    #[serde(default)]
    adt_registry: AdtRegistry,
    #[serde(default)]
    infer_stats: InferStats,
    #[serde(default)]
    library_proof_id: Option<LibraryProofId>,
    #[serde(default)]
    context_library_proof_id: Option<LibraryProofId>,
}
impl Serialize for CheckedProgram {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        CheckedProgramWireRef {
            annotated_exprs: &self.facts.annotated_exprs,
            type_env: &self.facts.type_env,
            local_ascriptions: &self.facts.local_ascriptions,
            linearity: &self.facts.linearity,
            signature_inference: &self.facts.signature_inference,
            type_headers: &self.facts.type_headers,
            adt_registry: &self.facts.adt_registry,
            infer_stats: self.facts.infer_stats,
            library_proof_id: self.library_proof_id,
            context_library_proof_id: self.context_library_proof_id,
        }
        .serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for CheckedProgram {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = CheckedProgramWire::deserialize(deserializer)?;
        Ok(Self {
            facts: CheckFacts {
                annotated_exprs: wire.annotated_exprs,
                type_env: wire.type_env,
                local_ascriptions: wire.local_ascriptions,
                linearity: wire.linearity,
                signature_inference: wire.signature_inference,
                type_headers: wire.type_headers,
                adt_registry: wire.adt_registry,
                infer_stats: wire.infer_stats,
            },
            library_proof_id: wire.library_proof_id,
            context_library_proof_id: wire.context_library_proof_id,
        })
    }
}

impl CheckedProgram {
    fn from_completion(completion: CompletedCheck) -> Self {
        Self {
            facts: completion.into_facts(),
            library_proof_id: None,
            context_library_proof_id: None,
        }
    }
    pub(crate) fn facts(&self) -> &CheckFacts {
        &self.facts
    }
    pub(crate) fn with_facts(&self, facts: CheckFacts) -> Self {
        Self {
            facts,
            library_proof_id: self.library_proof_id,
            context_library_proof_id: self.context_library_proof_id,
        }
    }
    pub fn exprs(&self) -> &[deep::Expr] {
        self.facts.exprs()
    }
    pub fn annotated_exprs(&self) -> &[deep::Expr] {
        self.facts.annotated_exprs()
    }
    pub fn type_env(&self) -> &BTreeMap<String, deep::Expr> {
        self.facts.type_env()
    }
    pub fn local_ascriptions(&self) -> &[CheckedLocalAscription] {
        self.facts.local_ascriptions()
    }
    pub fn linearity(&self) -> &LinearityInfo {
        self.facts.linearity()
    }
    pub fn signature_inference(&self) -> &SignatureInferenceMetadata {
        self.facts.signature_inference()
    }
    pub fn adt_registry(&self) -> &AdtRegistry {
        self.facts.adt_registry()
    }
    pub fn infer_stats(&self) -> InferStats {
        self.facts.infer_stats()
    }
    pub fn library_proof_id(&self) -> Option<LibraryProofId> {
        self.library_proof_id
    }
    pub fn with_linearity(mut self, linearity: LinearityInfo) -> Self {
        self.facts = self.facts.with_linearity(linearity);
        self
    }
    pub fn try_with_effect_annotations(
        &self,
        annotated_exprs: Vec<deep::Expr>,
    ) -> Result<Self, InferResult> {
        with_resources(|resources| {
            self.facts
                .try_with_effect_annotations(annotated_exprs, resources)
        })
        .map(|facts| self.with_facts(facts))
    }
    pub fn compose(library: &Self, new_code: &Self) -> Option<Self> {
        if library.library_proof_id.is_none()
            || new_code.context_library_proof_id != library.library_proof_id
        {
            return None;
        }
        let facts = with_resources(|resources| {
            CheckFacts::compose(&library.facts, &new_code.facts, resources)
        })?;
        Some(Self {
            facts,
            library_proof_id: new_code.library_proof_id,
            context_library_proof_id: None,
        })
    }
}

fn bind_library_products(
    mut type_env: TypeEnv,
    mut checked: CheckedProgram,
    base: Option<LibraryProofId>,
) -> (TypeEnv, CheckedProgram) {
    let proof = LibraryProofId::for_library(
        checked.exprs(),
        base,
        type_env.inner().selector_context_digest(),
        type_env.inner().rank_context_digest(),
    );
    type_env.bind_library_proof(proof);
    checked.library_proof_id = Some(proof);
    checked.context_library_proof_id = base;
    (type_env, checked)
}

pub fn infer_program(exprs: &[deep::Expr]) -> InferResult {
    with_resources(|resources| chelis_checker_core::infer_program(exprs, resources))
}
pub fn infer_ir_program(exprs: &[deep::Expr]) -> InferResult {
    with_profile_resources(|resources| chelis_checker_core::infer_ir_program(exprs, resources))
}
pub fn check_ir_program(exprs: &[deep::Expr]) -> Result<CheckedProgram, InferResult> {
    #[cfg(test)]
    TYPE_ANALYSIS_SESSION_COUNT.with(|count| count.set(count.get() + 1));
    with_profile_resources(|resources| chelis_checker_core::check_ir_program(exprs, resources))
        .map(CheckedProgram::from_completion)
}
pub fn check_typed_program(exprs: &[deep::Expr]) -> Result<CheckedProgram, InferResult> {
    with_resources(|resources| chelis_checker_core::check_typed_program(exprs, resources))
        .map(CheckedProgram::from_completion)
}
pub fn check_ir_with_context(
    context: &TypeEnv,
    exprs: &[deep::Expr],
) -> Result<CheckedProgram, InferResult> {
    check_ir_with_signature_context(context, &SignatureInferenceMetadata::default(), exprs)
}
pub fn check_ir_with_signature_context(
    context: &TypeEnv,
    signatures: &SignatureInferenceMetadata,
    exprs: &[deep::Expr],
) -> Result<CheckedProgram, InferResult> {
    #[cfg(test)]
    TYPE_ANALYSIS_SESSION_COUNT.with(|count| count.set(count.get() + 1));
    let completion = with_profile_resources(|resources| {
        chelis_checker_core::check_ir_with_signature_context(
            context.inner(),
            signatures,
            exprs,
            resources,
        )
    })?;
    let mut checked = CheckedProgram::from_completion(completion);
    checked.context_library_proof_id = context.library_proof_id();
    Ok(checked)
}
pub fn build_type_env_from_library(exprs: &[deep::Expr]) -> Result<TypeEnv, InferResult> {
    with_profile_resources(|resources| {
        chelis_checker_core::build_type_env_from_library_with_source(
            exprs,
            |snapshot, annotated| {
                let proof = LibraryProofId::for_library(
                    annotated,
                    None,
                    snapshot.selector_context_digest(),
                    snapshot.rank_context_digest(),
                );
                let mut context = TypeEnv::from_inner(snapshot);
                context.bind_library_proof(proof);
                context
            },
            resources,
        )
    })
}
pub fn build_compiled_library_context(
    exprs: &[deep::Expr],
) -> Result<(TypeEnv, CheckedProgram), InferResult> {
    let (snapshot, completion) = with_profile_resources(|resources| {
        chelis_checker_core::build_compiled_library_context(exprs, resources)
    })?;
    Ok(bind_library_products(
        TypeEnv::from_inner(snapshot),
        CheckedProgram::from_completion(completion),
        None,
    ))
}
pub fn build_compiled_library_context_with_base(
    base: &TypeEnv,
    exprs: &[deep::Expr],
) -> Result<(TypeEnv, CheckedProgram), InferResult> {
    let (snapshot, completion) = with_profile_resources(|resources| {
        chelis_checker_core::build_compiled_library_context_with_base(
            base.inner(),
            exprs,
            resources,
        )
    })?;
    Ok(bind_library_products(
        TypeEnv::from_inner(snapshot),
        CheckedProgram::from_completion(completion),
        base.library_proof_id(),
    ))
}
pub fn resolve_declared_surface(exprs: &[deep::Expr]) -> Result<DeclaredTypeSurface, InferResult> {
    with_resources(|resources| chelis_checker_core::resolve_declared_surface(exprs, resources))
}
pub fn fold_static_int_expr<F>(expr: &deep::Expr, resolve_name: F) -> Option<i64>
where
    F: FnMut(&str) -> Option<i64>,
{
    with_resources(|resources| {
        chelis_checker_core::fold_static_int_expr(expr, resolve_name, resources)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_type_core::{Dim, NominalArg, NominalParamKind, TensorPrec, Type};
    fn checked_surf(src: &str) -> CheckedProgram {
        let decls = chelis_surf::parser::parse_str(src, &chelis_syntax_host::NativeStack)
            .expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls, &chelis_syntax_host::NativeStack)
            .expect("Surf fixture must desugar");
        check_ir_program(&exprs).expect("IR check")
    }
    #[test]
    fn checked_program_preserves_alias_resolved_generic_adt_registry() {
        let checked = checked_surf(
            r#"
type Payload[a] = a
type Boxed[a] = | Boxed(Payload[a])
def identity[a](value: Boxed[a]) -> Boxed[a] = value
"#,
        );
        let assert_checker_metadata = |stage: &str, program: &CheckedProgram| {
            assert!(
                program.adt_registry().lookup("Boxed").is_some(),
                "{stage}: checker-owned ADT registry must survive"
            );
            let identity = program
                .signature_inference()
                .functions
                .get("identity")
                .unwrap_or_else(|| panic!("{stage}: identity signature must survive"));
            assert!(
                identity.authored_signature,
                "{stage}: authored-signature provenance must survive"
            );
            assert!(
                identity.authored_signature_type.is_some(),
                "{stage}: checker-decoded authored signature must survive"
            );
        };
        assert_checker_metadata("initial checking", &checked);
        let boxed = checked
            .adt_registry()
            .lookup("Boxed")
            .expect("checked registry must retain Boxed");
        assert_eq!(boxed.type_params, ["a"]);
        assert_eq!(boxed.variants.len(), 1);
        let field = &boxed.variants[0].fields[0].1;
        assert!(
            matches!(field, Type::Var(_)),
            "aliases must be checker-resolved and the stored payload retained: {field:?}"
        );

        let encoded = bincode::serialize(&checked).expect("CheckedProgram serializes");
        let decoded: CheckedProgram =
            bincode::deserialize(&encoded).expect("CheckedProgram deserializes");
        assert_checker_metadata("serialization", &decoded);
        assert_eq!(
            decoded
                .adt_registry()
                .lookup("Boxed")
                .expect("registry survives serialization")
                .variants[0]
                .fields[0]
                .1,
            *field
        );
        assert_eq!(
            decoded
                .signature_inference()
                .functions
                .get("identity")
                .and_then(|signature| signature.authored_signature_type.as_ref()),
            checked
                .signature_inference()
                .functions
                .get("identity")
                .and_then(|signature| signature.authored_signature_type.as_ref()),
            "checker-decoded authored signatures survive serialization"
        );
        let effects_reannotated = checked
            .try_with_effect_annotations(checked.annotated_exprs().to_vec())
            .expect("effects-only rewrite preserves checked metadata");
        assert_checker_metadata("effects reannotation", &effects_reannotated);
        let linearity_checked = crate::linearity::check_linearity(&effects_reannotated)
            .expect("linearity preserves checked metadata");
        assert_checker_metadata("linearity", &linearity_checked);
        assert!(
            CheckedProgram::compose(&checked, &effects_reannotated).is_none(),
            "independent checked programs must not compose"
        );

        let dimensional = checked_surf(
            r#"
type Column[n, a] = | Column(tensor[n, a])
def identity[n, a](column: Column[n, a]) -> Column[n, a] = column
"#,
        );
        let column = dimensional
            .adt_registry()
            .lookup("Column")
            .expect("checked registry must retain dimensional Column");
        let Type::Tensor(stored_dims, TensorPrec::Var(stored_precision)) =
            &column.variants[0].fields[0].1
        else {
            panic!("Column must retain its checker-owned polymorphic tensor field");
        };
        assert_eq!(
            column.param_kinds,
            vec![NominalParamKind::Dimension, NominalParamKind::Type],
            "the registry must preserve each source parameter's checker-owned kind"
        );
        let NominalArg::Dimension(Dim::Var(parameter_dimension)) = &column.param_args[0] else {
            panic!("Column's first parameter must retain its dimension variable");
        };
        let NominalArg::Type(Type::Var(parameter_precision)) = &column.param_args[1] else {
            panic!("Column's second parameter must retain its type variable");
        };
        let Some(Dim::Var(field_dimension)) = stored_dims.first() else {
            panic!("Column's field must retain its parameterized dimension");
        };
        assert_eq!(
            stored_precision, parameter_precision,
            "dtype parameter is the tensor's stored precision"
        );
        assert_eq!(
            field_dimension, parameter_dimension,
            "dimension parameter is the tensor's stored extent"
        );
        assert_eq!(
            column.param_vars,
            vec![*parameter_precision],
            "dimension parameters must not be reconstructed as stored dtypes"
        );
    }
    mod more {
        use super::*;
        use chelis_checker_core::CheckErrorKind;
        fn check(source: &str) -> InferResult {
            let exprs =
                chelis_deep::parser::parse_str(source, &chelis_syntax_host::NativeStack).unwrap();
            infer_program(&exprs)
        }

        #[test]
        fn fix7_fitness_unresolved_names() {
            let exprs = chelis_deep::parser::parse_str(
                "(def {} x (var {} unknown))",
                &chelis_syntax_host::NativeStack,
            )
            .unwrap();
            let result = crate::infer::infer_program(&exprs);
            let report = crate::fitness::FitnessReport::from_infer_result(&result);
            assert!(
                report.unresolved_names.contains(&"unknown".to_string()),
                "expected 'unknown' in unresolved_names, got {:?}",
                report.unresolved_names
            );

            assert!(report.errors.iter().all(|e| e.severity > 0.0));
        }
        #[test]
        fn fitness_uses_the_structured_identifier_instead_of_rendered_prose() {
            let exprs = chelis_deep::parser::parse_str(
                "(def {} x (var {} unknown))",
                &chelis_syntax_host::NativeStack,
            )
            .unwrap();
            let mut result = crate::infer::infer_program(&exprs);
            let error = result
                .errors
                .iter_mut()
                .find(|error| matches!(error.kind, CheckErrorKind::UnboundVariable { .. }))
                .expect("unbound diagnostic");
            error.message = "localized name-resolution rendering".to_string();

            let report = crate::fitness::FitnessReport::from_infer_result(&result);
            assert_eq!(report.unresolved_names, vec!["unknown"]);
            assert!(report.components.names < 1.0);
        }
        #[test]
        fn ir_preserves_unresolved_name_errors() {
            let decls = chelis_surf::parser::parse_str(
                "def probe(x: f32) -> f32 = sub(x, frobnicate(x))\n\
     def main() -> f32 = probe(cast(1.0, f32))\n",
                &chelis_syntax_host::NativeStack,
            )
            .expect("surf parse");
            let exprs =
                chelis_surf::desugar::desugar_program(&decls, &chelis_syntax_host::NativeStack)
                    .expect("Surf fixture must desugar");

            let result = infer_ir_program(&exprs);
            assert!(
                result
                    .errors
                    .iter()
                    .any(|error| matches!(error.kind, CheckErrorKind::UnboundVariable { .. })),
                "expected ir inference to preserve unresolved-name errors, got {:?}",
                result.errors
            );

            let report = crate::fitness::check_ir_program(&exprs);
            assert!(
                report.unresolved_names.contains(&"frobnicate".to_string()),
                "expected ir fitness report to include unresolved frobnicate, got {:?}",
                report.unresolved_names
            );
        }
        #[test]
        fn ir_rejects_out_of_scope_terminal_constructor_name() {
            let decls = chelis_surf::parser::parse_str(
                "type Pkg__chelis__std__Std__Nn__Generate__KVCache[a] = \
        | Pkg__chelis__std__Std__Nn__Generate__KVCache(List[a])\n\
     def make_cache[p]() -> \
         Pkg__chelis__std__Std__Nn__Generate__KVCache[p] = KVCache([])\n",
                &chelis_syntax_host::NativeStack,
            )
            .expect("surf parse");
            let exprs =
                chelis_surf::desugar::desugar_program(&decls, &chelis_syntax_host::NativeStack)
                    .expect("Surf fixture must desugar");

            let result = infer_ir_program(&exprs);
            assert!(
                result.errors.iter().any(|error| matches!(
                    error.kind,
                    CheckErrorKind::UnknownConstructor { .. }
                ) && error.message.contains("KVCache")),
                "expected an UnknownConstructor for the out-of-scope bare `KVCache`, got {:?}",
                result.errors
            );

            let diagnostic = result
                .errors
                .iter()
                .find(|error| matches!(error.kind, CheckErrorKind::UnknownConstructor { .. }))
                .expect("unknown constructor diagnostic");
            assert_eq!(diagnostic.kind.unresolved_identifier(), Some("KVCache"));
            let report = crate::fitness::FitnessReport::from_infer_result(&result);
            assert_eq!(report.unresolved_names, vec!["KVCache"]);
            assert!(report.components.names < 1.0);
        }
        #[test]
        fn fix10_fitness_has_untyped_nodes() {
            let result = check(
                "(def {} good (lit {type: (t-prim {} i32)} 42)) \
                        (def {} bad (var {} nope))",
            );
            let report = crate::fitness::FitnessReport::from_infer_result(&result);
            assert!(report.untyped_nodes > 0, "expected untyped_nodes > 0");

            assert!(
                report.errors.iter().all(|e| e.severity > 0.0),
                "all errors should have severity > 0"
            );
        }
        #[test]
        fn fix12_structure_score_measured() {
            let exprs = chelis_deep::parser::parse_str(
                "(def {} x (lit {type: (t-prim {} i32)} 42))",
                &chelis_syntax_host::NativeStack,
            )
            .unwrap();
            let report = crate::fitness::check_program(&exprs);
            assert!(
                (report.components.structure - 1.0).abs() < 0.01,
                "valid program structure should be ~1.0, got {}",
                report.components.structure
            );
        }
    }
}

#[cfg(test)]
thread_local! { static TYPE_ANALYSIS_SESSION_COUNT: Cell<usize> = const { Cell::new(0) }; }
#[cfg(test)]
pub(crate) fn reset_type_analysis_session_count() {
    TYPE_ANALYSIS_SESSION_COUNT.with(|count| count.set(0));
}
#[cfg(test)]
pub(crate) fn type_analysis_session_count() -> usize {
    TYPE_ANALYSIS_SESSION_COUNT.with(Cell::get)
}
