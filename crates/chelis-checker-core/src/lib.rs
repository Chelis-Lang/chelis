//! Portable semantic checking with explicit caller resources.
//! Ordinary facts do not grant native checked-program authority.

#![no_std]

#[macro_use]
extern crate alloc;
#[cfg(test)]
#[macro_use(thread_local, eprintln)]
extern crate std;

pub(crate) mod prelude {
    pub(crate) use alloc::borrow::ToOwned;
    pub(crate) use alloc::boxed::Box;
    pub(crate) use alloc::string::{String, ToString};
    pub(crate) use alloc::vec::Vec;
}

pub mod adt;
pub mod bitwise;
pub mod builtin_discovery;
#[cfg(test)]
mod builtin_value_contract_tests;
mod builtins;
pub mod cancel;
pub mod context;
pub mod deep_type;
pub mod env;
pub mod errors;
pub mod infer;
pub mod invariants;
pub mod key_admission;
pub mod known_tags;
pub mod linearity;
pub mod opacity;
mod rejection_registry_generated;
pub mod resources;
#[cfg(test)]
mod result_scope;
pub(crate) mod session;
pub mod static_seed;
pub mod unsupported;

pub use bitwise::{BitwiseError, bitwise_scalar, bitwise_tensor};
pub use builtins::{
    AdmittedBuiltin, AxisArgumentLayout, BUILTIN_NAMES, BUILTINS, BuiltinCapabilityDecl,
    BuiltinDecl, BuiltinInferenceRule, BuiltinSemanticDomain, BuiltinSiblingCaseDecl,
    BuiltinSiblingCaseId, COMPARISON_OPS, CaseKeys, InferenceDisposition, KeyParameter,
    KeyParameterSite, KeyRouting, Realizability, ShapeClass, axis_argument_layout, builtin_decl,
    builtin_env, builtin_env_names, builtin_value_contract_carried, case_keys, prelude_adt_defs,
    realizability, shape_class,
};
pub use context::ContextSnapshot;
pub use errors::{CheckError, CheckErrorKind, ErrorWitness};
pub use infer::{
    CheckFacts, CheckedLocalAscription, CompletedCheck, DeclaredSignature, DeclaredTypeSurface,
    InferResult, InferStats, LocalAscriptionAxisClaim, LocalAscriptionComposition,
    LocalAscriptionId, LocalAscriptionOrigin, build_compiled_library_context,
    build_compiled_library_context_with_base, build_type_env_from_library, check_ir_program,
    check_ir_with_context, check_ir_with_signature_context, check_typed_program,
    compose_local_ascriptions, fold_static_int_expr, infer_ir_program, infer_program,
    resolve_declared_surface,
};
pub use linearity::{
    CopyRepair, CopyRepairUse, CopyRepairUseKind, LinearityInfo, check_linearity,
    check_linearity_with_context, copy_repairs,
};
pub use resources::{Cancellation, CheckObserver, CheckPolicy, Resources};

#[cfg(feature = "checkpoint-compile-probe")]
#[doc(hidden)]
pub struct DiagnosticCheckpoint(session::DiagnosticCheckpoint);
#[cfg(feature = "checkpoint-compile-probe")]
#[doc(hidden)]
pub fn checkpoint_iter_compile_probe(
    sink: &errors::DiagnosticSink<'_>,
    checkpoint: DiagnosticCheckpoint,
) {
    let _ = sink.iter_since(checkpoint.0);
}

/// Borrow annotation output on its operation stack for native context admission.
pub fn build_type_env_from_library_with_source<S: chelis_stack_services::StackServices, R>(
    exprs: &[chelis_deep::Expr],
    receive: impl FnOnce(ContextSnapshot, &[chelis_deep::Expr]) -> R,
    resources: &Resources<'_, S>,
) -> Result<R, InferResult> {
    resources.run(|operation| {
        let (snapshot, annotated) =
            session::build_type_env_from_library_with_source(exprs, operation)?;
        Ok(receive(snapshot, &annotated))
    })
}

#[cfg(test)]
mod test_support;

pub use opacity::{
    demangle_ident, is_linker_format_name, linked_binding_in_module_of,
    linked_constructor_source_name,
};
