//! Type inference engine for the Chelis type checker.
//!
//! Walks Deep AST nodes and assigns types using Hindley-Milner inference.

#[macro_use]
mod stack;

use stack::*;
pub use stack::{
    reset_grow_segment_bytes_for_test, run_on_grown_stack, set_grow_segment_bytes_for_test,
};

use chelis_unord::{UnordMap, UnordSet};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use chelis_deep::annotations::BindingTypeOrigin;
use chelis_deep::ast as deep;
use chelis_deep::role::SlotShape;
use chelis_deep::{DeepTag, Span, decode_effect_kind};
use chelis_vocab::EffectKind;

use crate::adt::{AdtRegistry, CallShape};
use crate::builtins;
use crate::cancel::CancelToken;
use crate::context::{TypeEnv, TypeEnvInner};
use crate::deep_type::{
    BinderMode, DeepTypeResolver, ResolvedCastTarget, TypeDiagnosticLocation, TypeResolutionEnv,
    TypeUseSite, deferred_family_diagnostic, is_deferred_dtype_name, is_unsigned_dtype_name,
    unsigned_family_diagnostic,
};
use crate::env::{Env, TopLevelValueVisibility};
use crate::errors::*;
use crate::linearity::LinearityInfo;
use crate::session::{DeclarationDiagnosticOwner, DiagnosticSink};
use crate::types::*;
use crate::unify::*;

trait DiagnosticOutput {
    fn push(&mut self, error: CheckError);
}

#[cfg(test)]
impl DiagnosticOutput for Vec<CheckError> {
    fn push(&mut self, error: CheckError) {
        Vec::push(self, error);
    }
}

impl DiagnosticOutput for DiagnosticSink<'_> {
    fn push(&mut self, error: CheckError) {
        DiagnosticSink::push(self, error);
    }
}

use std::cell::{Cell, RefCell};

mod annotate;
mod app;
mod app_collection;
mod app_helpers;
mod app_hostio;
mod app_numeric;
mod app_operand_dtype;
mod app_post;
mod app_route;
mod app_scatter;
mod app_shape;
mod app_shape_helpers;
mod app_string;
mod app_tensor;
mod binder_literal;
mod checked;
mod common;
mod declarations;
mod declared_surface;
mod declared_type;
pub(crate) use declared_surface::resolve_declared_surface_in_session;
pub use declared_surface::{DeclaredSignature, DeclaredTypeSurface, resolve_declared_surface};
mod deferred_operands;
pub(crate) mod expr;
mod expr_function;
mod expr_pattern;
pub(crate) mod expr_record;
mod expr_transform;
mod operand_deferral;
mod program;
pub(crate) mod recursion;
mod rigid;
mod shape_honesty;
mod slot;
mod static_int;
mod static_value;
mod validate;
mod vmap_extent;

use annotate::*;
use app::*;
use app_collection::*;
use app_helpers::*;
use app_hostio::*;
use app_numeric::*;
use app_operand_dtype::*;
use app_post::*;
use app_route::*;
use app_scatter::*;
use app_shape::*;
use app_shape_helpers::*;
use app_string::*;
use app_tensor::*;
use binder_literal::*;
use checked::*;
use common::*;
pub(crate) use common::{decide_shape_route, shape_route_result};
use declared_type::*;
// chelis#1654: the settled decision for transported checked collection
// contracts. Direct syntactic calls keep the better-informed eager routes.
pub(crate) use app_collection::{TensorConcatCallEvidence, decide_collection_constraint};
use declarations::*;
use deferred_operands::*;
use expr::*;
use expr_function::*;
use expr_pattern::*;
use expr_record::*;
use expr_transform::*;
use operand_deferral::*;
use program::*;
use rigid::*;
use slot::*;
pub use static_int::fold_static_int_expr;
use static_value::*;
use validate::*;
use vmap_extent::*;

pub use checked::{
    CheckedLocalTensorAscription, CheckedProgram, FunctionSignatureInference, InferResult,
    InferStats, LocalAscriptionAxisClaim, LocalAscriptionId, LocalTensorAscriptionOrigin,
    ParamSignatureInference, SignatureInferenceMetadata, compose_local_tensor_ascriptions,
};
pub use program::{
    build_compiled_library_context, build_compiled_library_context_with_base,
    build_type_env_from_library, check_ir_program, check_ir_with_context,
    check_ir_with_signature_context, check_typed_program, infer_ir_program, infer_program,
};
pub use validate::type_to_deep_expr;

pub(crate) use checked::checked_program_with_effect_annotations_in_session;
#[cfg(test)]
pub(crate) use checked::take_post_app_key_log;
#[cfg(test)]
pub(crate) use checked::{
    FinalizationMutationCase, ReconcileMutationCase, TypeStampMutationCase,
    run_finalization_mutation_case, run_reconcile_mutation_case, run_type_stamp_mutation_case,
};
pub(crate) use declarations::{param_has_consuming_use, param_has_consuming_use_in_session};
pub(crate) use program::{
    build_compiled_library_context_in_session, build_compiled_library_context_with_base_in_session,
    build_type_env_from_library_in_session, check_ir_with_signature_context_in_session,
    check_typed_program_in_session, infer_ir_program_in_session, infer_program_in_session,
};

#[cfg(test)]
pub(crate) use program::builtin_selection_probe;

#[cfg(test)]
mod tests;
