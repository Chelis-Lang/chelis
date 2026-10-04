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
use chelis_deep::node::Node as DeepNode;
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
use crate::env::{BindingFacts, DeclarationBinderIdentities, Env, TopLevelValueVisibility};
use crate::errors::*;
use crate::linearity::LinearityInfo;
use crate::session::{DeclarationDiagnosticOwner, DeclarationTypeDiagnosticClass, DiagnosticSink};
use crate::types::*;
use crate::unify::*;

trait DiagnosticOutput {
    fn push(&mut self, error: CheckError);

    fn declaration_owns_unknown_primitive(
        &self,
        _owner: &DeclarationDiagnosticOwner,
        _primitive_name: &str,
    ) -> bool {
        false
    }

    fn resolved_unknown_primitive_at(
        &self,
        _location: &TypeDiagnosticLocation,
        _primitive_name: &str,
    ) -> bool {
        false
    }
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

    fn declaration_owns_unknown_primitive(
        &self,
        owner: &DeclarationDiagnosticOwner,
        primitive_name: &str,
    ) -> bool {
        self.declaration_type_witness(
            owner,
            primitive_name,
            DeclarationTypeDiagnosticClass::UnknownPrimitive,
        )
        .is_some()
    }

    fn resolved_unknown_primitive_at(
        &self,
        location: &TypeDiagnosticLocation,
        primitive_name: &str,
    ) -> bool {
        let (span_offset, span_id) = location.stable_key();
        self.unknown_primitive_site_witness(span_offset, span_id, primitive_name)
            .is_some()
    }
}

use std::cell::{Cell, RefCell};

mod annotate;
mod app;
mod app_collection;
mod app_equality;
mod app_helpers;
mod app_hostio;
mod app_numeric;
mod app_operand_dtype;
mod app_post;
mod app_post_diagonal;
mod app_route;
mod app_scatter;
mod app_shape;
mod app_shape_helpers;
mod app_string;
mod app_tensor;
mod binder_literal;
mod checked;
mod common;
mod declaration_close;
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
mod grad_selector;
mod group_link;
mod literal_width;
mod operand_deferral;
mod program;
pub(crate) mod recursion;
mod rigid;
mod shape_honesty;
mod slot;
mod static_int;
mod static_value;
mod type_derivation;
mod validate;
mod validate_core_transform;
mod vmap_extent;

use annotate::*;
use app::*;
use app_collection::*;
use app_equality::*;
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
use declaration_close::*;
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
use grad_selector::validate_grad_selector_identity;
pub(crate) use grad_selector::{
    SelectorCallableContext, extend_selector_callable_context, selector_callable_context_digest,
};
use group_link::*;
use operand_deferral::*;
use program::*;
use rigid::*;
use slot::*;
pub use static_int::fold_static_int_expr;
use static_value::*;
use type_derivation::*;
use validate::*;
use validate_core_transform::*;
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
