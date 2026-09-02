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
use std::collections::{BTreeMap, VecDeque};

use chelis_deep::ast as deep;
use chelis_deep::{DeepTag, Span, decode_effect_kind};
use chelis_vocab::EffectKind;

use crate::adt::{AdtRegistry, CallShape};
use crate::builtins;
use crate::cancel::CancelToken;
use crate::context::{TypeEnv, TypeEnvInner};
use crate::deep_type::{
    BinderMode, DeepTypeResolver, ResolvedCastTarget, TypeDiagnosticLocation, TypeResolutionEnv,
    TypeUseSite,
};
use crate::env::{Env, TopLevelValueVisibility};
use crate::errors::*;
use crate::linearity::LinearityInfo;
use crate::session::DiagnosticSink;
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
mod app_tensor;
mod checked;
mod common;
mod declarations;
mod expr;
mod expr_function;
mod expr_pattern;
mod expr_record;
mod expr_transform;
mod program;
pub(crate) mod recursion;
mod static_value;
mod validate;

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
use app_tensor::*;
use checked::*;
use common::*;
use declarations::*;
use expr::*;
use expr_function::*;
use expr_pattern::*;
use expr_record::*;
use expr_transform::*;
use program::*;
use static_value::*;
use validate::*;

pub use checked::{
    CheckedProgram, FunctionSignatureInference, InferResult, InferStats, ParamSignatureInference,
    SignatureInferenceMetadata,
};
pub use program::{
    build_compiled_library_context, build_compiled_library_context_with_base,
    build_type_env_from_library, check_ir_program, check_ir_with_context,
    check_ir_with_signature_context, check_typed_program, infer_ir_program, infer_program,
};
pub use validate::type_to_deep_expr;

pub(crate) use checked::checked_program_with_effect_annotations_in_session;
#[cfg(test)]
pub(crate) use checked::{
    FinalizationMutationCase, TypeStampMutationCase, run_finalization_mutation_case,
    run_type_stamp_mutation_case,
};
pub(crate) use declarations::{param_has_consuming_use, param_has_consuming_use_in_session};
pub(crate) use program::{
    build_compiled_library_context_in_session, build_compiled_library_context_with_base_in_session,
    build_type_env_from_library_in_session, check_ir_with_signature_context_in_session,
    check_typed_program_in_session, infer_ir_program_in_session, infer_program_in_session,
};

#[cfg(test)]
mod tests;
