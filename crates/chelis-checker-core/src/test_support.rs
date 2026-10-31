use crate::errors::CheckError;
use crate::infer::DeclaredTypeSurface;
use crate::prelude::*;
use crate::resources::Operation;
use crate::{CheckFacts, CompletedCheck, ContextSnapshot, InferResult, Resources};
use chelis_deep::ast as deep;
use chelis_syntax_host::NativeStack;

fn with_resources<R>(
    parent: Option<&Operation<'_>>,
    run: impl FnOnce(&Resources<'_, NativeStack>) -> R,
) -> R {
    let cancellation = crate::cancel::current_cancel_token();
    let mut resources = Resources::new(&NativeStack);
    resources.cancellation = cancellation
        .as_ref()
        .map(|token| token as &dyn crate::Cancellation);
    if let Some(parent) = parent {
        resources.generalize_sweep_oracle = parent.sweep_enabled.get();
        resources.generalize_sweep_visits = Some(parent.sweep_visits);
    }
    run(&resources)
}

pub(crate) fn run<R>(run: impl FnOnce(&Operation<'_>) -> R) -> R {
    with_resources(None, |resources| resources.run_borrowed(run))
}

pub(crate) fn check_typed_program(
    exprs: &[deep::Expr],
    operation: &Operation<'_>,
) -> Result<CheckFacts, InferResult> {
    with_resources(Some(operation), |resources| {
        crate::infer::check_typed_program(exprs, resources).map(CompletedCheck::into_facts)
    })
}

pub(crate) fn infer_program(exprs: &[deep::Expr], operation: &Operation<'_>) -> InferResult {
    with_resources(Some(operation), |resources| {
        crate::infer::infer_program(exprs, resources)
    })
}

pub(crate) fn check_ir_program(
    exprs: &[deep::Expr],
    operation: &Operation<'_>,
) -> Result<CheckFacts, InferResult> {
    with_resources(Some(operation), |resources| {
        crate::infer::check_ir_program(exprs, resources).map(CompletedCheck::into_facts)
    })
}

pub(crate) fn infer_ir_program(exprs: &[deep::Expr], operation: &Operation<'_>) -> InferResult {
    with_resources(Some(operation), |resources| {
        crate::infer::infer_ir_program(exprs, resources)
    })
}

pub(crate) fn build_type_env_from_library(
    exprs: &[deep::Expr],
    operation: &Operation<'_>,
) -> Result<ContextSnapshot, InferResult> {
    with_resources(Some(operation), |resources| {
        crate::infer::build_type_env_from_library(exprs, resources)
    })
}

pub(crate) fn check_ir_with_context(
    context: &ContextSnapshot,
    exprs: &[deep::Expr],
    operation: &Operation<'_>,
) -> Result<CheckFacts, InferResult> {
    with_resources(Some(operation), |resources| {
        crate::infer::check_ir_with_context(context, exprs, resources)
            .map(CompletedCheck::into_facts)
    })
}

pub(crate) fn check_linearity(
    program: &CheckFacts,
    operation: &Operation<'_>,
) -> Result<CheckFacts, Vec<CheckError>> {
    with_resources(Some(operation), |resources| {
        crate::linearity::check_linearity(program, resources)
    })
}

pub(crate) fn resolve_declared_surface(
    exprs: &[deep::Expr],
    operation: &Operation<'_>,
) -> Result<DeclaredTypeSurface, InferResult> {
    with_resources(Some(operation), |resources| {
        crate::infer::resolve_declared_surface(exprs, resources)
    })
}
