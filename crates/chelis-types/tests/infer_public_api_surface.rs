//! Public API lock for `chelis_types::infer` (openspec
//! `modularize-type-inference`).
//!
//! The module split moves inference code between files. Every item below is
//! reachable today at `chelis_types::infer::<name>`, and the split is not
//! allowed to remove one, rename one, or move one to a different path. This
//! file is the executable form of that promise: it imports each item by its
//! exact path and binds each function to a `const` of its exact signature, so
//! a changed argument or return type is a compile error rather than a silent
//! source-compatible drift.
//!
//! A new public item is not a failure here. A *missing or moved* one is.

use std::collections::HashMap;

// Every public item of `chelis_types::infer`, imported by exact path.
use chelis_types::infer::{
    CheckedProgram, FunctionSignatureInference, InferResult, InferStats, ParamSignatureInference,
    SignatureInferenceMetadata, build_compiled_library_context,
    build_compiled_library_context_with_base, build_type_env_from_library, check_ir_program,
    check_ir_with_context, check_ir_with_signature_context, check_typed_program, infer_ir_program,
    infer_program, run_on_grown_stack, set_grow_segment_bytes_for_test, type_to_deep_expr,
};

use chelis_deep::ast as deep;
use chelis_types::TypeEnv;
use chelis_types::types::Type;

type CompiledLibraryContextResult = Result<(TypeEnv, CheckedProgram), InferResult>;

// ── Function signatures ─────────────────────────────────────────────────────
//
// Binding each function to a `const` of its written-out type pins the
// signature. A changed parameter or return type fails to compile here.

const _INFER_PROGRAM: fn(&[deep::Expr]) -> InferResult = infer_program;

const _CHECK_IR_PROGRAM: fn(&[deep::Expr]) -> Result<CheckedProgram, InferResult> =
    check_ir_program;

const _CHECK_TYPED_PROGRAM: fn(&[deep::Expr]) -> Result<CheckedProgram, InferResult> =
    check_typed_program;

const _INFER_IR_PROGRAM: fn(&[deep::Expr]) -> InferResult = infer_ir_program;

const _BUILD_TYPE_ENV_FROM_LIBRARY: fn(&[deep::Expr]) -> Result<TypeEnv, InferResult> =
    build_type_env_from_library;

const _TYPE_TO_DEEP_EXPR: fn(&Type) -> deep::Expr = type_to_deep_expr;

const _SET_GROW_SEGMENT_BYTES_FOR_TEST: fn(usize) = set_grow_segment_bytes_for_test;

const _BUILD_COMPILED_LIBRARY_CONTEXT: fn(&[deep::Expr]) -> CompiledLibraryContextResult =
    build_compiled_library_context;

const _BUILD_COMPILED_LIBRARY_CONTEXT_WITH_BASE: fn(
    &TypeEnv,
    &[deep::Expr],
) -> CompiledLibraryContextResult = build_compiled_library_context_with_base;

const _CHECK_IR_WITH_CONTEXT: fn(&TypeEnv, &[deep::Expr]) -> Result<CheckedProgram, InferResult> =
    check_ir_with_context;

const _CHECK_IR_WITH_SIGNATURE_CONTEXT: fn(
    &TypeEnv,
    &SignatureInferenceMetadata,
    &[deep::Expr],
) -> Result<CheckedProgram, InferResult> = check_ir_with_signature_context;

// `run_on_grown_stack` is generic over its closure, so it is exercised by a
// call below rather than pinned as a function pointer.

// ── Types ───────────────────────────────────────────────────────────────────

/// Reads every public field of the signature-metadata types. A removed or
/// renamed field is a compile error here.
fn _read_signature_metadata_fields(meta: &SignatureInferenceMetadata) {
    let _empty: bool = meta.is_empty();
    for function in meta.functions.values() {
        let _: &String = &function.name;
        let _: bool = function.authored_signature;
        let _: &Option<Type> = &function.authored_signature_type;
        let _: bool = function.recursive_cycle;
        let _: &Type = &function.checked_signature;
        let _: &Type = &function.display_signature;
        let params: &Vec<ParamSignatureInference> = &function.params;
        for param in params {
            let _: usize = param.index;
            let _: &String = &param.name;
            let _: bool = param.written;
            let _: bool = param.inferred_read_only;
            let _: &Type = &param.checked_type;
            let _: &Type = &param.display_type;
        }
    }
}

/// Pins `FunctionSignatureInference` as a nameable type in its own right.
fn _name_function_signature_inference(value: &FunctionSignatureInference) -> &Type {
    &value.checked_signature
}

/// Public field access on the result types, and the public accessors on
/// `CheckedProgram`. These are part of the surface a consumer reads.
#[test]
fn public_result_shape_is_reachable() {
    let empty: Vec<deep::Expr> = Vec::new();

    let report: InferResult = infer_program(&empty);
    let _errors: &Vec<chelis_types::errors::CheckError> = &report.errors;
    let _typed: usize = report.typed_nodes;
    let _total: usize = report.total_nodes;

    let checked: CheckedProgram = check_ir_program(&empty).expect("empty program checks");
    let _exprs: &[deep::Expr] = checked.exprs();
    let _annotated: &[deep::Expr] = checked.annotated_exprs();
    let _type_env: &HashMap<String, deep::Expr> = checked.type_env();
    let _linearity: &chelis_types::LinearityInfo = checked.linearity();
    let _sig: &SignatureInferenceMetadata = checked.signature_inference();
    let _adt: &chelis_types::adt::AdtRegistry = checked.adt_registry();

    let stats: InferStats = checked.infer_stats();
    let _typed_nodes: usize = stats.typed_nodes;
    let _total_nodes: usize = stats.total_nodes;

    _read_signature_metadata_fields(checked.signature_inference());
}

/// The context-building entry points keep their argument shape and their
/// `Result` return.
#[test]
fn public_context_entry_points_are_reachable() {
    let empty: Vec<deep::Expr> = Vec::new();

    let context: CompiledLibraryContextResult = build_compiled_library_context(&empty);
    let (env, library) = context.expect("empty library context builds");

    let (based_env, _based): (TypeEnv, CheckedProgram) =
        build_compiled_library_context_with_base(&env, &empty)
            .expect("empty based library context builds");

    let _with_context: Result<CheckedProgram, InferResult> =
        check_ir_with_context(&based_env, &empty);

    let _with_signature_context: Result<CheckedProgram, InferResult> =
        check_ir_with_signature_context(&env, library.signature_inference(), &empty);

    let _env: Result<TypeEnv, InferResult> = build_type_env_from_library(&empty);
}

/// `run_on_grown_stack` stays generic over its closure and returns the
/// closure's value.
#[test]
fn public_grown_stack_runner_is_reachable() {
    let value: u32 = run_on_grown_stack(|| 7u32);
    assert_eq!(
        value, 7,
        "run_on_grown_stack must return the closure's value"
    );
}

/// The crate root keeps re-exporting the inference surface it exported
/// before the split, so a consumer importing from `chelis_types::` directly
/// is unaffected too.
#[test]
fn crate_root_reexports_are_reachable() {
    use chelis_types::{
        CheckedProgram as RootChecked, InferResult as RootResult,
        build_compiled_library_context as root_build_context,
        build_compiled_library_context_with_base as root_build_context_with_base,
        build_type_env_from_library as root_build_env, check_ir_program as root_check,
        check_ir_with_context as root_check_with_context,
        check_ir_with_signature_context as root_check_with_sig_context,
        check_typed_program as root_check_typed, infer_ir_program as root_infer_ir,
        infer_program as root_infer, run_on_grown_stack as root_grown,
        set_grow_segment_bytes_for_test as root_set_grow,
    };

    let empty: Vec<deep::Expr> = Vec::new();
    let _: RootResult = root_infer(&empty);
    let _: RootResult = root_infer_ir(&empty);
    let _: Result<RootChecked, RootResult> = root_check(&empty);
    let _: Result<RootChecked, RootResult> = root_check_typed(&empty);
    let _: Result<TypeEnv, RootResult> = root_build_env(&empty);

    let (env, library) = root_build_context(&empty).expect("root context builds");
    let _ = root_build_context_with_base(&env, &empty);
    let _ = root_check_with_context(&env, &empty);
    let _ = root_check_with_sig_context(&env, library.signature_inference(), &empty);
    let _: u8 = root_grown(|| 1u8);
    let _: fn(usize) = root_set_grow;
}
