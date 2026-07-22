//! PR #799 / chelis#730: the public compiler API must preserve its declared
//! `Result` failure channel when checked host types remain unresolved.
//!
//! Empty-list literals are valid through checking but deliberately retain a
//! host inference variable until context resolves their element type. At the
//! code-generation boundary an unresolved variable is a typed lowering
//! rejection under [05-UNS-1], never a panic and never an invented ABI type.

use chelis_compiler_api::compiler::{CompilerError, compile, compile_for_execution};
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, SourceKind};

fn c_request(source: &str) -> CompileRequest {
    CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        target: CompileTarget::C,
        entry_name: None,
    }
}

fn assert_unresolved_host_type<T: std::fmt::Debug>(result: Result<T, CompilerError>) {
    let error = result.expect_err("an unresolved host type must be returned as an error");
    assert_eq!(error.stage, "lower");
    assert_eq!(
        error.errors.len(),
        1,
        "unexpected error envelope: {error:?}"
    );
    let diagnostic = &error.errors[0];
    assert_eq!(diagnostic.kind, "lower_error");
    assert!(
        diagnostic
            .message
            .contains("host type did not resolve before the code-generation boundary"),
        "diagnostic must name the failed structural boundary: {diagnostic:?}"
    );
    assert!(
        diagnostic
            .message
            .contains("unresolved host inference variable"),
        "diagnostic must preserve the typed resolution cause: {diagnostic:?}"
    );
    assert!(
        diagnostic.message.contains("[05-UNS-1]") && diagnostic.message.contains("chelis#730"),
        "diagnostic must retain the owning contract citation: {diagnostic:?}"
    );
}

fn assert_unrepresentable_function_value(result: Result<impl std::fmt::Debug, CompilerError>) {
    let error = result.expect_err("a first-class anonymous function must not reach host codegen");
    assert_eq!(error.stage, "lower", "{error:?}");
    assert_eq!(error.errors.len(), 1, "{error:?}");
    let diagnostic = &error.errors[0];
    assert_eq!(diagnostic.kind, "lower_error");
    assert!(
        diagnostic.message.contains("unsupported:")
            && diagnostic.message.contains("anonymous function value `fn`")
            && diagnostic.message.contains("host expression lowering")
            && diagnostic.message.contains("[05-UNS-1]")
            && diagnostic.message.contains("chelis#730"),
        "the rejection must come from the fallible host-expression boundary: {diagnostic:?}"
    );
}

fn generic_record_source(dtype: &str, literal: &str) -> String {
    format!(
        "type ReviewBox[a] =\n\
           | ReviewBox {{ value: a }}\n\
         def unbox(box: ReviewBox[{dtype}]) -> {dtype} = match box with {{\n\
           | ReviewBox {{ value }} => value\n\
         }}\n\
         def out = print(unbox(ReviewBox {{ value: cast({literal}, {dtype}) }}))\n"
    )
}

#[test]
fn compile_returns_empty_list_host_resolution_error_instead_of_panicking() {
    assert_unresolved_host_type(compile(c_request("values = []\n")));
}

#[test]
fn compile_for_execution_returns_nested_empty_list_error_instead_of_panicking() {
    assert_unresolved_host_type(compile_for_execution(c_request("values = [[]]\n")));
}

#[test]
fn generic_adt_int8_specialization_reaches_exact_host_abi() {
    let result = compile(c_request(&generic_record_source("int8", "7")))
        .expect("ReviewBox[int8] must specialize its field before host resolution");
    let generated_c = result
        .files
        .iter()
        .find(|file| file.path.ends_with(".c"))
        .expect("C compilation must emit a translation unit");
    assert!(
        generated_c.contents.contains("int8_t"),
        "the specialized field must retain its exact int8 ABI:\n{}",
        generated_c.contents
    );
}

#[test]
fn generic_adt_bf16_specialization_reaches_structured_abi_rejection() {
    let error = compile(c_request(&generic_record_source("bf16", "7.0")))
        .expect_err("ReviewBox[bf16] must reach the C-host target decision");
    assert_eq!(error.stage, "compile");
    assert_eq!(
        error.errors.len(),
        1,
        "unexpected error envelope: {error:?}"
    );
    let diagnostic = &error.errors[0];
    assert_eq!(diagnostic.kind, "unsupported_feature");
    for expected in [
        "unsupported:",
        "dtype `bf16`",
        "C host ABI selection",
        "(codegen:c)",
    ] {
        assert!(
            diagnostic.message.contains(expected),
            "generic ADT rejection must contain {expected:?}: {diagnostic:?}"
        );
    }
    assert!(
        !diagnostic.message.contains("unresolved host type variable"),
        "generic substitution must finish before ABI selection: {diagnostic:?}"
    );
}

#[test]
fn bf16_callback_reaches_structured_abi_rejection_without_placeholder() {
    let source = "def apply(f: bf16 -> bf16, x: bf16) -> bf16 = f(x)\n\
                  out = print(apply(fn (x: bf16) -> x, cast(6.0, bf16)))\n";
    let error =
        compile(c_request(source)).expect_err("a bf16 callback has no grounded C-host ABI yet");
    assert_eq!(error.stage, "compile");
    assert_eq!(
        error.errors.len(),
        1,
        "unexpected error envelope: {error:?}"
    );
    let diagnostic = &error.errors[0];
    assert_eq!(diagnostic.kind, "unsupported_feature");
    assert!(
        diagnostic.message.contains("unsupported:")
            && diagnostic.message.contains("dtype `bf16`")
            && diagnostic.message.contains("C host ABI selection")
            && diagnostic.message.contains("(codegen:c)"),
        "callback rejection must come from ABI selection: {diagnostic:?}"
    );
}

#[test]
fn returned_anonymous_function_rejects_before_codegen() {
    assert_unrepresentable_function_value(compile(c_request(
        "module M.Main\n\
         def make() -> int8 -> int8 = fn (x: int8) -> add(x, cast(1, int8))\n\
         out = print(\"ok\")\n",
    )));
}

#[test]
fn nested_capturing_function_rejects_before_codegen() {
    assert_unrepresentable_function_value(compile(c_request(
        "module M.Main\n\
         def make(x: int8) -> int8 -> int8 = fn (y: int8) -> add(x, y)\n\
         out = print(\"ok\")\n",
    )));
}

#[test]
fn function_value_stored_in_adt_rejects_before_codegen() {
    assert_unrepresentable_function_value(compile(c_request(
        "module M.Main\n\
         type FnBox =\n\
           | FnBox { callback: int8 -> int8 }\n\
         saved = FnBox { callback: fn (x: int8) -> add(x, cast(1, int8)) }\n\
         out = print(\"ok\")\n",
    )));
}

fn generic_access_source(dtype: &str, literal: &str) -> String {
    format!(
        "type ReviewBox[a] =\n\
           | ReviewBox {{ value: a }}\n\
         type ReviewEnvelope[a] =\n\
           | ReviewEnvelope {{ inner: ReviewBox[a] }}\n\
         def open(envelope: ReviewEnvelope[{dtype}]) -> {dtype} = envelope.inner.value\n\
         def out = print(open(ReviewEnvelope {{\n\
           inner: ReviewBox {{ value: cast({literal}, {dtype}) }}\n\
         }}))\n"
    )
}

#[test]
fn nested_generic_adt_access_substitutes_int8_through_every_field() {
    let result = compile(c_request(&generic_access_source("int8", "7")))
        .expect("nested generic fields must instantiate before host resolution");
    let generated_c = result
        .files
        .iter()
        .find(|file| file.path.ends_with(".c"))
        .expect("C translation unit");
    assert!(generated_c.contents.contains("int8_t"));
}

#[test]
fn nested_generic_adt_access_preserves_bf16_until_abi_rejection() {
    let error = compile(c_request(&generic_access_source("bf16", "7.0")))
        .expect_err("nested generic bf16 must reach target selection");
    assert_eq!(error.stage, "compile", "{error:?}");
    let diagnostic = error.errors.first().expect("one diagnostic");
    assert_eq!(diagnostic.kind, "unsupported_feature");
    assert!(
        diagnostic.message.contains("dtype `bf16`")
            && diagnostic.message.contains("C host ABI selection")
            && !diagnostic.message.contains("unresolved host type variable"),
        "nested generic substitution must finish before target selection: {diagnostic:?}"
    );
}
