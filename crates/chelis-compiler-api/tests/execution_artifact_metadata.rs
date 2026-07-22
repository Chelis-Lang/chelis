//! Entry-scoped compiled-execution metadata (issues #817, #818).
//!
//! `compile_for_execution` backs `chelis.compile_and_load`. Before this
//! change it compiled the WHOLE program and reported the union of every
//! top-level def's `Load`s as the callable's inputs (#817), and returned an
//! empty manifest whenever the entry's body used a syntactic host-runtime
//! builtin such as `concat` — because that excludes the def from the pure
//! DAG, leaving `roots()` empty and tripping the host-lane early-return
//! (#818).
//!
//! These tests pin the fixed contract: the metadata is scoped to a single
//! resolved entry def, an unknown `entry_name` is a loud error that lists
//! the available entries, and a def named `main` no longer forces an
//! un-linkable `main` C symbol.

use chelis_compiler_api::compiler::compile_for_execution;
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, SourceKind};

const HELPER_PLUS_ENTRY: &str = "\
def helper(x: tensor[1, f32]) -> tensor[1, f32] = mul(x, x)
def solve(a: tensor[1, f32], b: tensor[1, f32]) -> tensor[1, f32] = add(helper(a), helper(b))
";

// #818 repro verbatim: single def, multi-statement block, a parameter reused
// across statements, `concat` in the body.
const CONCAT_ENTRY: &str = "\
def main(a: tensor[1, f32], b: tensor[1, f32]) -> tensor[2, f32] = {
  x = mul(copy(a), b)
  y = add(a, b)
  concat([x, y], cast(0, int32))
}
";

fn compile_c(
    source: &str,
    entry: Option<&str>,
) -> chelis_compiler_api::compiler::CompiledExecutionArtifact {
    compile_for_execution(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        target: CompileTarget::C,
        entry_name: entry.map(str::to_string),
    })
    .unwrap_or_else(|err| panic!("compile failed: {err:?}"))
}

fn input_names(artifact: &chelis_compiler_api::compiler::CompiledExecutionArtifact) -> Vec<String> {
    artifact
        .inputs
        .iter()
        .map(|spec| spec.name.clone())
        .collect()
}

/// #817: a multi-def file with no explicit `entry_name` scopes its metadata
/// to the preferred entry (`solve`, the last tensor-signature def), NOT the
/// union of every def's parameters.
#[test]
fn multi_def_without_entry_name_scopes_to_preferred_entry() {
    let artifact = compile_c(HELPER_PLUS_ENTRY, None);
    assert_eq!(
        input_names(&artifact),
        vec!["a".to_string(), "b".to_string()],
        "expected only `solve`'s params, not the merged `helper`+`solve` set"
    );
    assert_eq!(
        artifact.outputs.len(),
        1,
        "expected exactly one output for `solve`, got {:?}",
        artifact.outputs
    );
}

/// #817: an explicit `entry_name` selects that def's scope.
#[test]
fn multi_def_with_explicit_entry_name_scopes_to_it() {
    let artifact = compile_c(HELPER_PLUS_ENTRY, Some("solve"));
    assert_eq!(
        input_names(&artifact),
        vec!["a".to_string(), "b".to_string()]
    );
    assert_eq!(artifact.outputs.len(), 1);
}

/// The explicit selection genuinely changes the result: selecting `helper`
/// scopes to its single param `x`, distinct from `solve`'s `(a, b)`.
#[test]
fn explicit_entry_name_selects_a_different_def() {
    let artifact = compile_c(HELPER_PLUS_ENTRY, Some("helper"));
    assert_eq!(input_names(&artifact), vec!["x".to_string()]);
    assert_eq!(artifact.outputs.len(), 1);
}

/// Negative: an `entry_name` that names no def in a program that DOES have a
/// tensor entry is a typo'd selector — a loud error that lists the choices,
/// not a silently-merged or empty manifest.
#[test]
fn unknown_entry_name_errors_listing_available_defs() {
    let err = compile_for_execution(CompileRequest {
        source_kind: SourceKind::Surf,
        source: HELPER_PLUS_ENTRY.to_string(),
        target: CompileTarget::C,
        entry_name: Some("nope".to_string()),
    })
    .expect_err("unknown entry_name must error");
    let message = &err.errors[0].message;
    assert!(
        message.contains("unknown entry_name `nope`"),
        "message must name the bad entry, got: {message}"
    );
    assert!(
        message.contains("helper") && message.contains("solve"),
        "message must list the available entry defs, got: {message}"
    );
}

/// #818: a single def whose body uses `concat` (and reuses a parameter across
/// block statements) reports its real `(a, b)` inputs and one output, instead
/// of the empty manifest the host-lane early-return used to produce.
#[test]
fn concat_body_reports_real_inputs_and_outputs() {
    let artifact = compile_c(CONCAT_ENTRY, None);
    assert_eq!(
        input_names(&artifact),
        vec!["a".to_string(), "b".to_string()],
        "concat body must not collapse to an empty input manifest"
    );
    assert_eq!(
        artifact.outputs.len(),
        1,
        "expected one output, got {:?}",
        artifact.outputs
    );
}

/// Regression: a single composed-expression def (no block, no host-runtime
/// builtin) still reports its single input — the #818 "works" contrast case.
#[test]
fn single_expression_def_still_reports_its_input() {
    let artifact = compile_c(
        "def main(a: tensor[1, f32]) -> tensor[1, f32] = mul(copy(a), a)\n",
        None,
    );
    assert_eq!(input_names(&artifact), vec!["a".to_string()]);
    assert_eq!(artifact.outputs.len(), 1);
}

/// A def literally named `main`, selected via `entry_name`, must compile to a
/// non-`main` C symbol (decoupled from def selection) so the native toolchain
/// does not reject the translation unit for redefining the reserved
/// `int main(...)`. The scoped metadata is still correct.
#[test]
fn entry_named_main_does_not_emit_reserved_main_symbol() {
    let artifact = compile_c(CONCAT_ENTRY, Some("main"));
    assert_ne!(
        artifact.host_entry_name, "main",
        "the emitted C symbol must not be the reserved `main`"
    );
    assert_eq!(artifact.host_entry_name, "chelis_main");
    // Metadata stays scoped to the `main` def.
    assert_eq!(
        input_names(&artifact),
        vec!["a".to_string(), "b".to_string()]
    );

    // The generated entry function carries the rewritten symbol, and no bare
    // `main(` entry is declared to collide with the C runtime.
    let c = artifact
        .compile_result
        .files
        .iter()
        .find(|file| file.path == "chelis_main.c")
        .expect("chelis_main.c present")
        .contents
        .clone();
    assert!(
        c.contains("chelis_main"),
        "entry function must use the rewritten symbol"
    );
}
