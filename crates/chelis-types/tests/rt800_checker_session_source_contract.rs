//! Structural tripwires supporting the executable compile-fail oracles on
//! `errors::report`. These scans do not replace the compiler tests; they make
//! accidental reintroduction of the old ambient and throwaway-vector seams
//! fail with a targeted explanation.
//!
//! The inference source is read as the concatenation of every module under
//! `src/infer/`, in sorted path order. The scans below ask "does the
//! inference implementation contain (or avoid) this construct", which is a
//! property of the implementation as a whole, not of any one file it happens
//! to be split across.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

/// Every module of the inference tree, concatenated in sorted path order.
static INFER_SOURCE: LazyLock<String> = LazyLock::new(read_infer_tree);

fn read_infer_tree() -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/infer");
    let mut files = Vec::new();
    collect_rs_files(&root, &mut files);
    files.sort();
    assert!(
        !files.is_empty(),
        "found no inference source under {}; these scans would pass vacuously",
        root.display()
    );
    files
        .iter()
        .map(|path| {
            std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("read dir entry").path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

fn infer() -> &'static str {
    &INFER_SOURCE
}

const ERRORS: &str = include_str!("../src/errors.rs");
const DEEP_TYPE: &str = include_str!("../src/deep_type.rs");
const INFER_PROGRAM: &str = include_str!("../src/infer/program.rs");
const OPACITY: &str = include_str!("../src/opacity.rs");
const SESSION: &str = include_str!("../src/session.rs");
const DEEP_VALIDATE: &str = include_str!("../../chelis-deep/src/validate.rs");

#[test]
fn every_production_sink_constructor_has_a_returned_diagnostic_owner() {
    use quote::ToTokens;
    use syn::Item;

    let file = syn::parse_file(SESSION).expect("session.rs must remain valid Rust");
    let mut owners = Vec::new();
    for item in file.items {
        let Item::Fn(function) = item else {
            continue;
        };
        if !function
            .block
            .to_token_stream()
            .to_string()
            .contains("DiagnosticSink")
        {
            continue;
        }
        let return_type = function.sig.output.to_token_stream().to_string();
        assert!(
            return_type.contains("InferResult") || return_type.contains("Result"),
            "production sink constructor `{}` must return the diagnostics it owns, got `{return_type}`",
            function.sig.ident
        );
        owners.push(function.sig.ident.to_string());
    }
    owners.sort();
    assert_eq!(
        owners,
        ["infer_ir_program", "infer_program", "run_result"],
        "new production sink construction requires an explicit returned-error owner audit"
    );
}

#[test]
fn arbitrary_vec_diagnostic_output_is_test_only() {
    assert!(
        infer().contains("#[cfg(test)]\nimpl DiagnosticOutput for Vec<CheckError>"),
        "production validation must only emit through DiagnosticSink"
    );
}

#[test]
fn binder_and_signature_resolution_have_no_ambient_symbols() {
    for (label, source) in [
        ("infer", infer()),
        ("deep_type", DEEP_TYPE),
        ("opacity", OPACITY),
    ] {
        assert!(
            !source.contains("CURRENT_TYPE_BINDERS"),
            "{label} must not restore ambient binder state"
        );
        assert!(
            !source.contains("DECLARED_SIG_PARAM_TYPES"),
            "{label} must not restore ambient signature state"
        );
    }
}

#[test]
fn witness_minting_requires_the_session_sink() {
    assert!(
        ERRORS.contains("pub fn report(errors: &mut DiagnosticSink<'_>"),
        "report must require the unforgeable session capability"
    );
    assert!(
        ERRORS.contains("errors: &mut DiagnosticSink<'_>"),
        "report_witness must require the same capability"
    );
    assert!(
        !ERRORS.contains("report(errors: &mut Vec<CheckError>"),
        "an arbitrary diagnostic vector must never mint a witness"
    );
    assert!(
        !infer().contains("let mut hidden_errors"),
        "the retained reviewer mutation must remain uncompilable, not planted"
    );
}

#[test]
fn sink_construction_and_storage_stay_inside_the_owner() {
    for (label, source) in [
        ("infer", infer()),
        ("errors", ERRORS),
        ("deep_type", DEEP_TYPE),
    ] {
        assert!(
            !source.contains("let mut sink = DiagnosticSink {"),
            "{label} must only receive a sink; construction belongs to session.rs"
        );
    }
    assert!(
        SESSION.contains("pub struct DiagnosticSink<'session> {\n    errors:"),
        "sink storage must remain a private field"
    );
    for forbidden in [
        "impl Default for DiagnosticSink",
        "impl Clone for DiagnosticSink",
        "impl From<Vec<CheckError>> for DiagnosticSink",
        "impl DerefMut for DiagnosticSink",
        "fn into_errors(",
    ] {
        assert!(
            !SESSION.contains(forbidden),
            "session sink reopened an extraction/construction loophole: {forbidden}"
        );
    }
}

#[test]
fn annotation_consumes_owner_stamps_without_semantic_reinference() {
    let annotation = infer()
        .split_once("fn annotate_expr_with_scope(")
        .expect("annotation entry should exist")
        .1
        .split_once("fn should_attach_type_metadata(")
        .expect("annotation section should have a stable end")
        .0;

    for forbidden in [
        "infer_expr_in_scope(",
        "pattern_bindings(",
        "DeepTypeResolver::new(",
        "extract_params(",
    ] {
        assert!(
            !annotation.contains(forbidden),
            "annotation must consume the owning result instead of calling `{forbidden}`"
        );
    }
    assert!(
        !infer().contains("fn infer_expr_in_scope("),
        "the old fresh-substitution annotation inference seam must be deleted"
    );
}

#[test]
fn checked_results_share_one_totality_finalizer() {
    assert!(
        infer().contains("fn finalize_checked_program("),
        "all CheckedProgram results need one totality finalization boundary"
    );
    assert!(
        infer().contains("annotated_totality_invariant_traces("),
        "finalization must inspect the authoritative annotated tree"
    );

    let constructor_calls = infer().matches("CheckedProgram::from_parts_with_").count();
    assert_eq!(
        constructor_calls, 0,
        "result paths must not bypass finalize_checked_program; found {constructor_calls} direct constructors"
    );
}

#[test]
fn checked_result_reconstruction_is_effect_only_fallible_and_session_owned() {
    assert!(
        infer().contains("pub fn try_with_effect_annotations("),
        "the only public reconstruction seam must be effects-owned and fallible"
    );
    for forbidden in [
        "pub fn try_from_parts(",
        "pub fn try_from_parts_with_signature_context(",
        "pub fn from_parts(",
        "pub fn from_parts_with_signature_context(",
        "pub(crate) fn try_checked_program_from_parts(",
        "pub(crate) fn try_checked_program_from_parts_with_signature_context(",
        "pub(crate) fn checked_program_from_parts(",
        "pub(crate) fn checked_program_from_parts_with_signature_context(",
    ] {
        assert!(
            !infer().contains(forbidden) && !SESSION.contains(forbidden),
            "infallible/discarding reconstruction seam must be absent: `{forbidden}`"
        );
    }
    assert!(
        SESSION.contains("Ok(_) if !errors.is_empty()"),
        "run_result must centrally veto success after any authoritative diagnostic"
    );
}

#[test]
fn diagnostic_sink_is_append_only_and_cycle_errors_are_never_erased() {
    use syn::visit::Visit;
    use syn::{Expr, Item};

    #[derive(Default)]
    struct CyclicPrebindCalls(usize);

    impl<'ast> Visit<'ast> for CyclicPrebindCalls {
        fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
            if let Expr::Path(path) = call.func.as_ref()
                && path.qself.is_none()
                && path.path.is_ident("prebind_cyclic_component_schemes")
            {
                self.0 += 1;
            }
            syn::visit::visit_expr_call(self, call);
        }
    }

    for forbidden in [
        "pub(crate) fn retain(",
        "pub(crate) fn clear(",
        "pub(crate) fn truncate(",
        "pub(crate) fn drain(",
        "pub(crate) fn take(",
        "pub(crate) fn replace(",
        "suppress_unbound_for_cycle_members",
    ] {
        assert!(
            !SESSION.contains(forbidden) && !infer().contains(forbidden),
            "the witness-owning diagnostic session must be monotonic: `{forbidden}`"
        );
    }
    let file = syn::parse_file(INFER_PROGRAM).expect("infer/program.rs must remain valid Rust");
    for driver in [
        "infer_program_with_product_in_session",
        "infer_ir_program_with_state",
    ] {
        let function = file
            .items
            .iter()
            .find_map(|item| match item {
                Item::Fn(function) if function.sig.ident == driver => Some(function),
                _ => None,
            })
            .unwrap_or_else(|| panic!("missing production inference driver `{driver}`"));
        let mut calls = CyclicPrebindCalls::default();
        calls.visit_block(&function.block);
        assert_eq!(
            calls.0, 1,
            "production driver `{driver}` must call the cyclic-component prebind exactly once"
        );
    }
    assert!(
        !infer().contains("prebind_recursive_function_schemes"),
        "the source contract must not preserve the removed recursive-only prebind seam"
    );
}

#[test]
fn annotation_ownership_uses_the_canonical_exhaustive_child_role_table() {
    for role in [
        "RuntimeExpr",
        "Syntax",
        "Selector",
        "EffectHandler",
        "Binder",
        "Type",
        "ExplicitInferenceBypass",
    ] {
        assert!(
            infer().contains(role),
            "the shared child ownership classifier must name the `{role}` role"
        );
    }
    assert!(
        infer().contains("child_stamp_role("),
        "registration, annotation, and finalization need one shared child-role classifier"
    );
    assert!(
        infer().contains("chelis_deep::validate::VALID_TAGS"),
        "classifier completeness must be checked against the canonical Deep vocabulary"
    );
    assert!(
        DEEP_VALIDATE.contains("pub const VALID_TAGS"),
        "the source contract expects chelis-deep to remain the vocabulary owner"
    );
    assert!(
        !infer().contains("unwrap_or(ChildStampRole::RuntimeExpr)"),
        "unknown child roles must never silently default to runtime ownership"
    );
}
