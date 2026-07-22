//! Structural tripwires supporting the executable compile-fail oracles on
//! `errors::report`. These scans do not replace the compiler tests; they make
//! accidental reintroduction of the old ambient and throwaway-vector seams
//! fail with a targeted explanation.

const INFER: &str = include_str!("../src/infer.rs");
const ERRORS: &str = include_str!("../src/errors.rs");
const DEEP_TYPE: &str = include_str!("../src/deep_type.rs");
const OPACITY: &str = include_str!("../src/opacity.rs");
const SESSION: &str = include_str!("../src/session.rs");
const DEEP_VALIDATE: &str = include_str!("../../chelis-deep/src/validate.rs");

#[test]
fn binder_and_signature_resolution_have_no_ambient_symbols() {
    for (label, source) in [
        ("infer", INFER),
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
        !INFER.contains("let mut hidden_errors"),
        "the retained reviewer mutation must remain uncompilable, not planted"
    );
}

#[test]
fn sink_construction_and_storage_stay_inside_the_owner() {
    for (label, source) in [("infer", INFER), ("errors", ERRORS), ("deep_type", DEEP_TYPE)] {
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
    let annotation = INFER
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
        !INFER.contains("fn infer_expr_in_scope("),
        "the old fresh-substitution annotation inference seam must be deleted"
    );
}

#[test]
fn checked_results_share_one_totality_finalizer() {
    assert!(
        INFER.contains("fn finalize_checked_program("),
        "all CheckedProgram results need one totality finalization boundary"
    );
    assert!(
        INFER.contains("annotated_totality_invariant_traces("),
        "finalization must inspect the authoritative annotated tree"
    );

    let constructor_calls = INFER.matches("CheckedProgram::from_parts_with_").count();
    assert_eq!(
        constructor_calls, 0,
        "result paths must not bypass finalize_checked_program; found {constructor_calls} direct constructors"
    );
}

#[test]
fn annotation_ownership_uses_the_canonical_exhaustive_child_role_table() {
    for role in [
        "RuntimeExpr",
        "Syntax",
        "Selector",
        "Binder",
        "Type",
        "ExplicitInferenceBypass",
    ] {
        assert!(
            INFER.contains(role),
            "the shared child ownership classifier must name the `{role}` role"
        );
    }
    assert!(
        INFER.contains("child_stamp_role("),
        "registration, annotation, and finalization need one shared child-role classifier"
    );
    assert!(
        INFER.contains("chelis_deep::validate::VALID_TAGS"),
        "classifier completeness must be checked against the canonical Deep vocabulary"
    );
    assert!(
        DEEP_VALIDATE.contains("pub const VALID_TAGS"),
        "the source contract expects chelis-deep to remain the vocabulary owner"
    );
    assert!(
        !INFER.contains("unwrap_or(ChildStampRole::RuntimeExpr)"),
        "unknown child roles must never silently default to runtime ownership"
    );
}
