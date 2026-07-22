//! Structural tripwires supporting the executable compile-fail oracles on
//! `errors::report`. These scans do not replace the compiler tests; they make
//! accidental reintroduction of the old ambient and throwaway-vector seams
//! fail with a targeted explanation.

const INFER: &str = include_str!("../src/infer.rs");
const ERRORS: &str = include_str!("../src/errors.rs");
const DEEP_TYPE: &str = include_str!("../src/deep_type.rs");
const OPACITY: &str = include_str!("../src/opacity.rs");
const SESSION: &str = include_str!("../src/session.rs");

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
