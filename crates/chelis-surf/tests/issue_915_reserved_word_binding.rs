//! chelis#915: a reserved declaration keyword used as a value-binding name
//! must name the reserved word, not report `expected identifier, found Eq`
//! several bytes downstream.
//!
//! Root cause: `sig` lexes to `TokenKind::Sig` (`lexer.rs:500`), which is a
//! valid declaration head, so `parse_decl` dispatches into `parse_sig_decl`.
//! That consumes the keyword and then demands the signature's name, so the
//! failure surfaces at the `=` with the reserved word already consumed. The
//! *found* token is therefore `Eq`, never the keyword — which is why a
//! special case keyed on the found token would not fire on this form.
//!
//! The fix guards the three declaration heads whose keyword is a plausible
//! value name (`sig`, `type`, `dim`) on "next token is `=`", which is
//! unambiguously an attempted value binding, and points the offset at the
//! keyword instead of the `=`.
//!
//! This is a diagnostic-only change: no grammar change and no token
//! reclassification, so every program that parsed before still parses. The
//! controls below lock that.

use chelis_surf::parser::parse_str;

fn parse_err(src: &str) -> String {
    parse_str(src)
        .map(|decls| panic!("expected a parse error, got {} decl(s)", decls.len()))
        .unwrap_err()
        .to_string()
}

// ── The reported defect ─────────────────────────────────────────────

/// chelis#915's exact reproducer. Before the fix this read
/// `expected identifier, found Eq at byte 4`.
#[test]
fn sig_as_value_binding_names_the_reserved_word() {
    let msg = parse_err("sig = 0.2f64\n");
    assert!(
        msg.contains("`sig` is a reserved word"),
        "message must name the reserved word; got: {msg}"
    );
    assert!(
        msg.contains("signature declaration"),
        "message must say what `sig` introduces; got: {msg}"
    );
    assert!(
        msg.contains("sigma"),
        "message must offer a rename; got: {msg}"
    );
    // The offset points at the keyword, not at the `=` (which was byte 4).
    assert!(
        msg.contains("at byte 0"),
        "offset must point at the keyword; got: {msg}"
    );
}

/// The same shape for the other two declaration heads that can be reached
/// with `<keyword> =`. These were `expected type identifier, found Eq at
/// byte 5` and `expected identifier or type identifier, found Eq at byte 4`.
#[test]
fn type_and_dim_as_value_bindings_name_the_reserved_word() {
    let type_msg = parse_err("type = 1\n");
    assert!(
        type_msg.contains("`type` is a reserved word") && type_msg.contains("at byte 0"),
        "got: {type_msg}"
    );

    let dim_msg = parse_err("dim = 1\n");
    assert!(
        dim_msg.contains("`dim` is a reserved word") && dim_msg.contains("at byte 0"),
        "got: {dim_msg}"
    );
}

// ── Controls: nothing that parsed before may stop parsing ───────────

/// A legitimate signature declaration is untouched.
#[test]
fn legitimate_sig_declaration_still_parses() {
    let src = "sig f : f64\ndef f() -> f64 = 0.2f64\n";
    parse_str(src).expect("a real `sig` declaration must still parse");
}

/// A legitimate type declaration is untouched.
#[test]
fn legitimate_type_declaration_still_parses() {
    parse_str("type Weights = tensor[hidden, hidden, f32]\n")
        .expect("`type` alias declaration must still parse");
    parse_str("type Probability = | Probability { value: f32 }\n")
        .expect("`type` sum declaration must still parse");
}

/// A malformed `sig` declaration that is *not* an attempted value binding
/// falls through to the ordinary `expected …, found …` path. The guard is
/// keyed on `=`, so it must not swallow unrelated declaration errors.
#[test]
fn malformed_sig_declaration_keeps_the_generic_message() {
    let msg = parse_err("sig : f64\n");
    assert!(
        !msg.contains("reserved word"),
        "a non-binding malformed sig must keep the generic path; got: {msg}"
    );
    assert!(
        msg.contains("expected identifier"),
        "expected the generic identifier diagnostic; got: {msg}"
    );
}

/// Class B (chelis#915's terminology): the keyword in a non-declaration
/// position already names the token, and is deliberately left alone. This
/// locks the generic mechanism so a later change cannot quietly reroute it.
#[test]
fn reserved_word_in_parameter_position_is_unchanged() {
    let msg = parse_err("def f(sig) -> f64 = sig\n");
    assert!(
        msg.contains("expected identifier") && msg.contains("Sig"),
        "parameter-position diagnostic must be unchanged; got: {msg}"
    );
    assert!(
        !msg.contains("reserved word"),
        "the fix must not reach parameter position; got: {msg}"
    );
}
