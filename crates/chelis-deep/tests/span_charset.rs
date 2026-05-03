//! Span-charset enforcement at the Deep parser, per
//! `spec/03-deep-syntax.md` §1.1.1.
//!
//! Span ID strings MUST NOT contain ASCII control characters in U+0000..=U+001F
//! (except U+0020 space) or U+007F (DEL). The parser rejects forbidden
//! code points at parse time so backend emitters can interpolate spans
//! into `// span: <id>` comments without a forbidden byte (notably `\n`)
//! terminating the comment and turning the rest into live source.
//!
//! These tests cover (a) every named forbidden byte category from §1.1.1,
//! (b) the parser-decision lock for empty span ID + leading/trailing
//! whitespace, and (c) round-trip preservation of every allowed code-point
//! shape (ASCII identifier, Unicode, mathematical symbol, dot-path).

use chelis_deep::parser::{ParseError, parse_str};
use chelis_deep::printer::print_canonical;

// ── Forbidden character rejection ─────────────────────────────────

fn assert_forbidden(src: &str, want_byte: u8, want_repr: &str) {
    let err = parse_str(src).expect_err("parser must reject forbidden span char");
    let display = err.to_string();
    match err {
        ParseError::ForbiddenSpanChar {
            code_point, repr, ..
        } => {
            assert_eq!(
                code_point,
                u32::from(want_byte),
                "wrong code point in error: {display}"
            );
            assert_eq!(repr, want_repr, "wrong repr in error: {display}");
        }
        other => panic!("expected ForbiddenSpanChar error, got: {other:?}"),
    }
}

#[test]
fn newline_in_span_value_rejected() {
    // Direct injection probe from the S4 red-team finding.
    let src = "(def {span: \"op\\nINJECT\"} c (lit {} 0))";
    assert_forbidden(src, b'\n', "\\n");
}

#[test]
fn carriage_return_in_span_value_rejected() {
    let src = "(def {span: \"op\\rINJECT\"} c (lit {} 0))";
    assert_forbidden(src, b'\r', "\\r");
}

#[test]
fn nul_in_span_value_rejected() {
    let src = "(def {span: \"op\\0INJECT\"} c (lit {} 0))";
    assert_forbidden(src, 0x00, "\\0");
}

#[test]
fn tab_in_span_value_rejected() {
    let src = "(def {span: \"op\\tINJECT\"} c (lit {} 0))";
    assert_forbidden(src, b'\t', "\\t");
}

#[test]
fn other_c0_control_in_span_value_rejected() {
    // U+0001 — randomly chosen from the C0 range to lock the entire
    // forbidden range, not just the obvious named characters. Built via a
    // raw string + manual escape-by-construction so the lexer doesn't
    // reject the source on a different code path.
    //
    // The string-literal escape-set in the Deep lexer is intentionally
    // narrow, so to introduce a literal U+0001 we construct the source
    // programmatically with the byte spliced in.
    let mut src = String::from("(def {span: \"op");
    src.push('\u{0001}');
    src.push_str("INJECT\"} c (lit {} 0))");
    assert_forbidden(&src, 0x01, "\\x01");
}

#[test]
fn del_byte_in_span_value_rejected() {
    let mut src = String::from("(def {span: \"op");
    src.push('\u{007F}');
    src.push_str("INJECT\"} c (lit {} 0))");
    assert_forbidden(&src, 0x7F, "\\x7f");
}

#[test]
fn newline_in_legacy_meta_expr_form_also_rejected() {
    // Defense-in-depth: the legacy `^{:span "..."}` prefix metadata form
    // must apply the same rule so producers can't bypass the check via
    // that entry point.
    let src = "^{:span \"op\\nINJECT\"} (def {} c (lit {} 0))";
    assert_forbidden(src, b'\n', "\\n");
}

#[test]
fn forbidden_char_error_message_cites_spec_and_byte_position() {
    // Lock the diagnostic shape: the message must include the byte
    // position inside the value, the canonical representation, and a
    // pointer back to spec/03-deep-syntax.md §1.1.1.
    let src = "(def {span: \"abc\\ndef\"} c (lit {} 0))";
    let err = parse_str(src).expect_err("parser must reject forbidden span char");
    let msg = err.to_string();
    assert!(
        msg.contains("byte 3"),
        "diagnostic missing in-value byte position: {msg}"
    );
    assert!(
        msg.contains("\\n"),
        "diagnostic missing canonical \\n form: {msg}"
    );
    assert!(
        msg.contains("U+000A"),
        "diagnostic missing U+000A code point: {msg}"
    );
    assert!(
        msg.contains("spec/03-deep-syntax.md"),
        "diagnostic missing spec pointer: {msg}"
    );
}

// ── Allowed shapes (positive parity) ──────────────────────────────

#[test]
fn empty_span_id_is_valid() {
    // Spec decision (§1.1.1): empty span ID is a VALID "no-provenance"
    // sentinel. Lock here so a future refactor can't silently flip it.
    let src = r#"(def {span: ""} c (lit {} 0))"#;
    let exprs = parse_str(src).expect("empty span ID must parse");
    assert_eq!(exprs.len(), 1);
}

#[test]
fn leading_and_trailing_whitespace_in_span_id_is_valid() {
    // U+0020 (ASCII space) is explicitly NOT forbidden. Discouraged but
    // valid per spec §1.1.1.
    let src = r#"(def {span: " eq1.body "} c (lit {} 0))"#;
    let exprs = parse_str(src).expect("space-padded span ID must parse");
    assert_eq!(exprs[0].span_id(), Some(" eq1.body "));
}

#[test]
fn unicode_dot_path_span_id_round_trips_byte_identical() {
    // The producer-canonical shape — Greek letters, mathematical
    // symbols, dot-path identifiers — all U+0020+, all allowed.
    let originals = [
        "n_001",
        "expr.5.lhs",
        "α.β.γ",
        "eq1.σ_body",
        "∇.∂.∂x",
        "obj_func.return_value",
    ];
    for original in &originals {
        let src = format!(r#"(def {{span: "{original}"}} c (lit {{}} 0))"#);
        let exprs1 = parse_str(&src).expect("unicode span ID must parse");
        assert_eq!(exprs1[0].span_id(), Some(*original));
        let printed = print_canonical(&exprs1);
        let exprs2 = parse_str(&printed).expect("reprint must re-parse");
        assert_eq!(exprs2[0].span_id(), Some(*original));
        // Fixed-point: reprint = print(parse(print(...))).
        let reprinted = print_canonical(&exprs2);
        assert_eq!(
            printed, reprinted,
            "printer not fixed point on `{original}`"
        );
    }
}

#[test]
fn synthesized_marker_span_ids_pass_validation() {
    // Synthesized markers (`__synthesized_<pass>__`) are pure ASCII
    // identifiers so they must pass the charset check. Lock to prevent
    // accidental over-restriction (e.g. forbidding `_`).
    let src = r#"(def {span: "__synthesized_grad__"} c (lit {} 0))"#;
    let exprs = parse_str(src).expect("synthesized marker span must parse");
    assert_eq!(exprs[0].span_id(), Some("__synthesized_grad__"));
}

#[test]
fn other_meta_keys_with_control_chars_unaffected() {
    // The charset rule applies ONLY to `span`. Other arbitrary string
    // metadata values are unconstrained — verify a `\n` in some other
    // key is still accepted.
    let src = "(def {note: \"line1\\nline2\"} c (lit {} 0))";
    parse_str(src).expect("non-span string metadata must accept newlines");
}
