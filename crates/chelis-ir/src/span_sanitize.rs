//! Comment-safe span sanitization for backend emission.
//!
//! Per `spec/03-deep-syntax.md` §1.1.1, span ID strings MUST NOT contain
//! ASCII control characters in U+0000..=U+001F (except space, U+0020) or
//! U+007F (DEL). The Deep parser rejects forbidden characters at parse
//! time, so well-formed input cannot reach codegen with a forbidden code
//! point. This module provides a defense-in-depth sanitizer for the case
//! where IR is constructed programmatically (bypassing the parser) and a
//! forbidden code point reaches a backend's `// span: <id>` emission.
//!
//! The contract for a clean span ID — every code point allowed by the
//! spec — is **verbatim preservation**. Audit invariant: the `// span:
//! <id>` text in generated C / HIP / Metal source is byte-identical to the
//! span ID a customer sees in the upstream `.spans.json` sidecar for any
//! well-behaved input. The sanitizer returns `Cow::Borrowed(span)` in that
//! case, both as a zero-copy fast path and as an enforcement of the
//! invariant — clean spans are not rewritten.
//!
//! For forbidden code points, each is replaced with its canonical
//! backslash-escape (`\n` for U+000A LF, `\r` for U+000D CR, `\0` for
//! U+0000 NUL, `\t` for U+0009 TAB) or `\xNN` (two-digit hex) for any
//! other forbidden control byte.

use std::borrow::Cow;

/// Returns true when `b` is a forbidden ASCII byte per
/// `spec/03-deep-syntax.md` §1.1.1: U+0000..=U+001F except U+0020 (space),
/// or U+007F (DEL).
#[inline]
pub fn is_forbidden_byte(b: u8) -> bool {
    // U+0000..=U+001F except U+0020 (space). Space is 0x20 which is not
    // in 0x00..=0x1F, so we can just check the low control range.
    b <= 0x1F || b == 0x7F
}

/// Look for the first forbidden byte in `s`. Returns the byte position
/// and the offending byte if any.
///
/// The sanitizer and the Deep-side parser validation both use this so the
/// definitions can never drift.
#[inline]
pub fn find_forbidden_byte(s: &str) -> Option<(usize, u8)> {
    s.as_bytes()
        .iter()
        .copied()
        .enumerate()
        .find(|(_, b)| is_forbidden_byte(*b))
}

/// Render a forbidden byte as its canonical backslash escape, e.g. `\n`,
/// `\r`, `\0`, `\t`, or `\xNN` (two-digit lowercase hex) for other C0
/// controls and U+007F.
fn escape_forbidden(b: u8) -> String {
    match b {
        0x00 => "\\0".to_string(),
        0x09 => "\\t".to_string(),
        0x0A => "\\n".to_string(),
        0x0D => "\\r".to_string(),
        other => format!("\\x{other:02x}"),
    }
}

/// Sanitize a span ID for safe interpolation into a `// span: <id>` line
/// comment.
///
/// Returns `Cow::Borrowed(span)` (zero-copy) when `span` contains no
/// forbidden code points — this is the common case (well-formed Deep
/// input cannot carry forbidden code points, since the parser rejects
/// them per `spec/03-deep-syntax.md` §1.1.1). The borrowed case also locks
/// the audit invariant: clean span IDs are emitted verbatim, identical to
/// the upstream `.spans.json` sidecar.
///
/// Returns `Cow::Owned(escaped)` when one or more forbidden bytes are
/// present, with each forbidden byte replaced by its canonical
/// backslash-escape form (`\n`, `\r`, `\0`, `\t`, or `\xNN`).
pub fn sanitize_for_comment(span: &str) -> Cow<'_, str> {
    if find_forbidden_byte(span).is_none() {
        return Cow::Borrowed(span);
    }
    // Slow path: rebuild byte-by-byte (forbidden bytes are all single-byte
    // ASCII, so byte-level processing preserves UTF-8 of allowed code
    // points untouched).
    let bytes = span.as_bytes();
    let mut out = String::with_capacity(bytes.len() + 4);
    for &b in bytes {
        if is_forbidden_byte(b) {
            out.push_str(&escape_forbidden(b));
        } else {
            out.push(b as char);
        }
    }
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_id_returns_borrowed() {
        let s = "clean.id";
        let out = sanitize_for_comment(s);
        assert_eq!(out, "clean.id");
        // Zero-copy: pointer identity confirms Borrowed variant.
        assert!(matches!(out, Cow::Borrowed(_)));
        if let Cow::Borrowed(b) = out {
            assert_eq!(b.as_ptr(), s.as_ptr());
        }
    }

    #[test]
    fn empty_string_returns_borrowed() {
        let s = "";
        let out = sanitize_for_comment(s);
        assert_eq!(out, "");
        assert!(matches!(out, Cow::Borrowed(_)));
    }

    #[test]
    fn unicode_id_returns_borrowed() {
        // Greek letters + mathematical symbols + dot-path = the canonical
        // shape of producer-emitted IDs. Every code point is U+0020+ and
        // not U+007F, so the sanitizer must return Borrowed (verbatim).
        let s = "eq1.σ_body.α∇β";
        let out = sanitize_for_comment(s);
        assert_eq!(out, s);
        assert!(matches!(out, Cow::Borrowed(_)));
        if let Cow::Borrowed(b) = out {
            assert_eq!(b.as_ptr(), s.as_ptr());
        }
    }

    #[test]
    fn newline_is_escaped_to_backslash_n() {
        // The original injection vector: \n terminates the // line
        // comment and emits the rest as live code. Sanitize must replace
        // it with the literal two-character sequence \n.
        let out = sanitize_for_comment("op\nINJECT");
        assert!(matches!(out, Cow::Owned(_)));
        assert_eq!(out, "op\\nINJECT");
    }

    #[test]
    fn carriage_return_is_escaped() {
        let out = sanitize_for_comment("op\rINJECT");
        assert!(matches!(out, Cow::Owned(_)));
        assert_eq!(out, "op\\rINJECT");
    }

    #[test]
    fn nul_is_escaped() {
        let out = sanitize_for_comment("op\0INJECT");
        assert!(matches!(out, Cow::Owned(_)));
        assert_eq!(out, "op\\0INJECT");
    }

    #[test]
    fn tab_is_escaped() {
        let out = sanitize_for_comment("op\tINJECT");
        assert!(matches!(out, Cow::Owned(_)));
        assert_eq!(out, "op\\tINJECT");
    }

    #[test]
    fn other_control_chars_use_hex_escape() {
        // U+0001 has no canonical backslash form so we use \x01.
        let out = sanitize_for_comment("op\x01INJECT");
        assert!(matches!(out, Cow::Owned(_)));
        assert_eq!(out, "op\\x01INJECT");
    }

    #[test]
    fn del_byte_is_escaped() {
        // U+007F (DEL) is the lone forbidden byte outside the C0 range.
        let out = sanitize_for_comment("op\x7fINJECT");
        assert!(matches!(out, Cow::Owned(_)));
        assert_eq!(out, "op\\x7fINJECT");
    }

    #[test]
    fn space_is_not_escaped() {
        // U+0020 is explicitly NOT forbidden — leading/trailing/internal
        // space passes through verbatim.
        let s = " eq1.body ";
        let out = sanitize_for_comment(s);
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(out, " eq1.body ");
    }

    #[test]
    fn multiple_forbidden_bytes_all_escaped() {
        let out = sanitize_for_comment("a\nb\rc\0d\te\x01f");
        assert!(matches!(out, Cow::Owned(_)));
        assert_eq!(out, "a\\nb\\rc\\0d\\te\\x01f");
    }

    #[test]
    fn forbidden_at_start_and_end() {
        let out = sanitize_for_comment("\nstart_end\n");
        assert_eq!(out, "\\nstart_end\\n");
    }

    #[test]
    fn find_forbidden_byte_reports_first_position() {
        assert_eq!(find_forbidden_byte("clean"), None);
        assert_eq!(find_forbidden_byte("op\nbad"), Some((2, b'\n')));
        assert_eq!(find_forbidden_byte("\rfirst"), Some((0, b'\r')));
        // Unicode prefix shouldn't shift things.
        assert_eq!(find_forbidden_byte("σ"), None);
    }

    #[test]
    fn is_forbidden_byte_classifies_correctly() {
        // C0 controls except space.
        for b in 0x00u8..=0x1F {
            assert!(is_forbidden_byte(b), "byte 0x{b:02x} should be forbidden");
        }
        // Space is allowed.
        assert!(!is_forbidden_byte(0x20));
        // Printable ASCII allowed.
        for b in 0x21u8..0x7F {
            assert!(!is_forbidden_byte(b), "byte 0x{b:02x} should be allowed");
        }
        // DEL forbidden.
        assert!(is_forbidden_byte(0x7F));
        // High bytes (UTF-8 continuation / leading bytes) are not control
        // bytes; the sanitizer passes them through unchanged.
        for b in 0x80u8..=0xFF {
            assert!(!is_forbidden_byte(b), "byte 0x{b:02x} should be allowed");
        }
    }
}
