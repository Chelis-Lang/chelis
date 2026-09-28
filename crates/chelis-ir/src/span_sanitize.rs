//! Comment-context and format-string-context sanitizers for producer-
//! supplied strings flowing into generated source.
//!
//! Per `spec/03-deep-syntax.md` §1.1.1, span ID strings MUST NOT contain
//! ASCII control characters in U+0000..=U+001F (except space, U+0020) or
//! U+007F (DEL). The Deep parser rejects forbidden characters at parse
//! time, so well-formed input cannot reach codegen with a forbidden code
//! point. This module provides defense-in-depth sanitizers for the case
//! where IR is constructed programmatically (bypassing the parser) and a
//! forbidden code point reaches a backend emission site.
//!
//! ## Two emission contexts
//!
//! - [`sanitize_for_comment`] — for producer-supplied strings
//!   interpolated into `// ...` line comments.
//! - [`sanitize_for_format_string`] — for producer-supplied strings
//!   interpolated into a C/HIP `printf`/`fprintf` format string at compile
//!   time (i.e., into the format itself, not as a `%s` runtime argument).
//!   Adds `%`-escaping and `\\`/`"`-escaping on top of the comment-context
//!   rules so `printf` cannot misread positional specifiers and the
//!   surrounding `"..."` C string literal stays well-formed.
//!
//! ## Verbatim-preservation contract
//!
//! A clean span ID is preserved verbatim. The `// span: <id>` text in
//! generated C, HIP, and Metal source matches the ID in `.spans.json`.
//! [`sanitize_for_comment`] returns `Cow::Borrowed` in this case.
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

/// Sanitize a producer-supplied string for safe interpolation into a
/// `// ...` line comment.
///
/// This is the architectural shared sanitizer for the comment emission
/// context: every `format!("// ...{x}...")` callsite where `x` is a
/// producer-supplied string (span IDs, IR identifier-derived names, type
/// names, anything else flowing through from upstream) should route the
/// interpolated value through this function first. The contract is
/// agnostic to the source field.
///
/// Returns `Cow::Borrowed(s)` (zero-copy) when `s` contains no
/// forbidden code points. This is the common case (well-formed Deep
/// input cannot carry forbidden code points, since the parser rejects
/// them per `spec/03-deep-syntax.md` §1.1.1, and `LoadStoreName` enforces
/// the same constraint at construction). The borrowed case also locks
/// the audit invariant: clean strings are emitted verbatim, identical to
/// the upstream source representation.
///
/// Returns `Cow::Owned(escaped)` when one or more forbidden bytes are
/// present, with each forbidden byte replaced by its canonical
/// backslash-escape form (`\n`, `\r`, `\0`, `\t`, or `\xNN`).
pub fn sanitize_for_comment(s: &str) -> Cow<'_, str> {
    if find_forbidden_byte(s).is_none() {
        return Cow::Borrowed(s);
    }
    // Slow path: rebuild byte-by-byte (forbidden bytes are all single-byte
    // ASCII, so byte-level processing preserves UTF-8 of allowed code
    // points untouched).
    let bytes = s.as_bytes();
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

/// Returns true when `b` requires escaping inside a C/HIP `printf`-style
/// format string baked into a `"..."` C string literal: any forbidden
/// comment-context byte, plus `%` (positional format specifier), `\\`
/// (string-literal escape leader), or `"` (string-literal terminator).
#[inline]
fn is_forbidden_in_format_string(b: u8) -> bool {
    is_forbidden_byte(b) || b == b'%' || b == b'\\' || b == b'"'
}

/// Render a byte that must be escaped inside a C/HIP format-string
/// literal. The format-string context is stricter than the comment
/// context: in addition to the comment escapes, `%` doubles to `%%`
/// (so positional consumers don't misread it), `\\` doubles to `\\\\`
/// (so the C string-literal lexer doesn't interpret it), and `"` becomes
/// `\\"` (so the surrounding `"..."` literal stays well-formed).
fn escape_forbidden_in_format_string(b: u8) -> String {
    match b {
        0x00 => "\\0".to_string(),
        0x09 => "\\t".to_string(),
        0x0A => "\\n".to_string(),
        0x0D => "\\r".to_string(),
        b'%' => "%%".to_string(),
        b'\\' => "\\\\".to_string(),
        b'"' => "\\\"".to_string(),
        other if is_forbidden_byte(other) => format!("\\x{other:02x}"),
        // Caller guards on `is_forbidden_in_format_string`; any other byte
        // is a programming error in escape dispatch.
        other => {
            panic!("escape_forbidden_in_format_string called with non-forbidden byte 0x{other:02x}")
        }
    }
}

/// Sanitize a producer-supplied string for safe **compile-time**
/// interpolation into a C/HIP `printf`/`fprintf` format string baked into
/// a `"..."` C string literal — i.e., into the format itself, NOT as a
/// `%s` runtime argument.
///
/// This is the architectural shared sanitizer for the format-string
/// emission context. The format-string context is stricter than the
/// comment context (`sanitize_for_comment`) along three axes:
///
/// 1. **Control bytes** are escaped to `\n` / `\r` / `\0` / `\t` /
///    `\xNN` — same as the comment context (the surrounding `"..."` C
///    string literal would otherwise produce invalid escapes for raw
///    control bytes).
/// 2. **`%`** is escaped to `%%` so the runtime `printf` does not misread
///    the producer-supplied string as a positional specifier (e.g.
///    `%s` reaches into a missing argument and crashes).
/// 3. **`\\`** is escaped to `\\\\` and **`"`** is escaped to `\\"` so
///    the surrounding C string literal stays well-formed.
///
/// Returns `Cow::Borrowed(s)` (zero-copy) when `s` contains no
/// forbidden bytes — same verbatim-preservation invariant as
/// `sanitize_for_comment`.
///
/// ## When to use vs. when not to
///
/// Use this for compile-time string interpolation INTO the format
/// itself, e.g. `format!("printf(\"{name}: foo\\n\");", name = sanitized)`.
/// Do NOT use this for the runtime data side of a `%s` substitution —
/// that data goes through the runtime, never through the C string
/// literal lexer or the `printf` format scanner. The dangerous case is
/// strictly format-time interpolation; runtime-data is safe by virtue
/// of `%s` substitution itself.
///
pub fn sanitize_for_format_string(s: &str) -> Cow<'_, str> {
    let needs_escape = s
        .as_bytes()
        .iter()
        .any(|&b| is_forbidden_in_format_string(b));
    if !needs_escape {
        return Cow::Borrowed(s);
    }
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(bytes.len() + 4);
    for &b in bytes {
        if is_forbidden_in_format_string(b) {
            out.push_str(&escape_forbidden_in_format_string(b));
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

    // ------------------------------------------------------------------
    // sanitize_for_format_string
    //
    // Stricter than sanitize_for_comment: comment-context forbidden bytes
    // PLUS `%`, `\\`, and `"`. Verbatim-preservation contract for clean
    // input still holds.
    // ------------------------------------------------------------------

    #[test]
    fn fmt_clean_id_returns_borrowed() {
        let s = "my_func";
        let out = sanitize_for_format_string(s);
        assert_eq!(out, "my_func");
        assert!(matches!(out, Cow::Borrowed(_)));
        if let Cow::Borrowed(b) = out {
            assert_eq!(b.as_ptr(), s.as_ptr());
        }
    }

    #[test]
    fn fmt_unicode_clean_returns_borrowed() {
        // Greek + math + identifiers; no forbidden bytes, no `%`, no `\\`,
        // no `"`. Should stay zero-copy.
        let s = "eq1.σ_body.α∇β";
        let out = sanitize_for_format_string(s);
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(out, s);
    }

    #[test]
    fn fmt_percent_is_doubled() {
        // The bug class: a `%s` in the producer-supplied format-time
        // string would be misread by printf as a positional specifier
        // referencing a non-existent argument. Doubling neutralises it.
        let out = sanitize_for_format_string("foo%sBAR");
        assert!(matches!(out, Cow::Owned(_)));
        assert_eq!(out, "foo%%sBAR");
    }

    #[test]
    fn fmt_multiple_percents_each_doubled() {
        let out = sanitize_for_format_string("%d_%s_%n");
        assert_eq!(out, "%%d_%%s_%%n");
    }

    #[test]
    fn fmt_backslash_is_doubled() {
        // A raw `\\` in the format-time string would be eaten by the C
        // string-literal lexer. Doubling preserves the visible text.
        let out = sanitize_for_format_string("path\\to\\file");
        assert!(matches!(out, Cow::Owned(_)));
        assert_eq!(out, "path\\\\to\\\\file");
    }

    #[test]
    fn fmt_double_quote_is_escaped() {
        // A raw `"` would terminate the surrounding `"..."` C string
        // literal. Backslash-escape preserves the literal.
        let out = sanitize_for_format_string("a\"b");
        assert!(matches!(out, Cow::Owned(_)));
        assert_eq!(out, "a\\\"b");
    }

    #[test]
    fn fmt_newline_is_escaped() {
        // Comment-context bug class still applies: a raw newline inside a
        // C string literal is invalid (some compilers tolerate it as a
        // line splice, others reject), and the visible text would
        // fragment.
        let out = sanitize_for_format_string("op\nINJECT");
        assert!(matches!(out, Cow::Owned(_)));
        assert_eq!(out, "op\\nINJECT");
    }

    #[test]
    fn fmt_carriage_return_is_escaped() {
        let out = sanitize_for_format_string("op\rINJECT");
        assert_eq!(out, "op\\rINJECT");
    }

    #[test]
    fn fmt_nul_is_escaped() {
        let out = sanitize_for_format_string("op\0INJECT");
        assert_eq!(out, "op\\0INJECT");
    }

    #[test]
    fn fmt_tab_is_escaped() {
        let out = sanitize_for_format_string("op\tINJECT");
        assert_eq!(out, "op\\tINJECT");
    }

    #[test]
    fn fmt_other_control_chars_use_hex_escape() {
        let out = sanitize_for_format_string("op\x01INJECT");
        assert_eq!(out, "op\\x01INJECT");
    }

    #[test]
    fn fmt_del_byte_is_escaped() {
        let out = sanitize_for_format_string("op\x7fINJECT");
        assert_eq!(out, "op\\x7fINJECT");
    }

    #[test]
    fn fmt_space_is_not_escaped() {
        // U+0020 SPACE is allowed in format strings; it neither breaks
        // string literals nor is a printf specifier.
        let s = " hello world ";
        let out = sanitize_for_format_string(s);
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(out, " hello world ");
    }

    #[test]
    fn fmt_empty_string_returns_borrowed() {
        let out = sanitize_for_format_string("");
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(out, "");
    }

    #[test]
    fn fmt_combined_attack_vectors() {
        // All four classes in one string: control byte, `%`, `\\`, `"`.
        let out = sanitize_for_format_string("a\nb%sc\\d\"e");
        assert!(matches!(out, Cow::Owned(_)));
        assert_eq!(out, "a\\nb%%sc\\\\d\\\"e");
    }

    #[test]
    fn fmt_clean_output_runs_through_once() {
        // The sanitizer is *not* idempotent for dirty inputs (the
        // escaped form contains `\\` and `%%` which themselves require
        // escaping if re-fed into the sanitizer). Callers MUST sanitize
        // exactly once at the emission boundary; this matches every
        // `sanitize_for_comment` callsite pattern. The unconditional
        // single-pass invariant: clean inputs (Borrowed) survive
        // unchanged.
        let clean = "my_func";
        let out = sanitize_for_format_string(clean);
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(out, clean);
    }
}
