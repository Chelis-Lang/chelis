//! Validating newtype for `RiscOp::Load`/`RiscOp::Store` `name` fields.
//!
//! Per `spec/03-deep-syntax.md` §1.1.1, the architectural rule is that
//! every producer-supplied string flowing into generated source is
//! validated either at the parse boundary (spans, via §1.1.1 + parser
//! rejection) or at the construction boundary. This module establishes
//! the construction-side trust boundary for `Load`/`Store` names.
//!
//! ## Grammar
//!
//! The accepted character set mirrors the Deep parser's identifier
//! grammar (`crates/chelis-deep/src/lexer.rs`: `is_ident_start`,
//! `is_ident_continue`) extended with `.` to admit the synthesized
//! tuple-flatten names (`foo.0`, `lib_double.0`, `grads.1`) that
//! `chelis-ir`'s lowering legitimately produces today:
//!
//! - first byte: ASCII letter or `_` (`[A-Za-z_]`)
//! - subsequent bytes: ASCII letters, digits, `_`, `-`, `.`
//!   (`[A-Za-z0-9_.-]`)
//!
//! Every other byte is rejected. The grammar is ASCII-only (the parser is
//! ASCII-only); high bytes from any UTF-8 multibyte character are not in
//! the allowed set, so non-ASCII identifiers like `σ` are rejected.
//! Empty strings are rejected.
//!
//! ## Why this kills the bug class
//!
//! The S4 re-red-team gate found that programmatic IR construction with
//! `RiscOp::Load { name: "x\nINJECT".to_string() }` injects across all
//! three backends (printf format strings, Metal node-comment lines, etc.)
//! because the producer-supplied string flows into generated source
//! without sanitization. By gating construction on parser-grammar
//! validation, the only way a forbidden byte (newline, NUL, DEL, etc.)
//! can reach codegen is for someone to construct the newtype in a way
//! that bypasses `LoadStoreName::new` — which the `LoadStoreName(String)`
//! private field forbids in safe Rust outside this module.
//!
//! ## Deferred deeper work
//!
//! `spec/upstream-bugs/producer-string-sanitization.md` tracks the
//! deferred per-emission-context sanitization (comment shared sanitizer,
//! format-string sanitizer, full audit). This newtype is the bounded
//! S4-close-out fix; it does not replace the deeper hardening.

use std::borrow::Borrow;
use std::fmt;

use serde::{Deserialize, Serialize};

/// A `RiscOp::Load`/`RiscOp::Store` `name`, validated against the Deep
/// parser's identifier grammar (extended with `.` for synthesized
/// tuple-flatten names).
///
/// Construct with [`LoadStoreName::new`]; the constructor enforces the
/// grammar described at the module level. The newtype is
/// `#[serde(transparent)]` so its serialized shape is identical to the
/// previous `String` field — bincode round-trip and JSON wire formats are
/// preserved.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LoadStoreName(String);

/// Reasons the constructor may reject a candidate name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadStoreNameError {
    /// The candidate string was empty. Empty names are not produced by
    /// the parser or by lowering, and would emit as malformed identifier
    /// text in every backend.
    Empty,
    /// The candidate contained a byte outside the allowed grammar at
    /// `byte_offset`. `character` is the first character starting at that
    /// byte; `code_point` is its Unicode scalar value (`u32`).
    InvalidCharacter {
        byte_offset: usize,
        character: char,
        code_point: u32,
    },
}

impl fmt::Display for LoadStoreNameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadStoreNameError::Empty => {
                write!(f, "Load/Store name must not be empty")
            }
            LoadStoreNameError::InvalidCharacter {
                byte_offset,
                character,
                code_point,
            } => {
                write!(
                    f,
                    "invalid character {character:?} (U+{code_point:04X}) at byte offset \
                     {byte_offset} in Load/Store name; allowed grammar is \
                     [A-Za-z_][A-Za-z0-9_.-]* (see spec/03-deep-syntax.md §1.1.1 / \
                     spec/upstream-bugs/producer-string-sanitization.md)"
                )
            }
        }
    }
}

impl std::error::Error for LoadStoreNameError {}

#[inline]
fn is_name_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

#[inline]
fn is_name_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.'
}

impl LoadStoreName {
    /// Construct a validated `LoadStoreName`, rejecting any candidate
    /// that does not match the IR identifier grammar.
    pub fn new(s: impl Into<String>) -> Result<Self, LoadStoreNameError> {
        let owned = s.into();
        Self::validate(&owned)?;
        Ok(Self(owned))
    }

    /// Convenience for sites where the name is statically known to be a
    /// valid IR identifier (lowering, tests, fixed backend kernels).
    /// Panics with the validation error if the input is invalid.
    ///
    /// Use this when the name is a literal or comes from a source that
    /// is itself parser-validated (e.g., an `Atom::Symbol` or a
    /// `tuple-flatten` prefix). Do NOT use this for producer-supplied
    /// strings where validation could legitimately fail at runtime —
    /// for those, propagate the `Result` from [`LoadStoreName::new`].
    pub fn must(s: impl Into<String>) -> Self {
        let owned = s.into();
        match Self::validate(&owned) {
            Ok(()) => Self(owned),
            Err(e) => panic!("invalid LoadStoreName {owned:?}: {e}"),
        }
    }

    fn validate(s: &str) -> Result<(), LoadStoreNameError> {
        let bytes = s.as_bytes();
        if bytes.is_empty() {
            return Err(LoadStoreNameError::Empty);
        }
        // First-byte check (start-character rule).
        if !is_name_start(bytes[0]) {
            let character = s
                .chars()
                .next()
                .expect("non-empty string has at least one character");
            return Err(LoadStoreNameError::InvalidCharacter {
                byte_offset: 0,
                character,
                code_point: character as u32,
            });
        }
        // Continue-character check for the remaining bytes. Treat each
        // disallowed byte at byte_offset N as the start of a (possibly
        // multibyte) character starting at the most recent valid char
        // boundary at-or-before N.
        for (offset, &b) in bytes.iter().enumerate().skip(1) {
            if !is_name_continue(b) {
                let character = s[offset..].chars().next().unwrap_or_else(|| {
                    // We landed in the middle of a UTF-8 multibyte
                    // sequence. Walk back to the start of the
                    // character so the diagnostic is meaningful.
                    let mut start = offset;
                    while start > 0 && !s.is_char_boundary(start) {
                        start -= 1;
                    }
                    s[start..]
                        .chars()
                        .next()
                        .expect("char boundary scan finds a character")
                });
                let char_offset = {
                    let mut start = offset;
                    while start > 0 && !s.is_char_boundary(start) {
                        start -= 1;
                    }
                    start
                };
                return Err(LoadStoreNameError::InvalidCharacter {
                    byte_offset: char_offset,
                    character,
                    code_point: character as u32,
                });
            }
        }
        Ok(())
    }

    /// Borrow the validated name as `&str`.
    #[inline]
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume the newtype, returning the underlying validated `String`.
    #[inline]
    #[must_use]
    pub fn into_string(self) -> String {
        self.0
    }
}

/// `From<&str>` delegates to [`LoadStoreName::must`] and panics on
/// invalid input. This exists so `"x".into()` survives at the many
/// statically-known construction sites in lowering, tier2, grad, the
/// backends, and tests; it is NOT the path for producer-supplied
/// strings that could legitimately be invalid (use
/// [`LoadStoreName::new`] there and propagate the `Result`).
impl From<&str> for LoadStoreName {
    fn from(s: &str) -> Self {
        Self::must(s)
    }
}

/// `From<String>` mirror of [`From<&str>`]. Panics on invalid input.
impl From<String> for LoadStoreName {
    fn from(s: String) -> Self {
        Self::must(s)
    }
}

impl AsRef<str> for LoadStoreName {
    #[inline]
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Borrow<str> for LoadStoreName {
    #[inline]
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for LoadStoreName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl PartialEq<str> for LoadStoreName {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for LoadStoreName {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

impl PartialEq<String> for LoadStoreName {
    #[inline]
    fn eq(&self, other: &String) -> bool {
        &self.0 == other
    }
}

impl PartialEq<LoadStoreName> for str {
    #[inline]
    fn eq(&self, other: &LoadStoreName) -> bool {
        self == other.0
    }
}

impl PartialEq<LoadStoreName> for &str {
    #[inline]
    fn eq(&self, other: &LoadStoreName) -> bool {
        *self == other.0
    }
}

impl PartialEq<LoadStoreName> for String {
    #[inline]
    fn eq(&self, other: &LoadStoreName) -> bool {
        self == &other.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Positive cases — every legitimate IR-name shape constructs.
    // ------------------------------------------------------------------

    #[test]
    fn single_letter_succeeds() {
        let n = LoadStoreName::new("x").expect("single letter is valid");
        assert_eq!(n.as_str(), "x");
    }

    #[test]
    fn underscore_start_succeeds() {
        let n = LoadStoreName::new("_foo").expect("underscore start is valid");
        assert_eq!(n.as_str(), "_foo");
    }

    #[test]
    fn underscores_and_digits_succeed() {
        let n = LoadStoreName::new("foo_bar_42").expect("underscores+digits valid");
        assert_eq!(n.as_str(), "foo_bar_42");
    }

    #[test]
    fn dash_in_continue_position_succeeds() {
        // Deep identifier grammar admits `-` after the first character.
        let n = LoadStoreName::new("Foo-Bar").expect("dash valid in continue position");
        assert_eq!(n.as_str(), "Foo-Bar");
    }

    #[test]
    fn dot_for_tuple_flatten_succeeds() {
        // Lowering produces `foo.0`, `grads.0`, `lib_double.0` — these
        // are not parser-produced but they are legitimate IR names.
        let n = LoadStoreName::new("foo.0").expect("tuple-flatten name valid");
        assert_eq!(n.as_str(), "foo.0");
    }

    #[test]
    fn nested_tuple_flatten_succeeds() {
        let n = LoadStoreName::new("grads.0.1").expect("nested tuple flatten valid");
        assert_eq!(n.as_str(), "grads.0.1");
    }

    #[test]
    fn black_scholes_identifiers_succeed() {
        // The Black-Scholes Phase 0 fixture uses these identifiers; lock
        // them as a positive corpus so a future grammar tightening can't
        // silently break the executable example surface.
        for name in ["s", "k", "r", "t", "d_1", "d_2", "e", "c"] {
            let n = LoadStoreName::new(name)
                .unwrap_or_else(|_| panic!("Black-Scholes name {name:?} must be valid"));
            assert_eq!(n.as_str(), name);
        }
    }

    // ------------------------------------------------------------------
    // Negative cases — every byte outside the grammar rejects.
    // ------------------------------------------------------------------

    #[test]
    fn empty_string_is_rejected() {
        assert_eq!(LoadStoreName::new(""), Err(LoadStoreNameError::Empty));
    }

    #[test]
    fn newline_in_continue_position_is_rejected() {
        // The original injection vector from the S4 re-red-team gate.
        let err = LoadStoreName::new("x\nINJECT")
            .expect_err("newline must reject so injection can't reach codegen");
        assert_eq!(
            err,
            LoadStoreNameError::InvalidCharacter {
                byte_offset: 1,
                character: '\n',
                code_point: 0x0A,
            }
        );
    }

    #[test]
    fn carriage_return_is_rejected() {
        let err = LoadStoreName::new("x\rINJECT").unwrap_err();
        assert_eq!(
            err,
            LoadStoreNameError::InvalidCharacter {
                byte_offset: 1,
                character: '\r',
                code_point: 0x0D,
            }
        );
    }

    #[test]
    fn nul_byte_is_rejected() {
        let err = LoadStoreName::new("x\0INJECT").unwrap_err();
        assert_eq!(
            err,
            LoadStoreNameError::InvalidCharacter {
                byte_offset: 1,
                character: '\0',
                code_point: 0x00,
            }
        );
    }

    #[test]
    fn tab_is_rejected() {
        let err = LoadStoreName::new("x\tINJECT").unwrap_err();
        assert_eq!(
            err,
            LoadStoreNameError::InvalidCharacter {
                byte_offset: 1,
                character: '\t',
                code_point: 0x09,
            }
        );
    }

    #[test]
    fn soh_control_byte_is_rejected() {
        let err = LoadStoreName::new("x\x01INJECT").unwrap_err();
        assert_eq!(
            err,
            LoadStoreNameError::InvalidCharacter {
                byte_offset: 1,
                character: '\x01',
                code_point: 0x01,
            }
        );
    }

    #[test]
    fn delete_byte_is_rejected() {
        let err = LoadStoreName::new("x\x7FINJECT").unwrap_err();
        assert_eq!(
            err,
            LoadStoreNameError::InvalidCharacter {
                byte_offset: 1,
                character: '\x7F',
                code_point: 0x7F,
            }
        );
    }

    #[test]
    fn space_is_rejected() {
        // U+0020 SPACE is allowed in spans (§1.1.1) but NOT in
        // identifier names. A space inside a Load/Store name would be
        // invalid in every emitted target language anyway.
        let err = LoadStoreName::new("foo bar").unwrap_err();
        assert_eq!(
            err,
            LoadStoreNameError::InvalidCharacter {
                byte_offset: 3,
                character: ' ',
                code_point: 0x20,
            }
        );
    }

    #[test]
    fn unicode_greek_is_rejected() {
        // The Deep parser identifier grammar is ASCII-only. Greek
        // letters and other non-ASCII characters do not pass through
        // the parser, so the newtype rejects them at construction.
        // Future producers wanting Unicode identifiers must extend the
        // parser grammar first; this test locks the current contract.
        let err = LoadStoreName::new("σ").unwrap_err();
        // 'σ' is U+03C3, encoded as 0xCF 0x83 in UTF-8; the disallowed
        // byte at offset 0 reports the *character*, not a half byte.
        assert_eq!(
            err,
            LoadStoreNameError::InvalidCharacter {
                byte_offset: 0,
                character: 'σ',
                code_point: 0x03C3,
            }
        );
    }

    #[test]
    fn unicode_after_ascii_prefix_is_rejected() {
        // Multibyte char in continue position. The diagnostic must
        // report the character offset (start of the multibyte sequence),
        // not a UTF-8 continuation-byte offset.
        let err = LoadStoreName::new("foo_σ").unwrap_err();
        assert_eq!(
            err,
            LoadStoreNameError::InvalidCharacter {
                byte_offset: 4,
                character: 'σ',
                code_point: 0x03C3,
            }
        );
    }

    #[test]
    fn leading_digit_is_rejected() {
        // Parser ident-start is `[A-Za-z_]` only — digits cannot start.
        let err = LoadStoreName::new("0foo").unwrap_err();
        assert_eq!(
            err,
            LoadStoreNameError::InvalidCharacter {
                byte_offset: 0,
                character: '0',
                code_point: 0x30,
            }
        );
    }

    #[test]
    fn leading_dash_is_rejected() {
        let err = LoadStoreName::new("-foo").unwrap_err();
        assert_eq!(
            err,
            LoadStoreNameError::InvalidCharacter {
                byte_offset: 0,
                character: '-',
                code_point: 0x2D,
            }
        );
    }

    #[test]
    fn leading_dot_is_rejected() {
        let err = LoadStoreName::new(".foo").unwrap_err();
        assert_eq!(
            err,
            LoadStoreNameError::InvalidCharacter {
                byte_offset: 0,
                character: '.',
                code_point: 0x2E,
            }
        );
    }

    #[test]
    fn percent_format_specifier_is_rejected() {
        // `%` is one of the format-string injection vectors flagged in
        // spec/upstream-bugs/producer-string-sanitization.md. It is not
        // a legal identifier byte in any emitted target language; the
        // newtype rejects it at construction.
        let err = LoadStoreName::new("foo%s").unwrap_err();
        assert_eq!(
            err,
            LoadStoreNameError::InvalidCharacter {
                byte_offset: 3,
                character: '%',
                code_point: 0x25,
            }
        );
    }

    #[test]
    fn double_slash_is_rejected() {
        // Comment-injection vector; `/` is not in the grammar.
        let err = LoadStoreName::new("foo//bar").unwrap_err();
        assert_eq!(
            err,
            LoadStoreNameError::InvalidCharacter {
                byte_offset: 3,
                character: '/',
                code_point: 0x2F,
            }
        );
    }

    // ------------------------------------------------------------------
    // Trait/serde locks.
    // ------------------------------------------------------------------

    #[test]
    fn serde_is_transparent_for_json() {
        // Wire compat: a LoadStoreName must serialize as a plain JSON
        // string, identical to the previous `String` field shape.
        let n = LoadStoreName::new("foo_bar").unwrap();
        let json = serde_json::to_string(&n).unwrap();
        assert_eq!(json, "\"foo_bar\"");
        let round: LoadStoreName = serde_json::from_str(&json).unwrap();
        assert_eq!(round, n);
    }

    #[test]
    fn serde_is_transparent_for_bincode() {
        // bincode is a positional encoder; a transparent newtype emits
        // the same bytes as its inner String. Lock that here so a
        // future #[derive] change doesn't silently shift the on-disk
        // shape (which would invalidate every cached compile artifact).
        let n = LoadStoreName::new("grads.0").unwrap();
        let raw = String::from("grads.0");
        let n_bytes = bincode::serialize(&n).unwrap();
        let raw_bytes = bincode::serialize(&raw).unwrap();
        assert_eq!(n_bytes, raw_bytes);
        let round: LoadStoreName = bincode::deserialize(&n_bytes).unwrap();
        assert_eq!(round, n);
    }

    #[test]
    fn deserialize_rejects_invalid_grammar() {
        // Serde transparency means we accept the raw String shape on the
        // wire. We do NOT re-validate on deserialize — the
        // construction-side trust boundary is the in-process boundary.
        // Cached/network data is assumed to come from a trusted
        // chelis-ir construction. Lock this so the contract is explicit.
        let bad = "\"x\\nINJECT\"";
        let n: LoadStoreName = serde_json::from_str(bad).expect(
            "deserialize is transparent and does NOT re-validate; \
             trust boundary is at construction, not deserialization",
        );
        // The struct holds the bad bytes — but the only way to get here
        // is via deserialize from already-trusted persisted IR.
        assert_eq!(n.as_str(), "x\nINJECT");
    }

    #[test]
    fn hash_and_eq_are_consistent_with_str() {
        use std::collections::HashMap;

        let n = LoadStoreName::new("foo").unwrap();

        // HashMap<String, _>::get takes an &str via Borrow<str>. Since
        // `LoadStoreName: Borrow<str>`, a HashMap<LoadStoreName, _> can
        // also be looked up by &str. Both directions matter at backend
        // call sites.
        let mut by_string: HashMap<String, usize> = HashMap::new();
        by_string.insert("foo".to_string(), 1);
        assert_eq!(by_string.get(n.as_str()), Some(&1));

        let mut by_name: HashMap<LoadStoreName, usize> = HashMap::new();
        by_name.insert(n.clone(), 2);
        assert_eq!(by_name.get("foo"), Some(&2));
    }

    #[test]
    fn display_matches_as_str() {
        let n = LoadStoreName::new("foo.0").unwrap();
        assert_eq!(format!("{n}"), "foo.0");
        assert_eq!(n.to_string(), "foo.0");
    }

    #[test]
    fn equality_with_str_and_string() {
        let n = LoadStoreName::new("foo").unwrap();
        assert_eq!(n, "foo");
        assert_eq!(n, *"foo");
        assert_eq!("foo", n);
        assert_eq!(n, String::from("foo"));
        assert_eq!(String::from("foo"), n);
        assert_ne!(n, "bar");
    }
}
