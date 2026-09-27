//! Core/alloc items of the production metadata checks (chelis-deep metadata,
//! annotations, extension_data): `Option::iter` as a collection source, the
//! consumers of `str::bytes`, `str::to_ascii_lowercase` and `String == &str`;
//! `String::new`, `Default for String`, `String::pop`, `char::len_utf8` and the
//! ASCII classes of `char` and `u8`.

pub struct Storage {
    pub values: Vec<u32>,
}

pub struct Metadata {
    pub storage: Option<Box<Storage>>,
    pub extensions: Vec<u32>,
}

/// As `Metadata::values` + `metadata::entries`: a flat_map over the optional
/// storage, chained with the extensions and collected.
pub fn entries(meta: &Metadata) -> Vec<u32> {
    meta.storage
        .iter()
        .flat_map(|s| s.values.iter())
        .map(|v| *v + 1)
        .chain(meta.extensions.iter().copied())
        .collect()
}

pub fn option_iter(value: Option<u32>) -> (Option<u32>, Option<u32>, usize) {
    let (lower, _) = value.iter().size_hint();
    let mut it = value.iter();
    let first = it.next().copied();
    let second = it.next().copied();
    (first, second, lower)
}

/// As `metadata::check_entries`: the first control byte and its offset.
pub fn forbidden(s: &str) -> Option<(usize, u8)> {
    s.bytes().enumerate().find(|(_, b)| *b <= 0x1f || *b == 0x7f)
}

/// As `annotations::MetadataKey` / `extension_data` name checks.
pub fn byte_checks(s: &str) -> (bool, bool, Option<u8>, usize) {
    let lower = s.bytes().all(|b| (b >= b'a' && b <= b'z') || b == b'_');
    let digit = s.bytes().any(|b| b >= b'0' && b <= b'9');
    let dot = s.bytes().find(|b| *b == b'.');
    (lower, digit, dot, s.bytes().count())
}

pub fn first_two(s: &str) -> (Option<u8>, Option<u8>) {
    let mut bytes = s.bytes();
    let first = bytes.next();
    (first, bytes.next())
}

/// As the `P::Path` placement check.
pub fn same_path(path: &str, name: &str) -> bool {
    path.to_ascii_lowercase() == name
}

pub fn lowercase(path: &str) -> String {
    path.to_ascii_lowercase()
}

/// As the production `LowerDiagnostic`: a derived `Clone` over `Box<str>`.
#[derive(Clone)]
pub struct Diagnostic {
    pub code: Box<str>,
    pub line: u32,
}

pub fn clone_diagnostic(diagnostic: &Diagnostic) -> Diagnostic {
    diagnostic.clone()
}

/// `String::new`, then pushes (chelis-deep printer.rs).
pub fn built(chars: &[char]) -> String {
    let mut out = String::new();
    for &c in chars {
        out.push(c);
    }
    out
}

/// `Default for String`.
pub fn defaulted() -> (String, usize) {
    let text: String = Default::default();
    let len = text.len();
    (text, len)
}

/// `String::pop` on a borrowed string: the last char and the given-back string.
pub fn pop_last(text: &mut String) -> Option<char> {
    text.pop()
}

/// `char::len_utf8`.
pub fn utf8_width(c: char) -> usize {
    c.len_utf8()
}

/// `char::is_ascii_digit` and `char::is_ascii_alphanumeric`.
pub fn char_classes(c: char) -> (bool, bool) {
    (c.is_ascii_digit(), c.is_ascii_alphanumeric())
}

/// `u8::is_ascii_digit`, `u8::is_ascii_alphabetic` and `u8::is_ascii_alphanumeric`.
pub fn byte_classes(b: u8) -> (bool, bool, bool) {
    (b.is_ascii_digit(), b.is_ascii_alphabetic(), b.is_ascii_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_storage_is_a_collection_source() {
        let full = Metadata { storage: Some(Box::new(Storage { values: vec![1, 2] })), extensions: vec![7] };
        assert_eq!(entries(&full), vec![2, 3, 7]);
        let empty = Metadata { storage: None, extensions: vec![7, 8] };
        assert_eq!(entries(&empty), vec![7, 8]);
        assert_eq!(option_iter(Some(5)), (Some(5), None, 1));
        assert_eq!(option_iter(None), (None, None, 0));
    }

    #[test]
    fn byte_consumers_follow_the_utf8_bytes() {
        assert_eq!(forbidden("ab\tc"), Some((2, 9)));
        assert_eq!(forbidden("abc\u{7f}"), Some((3, 0x7f)));
        assert_eq!(forbidden("é ok"), None);
        assert_eq!(byte_checks("snake_case"), (true, false, None, 10));
        assert_eq!(byte_checks("v1.2"), (false, true, Some(b'.'), 4));
        assert_eq!(byte_checks(""), (true, false, None, 0));
        assert_eq!(byte_checks("é"), (false, false, None, 2));
        assert_eq!(first_two("é"), (Some(0xc3), Some(0xa9)));
        assert_eq!(first_two("x"), (Some(b'x'), None));
    }

    #[test]
    fn ascii_lowercase_keeps_other_bytes() {
        assert_eq!(lowercase("Std::IO_x9"), "std::io_x9");
        assert_eq!(lowercase("ÉCOLE"), "École");
        assert!(same_path("Core::Mem", "core::mem"));
        assert!(!same_path("Core::Mem", "core::me"));
    }

    #[test]
    fn boxed_str_clone_keeps_the_contents() {
        let diagnostic = Diagnostic { code: "E0042 é".into(), line: 3 };
        let copy = clone_diagnostic(&diagnostic);
        assert_eq!((&*copy.code, copy.line), ("E0042 é", 3));
    }

    #[test]
    fn strings_and_char_classes() {
        assert_eq!(built(&['a', 'é', 'z']), "aéz");
        assert_eq!(built(&[]), "");
        assert_eq!(defaulted(), (String::new(), 0));
        let mut text = "abé".to_string();
        assert_eq!(pop_last(&mut text), Some('é'));
        assert_eq!(text, "ab");
        let mut empty = String::new();
        assert_eq!(pop_last(&mut empty), None);
        assert_eq!(empty, "");
        let widths: Vec<usize> = ['a', '\u{7f}', '\u{80}', 'é', '\u{7ff}', '\u{800}', '€', '\u{ffff}', '\u{10000}', '😀']
            .iter()
            .map(|&c| utf8_width(c))
            .collect();
        assert_eq!(widths, vec![1, 1, 2, 2, 2, 3, 3, 3, 4, 4]);
        let classes: Vec<(bool, bool)> =
            ['0', '9', 'a', 'z', 'A', 'Z', '/', ':', '@', '[', '`', '{', 'é'].iter().map(|&c| char_classes(c)).collect();
        assert_eq!(
            classes,
            vec![
                (true, true), (true, true), (false, true), (false, true), (false, true), (false, true),
                (false, false), (false, false), (false, false), (false, false), (false, false), (false, false),
                (false, false),
            ]
        );
        let bytes: Vec<(bool, bool, bool)> =
            [b'0', b'9', b'a', b'z', b'A', b'Z', b'/', b':', b'@', b'[', b'`', b'{', 0xc3].iter().map(|&b| byte_classes(b)).collect();
        assert_eq!(
            bytes,
            vec![
                (true, false, true), (true, false, true), (false, true, true), (false, true, true),
                (false, true, true), (false, true, true), (false, false, false), (false, false, false),
                (false, false, false), (false, false, false), (false, false, false), (false, false, false),
                (false, false, false),
            ]
        );
    }
}
