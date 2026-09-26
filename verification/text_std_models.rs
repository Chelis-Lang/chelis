//! Core/alloc items of the production metadata checks (chelis-deep metadata,
//! annotations, extension_data): `Option::iter` as a collection source, the
//! consumers of `str::bytes`, `str::to_ascii_lowercase` and `String == &str`.

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
}
