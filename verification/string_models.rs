use std::cmp::Ordering;

pub fn literal_text() -> &'static str {
    "§\0\t\n\r\"\\\x1b\x7f\u{10000}"
}

pub fn extent_source_diagnostic() -> &'static str {
    "give the axis a real extent source: size the consuming operation's output from the operand that declares the extent, or declare the extent on the producing node"
}

pub fn deferred_dtype_diagnostic() -> &'static str {
    "reduce_sum: operand dtype `f8e4m3` is deferred per spec/04-type-system.md §1.1.1 and is not part of the active numeric primitive set"
}

pub fn compare_owned(left: &str, right: &str) -> (bool, bool, bool, bool) {
    let owned = String::from(left);
    let other = String::from(right);
    (
        owned.eq(&other),
        owned.ne(&other),
        owned.eq(right),
        owned.ne(right),
    )
}

pub fn compare_order(left: &str, right: &str) -> (Ordering, Ordering) {
    (
        String::from(left).cmp(&String::from(right)),
        left.cmp(right),
    )
}

pub fn is_anonymous(text: &str) -> bool {
    text.is_empty() || text == "*"
}

pub fn cloned_view(text: &str) -> (String, bool, bool) {
    let owned = String::from(text);
    let copied = owned.clone();
    let empty_view = copied.as_str().is_empty();
    let empty_owned = copied.is_empty();
    (copied, empty_view, empty_owned)
}

pub fn dereferenced_owned(text: &str) -> String {
    let owned = text.to_owned();
    String::from(&*owned)
}

pub fn boxed_view(text: &str) -> Box<str> {
    Box::<str>::from(String::from(text))
}

pub fn joined(left: &str, right: &str, separator: &str) -> String {
    [String::from(left), String::from(right)].join(separator)
}

pub fn empty_join(separator: &str) -> String {
    let values: [String; 0] = [];
    values.join(separator)
}

pub fn singleton_join(text: &str, separator: &str) -> String {
    [String::from(text)].join(separator)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_text_preserves_unicode_and_control_bytes() {
        assert_eq!(
            literal_text().as_bytes(),
            [0xc2, 0xa7, 0, 9, 10, 13, 34, 92, 27, 127, 0xf0, 0x90, 0x80, 0x80]
        );
    }

    #[test]
    fn long_production_diagnostics_preserve_text_and_byte_lengths() {
        assert_eq!(
            extent_source_diagnostic(),
            "give the axis a real extent source: size the consuming operation's output from the operand that declares the extent, or declare the extent on the producing node"
        );
        assert_eq!(extent_source_diagnostic().len(), 160);
        assert_eq!(
            deferred_dtype_diagnostic(),
            "reduce_sum: operand dtype `f8e4m3` is deferred per spec/04-type-system.md §1.1.1 and is not part of the active numeric primitive set"
        );
        assert_eq!(deferred_dtype_diagnostic().len(), 133);
    }

    #[test]
    fn borrowed_equality_in_short_circuit_preserves_anonymous_names() {
        assert!(is_anonymous(""));
        assert!(is_anonymous("*"));
        assert!(!is_anonymous("axis"));
        assert!(!is_anonymous("*\0"));
    }

    #[test]
    fn equality_uses_exact_utf8_bytes_and_lengths() {
        assert_eq!(
            compare_owned("é\0\u{10000}", "é\0\u{10000}"),
            (true, false, true, false)
        );
        assert_eq!(compare_owned("é", "e\u{301}"), (false, true, false, true));
        assert_eq!(compare_owned("axis", "axes"), (false, true, false, true));
        assert_eq!(compare_owned("a", "ab"), (false, true, false, true));
        assert_eq!(compare_owned("", ""), (true, false, true, false));
    }

    #[test]
    fn ordering_and_ownership_preserve_text() {
        assert_eq!(
            compare_order("\u{e000}", "\u{10000}"),
            (Ordering::Less, Ordering::Less)
        );
        assert_eq!(
            compare_order("prefix", "prefix!"),
            (Ordering::Less, Ordering::Less)
        );
        assert_eq!(
            cloned_view("é\0\u{10000}"),
            (String::from("é\0\u{10000}"), false, false)
        );
        assert_eq!(cloned_view(""), (String::from(""), true, true));
        assert_eq!(dereferenced_owned("λ"), "λ");
        assert_eq!(&*boxed_view("é\0"), "é\0");
    }

    #[test]
    fn joining_handles_empty_parts_and_utf8_separators() {
        assert_eq!(joined("", "β", "λ"), "λβ");
        assert_eq!(joined("a\0", "b", ""), "a\0b");
        assert_eq!(empty_join("ignored"), "");
        assert_eq!(singleton_join("β", "ignored"), "β");
    }
}
