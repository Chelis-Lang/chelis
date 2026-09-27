//! `str::rfind`, `str::ends_with`, `<[T]>::split_last` and
//! `core::slice::from_ref` of the production crates (Aeneas.Std
//! StringIter.lean, Slice.lean), with `char` and `&str` patterns.

/// As chelis-types `deep_type::span_offset_from_id`: the start offset of a
/// span id, after its last `:`.
pub fn span_offset_from_id(span_id: &str) -> Option<usize> {
    span_id
        .rfind(':')
        .map(|index| &span_id[index + 1..])
        .unwrap_or(span_id)
        .split_once("..")
        .and_then(|(start, _)| start.parse::<usize>().ok())
}

/// `str::rfind` with a `&str` pattern: the start of the last `::`.
pub fn last_separator(path: &str) -> Option<usize> {
    path.rfind("::")
}

/// As chelis-types `observation::shortest_repr`: the digits without their
/// trailing zeros (`str::ends_with` of a `char`, through `String`'s `Deref`).
pub fn strip_zeros(text: &str) -> String {
    let mut digits = text.to_string();
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    digits
}

/// `str::ends_with` with a `&str` pattern.
pub fn has_suffix(text: &str, suffix: &str) -> bool {
    text.ends_with(suffix)
}

/// As chelis-types `validate::fn_parts`: the last child is the return type,
/// the others are the arguments.
pub fn fn_parts(kids: &[u32]) -> Option<(u32, Vec<u32>)> {
    let (ret, args) = kids.split_last()?;
    Some((*ret, args.to_vec()))
}

/// As chelis-types `linearity`: one value passed as a slice.
pub fn as_slice(x: &u32) -> (usize, u32) {
    let s = std::slice::from_ref(x);
    (s.len(), s[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfind() {
        assert_eq!(span_offset_from_id("surf:a:12..17"), Some(12));
        assert_eq!(span_offset_from_id("12..17"), Some(12));
        assert_eq!(span_offset_from_id("é:3..4"), Some(3));
        assert_eq!(span_offset_from_id("a:b..4"), None);
        assert_eq!(span_offset_from_id("a:12"), None);
        assert_eq!(last_separator("a::b::c"), Some(4));
        assert_eq!(last_separator("a:::b"), Some(2));
        assert_eq!(last_separator("ab"), None);
        assert_eq!(last_separator(""), None);
    }

    #[test]
    fn ends_with() {
        assert_eq!(strip_zeros("12500"), "125");
        assert_eq!(strip_zeros("000"), "0");
        assert_eq!(strip_zeros("7"), "7");
        assert_eq!(strip_zeros("é0"), "é");
        assert!(has_suffix("abc", "bc"));
        assert!(has_suffix("abc", ""));
        assert!(has_suffix("", ""));
        assert!(!has_suffix("abc", "abcd"));
        assert!(!has_suffix("abc", "ab"));
    }

    #[test]
    fn split_last_and_from_ref() {
        assert_eq!(fn_parts(&[1, 2, 3]), Some((3, vec![1, 2])));
        assert_eq!(fn_parts(&[4]), Some((4, vec![])));
        assert_eq!(fn_parts(&[]), None);
        assert_eq!(as_slice(&9), (1, 9));
    }
}
