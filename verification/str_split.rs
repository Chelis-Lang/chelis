// `str::split` with a `&str` pattern (core str/mod.rs, str/iter.rs,
// str/pattern.rs): these native results pin the pieces between the leftmost
// non-overlapping matches, the empty needle's match at every char boundary
// (both ends included), the trailing empty piece, the default size_hint of
// `Split` and its fused end.

/// chelis-types opacity.rs demangle_lowercase_binding.
pub fn segments(stem: &str) -> Vec<&str> {
    stem.split("__").collect::<Vec<_>>()
}

pub fn pieces<'a>(text: &'a str, pattern: &str) -> Vec<&'a str> {
    text.split(pattern).collect()
}

pub fn first_two<'a>(text: &'a str, pattern: &str) -> (Option<&'a str>, Option<&'a str>) {
    let mut iter = text.split(pattern);
    let first = iter.next();
    (first, iter.next())
}

/// The number of pieces, then two more calls after the end.
pub fn fused<'a>(text: &'a str, pattern: &str) -> (usize, Option<&'a str>, Option<&'a str>) {
    let mut iter = text.split(pattern);
    let mut count = 0;
    while iter.next().is_some() {
        count += 1;
    }
    (count, iter.next(), iter.next())
}

pub fn hint(text: &str, pattern: &str) -> (usize, Option<usize>) {
    text.split(pattern).size_hint()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_shape() {
        assert_eq!(segments("Pkg__Mod__name__x"), vec!["Pkg", "Mod", "name", "x"]);
        assert_eq!(segments("__Entry__run"), vec!["", "Entry", "run"]);
        assert_eq!(segments("plain"), vec!["plain"]);
        assert_eq!(segments("a____b"), vec!["a", "", "b"]);
    }

    #[test]
    fn pieces_between_matches() {
        assert_eq!(pieces("", "__"), vec![""]);
        assert_eq!(pieces("aaa", "aa"), vec!["", "a"]);
        assert_eq!(pieces("aaaa", "aa"), vec!["", "", ""]);
        assert_eq!(pieces("__a__b__", "__"), vec!["", "a", "b", ""]);
        assert_eq!(pieces("ab", "abc"), vec!["ab"]);
        assert_eq!(pieces("ab", "ab"), vec!["", ""]);
        assert_eq!(pieces("héllo wörld", "ö"), vec!["héllo w", "rld"]);
        assert_eq!(pieces("x→y→z", "→"), vec!["x", "y", "z"]);
    }

    #[test]
    fn empty_needle() {
        assert_eq!(pieces("", ""), vec!["", ""]);
        assert_eq!(pieces("abc", ""), vec!["", "a", "b", "c", ""]);
        assert_eq!(pieces("aé→", ""), vec!["", "a", "é", "→", ""]);
    }

    #[test]
    fn iterator() {
        assert_eq!(first_two("a,b", ","), (Some("a"), Some("b")));
        assert_eq!(first_two("", ","), (Some(""), None));
        assert_eq!(fused("a,b,,c", ","), (4, None, None));
        assert_eq!(fused("", ""), (2, None, None));
        assert_eq!(hint("a__b", "__"), (0, None));
    }
}
