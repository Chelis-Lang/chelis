// `char` patterns (`CharSearcher`, core str/pattern.rs) and `str::rsplit_once`
// with `&str` and `char` delimiters (core str/mod.rs): these native results pin
// the pieces between the occurrences of a char, the first match of the reverse
// searcher (the rightmost non-overlapping occurrence of a `&str` needle, the end
// for an empty needle, the last occurrence of a char) and the char prefix test.

/// chelis-types opacity.rs terminal_segment.
pub fn terminal_segment(name: &str) -> &str {
    name.rsplit_once("__")
        .map(|(_, tail)| tail)
        .or_else(|| name.rsplit_once('.').map(|(_, tail)| tail))
        .unwrap_or(name)
}

/// chelis-types opacity.rs reef_module_stem: the stem before the last `__`.
pub fn module_part(stem: &str) -> Option<&str> {
    let (module_part, _terminal) = stem.rsplit_once("__")?;
    Some(module_part)
}

/// chelis-types opacity.rs demangle_module: `key.split('.').collect()`.
pub fn dot_segments(key: &str) -> Vec<&str> {
    key.split('.').collect()
}

pub fn char_pieces(text: &str, pattern: char) -> Vec<&str> {
    text.split(pattern).collect()
}

pub fn rsplit_str<'a>(text: &'a str, pattern: &str) -> Option<(&'a str, &'a str)> {
    text.rsplit_once(pattern)
}

pub fn rsplit_char(text: &str, pattern: char) -> Option<(&str, &str)> {
    text.rsplit_once(pattern)
}

pub fn starts_with_char(text: &str, pattern: char) -> bool {
    text.starts_with(pattern)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_shape() {
        assert_eq!(terminal_segment("Pkg__opq__Demo__Types__Probability"), "Probability");
        assert_eq!(terminal_segment("opq.Demo.Types"), "Types");
        assert_eq!(terminal_segment("plain"), "plain");
        assert_eq!(terminal_segment("a__b.c"), "b.c");
        assert_eq!(module_part("opq__Demo__Types__Probability"), Some("opq__Demo__Types"));
        assert_eq!(module_part("__x"), Some(""));
        assert_eq!(module_part("x"), None);
        assert_eq!(dot_segments("opq.Demo.Types"), vec!["opq", "Demo", "Types"]);
        assert_eq!(dot_segments(""), vec![""]);
        assert_eq!(dot_segments(".a."), vec!["", "a", ""]);
    }

    #[test]
    fn char_occurrences() {
        assert_eq!(char_pieces("héllo wörld", 'ö'), vec!["héllo w", "rld"]);
        assert_eq!(char_pieces("aéaé", 'é'), vec!["a", "a", ""]);
        assert_eq!(char_pieces("abc", 'x'), vec!["abc"]);
        assert_eq!(char_pieces("", 'a'), vec![""]);
        assert_eq!(char_pieces("a→b", '→'), vec!["a", "b"]);
        assert_eq!(char_pieces("𝄞x𝄞", '𝄞'), vec!["", "x", ""]);
    }

    #[test]
    fn last_str_match() {
        assert_eq!(rsplit_str("a__b__c", "__"), Some(("a__b", "c")));
        assert_eq!(rsplit_str("aaa", "aa"), Some(("a", "")));
        assert_eq!(rsplit_str("aaaa", "aa"), Some(("aa", "")));
        assert_eq!(rsplit_str("a____b", "__"), Some(("a__", "b")));
        assert_eq!(rsplit_str("abc", ""), Some(("abc", "")));
        assert_eq!(rsplit_str("", ""), Some(("", "")));
        assert_eq!(rsplit_str("abc", "x"), None);
        assert_eq!(rsplit_str("ab", "abc"), None);
        assert_eq!(rsplit_str("x→y→z", "→"), Some(("x→y", "z")));
    }

    #[test]
    fn last_char_match() {
        assert_eq!(rsplit_char("a.b.c", '.'), Some(("a.b", "c")));
        assert_eq!(rsplit_char("xéyéz", 'é'), Some(("xéy", "z")));
        assert_eq!(rsplit_char("abc", '.'), None);
        assert_eq!(rsplit_char("", 'a'), None);
        assert_eq!(rsplit_char("𝄞a𝄞", '𝄞'), Some(("𝄞a", "")));
    }

    #[test]
    fn char_prefix() {
        assert!(starts_with_char("é!", 'é'));
        assert!(!starts_with_char("", 'a'));
        assert!(!starts_with_char("ab", 'b'));
        assert!(starts_with_char("𝄞", '𝄞'));
    }
}
