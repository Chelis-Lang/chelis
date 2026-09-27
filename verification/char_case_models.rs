//! `char::is_uppercase` and `char::is_lowercase` (Aeneas.Std Core/CharCase.lean), as the
//! production crates call them: chelis-ir `lower`/`host` and chelis-effects `realizability`
//! test whether a name starts with an uppercase (a constructor) or lowercase letter. The model
//! is the set of chars for which the native method is true; `case_digest` is the digest that
//! the Lean checks compute from the model's tables.

/// As chelis-ir `host::is_static_constructor`: the first char is uppercase.
pub fn starts_upper(name: &str) -> bool {
    name.chars().next().is_some_and(char::is_uppercase)
}

/// As chelis-ir `host`: the first char is lowercase.
pub fn starts_lower(name: &str) -> bool {
    name.chars().next().is_some_and(char::is_lowercase)
}

/// As chelis-effects `realizability::tagged_needs_host`, through a closure.
pub fn starts_upper_closure(name: &str) -> bool {
    name.chars().next().is_some_and(|c| c.is_uppercase())
}

/// Both properties of one char.
pub fn classify(c: char) -> (bool, bool) {
    (c.is_uppercase(), c.is_lowercase())
}

/// Counts uppercase and lowercase chars into caller-owned counters.
pub fn tally(name: &str, counts: &mut (u32, u32)) {
    for c in name.chars() {
        if c.is_uppercase() {
            counts.0 += 1;
        }
        if c.is_lowercase() {
            counts.1 += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The number of chars with the property and the digest `h = h * 1000003 + c + 1`
    /// (wrapping, u64) of their code points in increasing order.
    fn case_digest(p: fn(char) -> bool) -> (u32, u64) {
        let mut count = 0u32;
        let mut h = 0u64;
        for c in (0u32..=0x10FFFF).filter_map(char::from_u32) {
            if p(c) {
                count += 1;
                h = h.wrapping_mul(1000003).wrapping_add(c as u64 + 1);
            }
        }
        (count, h)
    }

    #[test]
    fn digests() {
        assert_eq!(case_digest(char::is_uppercase), (2006, 3862418619914882295));
        assert_eq!(case_digest(char::is_lowercase), (2595, 12857385794233412760));
        assert_eq!(char::UNICODE_VERSION, (17, 0, 0));
    }

    #[test]
    fn names() {
        assert!(starts_upper("Some"));
        assert!(!starts_upper("some"));
        assert!(!starts_upper(""));
        assert!(!starts_upper("_X"));
        assert!(starts_upper("Ωmega"));
        assert!(!starts_upper("ǅ"));
        assert!(starts_lower("x1"));
        assert!(!starts_lower("X1"));
        assert!(!starts_lower(""));
        assert!(starts_lower("ßx"));
        assert!(starts_lower("ª"));
        assert!(starts_upper_closure("Tensor"));
        assert!(!starts_upper_closure("tensor"));
        assert!(starts_upper_closure("\u{1D400}"));
    }

    #[test]
    fn chars() {
        let cases = [
            ('A', (true, false)), ('Z', (true, false)), ('@', (false, false)), ('[', (false, false)),
            ('a', (false, true)), ('z', (false, true)), ('`', (false, false)), ('{', (false, false)),
            ('0', (false, false)), (' ', (false, false)), ('\u{7f}', (false, false)),
            ('\u{AA}', (false, true)), ('\u{C0}', (true, false)), ('\u{D7}', (false, false)),
            ('\u{DE}', (true, false)), ('\u{DF}', (false, true)), ('\u{F7}', (false, false)),
            ('\u{FF}', (false, true)), ('\u{1C5}', (false, false)), ('\u{3A3}', (true, false)),
            ('\u{3C2}', (false, true)), ('\u{2160}', (true, false)), ('\u{2170}', (false, true)),
            ('\u{1D400}', (true, false)), ('\u{1E900}', (true, false)), ('\u{1E922}', (false, true)),
            ('\u{10FFFF}', (false, false)),
        ];
        for (c, expected) in cases {
            assert_eq!(classify(c), expected, "{c:?}");
        }
    }

    #[test]
    fn tallies() {
        let mut counts = (1, 2);
        tally("AbΣσǅ9", &mut counts);
        assert_eq!(counts, (3, 4));
        let mut counts = (0, 0);
        tally("", &mut counts);
        assert_eq!(counts, (0, 0));
    }
}
