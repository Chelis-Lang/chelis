//! Models of core's primitive orderings, as the evaluator kernel uses them:
//! `impl Ord for bool` (its `cmp` reads the `ub_checks` build configuration on
//! the `unreachable_unchecked` arm, and its `min`/`max` are `&`/`|`), the
//! non-short-circuiting logical operators of `bool`, and the ordering of
//! `char` (MIR comparisons, range patterns and `Ord`/`PartialOrd`).

use std::cmp::Ordering;

/// `impl Ord for bool`'s `cmp`: `(*self as i8) - (*other as i8)`, whose
/// impossible arm is `unreachable_unchecked` guarded by `ub_checks`.
pub fn bool_cmp(left: bool, right: bool) -> Ordering {
    left.cmp(&right)
}

/// `impl PartialOrd for bool`'s `partial_cmp` delegates to `cmp`.
pub fn bool_partial_cmp(left: bool, right: bool) -> Option<Ordering> {
    left.partial_cmp(&right)
}

/// `impl Ord for bool`'s `min` (`self & other`) and `max` (`self | other`).
pub fn bool_min_max(left: bool, right: bool) -> (bool, bool) {
    (left.min(right), left.max(right))
}

/// `impl Ord for bool`'s `clamp` asserts `min <= max`.
pub fn bool_clamp(value: bool, low: bool, high: bool) -> bool {
    value.clamp(low, high)
}

/// The `PartialOrd for bool` methods, called through the trait.
pub fn bool_trait_order(left: &bool, right: &bool) -> (bool, bool, bool, bool) {
    (
        PartialOrd::lt(left, right),
        PartialOrd::le(left, right),
        PartialOrd::gt(left, right),
        PartialOrd::ge(left, right),
    )
}

/// The non-short-circuiting logical operators: both operands are evaluated.
pub fn bool_bits(left: bool, right: bool) -> (bool, bool, bool) {
    (left & right, left | right, left ^ right)
}

/// Both operands of `&` are evaluated, so an overflow on the right still panics.
pub fn eager_and(flag: bool, count: u8) -> bool {
    flag & (count + 1 > 1)
}

/// The primitive ordered comparisons of `char` (scalar-value order).
pub fn char_order(left: char, right: char) -> (bool, bool, bool, bool) {
    (left < right, left <= right, left > right, left >= right)
}

/// `impl Ord for char` and `impl PartialOrd for char`.
pub fn char_cmp(left: char, right: char) -> (Ordering, Option<Ordering>) {
    (left.cmp(&right), left.partial_cmp(&right))
}

/// A lexer-style range pattern, compiled to `char` comparisons.
pub fn is_ident_start(c: char) -> bool {
    matches!(c, 'a'..='z' | 'A'..='Z' | '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bool_order_is_false_before_true() {
        assert_eq!(bool_cmp(false, true), Ordering::Less);
        assert_eq!(bool_cmp(true, false), Ordering::Greater);
        assert_eq!(bool_cmp(true, true), Ordering::Equal);
        assert_eq!(bool_cmp(false, false), Ordering::Equal);
        assert_eq!(bool_partial_cmp(false, true), Some(Ordering::Less));
        assert_eq!(bool_partial_cmp(true, true), Some(Ordering::Equal));
        assert_eq!(bool_min_max(false, true), (false, true));
        assert_eq!(bool_min_max(true, true), (true, true));
        assert_eq!(bool_min_max(false, false), (false, false));
        assert_eq!(bool_clamp(false, true, true), true);
        assert_eq!(bool_clamp(true, false, false), false);
        assert_eq!(bool_clamp(true, false, true), true);
        assert!(std::panic::catch_unwind(|| bool_clamp(false, true, false)).is_err());
        assert_eq!(bool_trait_order(&false, &true), (true, true, false, false));
        assert_eq!(bool_trait_order(&true, &false), (false, false, true, true));
        assert_eq!(bool_trait_order(&true, &true), (false, true, false, true));
    }

    #[test]
    fn bool_bit_operators_evaluate_both_operands() {
        assert_eq!(bool_bits(false, false), (false, false, false));
        assert_eq!(bool_bits(false, true), (false, true, true));
        assert_eq!(bool_bits(true, false), (false, true, true));
        assert_eq!(bool_bits(true, true), (true, true, false));
        assert_eq!(eager_and(true, 1), true);
        assert_eq!(eager_and(false, 1), false);
        assert!(std::panic::catch_unwind(|| eager_and(false, u8::MAX)).is_err());
    }

    #[test]
    fn char_order_is_scalar_value_order() {
        assert_eq!(char_order('a', 'b'), (true, true, false, false));
        assert_eq!(char_order('z', 'a'), (false, false, true, true));
        assert_eq!(char_order('q', 'q'), (false, true, false, true));
        assert_eq!(char_order('Z', 'a'), (true, true, false, false));
        assert_eq!(char_order('\u{10FFFF}', '\u{E000}'), (false, false, true, true));
        assert_eq!(char_cmp('a', 'b'), (Ordering::Less, Some(Ordering::Less)));
        assert_eq!(char_cmp('\u{E9}', 'e'), (Ordering::Greater, Some(Ordering::Greater)));
        assert_eq!(char_cmp('x', 'x'), (Ordering::Equal, Some(Ordering::Equal)));
        assert!(is_ident_start('a'));
        assert!(is_ident_start('Z'));
        assert!(is_ident_start('_'));
        assert!(!is_ident_start('0'));
        assert!(!is_ident_start('{'));
        assert!(!is_ident_start('\u{E9}'));
    }
}
