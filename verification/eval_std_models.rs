//! Models of the std/core items that the evaluator slice (chelis-ir eval.rs
//! and its callees) calls and that had no model: signed `checked_abs` and
//! `checked_neg`, `usize::is_multiple_of`, `RangeInclusive::contains`,
//! `<[T]>::split_first`, `Vec::append`, `BTreeSet::is_empty`, the forwarding
//! arithmetic impls on references, `Borrow<T> for &T`, `f32`/`f64` `recip`,
//! and the derived comparisons of `Result`, 4-tuples, `TryFromIntError` and
//! `Option`'s `Eq`.
use std::borrow::Borrow;
use std::collections::BTreeSet;
use std::ops::RangeInclusive;

/// `checked_abs` and `checked_neg` of every signed width: `None` exactly on
/// the minimum value.
pub fn signed_checks(a: i8, b: i16, c: i32, d: i64) -> (Option<i8>, Option<i16>, Option<i32>, Option<i64>, Option<i8>, Option<i16>, Option<i32>, Option<i64>) {
    (
        a.checked_abs(),
        b.checked_abs(),
        c.checked_abs(),
        d.checked_abs(),
        a.checked_neg(),
        b.checked_neg(),
        c.checked_neg(),
        d.checked_neg(),
    )
}

/// `usize::is_multiple_of`: `rhs == 0` means `self == 0`.
pub fn is_multiple(value: usize, rhs: usize) -> bool {
    value.is_multiple_of(rhs)
}

/// `RangeInclusive::contains` before and after the range is exhausted by
/// iteration.
pub fn range_contains(start: usize, end: usize, item: usize, consume: bool) -> bool {
    let mut range: RangeInclusive<usize> = start..=end;
    if consume {
        while range.next().is_some() {}
    }
    range.contains(&item)
}

/// `<[T]>::split_first`.
pub fn first_and_rest(values: &[u32]) -> Option<(u32, usize)> {
    values.split_first().map(|(first, rest)| (*first, rest.len()))
}

/// `Vec::append` moves the elements of `other`, leaving it empty.
pub fn append(mut left: Vec<u32>, mut right: Vec<u32>) -> (Vec<u32>, usize) {
    left.append(&mut right);
    (left, right.len())
}

/// `BTreeSet::is_empty`.
pub fn set_is_empty(values: &[u32]) -> bool {
    let set: BTreeSet<u32> = values.iter().copied().collect();
    set.is_empty()
}

/// The forwarding impls on references `Mul<usize> for &usize`,
/// `Sub<&usize> for usize` and `AddAssign<&f64> for f64` (`&a * b`, `a - &b`,
/// `x += &y`; floats as bits).
pub fn ref_arith(a: usize, b: usize, x: u64, y: u64) -> (usize, usize, u64) {
    let product = &a * b;
    let difference = a - &b;
    let mut sum = f64::from_bits(x);
    sum += &f64::from_bits(y);
    (product, difference, sum.to_bits())
}

/// `impl Borrow<T> for &T`.
pub fn borrow_ref(value: &u32) -> u32 {
    let r: &&u32 = &value;
    *<&u32 as Borrow<u32>>::borrow(r)
}

/// `f32::recip` and `f64::recip` (`1 / x`; floats as bits).
pub fn recips(x: u32, y: u64) -> (u32, u64) {
    (f32::from_bits(x).recip().to_bits(), f64::from_bits(y).recip().to_bits())
}

/// `PartialEq for Result<T, E>` (derived), with `Result`s of `TryFrom` whose
/// error `TryFromIntError` is compared by its derived `PartialEq`.
pub fn results_eq(a: i64, b: i64, c: Result<u8, u32>, d: Result<u8, u32>) -> (bool, bool) {
    (i8::try_from(a) == i8::try_from(b), c == d)
}

/// `PartialEq for (W, V, U, T)`.
pub fn quads_eq(a: (u8, bool, u32, i64), b: (u8, bool, u32, i64)) -> bool {
    a == b
}

/// `Eq for Option<T>`: `assert_receiver_is_total_eq` is the derive's
/// compile-time check, a no-op at run time.
pub fn option_total_eq(a: Option<u32>) -> Option<u32> {
    Eq::assert_receiver_is_total_eq(&a);
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed() {
        assert_eq!(signed_checks(-5, 7, -9, 11), (Some(5), Some(7), Some(9), Some(11), Some(5), Some(-7), Some(9), Some(-11)));
        assert_eq!(
            signed_checks(i8::MIN, i16::MIN, i32::MIN, i64::MIN),
            (None, None, None, None, None, None, None, None)
        );
        assert_eq!(
            signed_checks(i8::MAX, i16::MAX, i32::MAX, i64::MAX),
            (Some(i8::MAX), Some(i16::MAX), Some(i32::MAX), Some(i64::MAX), Some(-i8::MAX), Some(-i16::MAX), Some(-i32::MAX), Some(-i64::MAX))
        );
        assert_eq!(signed_checks(0, 0, 0, 0), (Some(0), Some(0), Some(0), Some(0), Some(0), Some(0), Some(0), Some(0)));
    }

    #[test]
    fn multiples_and_ranges() {
        assert!(is_multiple(12, 4));
        assert!(!is_multiple(12, 5));
        assert!(is_multiple(0, 0));
        assert!(!is_multiple(3, 0));
        assert!(is_multiple(0, 7));
        assert!(range_contains(2, 5, 5, false));
        assert!(range_contains(2, 5, 2, false));
        assert!(!range_contains(2, 5, 6, false));
        assert!(!range_contains(2, 5, 1, false));
        assert!(!range_contains(5, 2, 3, false));
        assert!(!range_contains(2, 5, 5, true));
        assert!(!range_contains(2, 5, 3, true));
        assert!(!range_contains(4, 4, 4, true));
    }

    #[test]
    fn collections() {
        assert_eq!(first_and_rest(&[3, 4, 5]), Some((3, 2)));
        assert_eq!(first_and_rest(&[9]), Some((9, 0)));
        assert_eq!(first_and_rest(&[]), None);
        assert_eq!(append(vec![1, 2], vec![3]), (vec![1, 2, 3], 0));
        assert_eq!(append(vec![], vec![]), (vec![], 0));
        assert!(set_is_empty(&[]));
        assert!(!set_is_empty(&[1, 1]));
    }

    #[test]
    fn derived_comparisons() {
        assert_eq!(results_eq(3, 3, Ok(1), Ok(1)), (true, true));
        assert_eq!(results_eq(300, -200, Ok(1), Err(1)), (true, false));
        assert_eq!(results_eq(3, 300, Err(2), Err(3)), (false, false));
        assert_eq!(results_eq(3, 4, Err(7), Err(7)), (false, true));
        assert!(quads_eq((1, true, 2, -3), (1, true, 2, -3)));
        assert!(!quads_eq((1, true, 2, -3), (1, false, 2, -3)));
        assert!(!quads_eq((1, true, 2, -3), (1, true, 2, 3)));
        assert_eq!(option_total_eq(Some(4)), Some(4));
        assert_eq!(option_total_eq(None), None);
    }

    #[test]
    fn references_and_floats() {
        assert_eq!(ref_arith(6, 4, 1.5f64.to_bits(), 2.25f64.to_bits()), (24, 2, 3.75f64.to_bits()));
        assert_eq!(ref_arith(6, 4, 0x3ff0000000000000, 0x3ca0000000000000), (24, 2, 0x3ff0000000000000));
        assert!(std::panic::catch_unwind(|| ref_arith(usize::MAX, 2, 0, 0)).is_err());
        assert!(std::panic::catch_unwind(|| ref_arith(1, 2, 0, 0)).is_err());
        assert_eq!(borrow_ref(&42), 42);
        assert_eq!(recips(4.0f32.to_bits(), 0.5f64.to_bits()), (0.25f32.to_bits(), 2.0f64.to_bits()));
        assert_eq!(recips(0, 0x8000000000000000), (f32::INFINITY.to_bits(), f64::NEG_INFINITY.to_bits()));
        assert_eq!(recips(3.0f32.to_bits(), 3.0f64.to_bits()), (0x3eaaaaab, 0x3fd5555555555555));
        assert_eq!(recips(f32::INFINITY.to_bits(), 0x7ff8000000000000).0, 0);
    }
}
