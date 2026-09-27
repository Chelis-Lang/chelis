//! Integer methods of the production crates (Aeneas.Std Scalar): `u64::pow`
//! (checked, as in the production crates), `usize::saturating_mul`,
//! `u64::trailing_zeros` and `i64::checked_div_euclid` / `checked_rem_euclid`.

/// As chelis-types `observation::shortest_repr`: the power of ten of a digit
/// count, and the largest mantissa of that many digits.
pub fn decimal_bounds(p: u32) -> (u64, u64) {
    let p10 = 10u64.pow(p - 1);
    (p10, 10u64.pow(p) - 1)
}

/// `u64::pow` of any base.
pub fn power(base: u64, exp: u32) -> u64 {
    base.pow(exp)
}

/// As chelis-ir `analysis::copy_bytes`: the element count of the concrete
/// dimensions and the byte size, saturating.
pub fn copy_bytes(sizes: &[usize], element_size: usize) -> (usize, usize) {
    let mut concrete_elements = 1usize;
    for size in sizes {
        concrete_elements = concrete_elements.saturating_mul(*size);
    }
    (concrete_elements, concrete_elements.saturating_mul(element_size))
}

/// As chelis-types `dtype_semantics`: a magnitude is exact at a precision
/// when its low bits beyond the precision are zero.
pub fn exact_at(magnitude: u64, significant_bits: u32, precision: u32) -> bool {
    magnitude == 0 || magnitude.trailing_zeros() >= significant_bits.saturating_sub(precision)
}

/// `u64::trailing_zeros`.
pub fn trailing(x: u64) -> u32 {
    x.trailing_zeros()
}

/// As chelis-types `static_int::walk`: the Euclidean quotient and remainder
/// of static integers.
pub fn euclid(lhs: i64, rhs: i64) -> (Option<i64>, Option<i64>) {
    (lhs.checked_div_euclid(rhs), lhs.checked_rem_euclid(rhs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn panics<T>(f: impl FnOnce() -> T + std::panic::UnwindSafe) -> bool {
        std::panic::catch_unwind(f).is_err()
    }

    #[test]
    fn pow_is_checked() {
        assert_eq!(decimal_bounds(1), (1, 9));
        assert_eq!(decimal_bounds(19), (1_000_000_000_000_000_000, 9_999_999_999_999_999_999));
        assert!(panics(|| decimal_bounds(20)));
        assert!(panics(|| decimal_bounds(0)));
        assert_eq!(power(2, 63), 1 << 63);
        assert!(panics(|| power(2, 64)));
        assert_eq!(power(3, 40), 12_157_665_459_056_928_801);
        assert!(panics(|| power(3, 41)));
        assert_eq!(power(0, 0), 1);
        assert_eq!(power(0, u32::MAX), 0);
        assert_eq!(power(1, u32::MAX), 1);
        assert!(panics(|| power(2, u32::MAX)));
        assert_eq!(power(u64::MAX, 1), u64::MAX);
    }

    #[test]
    fn saturating_mul() {
        assert_eq!(copy_bytes(&[2, 3, 4], 8), (24, 192));
        assert_eq!(copy_bytes(&[], 8), (1, 8));
        assert_eq!(copy_bytes(&[1 << 32, 1 << 32], 1), (usize::MAX, usize::MAX));
        assert_eq!(copy_bytes(&[1 << 32, 1 << 31], 2), ((1 << 63), usize::MAX));
        assert_eq!(copy_bytes(&[usize::MAX, 0], 3), (0, 0));
    }

    #[test]
    fn trailing_zeros() {
        assert_eq!(trailing(0), 64);
        assert_eq!(trailing(1), 0);
        assert_eq!(trailing(40), 3);
        assert_eq!(trailing(1 << 63), 63);
        assert!(exact_at(0, 53, 24));
        assert!(exact_at(1 << 29, 53, 24));
        assert!(!exact_at((1 << 29) + (1 << 28), 53, 24));
        assert!(exact_at(3, 2, 24));
    }

    #[test]
    fn euclid_division() {
        assert_eq!(euclid(7, 2), (Some(3), Some(1)));
        assert_eq!(euclid(-7, 2), (Some(-4), Some(1)));
        assert_eq!(euclid(7, -2), (Some(-3), Some(1)));
        assert_eq!(euclid(-7, -2), (Some(4), Some(1)));
        assert_eq!(euclid(6, -3), (Some(-2), Some(0)));
        assert_eq!(euclid(5, 0), (None, None));
        assert_eq!(euclid(i64::MIN, -1), (None, None));
        assert_eq!(euclid(i64::MIN, 1), (Some(i64::MIN), Some(0)));
        assert_eq!(euclid(i64::MIN, i64::MIN), (Some(1), Some(0)));
        assert_eq!(euclid(-1, i64::MIN), (Some(1), Some(i64::MAX)));
        assert_eq!(euclid(i64::MAX, i64::MIN), (Some(0), Some(i64::MAX)));
    }
}
