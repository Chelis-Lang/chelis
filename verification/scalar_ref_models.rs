//! Integer, `bool` and ASCII helpers of the production crates (Aeneas.Std Scalar/Ops,
//! Scalar/Bitwise, Scalar/CoreConvertNum, String, Core/Iter): the operators on references
//! (`forward_ref_binop!`/`forward_ref_unop!`), `u8::is_ascii_hexdigit`/`is_ascii_whitespace`,
//! `u64::try_from(usize)` and the `Step` impl of `u32` behind `Range<u32>`.

/// As chelis-types `observation`: a remainder through a borrowed dividend.
pub fn rem_ref(value: &u64, modulus: u64) -> u64 {
    value % modulus
}

/// As chelis-ir `analysis` capacity masks: `&u128 & u128` and `&u128 & &u128`.
pub fn masks(bits: &u128, mask: u128, other: &u128) -> (u128, u128) {
    (bits & mask, bits & other)
}

/// As chelis-ir `lower`: `!flag` through a borrow.
pub fn not_ref(flag: &bool) -> bool {
    !flag
}

/// As chelis-types `static_int`: `&i64 - &i64`.
pub fn diff_refs(a: &i64, b: &i64) -> i64 {
    a - b
}

/// Counts of ASCII hex digits and ASCII whitespace bytes.
pub fn byte_classes(text: &str) -> (usize, usize) {
    let mut hex = 0;
    let mut space = 0;
    for b in text.bytes() {
        if b.is_ascii_hexdigit() {
            hex += 1;
        }
        if b.is_ascii_whitespace() {
            space += 1;
        }
    }
    (hex, space)
}

/// As chelis-types `dtype_semantics`: a `usize` widened to `u64`.
pub fn widen(n: usize) -> Option<u64> {
    u64::try_from(n).ok()
}

/// The number of values of a `u32` range (`Step::steps_between`).
pub fn span(lo: u32, hi: u32) -> usize {
    (lo..hi).len()
}

/// The `k`-th value of a `u32` range (`Step::forward_checked`) and the range left behind.
pub fn nth_from(range: &mut std::ops::Range<u32>, k: usize) -> Option<u32> {
    range.nth(k)
}

/// The `k`-th value from the back (`Step::backward_checked`).
pub fn nth_back_from(range: &mut std::ops::Range<u32>, k: usize) -> Option<u32> {
    range.nth_back(k)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn panics<T>(f: impl FnOnce() -> T + std::panic::UnwindSafe) -> bool {
        std::panic::catch_unwind(f).is_err()
    }

    #[test]
    fn operators() {
        assert_eq!(rem_ref(&17, 5), 2);
        assert!(panics(|| rem_ref(&17, 0)));
        assert_eq!(masks(&0xF0F0, 0xFF00, &0x0FF0), (0xF000, 0x00F0));
        assert_eq!(masks(&u128::MAX, 1 << 127, &u128::MAX), (1 << 127, u128::MAX));
        assert!(!not_ref(&true));
        assert!(not_ref(&false));
        assert_eq!(diff_refs(&5, &7), -2);
        assert_eq!(diff_refs(&i64::MIN, &-1), i64::MIN + 1);
        assert!(panics(|| diff_refs(&i64::MIN, &1)));
    }

    #[test]
    fn bytes() {
        assert_eq!(byte_classes("0x1F aG\tz\n\x0b\x0c\r"), (4, 5));
        assert_eq!(byte_classes(""), (0, 0));
        assert_eq!(byte_classes("é9"), (1, 0));
    }

    #[test]
    fn conversions() {
        assert_eq!(widen(0), Some(0));
        assert_eq!(widen(usize::MAX), Some(u64::MAX));
    }

    #[test]
    fn ranges() {
        assert_eq!(span(3, 10), 7);
        assert_eq!(span(10, 3), 0);
        assert_eq!(span(0, u32::MAX), u32::MAX as usize);
        let mut r = 3..10;
        assert_eq!(nth_from(&mut r, 2), Some(5));
        assert_eq!(r, 6..10);
        assert_eq!(nth_from(&mut r, 4), None);
        assert_eq!(r, 10..10);
        let mut r = 0..u32::MAX;
        assert_eq!(nth_from(&mut r, u32::MAX as usize + 1), None);
        assert_eq!(r, u32::MAX..u32::MAX);
        let mut r = 3..10;
        assert_eq!(nth_back_from(&mut r, 2), Some(7));
        assert_eq!(r, 3..7);
        assert_eq!(nth_back_from(&mut r, 9), None);
        assert_eq!(r, 3..3);
    }
}
