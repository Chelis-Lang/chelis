use std::cmp::Ordering;

/// The dropout-rate domain test of `DropoutParameters::new`.
pub fn dropout_rate_in_domain(bits: u64) -> bool {
    (0.0..1.0).contains(&f64::from_bits(bits))
}

pub fn contains_f64(start: u64, end: u64, item: u64) -> bool {
    (f64::from_bits(start)..f64::from_bits(end)).contains(&f64::from_bits(item))
}

pub fn contains_usize(start: usize, end: usize, item: usize) -> bool {
    (start..end).contains(&item)
}

/// A range bound and an item of distinct types. Comparing the value 99 panics,
/// so which `PartialOrd` implementation runs, and whether the second comparison
/// runs at all, is observable.
#[derive(Clone, Copy, PartialEq)]
pub struct Bound(pub u32);

#[derive(Clone, Copy)]
pub struct Item(pub u32);

impl PartialEq<Item> for Bound {
    fn eq(&self, other: &Item) -> bool {
        self.0 == other.0
    }
}

impl PartialEq<Bound> for Item {
    fn eq(&self, other: &Bound) -> bool {
        self.0 == other.0
    }
}

impl PartialOrd for Bound {
    fn partial_cmp(&self, other: &Bound) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl PartialOrd<Item> for Bound {
    fn partial_cmp(&self, other: &Item) -> Option<Ordering> {
        if self.0 == 99 {
            panic!("start compared");
        }
        self.0.partial_cmp(&other.0)
    }
}

impl PartialOrd<Bound> for Item {
    fn partial_cmp(&self, other: &Bound) -> Option<Ordering> {
        if other.0 == 99 {
            panic!("end compared");
        }
        self.0.partial_cmp(&other.0)
    }
}

pub fn contains_mixed(start: u32, end: u32, item: u32) -> bool {
    (Bound(start)..Bound(end)).contains(&Item(item))
}

/// `chelis_ir::eval::numel`'s product.
pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// The product continues from the iterator's current front and back.
pub fn product_after_ends(shape: &[usize]) -> usize {
    let mut iter = shape.iter();
    iter.next();
    iter.next_back();
    iter.product()
}

#[cfg(test)]
mod tests {
    use super::*;

    const ZERO: u64 = 0;
    const NEG_ZERO: u64 = 0x8000_0000_0000_0000;
    const HALF: u64 = 0x3fe0_0000_0000_0000;
    const ONE: u64 = 0x3ff0_0000_0000_0000;
    const NEG_TINY: u64 = 0x8000_0000_0000_0001;
    const BELOW_ONE: u64 = 0x3fef_ffff_ffff_ffff;
    const NAN: u64 = 0x7ff8_0000_0000_0000;
    const INF: u64 = 0x7ff0_0000_0000_0000;

    #[test]
    fn float_ranges_are_half_open_and_reject_nan() {
        assert!(dropout_rate_in_domain(ZERO));
        assert!(dropout_rate_in_domain(NEG_ZERO));
        assert!(dropout_rate_in_domain(HALF));
        assert!(dropout_rate_in_domain(BELOW_ONE));
        assert!(!dropout_rate_in_domain(ONE));
        assert!(!dropout_rate_in_domain(NEG_TINY));
        assert!(!dropout_rate_in_domain(NAN));
        assert!(!dropout_rate_in_domain(INF));
        assert!(!contains_f64(NAN, ONE, HALF));
        assert!(!contains_f64(ZERO, NAN, HALF));
        assert!(contains_f64(ZERO, INF, ONE));
        assert!(!contains_f64(ONE, ZERO, HALF));
    }

    #[test]
    fn integer_ranges_are_half_open() {
        assert!(contains_usize(3, 5, 3));
        assert!(contains_usize(3, 5, 4));
        assert!(!contains_usize(3, 5, 5));
        assert!(!contains_usize(3, 5, 2));
        assert!(!contains_usize(3, 3, 3));
        assert!(!contains_usize(5, 3, 4));
        assert!(contains_usize(0, usize::MAX, usize::MAX - 1));
    }

    #[test]
    fn mixed_ranges_compare_start_then_end() {
        assert!(contains_mixed(3, 7, 3));
        assert!(contains_mixed(3, 7, 6));
        assert!(!contains_mixed(3, 7, 7));
        assert!(!contains_mixed(7, 3, 5));
        // `start <= item` fails first, so the poisoned end is never compared.
        assert!(!contains_mixed(1, 99, 0));
    }

    #[test]
    #[should_panic(expected = "start compared")]
    fn mixed_ranges_compare_the_start_through_the_bound() {
        contains_mixed(99, 100, 5);
    }

    #[test]
    #[should_panic(expected = "end compared")]
    fn mixed_ranges_compare_the_end_through_the_item() {
        contains_mixed(1, 99, 5);
    }

    #[test]
    fn products_fold_from_one() {
        assert_eq!(numel(&[]), 1);
        assert_eq!(numel(&[7]), 7);
        assert_eq!(numel(&[2, 3, 4]), 24);
        assert_eq!(numel(&[0, usize::MAX, 2]), 0);
        assert_eq!(numel(&[1 << 32, (1 << 32) - 1]), usize::MAX - ((1 << 32) - 1));
        assert_eq!(product_after_ends(&[1000, 2, 3, 5, 1000]), 30);
        assert_eq!(product_after_ends(&[9]), 1);
        assert_eq!(product_after_ends(&[]), 1);
    }

    #[test]
    #[should_panic(expected = "overflow")]
    fn products_panic_at_the_first_overflow() {
        numel(&[usize::MAX, 2, 0]);
    }

    #[test]
    #[should_panic(expected = "overflow")]
    fn products_panic_on_overflow() {
        numel(&[1 << 32, 1 << 32]);
    }
}
