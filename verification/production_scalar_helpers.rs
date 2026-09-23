#![feature(hint_must_use)]

use std::num::TryFromIntError;

pub fn usize_to_i32(value: usize) -> Result<i32, TryFromIntError> {
    i32::try_from(value)
}

pub fn i32_to_usize(value: i32) -> Result<usize, TryFromIntError> {
    usize::try_from(value)
}

pub fn usize_to_i64(value: usize) -> Result<i64, TryFromIntError> {
    i64::try_from(value)
}

pub fn rounded_division(value: usize, divisor: usize) -> usize {
    value.div_ceil(divisor)
}

pub fn signed_magnitude(value: i64) -> (bool, u64) {
    (value.is_negative(), value.unsigned_abs())
}

pub fn multiply_borrowed(mut value: usize, factor: usize) -> usize {
    value *= &factor;
    value
}

// The argument must be evaluated even when condition is false.
pub fn eager_then_some(condition: bool, mut counter: u32) -> (Option<u32>, u32) {
    let chosen = condition.then_some({
        counter += 1;
        counter
    });
    (chosen, counter)
}

pub struct CheckedClone(pub u32);

impl Clone for CheckedClone {
    fn clone(&self) -> Self {
        Self(self.0 + 1)
    }
}

pub fn clone_optional(value: Option<u32>) -> Option<u32> {
    let wrapped = match value {
        Some(value) => Some(CheckedClone(value)),
        None => None,
    };
    match wrapped.clone() {
        Some(value) => Some(value.0),
        None => None,
    }
}

// Owned generic helpers must retain an embedded mutable borrow, not only its value.
pub fn owned_helpers_update(mut value: u32) -> u32 {
    {
        let boxed: Box<&mut u32> = Box::from(core::hint::must_use(&mut value));
        **boxed += 1;
    }
    value
}

pub fn then_some_borrow_update(condition: bool, mut value: u32) -> u32 {
    if let Some(target) = condition.then_some(&mut value) {
        *target += 1;
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion_bounds_are_normal_results() {
        assert_eq!(usize_to_i32(i32::MAX as usize), Ok(i32::MAX));
        assert!(usize_to_i32(i32::MAX as usize + 1).is_err());
        assert!(i32_to_usize(-1).is_err());
        assert_eq!(i32_to_usize(0), Ok(0));
        if usize::BITS == 64 {
            assert_eq!(usize_to_i64(i64::MAX as usize), Ok(i64::MAX));
            assert!(usize_to_i64(i64::MAX as usize + 1).is_err());
        }
    }

    #[test]
    fn integer_helpers_preserve_boundary_arithmetic() {
        assert_eq!(rounded_division(usize::MAX, 2), usize::MAX / 2 + 1);
        assert_eq!(rounded_division(12, 3), 4);
        assert!(std::panic::catch_unwind(|| rounded_division(0, 0)).is_err());
        assert_eq!(signed_magnitude(i64::MIN), (true, 1 << 63));
        assert_eq!(signed_magnitude(0), (false, 0));
        assert_eq!(multiply_borrowed(7, 6), 42);
        assert!(std::panic::catch_unwind(|| multiply_borrowed(usize::MAX, 2)).is_err());
    }

    #[test]
    fn owned_helpers_preserve_evaluation_and_borrowed_updates() {
        assert_eq!(eager_then_some(false, 4), (None, 5));
        assert!(std::panic::catch_unwind(|| eager_then_some(false, u32::MAX)).is_err());
        assert_eq!(clone_optional(Some(4)), Some(5));
        assert_eq!(clone_optional(None), None);
        assert!(std::panic::catch_unwind(|| clone_optional(Some(u32::MAX))).is_err());
        assert_eq!(owned_helpers_update(4), 5);
        assert_eq!(then_some_borrow_update(true, 4), 5);
        assert_eq!(then_some_borrow_update(false, 4), 4);
    }
}
