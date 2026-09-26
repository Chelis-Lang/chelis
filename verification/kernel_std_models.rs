//! Core/alloc items of the evaluator kernel (chelis-types dtype_semantics,
//! chelis-ir evaluation): the infallible `TryFrom` blanket, `TryFrom<i64>` for
//! the narrow integers, the f64 infinities, the `ToOwned` blanket, `Cow`'s
//! `Deref`, and the primitive ordered comparison of `bool`.

use std::borrow::Cow;

/// Generic over the conversion, as `dtype_semantics::int_buf`: `i64` goes
/// through the infallible blanket, the narrow integers through
/// `TryFromIntError`.
pub fn narrow<T: TryFrom<i64>>(value: i64) -> Option<T> {
    match T::try_from(value) {
        Ok(narrowed) => Some(narrowed),
        Err(_) => None,
    }
}

pub fn narrow_all(value: i64) -> (Option<i64>, Option<i8>, Option<i16>, Option<i32>) {
    (narrow::<i64>(value), narrow::<i8>(value), narrow::<i16>(value), narrow::<i32>(value))
}

/// A direct call, as the evaluator's shape checks.
pub fn index(value: i64) -> Option<usize> {
    match usize::try_from(value) {
        Ok(index) => Some(index),
        Err(_) => None,
    }
}

pub fn unbounded(negative: bool) -> u64 {
    let bound = if negative { f64::NEG_INFINITY } else { f64::INFINITY };
    bound.to_bits()
}

#[derive(Clone)]
pub struct Tensor {
    pub values: Vec<u32>,
    pub scale: u32,
}

pub fn owned_copy(tensor: &Tensor) -> Tensor {
    tensor.to_owned()
}

fn rescaled(tensor: &Tensor, owned: bool) -> Cow<'_, Tensor> {
    if owned {
        Cow::Owned(Tensor { values: tensor.values.clone(), scale: tensor.scale + 1 })
    } else {
        Cow::Borrowed(tensor)
    }
}

/// Both arms of `Cow`'s `Deref`: the borrowed tensor, or the owned one
/// through `Borrow<Tensor> for Tensor`.
pub fn cow_view(tensor: &Tensor, owned: bool) -> (u32, usize) {
    let view = rescaled(tensor, owned);
    let viewed: &Tensor = &view;
    (viewed.scale, viewed.values.len())
}

/// Ordered comparison of `bool`, as `dtype_semantics::compare_scalars`.
pub fn bool_order(left: bool, right: bool) -> (bool, bool, bool, bool) {
    (left < right, left <= right, left > right, left >= right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_conversions_check_the_target_range() {
        assert_eq!(narrow_all(5), (Some(5), Some(5), Some(5), Some(5)));
        assert_eq!(narrow_all(-128), (Some(-128), Some(-128), Some(-128), Some(-128)));
        assert_eq!(narrow_all(-129), (Some(-129), None, Some(-129), Some(-129)));
        assert_eq!(narrow_all(40_000), (Some(40_000), None, None, Some(40_000)));
        assert_eq!(narrow_all(-2_147_483_648), (Some(-2_147_483_648), None, None, Some(-2_147_483_648)));
        assert_eq!(narrow_all(1 << 40), (Some(1 << 40), None, None, None));
        assert_eq!(index(7), Some(7));
        assert_eq!(index(0), Some(0));
        assert_eq!(index(-1), None);
        assert_eq!(index(i64::MAX), Some(i64::MAX as usize));
    }

    #[test]
    fn infinities_have_their_ieee_bits() {
        assert_eq!(unbounded(false), 0x7ff0_0000_0000_0000);
        assert_eq!(unbounded(true), 0xfff0_0000_0000_0000);
    }

    #[test]
    fn borrowed_and_owned_views_agree_with_clone() {
        let tensor = Tensor { values: vec![1, 2, 3], scale: 4 };
        let copy = owned_copy(&tensor);
        assert_eq!((copy.values, copy.scale), (vec![1, 2, 3], 4));
        assert_eq!(cow_view(&tensor, false), (4, 3));
        assert_eq!(cow_view(&tensor, true), (5, 3));
    }

    #[test]
    fn false_is_less_than_true() {
        assert_eq!(bool_order(false, true), (true, true, false, false));
        assert_eq!(bool_order(true, false), (false, false, true, true));
        assert_eq!(bool_order(true, true), (false, true, false, true));
        assert_eq!(bool_order(false, false), (false, true, false, true));
    }
}
