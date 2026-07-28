//! Regression locks for the int32 accessor arms corrected in c9b99cae,
//! 521b4323, and 975e333b.
//!
//! Each of those arms decoded native int32 storage through an f32 view. The
//! `cmplt` case has its own file; this covers the rest of the ops that return
//! a tensor, so reverting any single arm fails here rather than silently
//! producing wrong numbers.
//!
//! Values are chosen to defeat the accident that hid the bug: small
//! non-negative integers survive an f32 reinterpretation because positive
//! IEEE-754 floats order and add roughly the way their bit patterns do.
//! Negatives, `i32::MIN`, and values past the denormal range do not.

use chelis_runtime::{
    chelis_alloc, chelis_tensor, chelis_tensor_clamp, chelis_tensor_cumsum, chelis_tensor_where,
    TensorElement, CHELIS_I32,
};

unsafe fn i32_vec(values: &[i32]) -> *mut chelis_tensor {
    unsafe {
        let shape = [values.len() as std::ffi::c_int];
        let t = chelis_alloc(1, shape.as_ptr(), CHELIS_I32);
        let p = i32::data_ptr_unchecked(t);
        for (i, v) in values.iter().enumerate() {
            *p.add(i) = *v;
        }
        t
    }
}

unsafe fn read_i32(t: *mut chelis_tensor, len: usize) -> Vec<i32> {
    unsafe {
        let p = i32::data_ptr_unchecked(t);
        (0..len).map(|i| *p.add(i)).collect()
    }
}

/// `where` selected the wrong branch for exactly one condition value.
///
/// The f32 truth test was `*p != 0.0`, which holds for every nonzero bit
/// pattern except `0x80000000` — that is `-0.0`, and `-0.0 != 0.0` is false
/// under IEEE-754. So a condition of `i32::MIN` took the else branch.
#[test]
fn where_treats_i32_min_condition_as_true() {
    unsafe {
        let cond = i32_vec(&[i32::MIN, 0, 1, -1]);
        let then_t = i32_vec(&[10, 20, 30, 40]);
        let else_t = i32_vec(&[-10, -20, -30, -40]);

        let out = chelis_tensor_where(cond, then_t, else_t);

        assert_eq!(
            read_i32(out, 4),
            vec![10, -20, 30, 40],
            "i32::MIN is a nonzero condition and must select the then branch; \
             an f32 truth test reads its bit pattern as -0.0 and selects else"
        );
    }
}

/// cumsum accumulated through an f32 view, summing denormal bit patterns
/// rather than the integers they encode.
#[test]
fn cumsum_accumulates_int32_values_not_bit_patterns() {
    unsafe {
        let t = i32_vec(&[1, 2, 3, 4]);
        let out = chelis_tensor_cumsum(t, 0);
        assert_eq!(read_i32(out, 4), vec![1, 3, 6, 10]);
    }
}

/// Past the denormal range an f32 accumulation stops even approximating the
/// integer sum, so this is the case the old arm could not fake.
#[test]
fn cumsum_is_correct_for_large_and_negative_int32() {
    unsafe {
        let t = i32_vec(&[1_000_000, -2_000_000, 3_000_000]);
        let out = chelis_tensor_cumsum(t, 0);
        assert_eq!(read_i32(out, 3), vec![1_000_000, -1_000_000, 2_000_000]);
    }
}

/// clamp compared and stored as f32. Negative bounds are where reinterpreting
/// the bytes stops being order-preserving.
#[test]
fn clamp_respects_negative_int32_bounds() {
    unsafe {
        let t = i32_vec(&[-100, -5, 0, 5, 100]);
        let lo = i32_vec(&[-10, -10, -10, -10, -10]);
        let hi = i32_vec(&[10, 10, 10, 10, 10]);

        let out = chelis_tensor_clamp(t, lo, hi);

        assert_eq!(read_i32(out, 5), vec![-10, -5, 0, 5, 10]);
    }
}

/// The extremes, which an f32 view maps onto NaN and infinity rather than
/// onto representable integers.
#[test]
fn clamp_handles_int32_extremes() {
    unsafe {
        let t = i32_vec(&[i32::MIN, i32::MAX]);
        let lo = i32_vec(&[-1_000, -1_000]);
        let hi = i32_vec(&[1_000, 1_000]);

        let out = chelis_tensor_clamp(t, lo, hi);

        assert_eq!(read_i32(out, 2), vec![-1_000, 1_000]);
    }
}
