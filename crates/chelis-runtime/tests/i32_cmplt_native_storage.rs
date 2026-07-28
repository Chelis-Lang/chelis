//! `cmplt` on int32 tensors must decode int32 storage, not f32 bit patterns.
//!
//! `chelis_scalar_tensor_from_i64` writes through `(int32_t*)` — its comment
//! records that the previous `value as f32` write "left the slot holding float
//! bit patterns that downstream readers interpreted as junk integers". So
//! CHELIS_I32 storage is native two's complement.
//!
//! `chelis_tensor_cmplt` nonetheless routes `RuntimeDType::I32` through
//! `data_as_f32_const`. Reinterpreting native int32 bytes as f32 happens to
//! give the right answer for small non-negative values, because IEEE-754
//! positive floats are monotonic in their bit patterns. It does not for
//! negative values: `-1i32` is `0xFFFFFFFF`, which is a NaN as f32, and every
//! comparison against a NaN is false.

use chelis_runtime::{chelis_scalar_tensor_from_i64, chelis_tensor_cmplt, chelis_tensor_to_f64};

/// Reads the single element of a rank-0 bool tensor as a truth value.
///
/// Bool storage is f32-encoded today (`CRuntime-BoolStorage-F1`), which
/// `chelis_tensor_to_f64` already accounts for, so this stays correct across
/// the pending storage migration.
unsafe fn truth_of(tensor: *mut chelis_runtime::chelis_tensor) -> bool {
    unsafe { chelis_tensor_to_f64(tensor) != 0.0 }
}

unsafe fn cmplt_i64(lhs: i64, rhs: i64) -> bool {
    unsafe {
        let l = chelis_scalar_tensor_from_i64(lhs);
        let r = chelis_scalar_tensor_from_i64(rhs);
        let out = chelis_tensor_cmplt(l, r);
        truth_of(out)
    }
}

/// The accidental-success region: small non-negative int32 values compare
/// correctly even through an f32 reinterpretation, because positive IEEE
/// floats order the same way their bit patterns do.
#[test]
fn small_non_negative_int32_comparisons_are_correct() {
    unsafe {
        assert!(cmplt_i64(1, 2), "1 < 2");
        assert!(!cmplt_i64(2, 1), "2 is not < 1");
        assert!(!cmplt_i64(3, 3), "3 is not < 3");
    }
}

/// The real case. `-1i32` is `0xFFFFFFFF`, a NaN when reinterpreted as f32,
/// and every NaN comparison is false — so an f32-strided read reports
/// `-1 < 0` as false.
#[test]
fn negative_int32_comparisons_are_correct() {
    unsafe {
        assert!(cmplt_i64(-1, 0), "-1 < 0");
        assert!(cmplt_i64(-5, -1), "-5 < -1");
        assert!(!cmplt_i64(0, -1), "0 is not < -1");
        assert!(!cmplt_i64(-1, -5), "-1 is not < -5");
    }
}

/// Mixed signs are the case an f32 reinterpretation gets most confidently
/// wrong: the sign bit makes the negative operand a NaN rather than a value
/// below the positive one.
#[test]
fn mixed_sign_int32_comparisons_are_correct() {
    unsafe {
        assert!(cmplt_i64(-1, 1), "-1 < 1");
        assert!(!cmplt_i64(1, -1), "1 is not < -1");
        assert!(cmplt_i64(i32::MIN as i64, i32::MAX as i64), "MIN < MAX");
        assert!(
            !cmplt_i64(i32::MAX as i64, i32::MIN as i64),
            "MAX is not < MIN"
        );
    }
}
