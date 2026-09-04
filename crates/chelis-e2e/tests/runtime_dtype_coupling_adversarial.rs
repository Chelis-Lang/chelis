//! Adversarial exact-storage checks for the runtime tensor boundary.
//!
//! These fixtures verify:
//! - `<T>::data_ptr` checked rejection across every cross-dtype pairing
//!   for `f32`, `f64`, and `i64`;
//! - exact tagged scalar extraction without an intermediate float carrier.

use std::ptr;

use chelis_runtime::{
    CHELIS_DTYPE_BOOL, CHELIS_DTYPE_F32, CHELIS_DTYPE_F64, CHELIS_DTYPE_I32, CHELIS_DTYPE_I64,
    RuntimeDType, TensorElement, chelis_alloc, chelis_dtype, chelis_tensor, chelis_tensor_release,
    chelis_tensor_to_scalar,
};

unsafe fn alloc_scalar(dtype: chelis_dtype) -> *mut chelis_tensor {
    unsafe { chelis_alloc(0, ptr::null(), dtype) }
}

// ============================================================
// §3.4 <T>::data_ptr checked rejection — full cross-dtype matrix
// ============================================================
//
// PR #84 added four `data_ptr` dtype-mismatch fixtures. We extend with
// the remaining cross-pairings between the three native-storage
// precisions (f32, f64, i64), plus the independent i32 and Bool8 stores.

#[test]
fn data_ptr_f32_rejects_f64_tensor() {
    unsafe {
        let t = alloc_scalar(CHELIS_DTYPE_F64);
        let res = <f32 as TensorElement>::data_ptr(t);
        assert!(res.is_err(), "f32::data_ptr must reject CHELIS_DTYPE_F64");
        let err = res.unwrap_err();
        assert_eq!(err.expected, RuntimeDType::F32);
        assert_eq!(err.actual, RuntimeDType::F64);
        chelis_tensor_release(t);
    }
}

#[test]
fn data_ptr_f32_rejects_i64_tensor() {
    unsafe {
        let t = alloc_scalar(CHELIS_DTYPE_I64);
        let res = <f32 as TensorElement>::data_ptr(t);
        assert!(res.is_err(), "f32::data_ptr must reject CHELIS_DTYPE_I64");
        chelis_tensor_release(t);
    }
}

#[test]
fn data_ptr_f64_rejects_f32_tensor() {
    unsafe {
        let t = alloc_scalar(CHELIS_DTYPE_F32);
        let res = <f64 as TensorElement>::data_ptr(t);
        assert!(res.is_err(), "f64::data_ptr must reject CHELIS_DTYPE_F32");
        let err = res.unwrap_err();
        assert_eq!(err.expected, RuntimeDType::F64);
        assert_eq!(err.actual, RuntimeDType::F32);
        chelis_tensor_release(t);
    }
}

#[test]
fn data_ptr_i64_rejects_f64_tensor() {
    unsafe {
        let t = alloc_scalar(CHELIS_DTYPE_F64);
        let res = <i64 as TensorElement>::data_ptr(t);
        assert!(res.is_err(), "i64::data_ptr must reject CHELIS_DTYPE_F64");
        let err = res.unwrap_err();
        assert_eq!(err.expected, RuntimeDType::I64);
        assert_eq!(err.actual, RuntimeDType::F64);
        chelis_tensor_release(t);
    }
}

#[test]
fn data_ptr_i64_rejects_i32_tensor() {
    unsafe {
        let t = alloc_scalar(CHELIS_DTYPE_I32);
        let res = <i64 as TensorElement>::data_ptr(t);
        assert!(res.is_err(), "i64::data_ptr must reject CHELIS_DTYPE_I32");
        let err = res.unwrap_err();
        assert_eq!(err.expected, RuntimeDType::I64);
        assert_eq!(err.actual, RuntimeDType::I32);
        chelis_tensor_release(t);
    }
}

#[test]
fn data_ptr_i64_rejects_bool_tensor() {
    unsafe {
        let t = alloc_scalar(CHELIS_DTYPE_BOOL);
        let res = <i64 as TensorElement>::data_ptr(t);
        assert!(res.is_err(), "i64::data_ptr must reject CHELIS_DTYPE_BOOL");
        let err = res.unwrap_err();
        assert_eq!(err.expected, RuntimeDType::I64);
        assert_eq!(err.actual, RuntimeDType::Bool);
        chelis_tensor_release(t);
    }
}

// ============================================================
// §3.4 i64-precision round-trip preservation
// ============================================================

#[test]
fn i64_round_trip_at_low_value_byte_exact() {
    unsafe {
        let t = alloc_scalar(CHELIS_DTYPE_I64);
        let raw = 1_000_000_000_000_i64;
        i64::fill(t, raw);
        let out = chelis_tensor_to_scalar(t);
        assert_eq!(out.dtype, CHELIS_DTYPE_I64);
        assert_eq!(out.bits, u64::from_ne_bytes(raw.to_ne_bytes()));
        chelis_tensor_release(t);
    }
}

#[test]
fn i64_above_2pow53_round_trips_without_float_conversion() {
    unsafe {
        let t = alloc_scalar(CHELIS_DTYPE_I64);
        let raw: i64 = (1i64 << 54) + 1;
        i64::fill(t, raw);
        let out = chelis_tensor_to_scalar(t);
        assert_eq!(out.dtype, CHELIS_DTYPE_I64);
        assert_eq!(out.bits, u64::from_ne_bytes(raw.to_ne_bytes()));
        chelis_tensor_release(t);
    }
}

#[test]
fn f64_precision_beyond_f32_round_trips() {
    unsafe {
        let t = alloc_scalar(CHELIS_DTYPE_F64);
        const VALUE: f64 = 1.234_567_890_123_456_7_f64;
        f64::fill(t, VALUE);
        let out = chelis_tensor_to_scalar(t);
        assert_eq!(out.dtype, CHELIS_DTYPE_F64);
        assert_eq!(out.bits, VALUE.to_bits());
        chelis_tensor_release(t);
    }
}
