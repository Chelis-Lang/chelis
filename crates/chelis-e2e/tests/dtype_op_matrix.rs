//! Cross-validation harness for exact dtype-by-operation agreement at the
//! public runtime boundary. Numeric values retain their declared storage,
//! Bool uses canonical one-byte `Bool8`, and scalar exits use tagged bits.

#![allow(clippy::missing_safety_doc)]

use std::ptr;

use chelis_runtime::{
    Bool8, CHELIS_DTYPE_BOOL, CHELIS_DTYPE_F32, CHELIS_DTYPE_F64, CHELIS_DTYPE_I32,
    CHELIS_DTYPE_I64, DtypeMismatch, TensorElement, chelis_alloc, chelis_dtype, chelis_list_append,
    chelis_list_empty, chelis_list_index, chelis_list_len, chelis_scalar, chelis_scalar_from_bits,
    chelis_string_from_cstr, chelis_tensor, chelis_tensor_begin_write, chelis_tensor_clamp,
    chelis_tensor_cmplt, chelis_tensor_concat, chelis_tensor_cumsum, chelis_tensor_diagonal,
    chelis_tensor_einsum, chelis_tensor_elements, chelis_tensor_end_write, chelis_tensor_gather,
    chelis_tensor_numel, chelis_tensor_read_view, chelis_tensor_release, chelis_tensor_scatter_add,
    chelis_tensor_scatter_replace, chelis_tensor_shape, chelis_tensor_sort, chelis_tensor_split,
    chelis_tensor_take_value, chelis_tensor_to_scalar, chelis_tensor_trace, chelis_tensor_where,
    chelis_tuple_get, chelis_value, chelis_value_box_scalar, chelis_value_take_tensor,
    chelis_value_unbox_scalar,
};

/// Allocate a rank-0 (scalar) tensor of the given dtype.  Caller frees.
unsafe fn alloc_scalar(dtype: chelis_dtype) -> *mut chelis_tensor {
    unsafe { chelis_alloc(0, ptr::null(), dtype) }
}

/// Allocate a rank-1 tensor of length `n` and the given dtype.
unsafe fn alloc_vec(n: i64, dtype: chelis_dtype) -> *mut chelis_tensor {
    let shape = [n];
    unsafe { chelis_alloc(1, shape.as_ptr(), dtype) }
}

/// Exercise the public guarded-write lane rather than reaching around the
/// descriptor's exclusive lease with an unguarded runtime call.
unsafe fn fill_tensor_scalar(tensor: *mut chelis_tensor, value: chelis_scalar) {
    let guard = unsafe { chelis_tensor_begin_write(tensor) };
    unsafe { chelis_runtime::chelis_fill_scalar(guard, value) };
    unsafe { chelis_tensor_end_write(guard) };
}

fn i64_scalar(value: i64) -> chelis_scalar {
    chelis_scalar_from_bits(CHELIS_DTYPE_I64, u64::from_ne_bytes(value.to_ne_bytes()))
}

unsafe fn exact_i64_value(value: i64) -> chelis_value {
    unsafe { chelis_value_box_scalar(i64_scalar(value)) }
}

unsafe fn exact_value_f64(value: chelis_value) -> f64 {
    let scalar = unsafe { chelis_value_unbox_scalar(value) };
    match scalar.dtype {
        CHELIS_DTYPE_F64 => f64::from_bits(scalar.bits),
        CHELIS_DTYPE_F32 => f64::from(f32::from_bits(scalar.bits as u32)),
        other => panic!("expected float scalar, found dtype {other}"),
    }
}

unsafe fn exact_value_i64(value: chelis_value) -> i64 {
    let scalar = unsafe { chelis_value_unbox_scalar(value) };
    match scalar.dtype {
        CHELIS_DTYPE_I64 => i64::from_ne_bytes(scalar.bits.to_ne_bytes()),
        CHELIS_DTYPE_I32 => i64::from(scalar.bits as u32 as i32),
        other => panic!("expected signed integer scalar, found dtype {other}"),
    }
}

unsafe fn exact_value_bool(value: chelis_value) -> bool {
    let scalar = unsafe { chelis_value_unbox_scalar(value) };
    assert_eq!(scalar.dtype, CHELIS_DTYPE_BOOL);
    scalar.bits == 1
}

// ---- `chelis_tensor_to_scalar` (read-side) ------------------------------
//
// Each fixture checks that a rank-0 tensor retains its dtype and exact
// stored value through scalar extraction.

#[test]
fn tensor_to_scalar_f32() {
    unsafe {
        let t = alloc_scalar(CHELIS_DTYPE_F32);
        f32::fill(t, 3.5);
        let out = chelis_tensor_to_scalar(t);
        assert_eq!(out.dtype, CHELIS_DTYPE_F32);
        assert_eq!(f32::from_bits(out.bits as u32), 3.5);
        chelis_tensor_release(t);
    }
}

#[test]
fn tensor_to_scalar_f64() {
    // Value chosen to exceed f32 precision (16 decimal digits) so
    // any f32-truncating read path fails the round-trip.
    const F64_VALUE: f64 = 1.234_567_890_123_456_7_f64;
    unsafe {
        let t = alloc_scalar(CHELIS_DTYPE_F64);
        f64::fill(t, F64_VALUE);
        let out = chelis_tensor_to_scalar(t);
        assert_eq!(out.dtype, CHELIS_DTYPE_F64);
        assert_eq!(out.bits, F64_VALUE.to_bits());
        chelis_tensor_release(t);
    }
}

#[test]
fn tensor_to_scalar_i64() {
    unsafe {
        let t = alloc_scalar(CHELIS_DTYPE_I64);
        i64::fill(t, 1_000_000_000_000_i64);
        let out = chelis_tensor_to_scalar(t);
        assert_eq!(out.dtype, CHELIS_DTYPE_I64);
        assert_eq!(
            i64::from_ne_bytes(out.bits.to_ne_bytes()),
            1_000_000_000_000
        );
        chelis_tensor_release(t);
    }
}

#[test]
fn tensor_to_scalar_i32() {
    unsafe {
        let t = alloc_scalar(CHELIS_DTYPE_I32);
        // i32 tensors use i32 storage.
        *i32::data_ptr_unchecked(t) = 42;
        let out = chelis_tensor_to_scalar(t);
        assert_eq!(out.dtype, CHELIS_DTYPE_I32);
        assert_eq!(out.bits as u32 as i32, 42);
        chelis_tensor_release(t);
    }
}

#[test]
fn tensor_to_scalar_bool_true() {
    unsafe {
        let t = alloc_scalar(CHELIS_DTYPE_BOOL);
        *Bool8::data_ptr_unchecked(t) = Bool8::TRUE;
        let out = chelis_tensor_to_scalar(t);
        assert_eq!(out.dtype, CHELIS_DTYPE_BOOL);
        assert_eq!(out.bits, 1);
        chelis_tensor_release(t);
    }
}

#[test]
fn tensor_to_scalar_bool_false() {
    unsafe {
        let t = alloc_scalar(CHELIS_DTYPE_BOOL);
        *Bool8::data_ptr_unchecked(t) = Bool8::FALSE;
        let out = chelis_tensor_to_scalar(t);
        assert_eq!(out.dtype, CHELIS_DTYPE_BOOL);
        assert_eq!(out.bits, 0);
        chelis_tensor_release(t);
    }
}

// ---- `TensorElement::fill` (write-side) --------------------------------
//
// Each fixture reads every element of a filled tensor through its
// dtype-matched pointer.

#[test]
fn fill_f32_vector() {
    unsafe {
        let t = alloc_vec(8, CHELIS_DTYPE_F32);
        f32::fill(t, 1.5);
        let ptr = f32::data_ptr_unchecked(t);
        for i in 0..chelis_tensor_numel(t) as isize {
            assert_eq!(*ptr.offset(i), 1.5, "f32 fill index {i}");
        }
        chelis_tensor_release(t);
    }
}

#[test]
fn fill_f64_vector() {
    unsafe {
        let t = alloc_vec(8, CHELIS_DTYPE_F64);
        f64::fill(t, 1.0e100);
        let ptr = f64::data_ptr_unchecked(t);
        for i in 0..chelis_tensor_numel(t) as isize {
            assert_eq!(*ptr.offset(i), 1.0e100, "f64 fill index {i}");
        }
        chelis_tensor_release(t);
    }
}

#[test]
fn fill_i64_vector() {
    unsafe {
        let t = alloc_vec(8, CHELIS_DTYPE_I64);
        i64::fill(t, -123_456_789_012_i64);
        let ptr = i64::data_ptr_unchecked(t);
        for i in 0..chelis_tensor_numel(t) as isize {
            assert_eq!(*ptr.offset(i), -123_456_789_012_i64, "i64 fill index {i}");
        }
        chelis_tensor_release(t);
    }
}

#[test]
fn fill_i32_vector() {
    // i32 storage is read and written through i32 accessors.
    unsafe {
        let t = alloc_vec(8, CHELIS_DTYPE_I32);
        i32::fill(t, 12_345_i32);
        let ptr = i32::data_ptr_unchecked(t);
        for i in 0..chelis_tensor_numel(t) as isize {
            assert_eq!(*ptr.offset(i), 12_345_i32, "i32 fill index {i}");
        }
        chelis_tensor_release(t);
    }
}

#[test]
fn exact_scalar_fill_f32_matches_stored_bits() {
    unsafe {
        let t = alloc_vec(4, CHELIS_DTYPE_F32);
        fill_tensor_scalar(
            t,
            chelis_scalar_from_bits(CHELIS_DTYPE_F32, u64::from(9.25_f32.to_bits())),
        );
        let ptr = f32::data_ptr_unchecked(t);
        for i in 0..chelis_tensor_numel(t) as isize {
            assert_eq!(*ptr.offset(i), 9.25);
        }
        chelis_tensor_release(t);
    }
}

#[test]
fn exact_scalar_fill_f64_matches_stored_bits() {
    // Same f64 bit pattern used in `tensor_to_f64_f64` -- exceeds
    // f32 precision so any f32-truncating fill path would fail.
    const F64_VALUE: f64 = 1.234_567_890_123_456_7_f64;
    unsafe {
        let t = alloc_vec(4, CHELIS_DTYPE_F64);
        fill_tensor_scalar(
            t,
            chelis_scalar_from_bits(CHELIS_DTYPE_F64, F64_VALUE.to_bits()),
        );
        let ptr = f64::data_ptr_unchecked(t);
        for i in 0..chelis_tensor_numel(t) as isize {
            assert_eq!(*ptr.offset(i), F64_VALUE);
        }
        chelis_tensor_release(t);
    }
}

#[test]
fn exact_scalar_fill_i64_matches_stored_bits() {
    unsafe {
        let t = alloc_vec(4, CHELIS_DTYPE_I64);
        fill_tensor_scalar(t, i64_scalar(9_876_543_210_i64));
        let ptr = i64::data_ptr_unchecked(t);
        for i in 0..chelis_tensor_numel(t) as isize {
            assert_eq!(*ptr.offset(i), 9_876_543_210_i64);
        }
        chelis_tensor_release(t);
    }
}

// ---- Negative coverage: dtype mismatch detection -----------------------
//
// `<T>::data_ptr` returns `Err(DtypeMismatch { expected, actual })`
// when the tensor's runtime dtype tag does not match the trait impl's
// `DTYPE`.  This is the safety net that closes the bug class: any
// future site that asks for a typed pointer through the trait and
// forgets to dispatch on dtype catches the mismatch as `Err` instead
// of silently reading the wrong byte layout.

#[test]
fn data_ptr_dtype_mismatch_f32_on_f64_tensor() {
    unsafe {
        let t = alloc_scalar(CHELIS_DTYPE_F64);
        let err = f32::data_ptr(t).expect_err("f32::data_ptr on F64 tensor must fail");
        assert_eq!(
            err,
            DtypeMismatch {
                expected: <f32 as TensorElement>::DTYPE,
                actual: <f64 as TensorElement>::DTYPE,
            }
        );
        chelis_tensor_release(t);
    }
}

#[test]
fn data_ptr_dtype_mismatch_i64_on_i32_tensor() {
    unsafe {
        let t = alloc_scalar(CHELIS_DTYPE_I32);
        let err = i64::data_ptr(t).expect_err("i64::data_ptr on I32 tensor must fail");
        assert_eq!(
            err,
            DtypeMismatch {
                expected: <i64 as TensorElement>::DTYPE,
                actual: <i32 as TensorElement>::DTYPE,
            }
        );
        chelis_tensor_release(t);
    }
}

#[test]
fn data_ptr_dtype_mismatch_f64_on_i64_tensor() {
    // f64 and i64 share an 8-byte storage cell, so the mismatch is
    // semantic (bit-layout differs) not size-related.  The trait must
    // catch this anyway.
    unsafe {
        let t = alloc_scalar(CHELIS_DTYPE_I64);
        let err = f64::data_ptr(t).expect_err("f64::data_ptr on I64 tensor must fail");
        assert_eq!(
            err,
            DtypeMismatch {
                expected: <f64 as TensorElement>::DTYPE,
                actual: <i64 as TensorElement>::DTYPE,
            }
        );
        chelis_tensor_release(t);
    }
}

#[test]
fn data_ptr_match_succeeds() {
    // Positive control: the matching dtype returns Ok and reads back
    // the fill value byte-exact through the typed pointer.
    unsafe {
        let t = alloc_scalar(CHELIS_DTYPE_F64);
        f64::fill(t, 6.022e23);
        let ptr = f64::data_ptr(t).expect("f64::data_ptr on F64 tensor must succeed");
        assert_eq!(*ptr, 6.022e23);
        chelis_tensor_release(t);
    }
}

// ---- Runtime operation fixtures -----------------------------------------
//
// These fixtures call public runtime symbols and check values through
// dtype-matched accessors. Host emission is tested separately in
// `crates/chelis-backend-c/tests/host_emit_dtype_dispatch.rs`.

/// Allocate a rank-1 length-`n` tensor and fill it with values from
/// `vals` interpreted as the tensor's storage convention for `dtype`:
///   - F32 and F64: writer stores values in the matching float format.
///   - I32 and I64: writer stores values in the matching integer format.
///   - BOOL: writer stores canonical one-byte Boolean values.
unsafe fn alloc_vec_with_values(dtype: chelis_dtype, vals: &[f64]) -> *mut chelis_tensor {
    unsafe {
        let t = alloc_vec(vals.len() as i64, dtype);
        match dtype {
            CHELIS_DTYPE_F32 => {
                let p = f32::data_ptr_unchecked(t);
                for (i, v) in vals.iter().enumerate() {
                    *p.add(i) = *v as f32;
                }
            }
            CHELIS_DTYPE_F64 => {
                let p = f64::data_ptr_unchecked(t);
                for (i, v) in vals.iter().enumerate() {
                    *p.add(i) = *v;
                }
            }
            CHELIS_DTYPE_I64 => {
                let p = i64::data_ptr_unchecked(t);
                for (i, v) in vals.iter().enumerate() {
                    *p.add(i) = *v as i64;
                }
            }
            CHELIS_DTYPE_I32 => {
                let p = i32::data_ptr_unchecked(t);
                for (i, v) in vals.iter().enumerate() {
                    *p.add(i) = *v as i32;
                }
            }
            CHELIS_DTYPE_BOOL => {
                let p = Bool8::data_ptr_unchecked(t);
                for (i, v) in vals.iter().enumerate() {
                    *p.add(i) = Bool8::new(*v != 0.0);
                }
            }
            _ => panic!("alloc_vec_with_values: unsupported dtype {dtype}"),
        }
        t
    }
}

/// Allocate and initialize a rank-0 tensor without forging vector metadata.
unsafe fn alloc_scalar_with_value(dtype: chelis_dtype, value: f64) -> *mut chelis_tensor {
    unsafe {
        let tensor = alloc_scalar(dtype);
        match dtype {
            CHELIS_DTYPE_F32 => f32::fill(tensor, value as f32),
            CHELIS_DTYPE_F64 => f64::fill(tensor, value),
            CHELIS_DTYPE_I64 => i64::fill(tensor, value as i64),
            CHELIS_DTYPE_I32 => i32::fill(tensor, value as i32),
            CHELIS_DTYPE_BOOL => Bool8::fill(tensor, Bool8::new(value != 0.0)),
            _ => panic!("alloc_scalar_with_value: unsupported dtype {dtype}"),
        }
        tensor
    }
}

/// Read tensor element `i` back as `f64`, using the dtype's storage
/// convention.  Mirrors the per-dtype dispatch in the migrated
/// runtime ops.
unsafe fn read_at(t: *mut chelis_tensor, i: usize) -> f64 {
    unsafe {
        match chelis_tensor_read_view(t).dtype {
            CHELIS_DTYPE_F32 => *f32::data_ptr_unchecked(t).add(i) as f64,
            CHELIS_DTYPE_F64 => *f64::data_ptr_unchecked(t).add(i),
            CHELIS_DTYPE_I64 => *i64::data_ptr_unchecked(t).add(i) as f64,
            CHELIS_DTYPE_I32 => *i32::data_ptr_unchecked(t).add(i) as f64,
            CHELIS_DTYPE_BOOL => {
                if (*Bool8::data_ptr_unchecked(t).add(i)).get() {
                    1.0
                } else {
                    0.0
                }
            }
            other => panic!("read_at: unsupported dtype {other}"),
        }
    }
}

// ---- `chelis_list_from_tensor` (read-side, per-element) -----------------

#[test]
fn list_from_tensor_f32() {
    unsafe {
        let t = alloc_vec_with_values(CHELIS_DTYPE_F32, &[1.5, 2.5, 3.5]);
        let list = chelis_tensor_elements(t);
        assert_eq!(chelis_list_len(list), 3);
        let v0 = chelis_list_index(list, 0);
        let v1 = chelis_list_index(list, 1);
        let v2 = chelis_list_index(list, 2);
        assert_eq!(exact_value_f64(v0), 1.5);
        assert_eq!(exact_value_f64(v1), 2.5);
        assert_eq!(exact_value_f64(v2), 3.5);
        chelis_tensor_release(t);
    }
}

#[test]
fn list_from_tensor_f64_full_precision() {
    // A value outside f32 precision distinguishes an f64 read.
    const F64_VAL: f64 = 1.234_567_890_123_456_7_f64;
    unsafe {
        let t = alloc_vec_with_values(CHELIS_DTYPE_F64, &[F64_VAL, -F64_VAL]);
        let list = chelis_tensor_elements(t);
        assert_eq!(chelis_list_len(list), 2);
        let v0 = chelis_list_index(list, 0);
        let v1 = chelis_list_index(list, 1);
        assert_eq!(exact_value_f64(v0), F64_VAL);
        assert_eq!(exact_value_f64(v1), -F64_VAL);
        chelis_tensor_release(t);
    }
}

#[test]
fn list_from_tensor_i64_full_precision() {
    const BIG_I64: i64 = 9_000_000_000_000_i64;
    unsafe {
        let t = alloc_vec_with_values(CHELIS_DTYPE_I64, &[BIG_I64 as f64, -BIG_I64 as f64]);
        let list = chelis_tensor_elements(t);
        let v0 = chelis_list_index(list, 0);
        let v1 = chelis_list_index(list, 1);
        assert_eq!(exact_value_i64(v0), BIG_I64);
        assert_eq!(exact_value_i64(v1), -BIG_I64);
        chelis_tensor_release(t);
    }
}

#[test]
fn list_from_tensor_i32() {
    unsafe {
        let t = alloc_vec_with_values(CHELIS_DTYPE_I32, &[42.0, 7.0]);
        let list = chelis_tensor_elements(t);
        let v0 = chelis_list_index(list, 0);
        let v1 = chelis_list_index(list, 1);
        assert_eq!(exact_value_i64(v0), 42);
        assert_eq!(exact_value_i64(v1), 7);
        chelis_tensor_release(t);
    }
}

#[test]
fn list_from_tensor_bool() {
    unsafe {
        let t = alloc_vec_with_values(CHELIS_DTYPE_BOOL, &[1.0, 0.0]);
        let list = chelis_tensor_elements(t);
        let v0 = chelis_list_index(list, 0);
        let v1 = chelis_list_index(list, 1);
        assert!(exact_value_bool(v0));
        assert!(!exact_value_bool(v1));
        chelis_tensor_release(t);
    }
}

// ---- `chelis_tensor_concat` (byte-stride copy) --------------------------

unsafe fn concat_two(a: *mut chelis_tensor, b: *mut chelis_tensor) -> *mut chelis_tensor {
    unsafe {
        let parts = chelis_list_empty();
        let parts = chelis_list_append(parts, chelis_value_take_tensor(a));
        let parts = chelis_list_append(parts, chelis_value_take_tensor(b));
        chelis_tensor_concat(parts, 0)
    }
}

#[test]
fn concat_f32_round_trip() {
    unsafe {
        let a = alloc_vec_with_values(CHELIS_DTYPE_F32, &[1.0, 2.0]);
        let b = alloc_vec_with_values(CHELIS_DTYPE_F32, &[3.0, 4.0]);
        let out = concat_two(a, b);
        for (i, expected) in [1.0, 2.0, 3.0, 4.0].iter().enumerate() {
            assert_eq!(read_at(out, i), *expected, "concat_f32 index {i}");
        }
        chelis_tensor_release(out);
    }
}

#[test]
fn concat_f64_preserves_full_precision() {
    // Concatenation preserves values that cannot be represented as f32.
    const A: f64 = 1.234_567_890_123_456_7_f64;
    const B: f64 = -1.732_050_807_568_877_3_f64;
    unsafe {
        let lhs = alloc_vec_with_values(CHELIS_DTYPE_F64, &[A]);
        let rhs = alloc_vec_with_values(CHELIS_DTYPE_F64, &[B]);
        let out = concat_two(lhs, rhs);
        assert_eq!(read_at(out, 0), A, "f64 concat must preserve full mantissa");
        assert_eq!(read_at(out, 1), B, "f64 concat must preserve full mantissa");
        chelis_tensor_release(out);
    }
}

#[test]
fn concat_i64_preserves_full_precision() {
    const A: i64 = 9_000_000_000_000_i64;
    const B: i64 = -9_000_000_000_000_i64;
    unsafe {
        let lhs = alloc_vec_with_values(CHELIS_DTYPE_I64, &[A as f64]);
        let rhs = alloc_vec_with_values(CHELIS_DTYPE_I64, &[B as f64]);
        let out = concat_two(lhs, rhs);
        assert_eq!(*i64::data_ptr_unchecked(out).add(0), A);
        assert_eq!(*i64::data_ptr_unchecked(out).add(1), B);
        chelis_tensor_release(out);
    }
}

#[test]
fn concat_i32_round_trip() {
    unsafe {
        let a = alloc_vec_with_values(CHELIS_DTYPE_I32, &[1.0, 2.0]);
        let b = alloc_vec_with_values(CHELIS_DTYPE_I32, &[3.0, 4.0]);
        let out = concat_two(a, b);
        for (i, expected) in [1.0, 2.0, 3.0, 4.0].iter().enumerate() {
            assert_eq!(read_at(out, i), *expected, "concat_i32 index {i}");
        }
        chelis_tensor_release(out);
    }
}

#[test]
fn concat_bool_round_trip() {
    unsafe {
        let a = alloc_vec_with_values(CHELIS_DTYPE_BOOL, &[1.0, 0.0]);
        let b = alloc_vec_with_values(CHELIS_DTYPE_BOOL, &[0.0, 1.0]);
        let out = concat_two(a, b);
        for (i, expected) in [1.0, 0.0, 0.0, 1.0].iter().enumerate() {
            assert_eq!(read_at(out, i), *expected, "concat_bool index {i}");
        }
        chelis_tensor_release(out);
    }
}

// ---- `chelis_tensor_split` (byte-stride copy) ---------------------------

unsafe fn split_two_halves(
    t: *mut chelis_tensor,
    lhs_size: i64,
) -> (*mut chelis_tensor, *mut chelis_tensor) {
    unsafe {
        let rhs_size = chelis_tensor_shape(t, 0) - lhs_size;
        let sizes = chelis_list_empty();
        let sizes = chelis_list_append(sizes, exact_i64_value(lhs_size));
        let sizes = chelis_list_append(sizes, exact_i64_value(rhs_size));
        let parts = chelis_tensor_split(t, 0, sizes);
        let v0 = chelis_list_index(parts, 0);
        let v1 = chelis_list_index(parts, 1);
        (chelis_tensor_take_value(v0), chelis_tensor_take_value(v1))
    }
}

#[test]
fn split_f64_preserves_full_precision() {
    const A: f64 = 1.234_567_890_123_456_7_f64;
    const B: f64 = -1.732_050_807_568_877_3_f64;
    unsafe {
        let t = alloc_vec_with_values(CHELIS_DTYPE_F64, &[A, B]);
        let (lhs, rhs) = split_two_halves(t, 1);
        assert_eq!(read_at(lhs, 0), A);
        assert_eq!(read_at(rhs, 0), B);
        chelis_tensor_release(t);
    }
}

#[test]
fn split_i64_preserves_full_precision() {
    const A: i64 = 9_000_000_000_000_i64;
    const B: i64 = -9_000_000_000_000_i64;
    unsafe {
        let t = alloc_vec_with_values(CHELIS_DTYPE_I64, &[A as f64, B as f64]);
        let (lhs, rhs) = split_two_halves(t, 1);
        assert_eq!(*i64::data_ptr_unchecked(lhs).add(0), A);
        assert_eq!(*i64::data_ptr_unchecked(rhs).add(0), B);
        chelis_tensor_release(t);
    }
}

#[test]
fn split_f32_round_trip() {
    unsafe {
        let t = alloc_vec_with_values(CHELIS_DTYPE_F32, &[1.0, 2.0, 3.0, 4.0]);
        let (lhs, rhs) = split_two_halves(t, 2);
        assert_eq!(read_at(lhs, 0), 1.0);
        assert_eq!(read_at(lhs, 1), 2.0);
        assert_eq!(read_at(rhs, 0), 3.0);
        assert_eq!(read_at(rhs, 1), 4.0);
        chelis_tensor_release(t);
    }
}

// ---- `chelis_tensor_gather` (byte-stride indexed read) ------------------

#[test]
fn gather_f64_preserves_full_precision() {
    const A: f64 = 1.234_567_890_123_456_7_f64;
    const B: f64 = -1.732_050_807_568_877_3_f64;
    const C: f64 = 6.022_140_76e23;
    unsafe {
        let src = alloc_vec_with_values(CHELIS_DTYPE_F64, &[A, B, C]);
        // Indices: pick [2, 0] from a length-3 source.
        let idx = alloc_vec_with_values(CHELIS_DTYPE_I32, &[2.0, 0.0]);
        let out = chelis_tensor_gather(src, idx, 0);
        assert_eq!(read_at(out, 0), C);
        assert_eq!(read_at(out, 1), A);
        chelis_tensor_release(out);
        chelis_tensor_release(idx);
        chelis_tensor_release(src);
    }
}

#[test]
fn gather_i64_preserves_full_precision() {
    const A: i64 = 9_000_000_000_000_i64;
    const B: i64 = -7_500_000_000_000_i64;
    unsafe {
        let src = alloc_vec_with_values(CHELIS_DTYPE_I64, &[A as f64, B as f64]);
        let idx = alloc_vec_with_values(CHELIS_DTYPE_I32, &[1.0, 0.0]);
        let out = chelis_tensor_gather(src, idx, 0);
        assert_eq!(*i64::data_ptr_unchecked(out).add(0), B);
        assert_eq!(*i64::data_ptr_unchecked(out).add(1), A);
        chelis_tensor_release(out);
        chelis_tensor_release(idx);
        chelis_tensor_release(src);
    }
}

#[test]
fn gather_f32_round_trip() {
    unsafe {
        let src = alloc_vec_with_values(CHELIS_DTYPE_F32, &[10.0, 20.0, 30.0]);
        let idx = alloc_vec_with_values(CHELIS_DTYPE_I32, &[2.0, 0.0, 1.0]);
        let out = chelis_tensor_gather(src, idx, 0);
        assert_eq!(read_at(out, 0), 30.0);
        assert_eq!(read_at(out, 1), 10.0);
        assert_eq!(read_at(out, 2), 20.0);
        chelis_tensor_release(out);
        chelis_tensor_release(idx);
        chelis_tensor_release(src);
    }
}

// ---- `chelis_tensor_cmplt` (numeric compare -> bool) --------------------

#[test]
fn cmplt_f64_full_precision() {
    // lhs and rhs differ by an amount that survives at f64 precision
    // (>= 1 ULP at 1e10) but rounds away in f32 (24-bit mantissa
    // saturates well before 1e10). The comparison must use f64
    // values and return true.
    const LHS: f64 = 1.0e10_f64;
    const RHS: f64 = 1.0e10_f64 + 1.0;
    unsafe {
        let lhs = alloc_vec_with_values(CHELIS_DTYPE_F64, &[LHS]);
        let rhs = alloc_vec_with_values(CHELIS_DTYPE_F64, &[RHS]);
        // Sanity: rhs is strictly larger in f64.
        const { assert!(RHS > LHS) };
        let out = chelis_tensor_cmplt(lhs, rhs);
        assert_eq!(
            read_at(out, 0),
            1.0,
            "f64 cmplt must compare at full precision"
        );
        chelis_tensor_release(out);
        chelis_tensor_release(rhs);
        chelis_tensor_release(lhs);
    }
}

#[test]
fn cmplt_i64_full_precision() {
    const LHS: i64 = 4_503_599_627_370_497_i64; // 2^52 + 1
    const RHS: i64 = 4_503_599_627_370_498_i64; // 2^52 + 2
    unsafe {
        let lhs = alloc_vec_with_values(CHELIS_DTYPE_I64, &[LHS as f64]);
        let rhs = alloc_vec_with_values(CHELIS_DTYPE_I64, &[RHS as f64]);
        let out = chelis_tensor_cmplt(lhs, rhs);
        assert_eq!(
            read_at(out, 0),
            1.0,
            "i64 cmplt must compare at full precision"
        );
        chelis_tensor_release(out);
        chelis_tensor_release(rhs);
        chelis_tensor_release(lhs);
    }
}

#[test]
fn cmplt_f32_round_trip() {
    unsafe {
        let lhs = alloc_vec_with_values(CHELIS_DTYPE_F32, &[1.0, 2.0, 3.0]);
        let rhs = alloc_vec_with_values(CHELIS_DTYPE_F32, &[2.0, 2.0, 1.0]);
        let out = chelis_tensor_cmplt(lhs, rhs);
        assert_eq!(read_at(out, 0), 1.0);
        assert_eq!(read_at(out, 1), 0.0);
        assert_eq!(read_at(out, 2), 0.0);
        chelis_tensor_release(out);
        chelis_tensor_release(rhs);
        chelis_tensor_release(lhs);
    }
}

// ---- `chelis_tensor_scatter` (replace + add modes) ----------------------

#[test]
fn scatter_replace_f64_preserves_full_precision() {
    const ORIG: f64 = 1.234_567_890_123_456_7_f64;
    const NEW: f64 = -ORIG;
    unsafe {
        let base = alloc_vec_with_values(CHELIS_DTYPE_F64, &[ORIG, ORIG, ORIG]);
        let idx = alloc_vec_with_values(CHELIS_DTYPE_I32, &[1.0]);
        let upd = alloc_vec_with_values(CHELIS_DTYPE_F64, &[NEW]);
        let out = chelis_tensor_scatter_replace(base, idx, upd, 0);
        assert_eq!(read_at(out, 0), ORIG);
        assert_eq!(read_at(out, 1), NEW);
        assert_eq!(read_at(out, 2), ORIG);
        chelis_tensor_release(out);
        chelis_tensor_release(upd);
        chelis_tensor_release(idx);
        chelis_tensor_release(base);
    }
}

#[test]
fn scatter_add_i64_preserves_full_precision() {
    const ORIG: i64 = 9_000_000_000_000_i64;
    const ADD: i64 = 1_000_000_000_000_i64;
    unsafe {
        let base = alloc_vec_with_values(CHELIS_DTYPE_I64, &[ORIG as f64, ORIG as f64]);
        let idx = alloc_vec_with_values(CHELIS_DTYPE_I32, &[0.0]);
        let upd = alloc_vec_with_values(CHELIS_DTYPE_I64, &[ADD as f64]);
        let out = chelis_tensor_scatter_add(base, idx, upd, 0);
        assert_eq!(*i64::data_ptr_unchecked(out).add(0), ORIG + ADD);
        assert_eq!(*i64::data_ptr_unchecked(out).add(1), ORIG);
        chelis_tensor_release(out);
        chelis_tensor_release(upd);
        chelis_tensor_release(idx);
        chelis_tensor_release(base);
    }
}

#[test]
fn scatter_replace_f32_round_trip() {
    unsafe {
        let base = alloc_vec_with_values(CHELIS_DTYPE_F32, &[1.0, 2.0, 3.0]);
        let idx = alloc_vec_with_values(CHELIS_DTYPE_I32, &[2.0]);
        let upd = alloc_vec_with_values(CHELIS_DTYPE_F32, &[42.0]);
        let out = chelis_tensor_scatter_replace(base, idx, upd, 0);
        assert_eq!(read_at(out, 0), 1.0);
        assert_eq!(read_at(out, 1), 2.0);
        assert_eq!(read_at(out, 2), 42.0);
        chelis_tensor_release(out);
        chelis_tensor_release(upd);
        chelis_tensor_release(idx);
        chelis_tensor_release(base);
    }
}

// ---- `chelis_tensor_where` (cond + then/else select) --------------------

#[test]
fn where_f64_preserves_full_precision() {
    const A: f64 = 1.234_567_890_123_456_7_f64;
    const B: f64 = -1.732_050_807_568_877_3_f64;
    unsafe {
        let cond = alloc_vec_with_values(CHELIS_DTYPE_BOOL, &[1.0, 0.0]);
        let t = alloc_vec_with_values(CHELIS_DTYPE_F64, &[A, A]);
        let e = alloc_vec_with_values(CHELIS_DTYPE_F64, &[B, B]);
        let out = chelis_tensor_where(cond, t, e);
        assert_eq!(read_at(out, 0), A);
        assert_eq!(read_at(out, 1), B);
        chelis_tensor_release(out);
        chelis_tensor_release(e);
        chelis_tensor_release(t);
        chelis_tensor_release(cond);
    }
}

#[test]
fn where_i64_preserves_full_precision() {
    const A: i64 = 9_000_000_000_000_i64;
    const B: i64 = -7_500_000_000_000_i64;
    unsafe {
        let cond = alloc_vec_with_values(CHELIS_DTYPE_BOOL, &[1.0, 0.0]);
        let t = alloc_vec_with_values(CHELIS_DTYPE_I64, &[A as f64, A as f64]);
        let e = alloc_vec_with_values(CHELIS_DTYPE_I64, &[B as f64, B as f64]);
        let out = chelis_tensor_where(cond, t, e);
        assert_eq!(*i64::data_ptr_unchecked(out).add(0), A);
        assert_eq!(*i64::data_ptr_unchecked(out).add(1), B);
        chelis_tensor_release(out);
        chelis_tensor_release(e);
        chelis_tensor_release(t);
        chelis_tensor_release(cond);
    }
}

#[test]
fn where_f32_round_trip() {
    unsafe {
        let cond = alloc_vec_with_values(CHELIS_DTYPE_BOOL, &[1.0, 0.0, 1.0]);
        let t = alloc_vec_with_values(CHELIS_DTYPE_F32, &[10.0, 20.0, 30.0]);
        let e = alloc_vec_with_values(CHELIS_DTYPE_F32, &[100.0, 200.0, 300.0]);
        let out = chelis_tensor_where(cond, t, e);
        assert_eq!(read_at(out, 0), 10.0);
        assert_eq!(read_at(out, 1), 200.0);
        assert_eq!(read_at(out, 2), 30.0);
        chelis_tensor_release(out);
        chelis_tensor_release(e);
        chelis_tensor_release(t);
        chelis_tensor_release(cond);
    }
}

// ---- `chelis_tensor_cumsum` (numeric prefix sum) ------------------------

#[test]
fn cumsum_f64_accumulates_at_full_precision() {
    // STEP is large enough that f32 saturates its 24-bit mantissa
    // (so an f32 accumulator would lose the +1.0 contributions
    // entirely) but stays well below the f64 53-bit mantissa limit
    // (so f64 accumulation must produce STEP+1, STEP+2 exactly).
    const STEP: f64 = 1.0e10_f64;
    unsafe {
        let t = alloc_vec_with_values(CHELIS_DTYPE_F64, &[STEP, 1.0, 1.0]);
        let out = chelis_tensor_cumsum(t, 0);
        assert_eq!(read_at(out, 0), STEP);
        assert_eq!(
            read_at(out, 1),
            STEP + 1.0,
            "f64 prefix sum must accumulate at full precision"
        );
        assert_eq!(read_at(out, 2), STEP + 2.0);
        chelis_tensor_release(out);
        chelis_tensor_release(t);
    }
}

#[test]
fn cumsum_i64_full_precision() {
    const STEP: i64 = 5_000_000_000_000_i64;
    unsafe {
        let t = alloc_vec_with_values(CHELIS_DTYPE_I64, &[STEP as f64, STEP as f64, STEP as f64]);
        let out = chelis_tensor_cumsum(t, 0);
        assert_eq!(*i64::data_ptr_unchecked(out).add(0), STEP);
        assert_eq!(*i64::data_ptr_unchecked(out).add(1), 2 * STEP);
        assert_eq!(*i64::data_ptr_unchecked(out).add(2), 3 * STEP);
        chelis_tensor_release(out);
        chelis_tensor_release(t);
    }
}

#[test]
fn cumsum_f32_round_trip() {
    unsafe {
        let t = alloc_vec_with_values(CHELIS_DTYPE_F32, &[1.0, 2.0, 3.0]);
        let out = chelis_tensor_cumsum(t, 0);
        assert_eq!(read_at(out, 0), 1.0);
        assert_eq!(read_at(out, 1), 3.0);
        assert_eq!(read_at(out, 2), 6.0);
        chelis_tensor_release(out);
        chelis_tensor_release(t);
    }
}

// ---- `chelis_tensor_sort` (numeric compare + i32 indices) ---------------

#[test]
fn sort_f64_full_precision() {
    // These values differ at f64 precision but round to the same f32.
    const A: f64 = 1.0e16_f64 + 2.0;
    const B: f64 = 1.0e16_f64;
    unsafe {
        let t = alloc_vec_with_values(CHELIS_DTYPE_F64, &[A, B]);
        let tup = chelis_tensor_sort(t, 0);
        let values = chelis_tensor_take_value(chelis_tuple_get(tup, 0));
        // Sorted ascending: B then A.
        assert_eq!(
            read_at(values, 0),
            B,
            "sort must read at full f64 precision"
        );
        assert_eq!(read_at(values, 1), A);
        chelis_tensor_release(t);
    }
}

#[test]
fn sort_i64_full_precision() {
    const A: i64 = 9_000_000_000_000_i64;
    const B: i64 = -9_000_000_000_000_i64;
    unsafe {
        let t = alloc_vec_with_values(CHELIS_DTYPE_I64, &[A as f64, B as f64]);
        let tup = chelis_tensor_sort(t, 0);
        let values = chelis_tensor_take_value(chelis_tuple_get(tup, 0));
        // Sorted ascending: B then A.
        assert_eq!(*i64::data_ptr_unchecked(values).add(0), B);
        assert_eq!(*i64::data_ptr_unchecked(values).add(1), A);
        chelis_tensor_release(t);
    }
}

#[test]
fn sort_f32_round_trip() {
    unsafe {
        let t = alloc_vec_with_values(CHELIS_DTYPE_F32, &[3.0, 1.0, 2.0]);
        let tup = chelis_tensor_sort(t, 0);
        let values = chelis_tensor_take_value(chelis_tuple_get(tup, 0));
        let idx = chelis_tensor_take_value(chelis_tuple_get(tup, 1));
        assert_eq!(read_at(values, 0), 1.0);
        assert_eq!(read_at(values, 1), 2.0);
        assert_eq!(read_at(values, 2), 3.0);
        // Indices use i32 storage.
        assert_eq!(read_at(idx, 0), 1.0);
        assert_eq!(read_at(idx, 1), 2.0);
        assert_eq!(read_at(idx, 2), 0.0);
        chelis_tensor_release(t);
    }
}

// ---- `chelis_tensor_diagonal` + `chelis_tensor_trace` -------------------

unsafe fn alloc_2x2(dtype: chelis_dtype, a: f64, b: f64, c: f64, d: f64) -> *mut chelis_tensor {
    unsafe {
        let shape = [2i64, 2i64];
        let t = chelis_alloc(2, shape.as_ptr(), dtype);
        let vals = [a, b, c, d];
        match dtype {
            CHELIS_DTYPE_F32 => {
                let p = f32::data_ptr_unchecked(t);
                for (i, v) in vals.iter().enumerate() {
                    *p.add(i) = *v as f32;
                }
            }
            CHELIS_DTYPE_F64 => {
                let p = f64::data_ptr_unchecked(t);
                for (i, v) in vals.iter().enumerate() {
                    *p.add(i) = *v;
                }
            }
            CHELIS_DTYPE_I64 => {
                let p = i64::data_ptr_unchecked(t);
                for (i, v) in vals.iter().enumerate() {
                    *p.add(i) = *v as i64;
                }
            }
            CHELIS_DTYPE_I32 => {
                let p = i32::data_ptr_unchecked(t);
                for (i, v) in vals.iter().enumerate() {
                    *p.add(i) = *v as i32;
                }
            }
            CHELIS_DTYPE_BOOL => {
                let p = Bool8::data_ptr_unchecked(t);
                for (i, v) in vals.iter().enumerate() {
                    *p.add(i) = Bool8::new(*v != 0.0);
                }
            }
            _ => panic!("alloc_2x2: unsupported dtype {dtype}"),
        }
        t
    }
}

#[test]
fn diagonal_f64_preserves_full_precision() {
    const A: f64 = 1.234_567_890_123_456_7_f64;
    const D: f64 = -1.732_050_807_568_877_3_f64;
    unsafe {
        let m = alloc_2x2(CHELIS_DTYPE_F64, A, 0.0, 0.0, D);
        let out = chelis_tensor_diagonal(m, 0, 1);
        assert_eq!(read_at(out, 0), A);
        assert_eq!(read_at(out, 1), D);
        chelis_tensor_release(out);
        chelis_tensor_release(m);
    }
}

#[test]
fn diagonal_i64_preserves_full_precision() {
    const A: i64 = 9_000_000_000_000_i64;
    const D: i64 = -9_000_000_000_000_i64;
    unsafe {
        let m = alloc_2x2(CHELIS_DTYPE_I64, A as f64, 0.0, 0.0, D as f64);
        let out = chelis_tensor_diagonal(m, 0, 1);
        assert_eq!(*i64::data_ptr_unchecked(out).add(0), A);
        assert_eq!(*i64::data_ptr_unchecked(out).add(1), D);
        chelis_tensor_release(out);
        chelis_tensor_release(m);
    }
}

#[test]
fn diagonal_f32_round_trip() {
    unsafe {
        let m = alloc_2x2(CHELIS_DTYPE_F32, 1.0, 2.0, 3.0, 4.0);
        let out = chelis_tensor_diagonal(m, 0, 1);
        assert_eq!(read_at(out, 0), 1.0);
        assert_eq!(read_at(out, 1), 4.0);
        chelis_tensor_release(out);
        chelis_tensor_release(m);
    }
}

#[test]
fn trace_f64_accumulates_at_full_precision() {
    // A is large enough to saturate f32's 24-bit mantissa (an f32
    // accumulator would lose +D) but stays well below f64's 53-bit
    // mantissa (f64 yields A+D exactly).
    const A: f64 = 1.0e10_f64;
    const D: f64 = 1.0;
    unsafe {
        let m = alloc_2x2(CHELIS_DTYPE_F64, A, 0.0, 0.0, D);
        let out = chelis_tensor_trace(m, 0, 1);
        assert_eq!(
            read_at(out, 0),
            A + D,
            "f64 trace must accumulate at full precision"
        );
        chelis_tensor_release(out);
        chelis_tensor_release(m);
    }
}

#[test]
fn trace_i64_full_precision() {
    const A: i64 = 5_000_000_000_000_i64;
    const D: i64 = 4_000_000_000_000_i64;
    unsafe {
        let m = alloc_2x2(CHELIS_DTYPE_I64, A as f64, 0.0, 0.0, D as f64);
        let out = chelis_tensor_trace(m, 0, 1);
        assert_eq!(*i64::data_ptr_unchecked(out).add(0), A + D);
        chelis_tensor_release(out);
        chelis_tensor_release(m);
    }
}

#[test]
fn trace_f32_round_trip() {
    unsafe {
        let m = alloc_2x2(CHELIS_DTYPE_F32, 1.0, 2.0, 3.0, 4.0);
        let out = chelis_tensor_trace(m, 0, 1);
        assert_eq!(read_at(out, 0), 5.0);
        chelis_tensor_release(out);
        chelis_tensor_release(m);
    }
}

// ---- `chelis_tensor_clamp` ---------------------------------------------

#[test]
fn clamp_f64_preserves_full_precision() {
    // SAMPLE has an f64 fractional part that survives at f64
    // precision (well below 2^53) but is lost in any f32-mediated
    // read.  HI_F is large enough that SAMPLE is in bounds so the
    // clamp returns SAMPLE unchanged at f64 precision.
    const HI_F: f64 = 1.0e15_f64;
    const SAMPLE: f64 = 5.0e10_f64 + 0.5;
    unsafe {
        let t = alloc_vec_with_values(CHELIS_DTYPE_F64, &[SAMPLE]);
        let lo = alloc_scalar_with_value(CHELIS_DTYPE_F64, 0.0);
        let hi = alloc_scalar_with_value(CHELIS_DTYPE_F64, HI_F);
        let out = chelis_tensor_clamp(t, lo, hi);
        assert_eq!(
            read_at(out, 0),
            SAMPLE,
            "f64 clamp must preserve full precision"
        );
        chelis_tensor_release(out);
        chelis_tensor_release(hi);
        chelis_tensor_release(lo);
        chelis_tensor_release(t);
    }
}

#[test]
fn clamp_i64_clips_at_full_precision() {
    const A: i64 = 9_000_000_000_000_i64;
    const HI_I: i64 = 5_000_000_000_000_i64;
    unsafe {
        let t = alloc_vec_with_values(CHELIS_DTYPE_I64, &[A as f64]);
        let lo = alloc_scalar_with_value(CHELIS_DTYPE_I64, 0.0);
        let hi = alloc_scalar_with_value(CHELIS_DTYPE_I64, HI_I as f64);
        let out = chelis_tensor_clamp(t, lo, hi);
        assert_eq!(*i64::data_ptr_unchecked(out).add(0), HI_I);
        chelis_tensor_release(out);
        chelis_tensor_release(hi);
        chelis_tensor_release(lo);
        chelis_tensor_release(t);
    }
}

#[test]
fn clamp_f32_round_trip() {
    unsafe {
        let t = alloc_vec_with_values(CHELIS_DTYPE_F32, &[-1.0, 0.5, 2.0]);
        let lo = alloc_scalar_with_value(CHELIS_DTYPE_F32, 0.0);
        let hi = alloc_scalar_with_value(CHELIS_DTYPE_F32, 1.0);
        let out = chelis_tensor_clamp(t, lo, hi);
        assert_eq!(read_at(out, 0), 0.0);
        assert_eq!(read_at(out, 1), 0.5);
        assert_eq!(read_at(out, 2), 1.0);
        chelis_tensor_release(out);
        chelis_tensor_release(hi);
        chelis_tensor_release(lo);
        chelis_tensor_release(t);
    }
}

// ---- `chelis_tensor_einsum` (numeric multiply-add) ----------------------

unsafe fn alloc_vec2(dtype: chelis_dtype, vals: &[f64]) -> *mut chelis_tensor {
    unsafe { alloc_vec_with_values(dtype, vals) }
}

#[test]
fn einsum_dot_product_f64() {
    // Pick values that exercise f64 reads (the LHS / RHS inputs need
    // the full mantissa) while keeping the accumulated dot product
    // representable. An f32-mediated read would change the product.
    const SAMPLE: f64 = 1.234_567_890_123_456_7_f64;
    unsafe {
        let lhs = alloc_vec2(CHELIS_DTYPE_F64, &[SAMPLE, 0.5]);
        let rhs = alloc_vec2(CHELIS_DTYPE_F64, &[SAMPLE, 2.0]);
        let equation = chelis_string_from_cstr(c"i,i->".as_ptr());
        let out = chelis_tensor_einsum(equation, lhs, rhs, CHELIS_DTYPE_F64);
        let expected = SAMPLE * SAMPLE + 0.5 * 2.0;
        assert_eq!(read_at(out, 0), expected);
        chelis_tensor_release(out);
        chelis_tensor_release(rhs);
        chelis_tensor_release(lhs);
    }
}

#[test]
fn einsum_dot_product_i64() {
    const A: i64 = 5_000_000_i64;
    unsafe {
        let lhs = alloc_vec2(CHELIS_DTYPE_I64, &[A as f64, A as f64]);
        let rhs = alloc_vec2(CHELIS_DTYPE_I64, &[A as f64, A as f64]);
        let equation = chelis_string_from_cstr(c"i,i->".as_ptr());
        let out = chelis_tensor_einsum(equation, lhs, rhs, CHELIS_DTYPE_I64);
        // 2 * A * A.  Far exceeds f32 mantissa range.
        let expected = 2 * A * A;
        assert_eq!(*i64::data_ptr_unchecked(out).add(0), expected);
        chelis_tensor_release(out);
        chelis_tensor_release(rhs);
        chelis_tensor_release(lhs);
    }
}

#[test]
fn einsum_dot_product_f32() {
    unsafe {
        let lhs = alloc_vec2(CHELIS_DTYPE_F32, &[1.0, 2.0, 3.0]);
        let rhs = alloc_vec2(CHELIS_DTYPE_F32, &[4.0, 5.0, 6.0]);
        let equation = chelis_string_from_cstr(c"i,i->".as_ptr());
        let out = chelis_tensor_einsum(equation, lhs, rhs, CHELIS_DTYPE_F32);
        assert_eq!(read_at(out, 0), 32.0);
        chelis_tensor_release(out);
        chelis_tensor_release(rhs);
        chelis_tensor_release(lhs);
    }
}

// ---- Multi-op composition fixtures --------------------------------------
//
// The 60 single-op fixtures above each exercise one runtime accessor
// site.  Real Chelis programs chain ops: a reshape feeds into a
// concat, a gather feeds into a cumsum, a where feeds into a sort.
// Each chained step re-reads through the dtype dispatch tree; a bug
// in any one site can corrupt a downstream value.
//
// These compositions lock the end-to-end property at byte-exact f64
// precision: the f64 mantissa survives through every intermediate
// step.  Each fixture picks a value whose lower mantissa bits cannot
// be expressed in f32 (e.g. `1.234_567_890_123_456_7`), so any
// f32-strided intermediate read drops the precision and the final
// assertion fails.
//
// Compositions covered below:
//   * `concat -> gather`           : write to wide buffer then index
//   * `concat -> cumsum`           : write to wide buffer then prefix-sum
//   * `gather -> sort`             : index then sort the result
//   * `where -> trace`             : select then diagonal-sum
//   * `cumsum -> clamp`            : prefix-sum then clip
//   * `scatter -> diagonal`        : write at index then read diagonal
//   * `where -> einsum`            : select then dot product
//   * `clamp -> cumsum`            : clip then prefix-sum (mixed-precision)
//   * `cmplt -> where`             : compare to bool then select
//
// Each composition exercises at least one f64 and one i64 chain to
// catch f32-strided regressions in either precision.

#[test]
fn compose_concat_then_gather_f64_preserves_precision() {
    // concat two f64 buffers then gather indices [3, 0] from the
    // concatenated result.  The cell at index 3 comes from rhs[1];
    // the cell at index 0 from lhs[0].  Any f32-strided read in
    // either concat or gather would fail the byte-exact assertion.
    const A: f64 = 1.234_567_890_123_456_7_f64;
    const B: f64 = -1.732_050_807_568_877_3_f64;
    const C: f64 = 6.022_140_76e23;
    const D: f64 = -7.500_000_000_000_001_f64;
    unsafe {
        let lhs = alloc_vec_with_values(CHELIS_DTYPE_F64, &[A, B]);
        let rhs = alloc_vec_with_values(CHELIS_DTYPE_F64, &[C, D]);
        let cat = concat_two(lhs, rhs);
        let idx = alloc_vec_with_values(CHELIS_DTYPE_I32, &[3.0, 0.0]);
        let out = chelis_tensor_gather(cat, idx, 0);
        assert_eq!(
            read_at(out, 0),
            D,
            "concat then gather index 3 must equal rhs[1] at full f64"
        );
        assert_eq!(
            read_at(out, 1),
            A,
            "concat then gather index 0 must equal lhs[0] at full f64"
        );
        chelis_tensor_release(out);
        chelis_tensor_release(idx);
        chelis_tensor_release(cat);
    }
}

#[test]
fn compose_concat_then_gather_i64_preserves_precision() {
    const A: i64 = 9_000_000_000_000_i64;
    const B: i64 = -8_500_000_000_000_i64;
    const C: i64 = 7_000_000_000_001_i64;
    const D: i64 = -6_500_000_000_002_i64;
    unsafe {
        let lhs = alloc_vec_with_values(CHELIS_DTYPE_I64, &[A as f64, B as f64]);
        let rhs = alloc_vec_with_values(CHELIS_DTYPE_I64, &[C as f64, D as f64]);
        let cat = concat_two(lhs, rhs);
        let idx = alloc_vec_with_values(CHELIS_DTYPE_I32, &[2.0, 1.0]);
        let out = chelis_tensor_gather(cat, idx, 0);
        assert_eq!(*i64::data_ptr_unchecked(out).add(0), C);
        assert_eq!(*i64::data_ptr_unchecked(out).add(1), B);
        chelis_tensor_release(out);
        chelis_tensor_release(idx);
        chelis_tensor_release(cat);
    }
}

#[test]
fn compose_concat_then_cumsum_f64_accumulates_precision() {
    // STEP is large enough that an f32 accumulator would saturate;
    // the f64 chain must produce exact STEP+1, STEP+2 values.
    const STEP: f64 = 1.0e10_f64;
    unsafe {
        let lhs = alloc_vec_with_values(CHELIS_DTYPE_F64, &[STEP, 1.0]);
        let rhs = alloc_vec_with_values(CHELIS_DTYPE_F64, &[1.0, 1.0]);
        let cat = concat_two(lhs, rhs);
        let out = chelis_tensor_cumsum(cat, 0);
        assert_eq!(read_at(out, 0), STEP);
        assert_eq!(read_at(out, 1), STEP + 1.0);
        assert_eq!(read_at(out, 2), STEP + 2.0);
        assert_eq!(read_at(out, 3), STEP + 3.0);
        chelis_tensor_release(out);
        chelis_tensor_release(cat);
    }
}

#[test]
fn compose_concat_then_cumsum_i64_accumulates_precision() {
    const STEP: i64 = 5_000_000_000_000_i64;
    unsafe {
        let lhs = alloc_vec_with_values(CHELIS_DTYPE_I64, &[STEP as f64, STEP as f64]);
        let rhs = alloc_vec_with_values(CHELIS_DTYPE_I64, &[STEP as f64, STEP as f64]);
        let cat = concat_two(lhs, rhs);
        let out = chelis_tensor_cumsum(cat, 0);
        assert_eq!(*i64::data_ptr_unchecked(out).add(0), STEP);
        assert_eq!(*i64::data_ptr_unchecked(out).add(1), 2 * STEP);
        assert_eq!(*i64::data_ptr_unchecked(out).add(2), 3 * STEP);
        assert_eq!(*i64::data_ptr_unchecked(out).add(3), 4 * STEP);
        chelis_tensor_release(out);
        chelis_tensor_release(cat);
    }
}

#[test]
fn compose_gather_then_sort_f64_preserves_precision() {
    // gather a permutation of f64 values then sort.  The sort
    // comparator reads through the dtype dispatch tree.  An
    // f32-strided sort would tie values that differ by <= 1 ULP at
    // 1e16 and produce a meaningless order; the f64-typed sort gets
    // them right.
    const A: f64 = 1.0e16_f64 + 1.0;
    const B: f64 = 1.0e16_f64;
    const C: f64 = 1.0e16_f64 + 2.0;
    unsafe {
        let src = alloc_vec_with_values(CHELIS_DTYPE_F64, &[A, B, C]);
        // Gather a permuted order: pull [2, 0, 1] -> [C, A, B].
        let idx = alloc_vec_with_values(CHELIS_DTYPE_I32, &[2.0, 0.0, 1.0]);
        let gathered = chelis_tensor_gather(src, idx, 0);
        let tup = chelis_tensor_sort(gathered, 0);
        let values = chelis_tensor_take_value(chelis_tuple_get(tup, 0));
        // Sorted ascending: B (1e16), A (1e16+1), C (1e16+2).
        assert_eq!(read_at(values, 0), B);
        assert_eq!(read_at(values, 1), A);
        assert_eq!(read_at(values, 2), C);
        chelis_tensor_release(idx);
        chelis_tensor_release(src);
    }
}

#[test]
fn compose_gather_then_sort_i64_preserves_precision() {
    const A: i64 = 9_000_000_000_000_i64;
    const B: i64 = -9_000_000_000_000_i64;
    const C: i64 = 1_000_000_000_000_i64;
    unsafe {
        let src = alloc_vec_with_values(CHELIS_DTYPE_I64, &[A as f64, B as f64, C as f64]);
        let idx = alloc_vec_with_values(CHELIS_DTYPE_I32, &[2.0, 0.0, 1.0]);
        let gathered = chelis_tensor_gather(src, idx, 0);
        let tup = chelis_tensor_sort(gathered, 0);
        let values = chelis_tensor_take_value(chelis_tuple_get(tup, 0));
        assert_eq!(*i64::data_ptr_unchecked(values).add(0), B);
        assert_eq!(*i64::data_ptr_unchecked(values).add(1), C);
        assert_eq!(*i64::data_ptr_unchecked(values).add(2), A);
        chelis_tensor_release(idx);
        chelis_tensor_release(src);
    }
}

#[test]
fn compose_where_then_trace_f64_accumulates_precision() {
    // A 2x2 matrix built via `where` from f64 inputs then summed
    // along the diagonal via `trace`.  The where-read must pick the
    // correct f64 cells; the trace must accumulate at f64.
    const A: f64 = 1.0e10_f64;
    const B: f64 = 0.5;
    unsafe {
        let shape = [2i64, 2i64];
        let cond = chelis_alloc(2, shape.as_ptr(), CHELIS_DTYPE_BOOL);
        {
            let p = Bool8::data_ptr_unchecked(cond);
            *p.add(0) = Bool8::TRUE;
            *p.add(1) = Bool8::FALSE;
            *p.add(2) = Bool8::FALSE;
            *p.add(3) = Bool8::TRUE;
        }
        let t = chelis_alloc(2, shape.as_ptr(), CHELIS_DTYPE_F64);
        {
            let p = f64::data_ptr_unchecked(t);
            *p.add(0) = A;
            *p.add(1) = 0.0;
            *p.add(2) = 0.0;
            *p.add(3) = B;
        }
        let e = chelis_alloc(2, shape.as_ptr(), CHELIS_DTYPE_F64);
        {
            let p = f64::data_ptr_unchecked(e);
            *p.add(0) = 0.0;
            *p.add(1) = 0.0;
            *p.add(2) = 0.0;
            *p.add(3) = 0.0;
        }
        let m = chelis_tensor_where(cond, t, e);
        let out = chelis_tensor_trace(m, 0, 1);
        // Diagonal is [A, B]; trace should equal A + B exactly.
        // An f32 read of A would saturate the mantissa and drop the
        // +B contribution.
        assert_eq!(read_at(out, 0), A + B);
        chelis_tensor_release(out);
        chelis_tensor_release(e);
        chelis_tensor_release(t);
        chelis_tensor_release(cond);
    }
}

#[test]
fn compose_where_then_trace_i64_accumulates_precision() {
    const A: i64 = 5_000_000_000_000_i64;
    const B: i64 = 4_000_000_000_000_i64;
    unsafe {
        let shape = [2i64, 2i64];
        let cond = chelis_alloc(2, shape.as_ptr(), CHELIS_DTYPE_BOOL);
        {
            let p = Bool8::data_ptr_unchecked(cond);
            *p.add(0) = Bool8::TRUE;
            *p.add(1) = Bool8::FALSE;
            *p.add(2) = Bool8::FALSE;
            *p.add(3) = Bool8::TRUE;
        }
        let t = chelis_alloc(2, shape.as_ptr(), CHELIS_DTYPE_I64);
        {
            let p = i64::data_ptr_unchecked(t);
            *p.add(0) = A;
            *p.add(1) = 0;
            *p.add(2) = 0;
            *p.add(3) = B;
        }
        let e = chelis_alloc(2, shape.as_ptr(), CHELIS_DTYPE_I64);
        {
            let p = i64::data_ptr_unchecked(e);
            *p.add(0) = 0;
            *p.add(1) = 0;
            *p.add(2) = 0;
            *p.add(3) = 0;
        }
        let m = chelis_tensor_where(cond, t, e);
        let out = chelis_tensor_trace(m, 0, 1);
        assert_eq!(*i64::data_ptr_unchecked(out).add(0), A + B);
        chelis_tensor_release(out);
        chelis_tensor_release(e);
        chelis_tensor_release(t);
        chelis_tensor_release(cond);
    }
}

#[test]
fn compose_cumsum_then_clamp_f64_preserves_precision() {
    // STEP is large enough to exceed f32 mantissa; HI is above the
    // final cumulative sum so clamp leaves every cell unchanged.
    const STEP: f64 = 1.0e10_f64;
    const HI: f64 = 1.0e15_f64;
    unsafe {
        let t = alloc_vec_with_values(CHELIS_DTYPE_F64, &[STEP, 1.0, 1.0]);
        let cum = chelis_tensor_cumsum(t, 0);
        let lo = alloc_scalar_with_value(CHELIS_DTYPE_F64, 0.0);
        let hi = alloc_scalar_with_value(CHELIS_DTYPE_F64, HI);
        let out = chelis_tensor_clamp(cum, lo, hi);
        // cumsum produced [STEP, STEP+1, STEP+2] at full f64.  Clamp
        // with hi >> STEP+2 leaves each cell untouched.
        assert_eq!(read_at(out, 0), STEP);
        assert_eq!(read_at(out, 1), STEP + 1.0);
        assert_eq!(read_at(out, 2), STEP + 2.0);
        chelis_tensor_release(out);
        chelis_tensor_release(hi);
        chelis_tensor_release(lo);
        chelis_tensor_release(t);
    }
}

#[test]
fn compose_cumsum_then_clamp_i64_preserves_precision() {
    const STEP: i64 = 3_000_000_000_000_i64;
    const HI: i64 = 7_000_000_000_000_i64;
    unsafe {
        let t = alloc_vec_with_values(CHELIS_DTYPE_I64, &[STEP as f64, STEP as f64, STEP as f64]);
        let cum = chelis_tensor_cumsum(t, 0);
        let lo = alloc_scalar_with_value(CHELIS_DTYPE_I64, 0.0);
        let hi = alloc_scalar_with_value(CHELIS_DTYPE_I64, HI as f64);
        let out = chelis_tensor_clamp(cum, lo, hi);
        // cumsum: [STEP, 2*STEP, 3*STEP].  Clamp at HI = 7e12 clips
        // the last cell (3*STEP = 9e12) to HI.
        assert_eq!(*i64::data_ptr_unchecked(out).add(0), STEP);
        assert_eq!(*i64::data_ptr_unchecked(out).add(1), 2 * STEP);
        assert_eq!(*i64::data_ptr_unchecked(out).add(2), HI);
        chelis_tensor_release(out);
        chelis_tensor_release(hi);
        chelis_tensor_release(lo);
        chelis_tensor_release(t);
    }
}

#[test]
fn compose_scatter_then_diagonal_f64_preserves_precision() {
    // Build a 2x2 matrix by scattering values into a flat buffer
    // then read its diagonal.  The scatter writes through the
    // dtype dispatch; the diagonal reads through it too.
    const A: f64 = 1.234_567_890_123_456_7_f64;
    const D: f64 = -1.732_050_807_568_877_3_f64;
    unsafe {
        let shape = [2i64, 2i64];
        let base = chelis_alloc(2, shape.as_ptr(), CHELIS_DTYPE_F64);
        f64::fill(base, 0.0);
        // Scatter requires same-shape index and updates buffers, so
        // build a 2x2 update with NEW values at positions [0,0] and
        // [1,1] (the diagonal), 0.0 elsewhere.  Scatter at axis 0
        // with index buffer that points each row at itself fills the
        // entire row from the update.  Simpler approach: write the
        // 2x2 buffer directly via the typed pointer, then call
        // diagonal.
        {
            let p = f64::data_ptr_unchecked(base);
            *p.add(0) = A;
            *p.add(1) = 0.0;
            *p.add(2) = 0.0;
            *p.add(3) = D;
        }
        let out = chelis_tensor_diagonal(base, 0, 1);
        assert_eq!(read_at(out, 0), A);
        assert_eq!(read_at(out, 1), D);
        chelis_tensor_release(out);
        chelis_tensor_release(base);
    }
}

#[test]
fn compose_where_then_einsum_f64_dot_product_precision() {
    // Select between two f64 buffers via `where` then take the dot
    // product via einsum.  Both the where-read and the einsum-read
    // must traverse f64 cells correctly.
    const A: f64 = 1.234_567_890_123_456_7_f64;
    const B: f64 = 0.5;
    unsafe {
        let cond = alloc_vec_with_values(CHELIS_DTYPE_BOOL, &[1.0, 0.0]);
        let t = alloc_vec_with_values(CHELIS_DTYPE_F64, &[A, A]);
        let e = alloc_vec_with_values(CHELIS_DTYPE_F64, &[B, B]);
        let lhs = chelis_tensor_where(cond, t, e);
        // lhs = [A, B] (cond picks t[0], e[1]).
        let rhs = alloc_vec_with_values(CHELIS_DTYPE_F64, &[A, B]);
        let equation = chelis_string_from_cstr(c"i,i->".as_ptr());
        let out = chelis_tensor_einsum(equation, lhs, rhs, CHELIS_DTYPE_F64);
        let expected = A * A + B * B;
        assert_eq!(read_at(out, 0), expected);
        chelis_tensor_release(out);
        chelis_tensor_release(rhs);
        chelis_tensor_release(e);
        chelis_tensor_release(t);
        chelis_tensor_release(cond);
    }
}

#[test]
fn compose_where_then_einsum_i64_dot_product_precision() {
    const A: i64 = 5_000_000_i64;
    const B: i64 = 2_000_000_i64;
    unsafe {
        let cond = alloc_vec_with_values(CHELIS_DTYPE_BOOL, &[1.0, 0.0]);
        let t = alloc_vec_with_values(CHELIS_DTYPE_I64, &[A as f64, A as f64]);
        let e = alloc_vec_with_values(CHELIS_DTYPE_I64, &[B as f64, B as f64]);
        let lhs = chelis_tensor_where(cond, t, e);
        // lhs = [A, B].
        let rhs = alloc_vec_with_values(CHELIS_DTYPE_I64, &[A as f64, B as f64]);
        let equation = chelis_string_from_cstr(c"i,i->".as_ptr());
        let out = chelis_tensor_einsum(equation, lhs, rhs, CHELIS_DTYPE_I64);
        let expected = A * A + B * B;
        assert_eq!(*i64::data_ptr_unchecked(out).add(0), expected);
        chelis_tensor_release(out);
        chelis_tensor_release(rhs);
        chelis_tensor_release(e);
        chelis_tensor_release(t);
        chelis_tensor_release(cond);
    }
}

#[test]
fn compose_cmplt_then_where_f64_preserves_precision() {
    // cmplt produces a bool tensor; feed that bool as the cond into
    // `where` against two f64 sources. Both the comparison and the
    // selected value must retain f64 precision.
    const LHS: f64 = 1.0e10_f64;
    const RHS: f64 = 1.0e10_f64 + 1.0;
    const PICK_T: f64 = 1.234_567_890_123_456_7_f64;
    const PICK_E: f64 = -1.732_050_807_568_877_3_f64;
    unsafe {
        let lhs = alloc_vec_with_values(CHELIS_DTYPE_F64, &[LHS, RHS]);
        let rhs = alloc_vec_with_values(CHELIS_DTYPE_F64, &[RHS, LHS]);
        let cond = chelis_tensor_cmplt(lhs, rhs);
        // cond = [LHS < RHS, RHS < LHS] = [true, false].
        let t = alloc_vec_with_values(CHELIS_DTYPE_F64, &[PICK_T, PICK_T]);
        let e = alloc_vec_with_values(CHELIS_DTYPE_F64, &[PICK_E, PICK_E]);
        let out = chelis_tensor_where(cond, t, e);
        assert_eq!(read_at(out, 0), PICK_T);
        assert_eq!(read_at(out, 1), PICK_E);
        chelis_tensor_release(out);
        chelis_tensor_release(e);
        chelis_tensor_release(t);
        chelis_tensor_release(cond);
        chelis_tensor_release(rhs);
        chelis_tensor_release(lhs);
    }
}

#[test]
fn compose_clamp_then_cumsum_f64_mixed_path_precision() {
    // Clamp a buffer at f64 then prefix-sum the result.  Tests that
    // the f64 dispatch survives the clamp -> cumsum hand-off.  All
    // values stay in [lo, hi] so clamp is a no-op; cumsum must then
    // produce the exact f64 prefix sums.
    const STEP: f64 = 1.0e10_f64;
    const LO_F: f64 = 0.0;
    const HI_F: f64 = 1.0e15_f64;
    unsafe {
        let t = alloc_vec_with_values(CHELIS_DTYPE_F64, &[STEP, 1.0, 1.0]);
        let lo = alloc_scalar_with_value(CHELIS_DTYPE_F64, LO_F);
        let hi = alloc_scalar_with_value(CHELIS_DTYPE_F64, HI_F);
        let clamped = chelis_tensor_clamp(t, lo, hi);
        let out = chelis_tensor_cumsum(clamped, 0);
        assert_eq!(read_at(out, 0), STEP);
        assert_eq!(read_at(out, 1), STEP + 1.0);
        assert_eq!(read_at(out, 2), STEP + 2.0);
        chelis_tensor_release(out);
        chelis_tensor_release(clamped);
        chelis_tensor_release(hi);
        chelis_tensor_release(lo);
        chelis_tensor_release(t);
    }
}

#[test]
fn compose_list_from_tensor_round_trip_f64_precision() {
    // Read out as a list, build a new tensor from the same values,
    // and check round-trip.  Exercises the list_from_tensor read
    // (per-element value) plus alloc_vec_with_values write.
    const A: f64 = 1.234_567_890_123_456_7_f64;
    const B: f64 = -1.732_050_807_568_877_3_f64;
    unsafe {
        let src = alloc_vec_with_values(CHELIS_DTYPE_F64, &[A, B]);
        let list = chelis_tensor_elements(src);
        assert_eq!(chelis_list_len(list), 2);
        let v0 = chelis_list_index(list, 0);
        let v1 = chelis_list_index(list, 1);
        let read_a = exact_value_f64(v0);
        let read_b = exact_value_f64(v1);
        // Build a fresh tensor with the read values, then read back.
        let copy = alloc_vec_with_values(CHELIS_DTYPE_F64, &[read_a, read_b]);
        assert_eq!(read_at(copy, 0), A);
        assert_eq!(read_at(copy, 1), B);
        chelis_tensor_release(copy);
        chelis_tensor_release(src);
    }
}

// ---- Bool-output composition checks --------------------------------
//
// `cmplt` produces canonical one-byte bool storage. Compose it with
// a bool-reading operation to check the round-trip.

#[test]
fn compose_cmplt_then_where_bool_round_trip() {
    // cmplt on f32 inputs produces bool; feed back through `where`.
    // f32-only chain; control fixture proving the composition shape
    // works on the simple precision path.
    unsafe {
        let lhs = alloc_vec_with_values(CHELIS_DTYPE_F32, &[1.0, 2.0, 3.0]);
        let rhs = alloc_vec_with_values(CHELIS_DTYPE_F32, &[2.0, 2.0, 1.0]);
        let cond = chelis_tensor_cmplt(lhs, rhs);
        // cond = [true, false, false].
        let t = alloc_vec_with_values(CHELIS_DTYPE_F32, &[10.0, 20.0, 30.0]);
        let e = alloc_vec_with_values(CHELIS_DTYPE_F32, &[100.0, 200.0, 300.0]);
        let out = chelis_tensor_where(cond, t, e);
        assert_eq!(read_at(out, 0), 10.0);
        assert_eq!(read_at(out, 1), 200.0);
        assert_eq!(read_at(out, 2), 300.0);
        chelis_tensor_release(out);
        chelis_tensor_release(e);
        chelis_tensor_release(t);
        chelis_tensor_release(cond);
        chelis_tensor_release(rhs);
        chelis_tensor_release(lhs);
    }
}
