//! Cross-validation harness for dtype x op precision agreement.
//!
//! Closes the foundational `CRuntime-F32Coupling` `docs/gap_synthesis.md`
//! §5 entry: C runtime's `chelis_tensor.data` was typed `*mut f32`
//! regardless of dtype, producing four distinct silent-data-corruption
//! bugs (cast PR #64, reshape PR #67, print PR #72, plus the int32-as-f32
//! storage bug PR #72 surfaced). PR 1 of the W2 series introduces the
//! `TensorElement` trait and migrates two anchor ops; PRs 2-4 (Agent B)
//! migrate the remaining ~37 access sites mechanically.
//!
//! Anchor ops covered by PR 1:
//!   * `chelis_tensor_to_f64` (read-side) at `lib.rs:519`.
//!   * `chelis_fill_i64` / `chelis_fill_f64` / `chelis_fill_f32`
//!     replaced by `T::fill(t, val)` defaults (write-side).
//!
//! Negative coverage:
//!   * `<T>::data_ptr(t)` on a tensor whose dtype is something other
//!     than `T` returns `Err(DtypeMismatch { expected, actual })`.
//!
//! Bool routing note: per the W2 PR 1 dispatch brief and orchestrator
//! decision on PR #79, bool storage today is 4-byte f32-encoded
//! (`chelis_alloc` allocates `size_of::<f32>()` bytes per element for
//! `CHELIS_BOOL`). The `TensorElement` trait does NOT provide a `bool`
//! impl; bool dtype-dispatched sites read through `f32::data_ptr`
//! internally and compare against 0.0. The same f32-routed convention
//! applies to int32 storage today, which is also 4-byte f32-encoded;
//! the i32 trait impl exists for future storage migration but is not
//! used at the current `chelis_tensor_to_f64` int32 arm.
//!
//! TODO entries below enumerate the ops Agent B's PRs 2-4 cover.
//! Each TODO names the runtime site and the migration template
//! (outer match on `(*t).dtype`, arms call `data_ptr_unchecked`).
//! See `docs/design/compiler_cleanup_0_7_8_spec_lock.md` Contract 3
//! for the full op enumeration.

#![allow(clippy::missing_safety_doc)]

use std::ptr;

use std::os::raw::c_int;

use chelis_runtime::{
    CHELIS_BOOL, CHELIS_F32, CHELIS_F64, CHELIS_I32, CHELIS_I64, DtypeMismatch, TensorElement,
    chelis_alloc, chelis_fill_f32, chelis_fill_f64, chelis_fill_i64, chelis_free,
    chelis_list_append, chelis_list_empty, chelis_list_from_tensor, chelis_list_index,
    chelis_list_len, chelis_string_from_cstr, chelis_tensor, chelis_tensor_clamp,
    chelis_tensor_cmplt, chelis_tensor_concat, chelis_tensor_cumsum, chelis_tensor_diagonal,
    chelis_tensor_einsum, chelis_tensor_gather, chelis_tensor_scatter, chelis_tensor_sort,
    chelis_tensor_split, chelis_tensor_to_f64, chelis_tensor_trace, chelis_tensor_where,
    chelis_tuple_get, chelis_value_as_bool, chelis_value_as_f64, chelis_value_as_int64,
    chelis_value_as_tensor, chelis_value_from_tensor, data_as_f32,
};

/// Allocate a rank-0 (scalar) tensor of the given dtype.  Caller frees.
unsafe fn alloc_scalar(dtype: c_int) -> *mut chelis_tensor {
    unsafe { chelis_alloc(0, ptr::null(), dtype) }
}

/// Allocate a rank-1 tensor of length `n` and the given dtype.
unsafe fn alloc_vec(n: c_int, dtype: c_int) -> *mut chelis_tensor {
    let shape = [n];
    unsafe { chelis_alloc(1, shape.as_ptr(), dtype) }
}

// ---- `chelis_tensor_to_f64` (read-side) --------------------------------
//
// Anchor op A.  Reads a rank-0 tensor of any supported precision and
// returns the value as f64.  Pre-migration the body reads `*data as
// f64` unconditionally (data was `*mut f32`); post-migration it
// dispatches on `(*t).dtype` and selects the typed read.  Bool and i32
// storage today is 4-byte f32-encoded so those arms route through
// `f32::data_ptr_unchecked`.

#[test]
fn tensor_to_f64_f32() {
    unsafe {
        let t = alloc_scalar(CHELIS_F32);
        f32::fill(t, 3.5);
        let out = chelis_tensor_to_f64(t);
        assert_eq!(
            out, 3.5,
            "f32 rank-0 tensor must read back as f64 without precision loss"
        );
        chelis_free(t);
    }
}

#[test]
fn tensor_to_f64_f64() {
    // Value chosen to exceed f32 precision (16 decimal digits) so
    // any f32-truncating read path fails the round-trip.
    const F64_VALUE: f64 = 1.234_567_890_123_456_7_f64;
    unsafe {
        let t = alloc_scalar(CHELIS_F64);
        f64::fill(t, F64_VALUE);
        let out = chelis_tensor_to_f64(t);
        assert_eq!(
            out, F64_VALUE,
            "f64 rank-0 tensor must round-trip with full f64 precision"
        );
        chelis_free(t);
    }
}

#[test]
fn tensor_to_f64_i64() {
    unsafe {
        let t = alloc_scalar(CHELIS_I64);
        i64::fill(t, 1_000_000_000_000_i64);
        let out = chelis_tensor_to_f64(t);
        assert_eq!(
            out, 1_000_000_000_000.0,
            "i64 rank-0 tensor must read back as f64 with the full i64 value preserved"
        );
        chelis_free(t);
    }
}

#[test]
fn tensor_to_f64_i32() {
    unsafe {
        let t = alloc_scalar(CHELIS_I32);
        // I32 storage today is 4-byte f32-encoded.  Write through
        // `data_as_f32` (the transition shim) rather than
        // `f32::fill` -- the latter would trip the trait's
        // debug_assert (DTYPE mismatch: F32 vs I32).
        *data_as_f32(t) = 42.0f32;
        let out = chelis_tensor_to_f64(t);
        assert_eq!(
            out, 42.0,
            "i32 rank-0 tensor (f32-encoded storage) must read back as f64"
        );
        chelis_free(t);
    }
}

#[test]
fn tensor_to_f64_bool_true() {
    unsafe {
        let t = alloc_scalar(CHELIS_BOOL);
        // Bool storage today is 4-byte f32-encoded (1.0f32 /
        // 0.0f32).  Write through `data_as_f32` to bypass the
        // trait's dtype assertion.
        *data_as_f32(t) = 1.0f32;
        let out = chelis_tensor_to_f64(t);
        assert_eq!(
            out, 1.0,
            "bool=true rank-0 tensor (f32-encoded storage) must read back as 1.0"
        );
        chelis_free(t);
    }
}

#[test]
fn tensor_to_f64_bool_false() {
    unsafe {
        let t = alloc_scalar(CHELIS_BOOL);
        *data_as_f32(t) = 0.0f32;
        let out = chelis_tensor_to_f64(t);
        assert_eq!(
            out, 0.0,
            "bool=false rank-0 tensor (f32-encoded storage) must read back as 0.0"
        );
        chelis_free(t);
    }
}

// ---- `TensorElement::fill` (write-side) --------------------------------
//
// Anchor op B.  Each precision's `chelis_fill_*` extern symbol thins
// to `T::fill(t, val)` after migration.  The pre-migration body and
// the post-migration default-trait expansion compile to bit-identical
// code (cast `.data` to typed pointer, loop over `size`).  This
// fixture asserts every element of a freshly-allocated tensor reads
// back as the filled value via the trait's typed pointer.

#[test]
fn fill_f32_vector() {
    unsafe {
        let t = alloc_vec(8, CHELIS_F32);
        f32::fill(t, 1.5);
        let ptr = f32::data_ptr_unchecked(t);
        for i in 0..(*t).size as isize {
            assert_eq!(*ptr.offset(i), 1.5, "f32 fill index {i}");
        }
        chelis_free(t);
    }
}

#[test]
fn fill_f64_vector() {
    unsafe {
        let t = alloc_vec(8, CHELIS_F64);
        f64::fill(t, 1.0e100);
        let ptr = f64::data_ptr_unchecked(t);
        for i in 0..(*t).size as isize {
            assert_eq!(*ptr.offset(i), 1.0e100, "f64 fill index {i}");
        }
        chelis_free(t);
    }
}

#[test]
fn fill_i64_vector() {
    unsafe {
        let t = alloc_vec(8, CHELIS_I64);
        i64::fill(t, -123_456_789_012_i64);
        let ptr = i64::data_ptr_unchecked(t);
        for i in 0..(*t).size as isize {
            assert_eq!(*ptr.offset(i), -123_456_789_012_i64, "i64 fill index {i}");
        }
        chelis_free(t);
    }
}

#[test]
fn fill_i32_vector() {
    // i32 storage today is 4-byte f32-encoded.  The trait impl for
    // i32 has `DTYPE = CHELIS_I32`, so `i32::fill` writes i32 bytes
    // into the buffer.  A round-trip through `i32::data_ptr_unchecked`
    // reads them back as i32 -- this exercises the trait surface
    // even though the runtime's other I32 accessors still treat the
    // storage as f32-encoded.  Filed under the §5 follow-on for i32
    // storage representation migration.
    unsafe {
        let t = alloc_vec(8, CHELIS_I32);
        i32::fill(t, 12_345_i32);
        let ptr = i32::data_ptr_unchecked(t);
        for i in 0..(*t).size as isize {
            assert_eq!(*ptr.offset(i), 12_345_i32, "i32 fill index {i}");
        }
        chelis_free(t);
    }
}

// Existing extern symbols must keep their bodies bit-identical after
// the trait-default migration so generated C drivers continue to link
// and produce identical output.  Each fixture writes a value through
// the extern wrapper and reads it back through the trait's typed
// pointer.

#[test]
fn fill_f32_extern_matches_trait() {
    unsafe {
        let t = alloc_vec(4, CHELIS_F32);
        chelis_fill_f32(t, 9.25);
        let ptr = f32::data_ptr_unchecked(t);
        for i in 0..(*t).size as isize {
            assert_eq!(*ptr.offset(i), 9.25);
        }
        chelis_free(t);
    }
}

#[test]
fn fill_f64_extern_matches_trait() {
    // Same f64 bit pattern used in `tensor_to_f64_f64` -- exceeds
    // f32 precision so any f32-truncating fill path would fail.
    const F64_VALUE: f64 = 1.234_567_890_123_456_7_f64;
    unsafe {
        let t = alloc_vec(4, CHELIS_F64);
        chelis_fill_f64(t, F64_VALUE);
        let ptr = f64::data_ptr_unchecked(t);
        for i in 0..(*t).size as isize {
            assert_eq!(*ptr.offset(i), F64_VALUE);
        }
        chelis_free(t);
    }
}

#[test]
fn fill_i64_extern_matches_trait() {
    unsafe {
        let t = alloc_vec(4, CHELIS_I64);
        chelis_fill_i64(t, 9_876_543_210_i64);
        let ptr = i64::data_ptr_unchecked(t);
        for i in 0..(*t).size as isize {
            assert_eq!(*ptr.offset(i), 9_876_543_210_i64);
        }
        chelis_free(t);
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
        let t = alloc_scalar(CHELIS_F64);
        let err = f32::data_ptr(t).expect_err("f32::data_ptr on F64 tensor must fail");
        assert_eq!(
            err,
            DtypeMismatch {
                expected: CHELIS_F32,
                actual: CHELIS_F64,
            }
        );
        chelis_free(t);
    }
}

#[test]
fn data_ptr_dtype_mismatch_i64_on_i32_tensor() {
    unsafe {
        let t = alloc_scalar(CHELIS_I32);
        let err = i64::data_ptr(t).expect_err("i64::data_ptr on I32 tensor must fail");
        assert_eq!(
            err,
            DtypeMismatch {
                expected: CHELIS_I64,
                actual: CHELIS_I32,
            }
        );
        chelis_free(t);
    }
}

#[test]
fn data_ptr_dtype_mismatch_f64_on_i64_tensor() {
    // f64 and i64 share an 8-byte storage cell, so the mismatch is
    // semantic (bit-layout differs) not size-related.  The trait must
    // catch this anyway.
    unsafe {
        let t = alloc_scalar(CHELIS_I64);
        let err = f64::data_ptr(t).expect_err("f64::data_ptr on I64 tensor must fail");
        assert_eq!(
            err,
            DtypeMismatch {
                expected: CHELIS_F64,
                actual: CHELIS_I64,
            }
        );
        chelis_free(t);
    }
}

#[test]
fn data_ptr_match_succeeds() {
    // Positive control: the matching dtype returns Ok and reads back
    // the fill value byte-exact through the typed pointer.
    unsafe {
        let t = alloc_scalar(CHELIS_F64);
        f64::fill(t, 6.022e23);
        let ptr = f64::data_ptr(t).expect("f64::data_ptr on F64 tensor must succeed");
        assert_eq!(*ptr, 6.022e23);
        chelis_free(t);
    }
}

// ---- Cross-validation fixtures for PR 2 migrated ops --------------------
//
// Each op gets one fixture per semantically-meaningful precision per
// Contract 3 of `docs/design/compiler_cleanup_0_7_8_spec_lock.md`.
// The fixtures exercise the runtime function directly via its extern
// symbol and assert byte-exact round-trip through the trait's typed
// pointer.
//
// I32 and BOOL fixtures write through `data_as_f32` (the f32-encoded
// storage convention) and read back the same way, matching the
// migrated runtime's CHELIS_I32 / CHELIS_BOOL dispatch arms.  The
// fixtures lock the post-migration behavior; pre-migration the F64
// and I64 fixtures would have failed because the f32-strided read
// silently truncated 8-byte storage to 4-byte chunks.
//
// PR 3 host_emit code-generation site fixtures and any matrix
// completion remain on the W2 PR 3 / PR 4 plan.

/// Allocate a rank-1 length-`n` tensor and fill it with values from
/// `vals` interpreted as the tensor's storage convention for `dtype`:
///   - F32 / I32 / BOOL: writer stores `value as f32` (existing
///     f32-encoded storage for I32 / BOOL).
///   - F64: writer stores `f64` bytes.
///   - I64: writer stores `i64` bytes (treating each `f64` slot as
///     `value as i64`).
unsafe fn alloc_vec_with_values(dtype: c_int, vals: &[f64]) -> *mut chelis_tensor {
    unsafe {
        let t = alloc_vec(vals.len() as c_int, dtype);
        match dtype {
            CHELIS_F32 => {
                let p = f32::data_ptr_unchecked(t);
                for (i, v) in vals.iter().enumerate() {
                    *p.add(i) = *v as f32;
                }
            }
            CHELIS_F64 => {
                let p = f64::data_ptr_unchecked(t);
                for (i, v) in vals.iter().enumerate() {
                    *p.add(i) = *v;
                }
            }
            CHELIS_I64 => {
                let p = i64::data_ptr_unchecked(t);
                for (i, v) in vals.iter().enumerate() {
                    *p.add(i) = *v as i64;
                }
            }
            CHELIS_I32 | CHELIS_BOOL => {
                let p = data_as_f32(t);
                for (i, v) in vals.iter().enumerate() {
                    *p.add(i) = *v as f32;
                }
            }
            _ => panic!("alloc_vec_with_values: unsupported dtype {dtype}"),
        }
        t
    }
}

/// Read tensor element `i` back as `f64`, using the dtype's storage
/// convention.  Mirrors the per-dtype dispatch in the migrated
/// runtime ops.
unsafe fn read_at(t: *mut chelis_tensor, i: usize) -> f64 {
    unsafe {
        match (*t).dtype {
            CHELIS_F32 => *f32::data_ptr_unchecked(t).add(i) as f64,
            CHELIS_F64 => *f64::data_ptr_unchecked(t).add(i),
            CHELIS_I64 => *i64::data_ptr_unchecked(t).add(i) as f64,
            CHELIS_I32 | CHELIS_BOOL => *data_as_f32(t).add(i) as f64,
            other => panic!("read_at: unsupported dtype {other}"),
        }
    }
}

// ---- `chelis_list_from_tensor` (read-side, per-element) -----------------

#[test]
fn list_from_tensor_f32() {
    unsafe {
        let t = alloc_vec_with_values(CHELIS_F32, &[1.5, 2.5, 3.5]);
        let list = chelis_list_from_tensor(t);
        assert_eq!(chelis_list_len(list), 3);
        let v0 = chelis_list_index(list, 0);
        let v1 = chelis_list_index(list, 1);
        let v2 = chelis_list_index(list, 2);
        assert_eq!(chelis_value_as_f64(v0), 1.5);
        assert_eq!(chelis_value_as_f64(v1), 2.5);
        assert_eq!(chelis_value_as_f64(v2), 3.5);
        chelis_free(t);
    }
}

#[test]
fn list_from_tensor_f64_full_precision() {
    // The f64 round-trip is the bug repro: pre-migration this would
    // have read 4 bytes as f32 and lost the full mantissa.
    const F64_VAL: f64 = 1.234_567_890_123_456_7_f64;
    unsafe {
        let t = alloc_vec_with_values(CHELIS_F64, &[F64_VAL, -F64_VAL]);
        let list = chelis_list_from_tensor(t);
        assert_eq!(chelis_list_len(list), 2);
        let v0 = chelis_list_index(list, 0);
        let v1 = chelis_list_index(list, 1);
        assert_eq!(chelis_value_as_f64(v0), F64_VAL);
        assert_eq!(chelis_value_as_f64(v1), -F64_VAL);
        chelis_free(t);
    }
}

#[test]
fn list_from_tensor_i64_full_precision() {
    const BIG_I64: i64 = 9_000_000_000_000_i64;
    unsafe {
        let t = alloc_vec_with_values(CHELIS_I64, &[BIG_I64 as f64, -BIG_I64 as f64]);
        let list = chelis_list_from_tensor(t);
        let v0 = chelis_list_index(list, 0);
        let v1 = chelis_list_index(list, 1);
        assert_eq!(chelis_value_as_int64(v0), BIG_I64);
        assert_eq!(chelis_value_as_int64(v1), -BIG_I64);
        chelis_free(t);
    }
}

#[test]
fn list_from_tensor_i32() {
    unsafe {
        let t = alloc_vec_with_values(CHELIS_I32, &[42.0, 7.0]);
        let list = chelis_list_from_tensor(t);
        let v0 = chelis_list_index(list, 0);
        let v1 = chelis_list_index(list, 1);
        assert_eq!(chelis_value_as_int64(v0), 42);
        assert_eq!(chelis_value_as_int64(v1), 7);
        chelis_free(t);
    }
}

#[test]
fn list_from_tensor_bool() {
    unsafe {
        let t = alloc_vec_with_values(CHELIS_BOOL, &[1.0, 0.0]);
        let list = chelis_list_from_tensor(t);
        let v0 = chelis_list_index(list, 0);
        let v1 = chelis_list_index(list, 1);
        assert!(chelis_value_as_bool(v0));
        assert!(!chelis_value_as_bool(v1));
        chelis_free(t);
    }
}

// ---- `chelis_tensor_concat` (byte-stride copy) --------------------------

unsafe fn concat_two(a: *mut chelis_tensor, b: *mut chelis_tensor) -> *mut chelis_tensor {
    unsafe {
        let parts = chelis_list_empty();
        let parts = chelis_list_append(parts, chelis_value_from_tensor(a));
        let parts = chelis_list_append(parts, chelis_value_from_tensor(b));
        chelis_tensor_concat(parts, 0)
    }
}

#[test]
fn concat_f32_round_trip() {
    unsafe {
        let a = alloc_vec_with_values(CHELIS_F32, &[1.0, 2.0]);
        let b = alloc_vec_with_values(CHELIS_F32, &[3.0, 4.0]);
        let out = concat_two(a, b);
        for (i, expected) in [1.0, 2.0, 3.0, 4.0].iter().enumerate() {
            assert_eq!(read_at(out, i), *expected, "concat_f32 index {i}");
        }
        chelis_free(out);
    }
}

#[test]
fn concat_f64_preserves_full_precision() {
    // Pre-migration this fixture would fail: each per-element copy
    // was a 4-byte f32-strided read, dropping the upper 4 bytes.
    const A: f64 = 1.234_567_890_123_456_7_f64;
    const B: f64 = -1.732_050_807_568_877_3_f64;
    unsafe {
        let lhs = alloc_vec_with_values(CHELIS_F64, &[A]);
        let rhs = alloc_vec_with_values(CHELIS_F64, &[B]);
        let out = concat_two(lhs, rhs);
        assert_eq!(read_at(out, 0), A, "f64 concat must preserve full mantissa");
        assert_eq!(read_at(out, 1), B, "f64 concat must preserve full mantissa");
        chelis_free(out);
    }
}

#[test]
fn concat_i64_preserves_full_precision() {
    const A: i64 = 9_000_000_000_000_i64;
    const B: i64 = -9_000_000_000_000_i64;
    unsafe {
        let lhs = alloc_vec_with_values(CHELIS_I64, &[A as f64]);
        let rhs = alloc_vec_with_values(CHELIS_I64, &[B as f64]);
        let out = concat_two(lhs, rhs);
        assert_eq!(*i64::data_ptr_unchecked(out).add(0), A);
        assert_eq!(*i64::data_ptr_unchecked(out).add(1), B);
        chelis_free(out);
    }
}

#[test]
fn concat_i32_round_trip() {
    unsafe {
        let a = alloc_vec_with_values(CHELIS_I32, &[1.0, 2.0]);
        let b = alloc_vec_with_values(CHELIS_I32, &[3.0, 4.0]);
        let out = concat_two(a, b);
        for (i, expected) in [1.0, 2.0, 3.0, 4.0].iter().enumerate() {
            assert_eq!(read_at(out, i), *expected, "concat_i32 index {i}");
        }
        chelis_free(out);
    }
}

#[test]
fn concat_bool_round_trip() {
    unsafe {
        let a = alloc_vec_with_values(CHELIS_BOOL, &[1.0, 0.0]);
        let b = alloc_vec_with_values(CHELIS_BOOL, &[0.0, 1.0]);
        let out = concat_two(a, b);
        for (i, expected) in [1.0, 0.0, 0.0, 1.0].iter().enumerate() {
            assert_eq!(read_at(out, i), *expected, "concat_bool index {i}");
        }
        chelis_free(out);
    }
}

// ---- `chelis_tensor_split` (byte-stride copy) ---------------------------

unsafe fn split_two_halves(
    t: *mut chelis_tensor,
    lhs_size: i64,
) -> (*mut chelis_tensor, *mut chelis_tensor) {
    unsafe {
        let rhs_size = (*t).shape[0] as i64 - lhs_size;
        let sizes = chelis_list_empty();
        let sizes = chelis_list_append(sizes, chelis_runtime::chelis_value_from_int64(lhs_size));
        let sizes = chelis_list_append(sizes, chelis_runtime::chelis_value_from_int64(rhs_size));
        let parts = chelis_tensor_split(t, 0, sizes);
        let v0 = chelis_list_index(parts, 0);
        let v1 = chelis_list_index(parts, 1);
        (chelis_value_as_tensor(v0), chelis_value_as_tensor(v1))
    }
}

#[test]
fn split_f64_preserves_full_precision() {
    const A: f64 = 1.234_567_890_123_456_7_f64;
    const B: f64 = -1.732_050_807_568_877_3_f64;
    unsafe {
        let t = alloc_vec_with_values(CHELIS_F64, &[A, B]);
        let (lhs, rhs) = split_two_halves(t, 1);
        assert_eq!(read_at(lhs, 0), A);
        assert_eq!(read_at(rhs, 0), B);
        chelis_free(t);
    }
}

#[test]
fn split_i64_preserves_full_precision() {
    const A: i64 = 9_000_000_000_000_i64;
    const B: i64 = -9_000_000_000_000_i64;
    unsafe {
        let t = alloc_vec_with_values(CHELIS_I64, &[A as f64, B as f64]);
        let (lhs, rhs) = split_two_halves(t, 1);
        assert_eq!(*i64::data_ptr_unchecked(lhs).add(0), A);
        assert_eq!(*i64::data_ptr_unchecked(rhs).add(0), B);
        chelis_free(t);
    }
}

#[test]
fn split_f32_round_trip() {
    unsafe {
        let t = alloc_vec_with_values(CHELIS_F32, &[1.0, 2.0, 3.0, 4.0]);
        let (lhs, rhs) = split_two_halves(t, 2);
        assert_eq!(read_at(lhs, 0), 1.0);
        assert_eq!(read_at(lhs, 1), 2.0);
        assert_eq!(read_at(rhs, 0), 3.0);
        assert_eq!(read_at(rhs, 1), 4.0);
        chelis_free(t);
    }
}

// ---- `chelis_tensor_gather` (byte-stride indexed read) ------------------

#[test]
fn gather_f64_preserves_full_precision() {
    const A: f64 = 1.234_567_890_123_456_7_f64;
    const B: f64 = -1.732_050_807_568_877_3_f64;
    const C: f64 = 6.022_140_76e23;
    unsafe {
        let src = alloc_vec_with_values(CHELIS_F64, &[A, B, C]);
        // Indices: pick [2, 0] from a length-3 source.
        let idx = alloc_vec_with_values(CHELIS_I32, &[2.0, 0.0]);
        let out = chelis_tensor_gather(src, idx, 0);
        assert_eq!(read_at(out, 0), C);
        assert_eq!(read_at(out, 1), A);
        chelis_free(out);
        chelis_free(idx);
        chelis_free(src);
    }
}

#[test]
fn gather_i64_preserves_full_precision() {
    const A: i64 = 9_000_000_000_000_i64;
    const B: i64 = -7_500_000_000_000_i64;
    unsafe {
        let src = alloc_vec_with_values(CHELIS_I64, &[A as f64, B as f64]);
        let idx = alloc_vec_with_values(CHELIS_I32, &[1.0, 0.0]);
        let out = chelis_tensor_gather(src, idx, 0);
        assert_eq!(*i64::data_ptr_unchecked(out).add(0), B);
        assert_eq!(*i64::data_ptr_unchecked(out).add(1), A);
        chelis_free(out);
        chelis_free(idx);
        chelis_free(src);
    }
}

#[test]
fn gather_f32_round_trip() {
    unsafe {
        let src = alloc_vec_with_values(CHELIS_F32, &[10.0, 20.0, 30.0]);
        let idx = alloc_vec_with_values(CHELIS_I32, &[2.0, 0.0, 1.0]);
        let out = chelis_tensor_gather(src, idx, 0);
        assert_eq!(read_at(out, 0), 30.0);
        assert_eq!(read_at(out, 1), 10.0);
        assert_eq!(read_at(out, 2), 20.0);
        chelis_free(out);
        chelis_free(idx);
        chelis_free(src);
    }
}

// ---- `chelis_tensor_cmplt` (numeric compare -> bool) --------------------

unsafe fn cmplt_replace_str() -> chelis_runtime::chelis_string {
    unsafe { chelis_string_from_cstr(c"replace".as_ptr()) }
}

#[test]
fn cmplt_f64_full_precision() {
    // lhs and rhs differ by an amount that survives at f64 precision
    // (>= 1 ULP at 1e10) but rounds away in f32 (24-bit mantissa
    // saturates well before 1e10).  Pre-migration read both as f32:
    // they tie and the strict `<` returns 0.0.  Post-migration reads
    // at f64 and returns 1.0.
    const LHS: f64 = 1.0e10_f64;
    const RHS: f64 = 1.0e10_f64 + 1.0;
    unsafe {
        let lhs = alloc_vec_with_values(CHELIS_F64, &[LHS]);
        let rhs = alloc_vec_with_values(CHELIS_F64, &[RHS]);
        // Sanity: rhs is strictly larger in f64.
        const { assert!(RHS > LHS) };
        let out = chelis_tensor_cmplt(lhs, rhs);
        assert_eq!(
            read_at(out, 0),
            1.0,
            "f64 cmplt must compare at full precision"
        );
        chelis_free(out);
        chelis_free(rhs);
        chelis_free(lhs);
    }
}

#[test]
fn cmplt_i64_full_precision() {
    const LHS: i64 = 4_503_599_627_370_497_i64; // 2^52 + 1
    const RHS: i64 = 4_503_599_627_370_498_i64; // 2^52 + 2
    unsafe {
        let lhs = alloc_vec_with_values(CHELIS_I64, &[LHS as f64]);
        let rhs = alloc_vec_with_values(CHELIS_I64, &[RHS as f64]);
        let out = chelis_tensor_cmplt(lhs, rhs);
        assert_eq!(
            read_at(out, 0),
            1.0,
            "i64 cmplt must compare at full precision"
        );
        chelis_free(out);
        chelis_free(rhs);
        chelis_free(lhs);
    }
}

#[test]
fn cmplt_f32_round_trip() {
    unsafe {
        let lhs = alloc_vec_with_values(CHELIS_F32, &[1.0, 2.0, 3.0]);
        let rhs = alloc_vec_with_values(CHELIS_F32, &[2.0, 2.0, 1.0]);
        let out = chelis_tensor_cmplt(lhs, rhs);
        assert_eq!(read_at(out, 0), 1.0);
        assert_eq!(read_at(out, 1), 0.0);
        assert_eq!(read_at(out, 2), 0.0);
        chelis_free(out);
        chelis_free(rhs);
        chelis_free(lhs);
    }
}

// ---- `chelis_tensor_scatter` (replace + add modes) ----------------------

#[test]
fn scatter_replace_f64_preserves_full_precision() {
    const ORIG: f64 = 1.234_567_890_123_456_7_f64;
    const NEW: f64 = -ORIG;
    unsafe {
        let base = alloc_vec_with_values(CHELIS_F64, &[ORIG, ORIG, ORIG]);
        let idx = alloc_vec_with_values(CHELIS_I32, &[1.0]);
        let upd = alloc_vec_with_values(CHELIS_F64, &[NEW]);
        let mode = cmplt_replace_str();
        let out = chelis_tensor_scatter(base, idx, upd, 0, mode);
        assert_eq!(read_at(out, 0), ORIG);
        assert_eq!(read_at(out, 1), NEW);
        assert_eq!(read_at(out, 2), ORIG);
        chelis_free(out);
        chelis_free(upd);
        chelis_free(idx);
        chelis_free(base);
    }
}

#[test]
fn scatter_add_i64_preserves_full_precision() {
    const ORIG: i64 = 9_000_000_000_000_i64;
    const ADD: i64 = 1_000_000_000_000_i64;
    unsafe {
        let base = alloc_vec_with_values(CHELIS_I64, &[ORIG as f64, ORIG as f64]);
        let idx = alloc_vec_with_values(CHELIS_I32, &[0.0]);
        let upd = alloc_vec_with_values(CHELIS_I64, &[ADD as f64]);
        let mode = chelis_string_from_cstr(c"add".as_ptr());
        let out = chelis_tensor_scatter(base, idx, upd, 0, mode);
        assert_eq!(*i64::data_ptr_unchecked(out).add(0), ORIG + ADD);
        assert_eq!(*i64::data_ptr_unchecked(out).add(1), ORIG);
        chelis_free(out);
        chelis_free(upd);
        chelis_free(idx);
        chelis_free(base);
    }
}

#[test]
fn scatter_replace_f32_round_trip() {
    unsafe {
        let base = alloc_vec_with_values(CHELIS_F32, &[1.0, 2.0, 3.0]);
        let idx = alloc_vec_with_values(CHELIS_I32, &[2.0]);
        let upd = alloc_vec_with_values(CHELIS_F32, &[42.0]);
        let mode = cmplt_replace_str();
        let out = chelis_tensor_scatter(base, idx, upd, 0, mode);
        assert_eq!(read_at(out, 0), 1.0);
        assert_eq!(read_at(out, 1), 2.0);
        assert_eq!(read_at(out, 2), 42.0);
        chelis_free(out);
        chelis_free(upd);
        chelis_free(idx);
        chelis_free(base);
    }
}

// ---- `chelis_tensor_where` (cond + then/else select) --------------------

#[test]
fn where_f64_preserves_full_precision() {
    const A: f64 = 1.234_567_890_123_456_7_f64;
    const B: f64 = -1.732_050_807_568_877_3_f64;
    unsafe {
        let cond = alloc_vec_with_values(CHELIS_BOOL, &[1.0, 0.0]);
        let t = alloc_vec_with_values(CHELIS_F64, &[A, A]);
        let e = alloc_vec_with_values(CHELIS_F64, &[B, B]);
        let out = chelis_tensor_where(cond, t, e);
        assert_eq!(read_at(out, 0), A);
        assert_eq!(read_at(out, 1), B);
        chelis_free(out);
        chelis_free(e);
        chelis_free(t);
        chelis_free(cond);
    }
}

#[test]
fn where_i64_preserves_full_precision() {
    const A: i64 = 9_000_000_000_000_i64;
    const B: i64 = -7_500_000_000_000_i64;
    unsafe {
        let cond = alloc_vec_with_values(CHELIS_BOOL, &[1.0, 0.0]);
        let t = alloc_vec_with_values(CHELIS_I64, &[A as f64, A as f64]);
        let e = alloc_vec_with_values(CHELIS_I64, &[B as f64, B as f64]);
        let out = chelis_tensor_where(cond, t, e);
        assert_eq!(*i64::data_ptr_unchecked(out).add(0), A);
        assert_eq!(*i64::data_ptr_unchecked(out).add(1), B);
        chelis_free(out);
        chelis_free(e);
        chelis_free(t);
        chelis_free(cond);
    }
}

#[test]
fn where_f32_round_trip() {
    unsafe {
        let cond = alloc_vec_with_values(CHELIS_BOOL, &[1.0, 0.0, 1.0]);
        let t = alloc_vec_with_values(CHELIS_F32, &[10.0, 20.0, 30.0]);
        let e = alloc_vec_with_values(CHELIS_F32, &[100.0, 200.0, 300.0]);
        let out = chelis_tensor_where(cond, t, e);
        assert_eq!(read_at(out, 0), 10.0);
        assert_eq!(read_at(out, 1), 200.0);
        assert_eq!(read_at(out, 2), 30.0);
        chelis_free(out);
        chelis_free(e);
        chelis_free(t);
        chelis_free(cond);
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
        let t = alloc_vec_with_values(CHELIS_F64, &[STEP, 1.0, 1.0]);
        let out = chelis_tensor_cumsum(t, 0);
        assert_eq!(read_at(out, 0), STEP);
        assert_eq!(
            read_at(out, 1),
            STEP + 1.0,
            "f64 prefix sum must accumulate at full precision"
        );
        assert_eq!(read_at(out, 2), STEP + 2.0);
        chelis_free(out);
        chelis_free(t);
    }
}

#[test]
fn cumsum_i64_full_precision() {
    const STEP: i64 = 5_000_000_000_000_i64;
    unsafe {
        let t = alloc_vec_with_values(CHELIS_I64, &[STEP as f64, STEP as f64, STEP as f64]);
        let out = chelis_tensor_cumsum(t, 0);
        assert_eq!(*i64::data_ptr_unchecked(out).add(0), STEP);
        assert_eq!(*i64::data_ptr_unchecked(out).add(1), 2 * STEP);
        assert_eq!(*i64::data_ptr_unchecked(out).add(2), 3 * STEP);
        chelis_free(out);
        chelis_free(t);
    }
}

#[test]
fn cumsum_f32_round_trip() {
    unsafe {
        let t = alloc_vec_with_values(CHELIS_F32, &[1.0, 2.0, 3.0]);
        let out = chelis_tensor_cumsum(t, 0);
        assert_eq!(read_at(out, 0), 1.0);
        assert_eq!(read_at(out, 1), 3.0);
        assert_eq!(read_at(out, 2), 6.0);
        chelis_free(out);
        chelis_free(t);
    }
}

// ---- `chelis_tensor_sort` (numeric compare + i32 indices) ---------------

#[test]
fn sort_f64_full_precision() {
    // Two f64 values that round to the same f32; sort would have
    // produced a stable-but-meaningless order under the
    // pre-migration f32 read.
    const A: f64 = 1.0e16_f64 + 1.0;
    const B: f64 = 1.0e16_f64;
    unsafe {
        let t = alloc_vec_with_values(CHELIS_F64, &[A, B]);
        let tup = chelis_tensor_sort(t, 0);
        let values = chelis_value_as_tensor(chelis_tuple_get(tup, 0));
        // Sorted ascending: B then A.
        assert_eq!(
            read_at(values, 0),
            B,
            "sort must read at full f64 precision"
        );
        assert_eq!(read_at(values, 1), A);
        chelis_free(t);
    }
}

#[test]
fn sort_i64_full_precision() {
    const A: i64 = 9_000_000_000_000_i64;
    const B: i64 = -9_000_000_000_000_i64;
    unsafe {
        let t = alloc_vec_with_values(CHELIS_I64, &[A as f64, B as f64]);
        let tup = chelis_tensor_sort(t, 0);
        let values = chelis_value_as_tensor(chelis_tuple_get(tup, 0));
        // Sorted ascending: B then A.
        assert_eq!(*i64::data_ptr_unchecked(values).add(0), B);
        assert_eq!(*i64::data_ptr_unchecked(values).add(1), A);
        chelis_free(t);
    }
}

#[test]
fn sort_f32_round_trip() {
    unsafe {
        let t = alloc_vec_with_values(CHELIS_F32, &[3.0, 1.0, 2.0]);
        let tup = chelis_tensor_sort(t, 0);
        let values = chelis_value_as_tensor(chelis_tuple_get(tup, 0));
        let idx = chelis_value_as_tensor(chelis_tuple_get(tup, 1));
        assert_eq!(read_at(values, 0), 1.0);
        assert_eq!(read_at(values, 1), 2.0);
        assert_eq!(read_at(values, 2), 3.0);
        // Indices tensor is CHELIS_I32 (f32-encoded).
        assert_eq!(read_at(idx, 0), 1.0);
        assert_eq!(read_at(idx, 1), 2.0);
        assert_eq!(read_at(idx, 2), 0.0);
        chelis_free(t);
    }
}

// ---- `chelis_tensor_diagonal` + `chelis_tensor_trace` -------------------

unsafe fn alloc_2x2(dtype: c_int, a: f64, b: f64, c: f64, d: f64) -> *mut chelis_tensor {
    unsafe {
        let shape = [2i32, 2i32];
        let t = chelis_alloc(2, shape.as_ptr(), dtype);
        let vals = [a, b, c, d];
        match dtype {
            CHELIS_F32 => {
                let p = f32::data_ptr_unchecked(t);
                for (i, v) in vals.iter().enumerate() {
                    *p.add(i) = *v as f32;
                }
            }
            CHELIS_F64 => {
                let p = f64::data_ptr_unchecked(t);
                for (i, v) in vals.iter().enumerate() {
                    *p.add(i) = *v;
                }
            }
            CHELIS_I64 => {
                let p = i64::data_ptr_unchecked(t);
                for (i, v) in vals.iter().enumerate() {
                    *p.add(i) = *v as i64;
                }
            }
            CHELIS_I32 | CHELIS_BOOL => {
                let p = data_as_f32(t);
                for (i, v) in vals.iter().enumerate() {
                    *p.add(i) = *v as f32;
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
        let m = alloc_2x2(CHELIS_F64, A, 0.0, 0.0, D);
        let out = chelis_tensor_diagonal(m, 0, 1);
        assert_eq!(read_at(out, 0), A);
        assert_eq!(read_at(out, 1), D);
        chelis_free(out);
        chelis_free(m);
    }
}

#[test]
fn diagonal_i64_preserves_full_precision() {
    const A: i64 = 9_000_000_000_000_i64;
    const D: i64 = -9_000_000_000_000_i64;
    unsafe {
        let m = alloc_2x2(CHELIS_I64, A as f64, 0.0, 0.0, D as f64);
        let out = chelis_tensor_diagonal(m, 0, 1);
        assert_eq!(*i64::data_ptr_unchecked(out).add(0), A);
        assert_eq!(*i64::data_ptr_unchecked(out).add(1), D);
        chelis_free(out);
        chelis_free(m);
    }
}

#[test]
fn diagonal_f32_round_trip() {
    unsafe {
        let m = alloc_2x2(CHELIS_F32, 1.0, 2.0, 3.0, 4.0);
        let out = chelis_tensor_diagonal(m, 0, 1);
        assert_eq!(read_at(out, 0), 1.0);
        assert_eq!(read_at(out, 1), 4.0);
        chelis_free(out);
        chelis_free(m);
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
        let m = alloc_2x2(CHELIS_F64, A, 0.0, 0.0, D);
        let out = chelis_tensor_trace(m, 0, 1);
        assert_eq!(
            read_at(out, 0),
            A + D,
            "f64 trace must accumulate at full precision"
        );
        chelis_free(out);
        chelis_free(m);
    }
}

#[test]
fn trace_i64_full_precision() {
    const A: i64 = 5_000_000_000_000_i64;
    const D: i64 = 4_000_000_000_000_i64;
    unsafe {
        let m = alloc_2x2(CHELIS_I64, A as f64, 0.0, 0.0, D as f64);
        let out = chelis_tensor_trace(m, 0, 1);
        assert_eq!(*i64::data_ptr_unchecked(out).add(0), A + D);
        chelis_free(out);
        chelis_free(m);
    }
}

#[test]
fn trace_f32_round_trip() {
    unsafe {
        let m = alloc_2x2(CHELIS_F32, 1.0, 2.0, 3.0, 4.0);
        let out = chelis_tensor_trace(m, 0, 1);
        assert_eq!(read_at(out, 0), 5.0);
        chelis_free(out);
        chelis_free(m);
    }
}

// ---- `chelis_tensor_clamp` ---------------------------------------------

#[test]
fn clamp_f64_preserves_full_precision() {
    // SAMPLE has an f64 fractional part that survives at f64
    // precision (well below 2^53) but is lost in any f32-mediated
    // read.  HI_F is large enough that SAMPLE is in bounds so the
    // clamp returns SAMPLE unchanged; pre-migration the f32 read
    // would have rounded it to the nearest representable f32.
    const HI_F: f64 = 1.0e15_f64;
    const SAMPLE: f64 = 5.0e10_f64 + 0.5;
    unsafe {
        let t = alloc_vec_with_values(CHELIS_F64, &[SAMPLE]);
        let lo = alloc_vec_with_values(CHELIS_F64, &[0.0]);
        let hi = alloc_vec_with_values(CHELIS_F64, &[HI_F]);
        // lo and hi must be scalars (rank-0) per `tensor_scalar_or_same_shape`.
        (*lo).ndim = 0;
        (*hi).ndim = 0;
        let out = chelis_tensor_clamp(t, lo, hi);
        assert_eq!(
            read_at(out, 0),
            SAMPLE,
            "f64 clamp must preserve full precision"
        );
        chelis_free(out);
        chelis_free(hi);
        chelis_free(lo);
        chelis_free(t);
    }
}

#[test]
fn clamp_i64_clips_at_full_precision() {
    const A: i64 = 9_000_000_000_000_i64;
    const HI_I: i64 = 5_000_000_000_000_i64;
    unsafe {
        let t = alloc_vec_with_values(CHELIS_I64, &[A as f64]);
        let lo = alloc_vec_with_values(CHELIS_I64, &[0.0]);
        let hi = alloc_vec_with_values(CHELIS_I64, &[HI_I as f64]);
        (*lo).ndim = 0;
        (*hi).ndim = 0;
        let out = chelis_tensor_clamp(t, lo, hi);
        assert_eq!(*i64::data_ptr_unchecked(out).add(0), HI_I);
        chelis_free(out);
        chelis_free(hi);
        chelis_free(lo);
        chelis_free(t);
    }
}

#[test]
fn clamp_f32_round_trip() {
    unsafe {
        let t = alloc_vec_with_values(CHELIS_F32, &[-1.0, 0.5, 2.0]);
        let lo = alloc_vec_with_values(CHELIS_F32, &[0.0]);
        let hi = alloc_vec_with_values(CHELIS_F32, &[1.0]);
        (*lo).ndim = 0;
        (*hi).ndim = 0;
        let out = chelis_tensor_clamp(t, lo, hi);
        assert_eq!(read_at(out, 0), 0.0);
        assert_eq!(read_at(out, 1), 0.5);
        assert_eq!(read_at(out, 2), 1.0);
        chelis_free(out);
        chelis_free(hi);
        chelis_free(lo);
        chelis_free(t);
    }
}

// ---- `chelis_tensor_einsum` (numeric multiply-add) ----------------------

unsafe fn alloc_vec2(dtype: c_int, vals: &[f64]) -> *mut chelis_tensor {
    unsafe { alloc_vec_with_values(dtype, vals) }
}

#[test]
fn einsum_dot_product_f64() {
    // Pick values that exercise f64 reads (the LHS / RHS inputs need
    // the full mantissa) while keeping the accumulated dot product
    // representable.  The pre-migration f32 read would have rounded
    // SAMPLE to the nearest f32, producing a different product.
    const SAMPLE: f64 = 1.234_567_890_123_456_7_f64;
    unsafe {
        let lhs = alloc_vec2(CHELIS_F64, &[SAMPLE, 0.5]);
        let rhs = alloc_vec2(CHELIS_F64, &[SAMPLE, 2.0]);
        let equation = chelis_string_from_cstr(c"i,i->".as_ptr());
        let out = chelis_tensor_einsum(equation, lhs, rhs);
        let expected = SAMPLE * SAMPLE + 0.5 * 2.0;
        assert_eq!(read_at(out, 0), expected);
        chelis_free(out);
        chelis_free(rhs);
        chelis_free(lhs);
    }
}

#[test]
fn einsum_dot_product_i64() {
    const A: i64 = 5_000_000_i64;
    unsafe {
        let lhs = alloc_vec2(CHELIS_I64, &[A as f64, A as f64]);
        let rhs = alloc_vec2(CHELIS_I64, &[A as f64, A as f64]);
        let equation = chelis_string_from_cstr(c"i,i->".as_ptr());
        let out = chelis_tensor_einsum(equation, lhs, rhs);
        // 2 * A * A.  Far exceeds f32 mantissa range.
        let expected = 2 * A * A;
        assert_eq!(*i64::data_ptr_unchecked(out).add(0), expected);
        chelis_free(out);
        chelis_free(rhs);
        chelis_free(lhs);
    }
}

#[test]
fn einsum_dot_product_f32() {
    unsafe {
        let lhs = alloc_vec2(CHELIS_F32, &[1.0, 2.0, 3.0]);
        let rhs = alloc_vec2(CHELIS_F32, &[4.0, 5.0, 6.0]);
        let equation = chelis_string_from_cstr(c"i,i->".as_ptr());
        let out = chelis_tensor_einsum(equation, lhs, rhs);
        assert_eq!(read_at(out, 0), 32.0);
        chelis_free(out);
        chelis_free(rhs);
        chelis_free(lhs);
    }
}

// ---- Bool runtime_fail assertions for ops that reject bool --------------
//
// `cumsum`, `sort`, `clamp`, `trace`, and `einsum` are documented in
// Contract 3 as numeric-only ops.  The migrated runtime calls
// `runtime_fail!` for the CHELIS_BOOL arm.  `runtime_fail!` invokes
// `std::process::exit(1)` which kills the test process, so these
// negative-path arms cannot be exercised from a unit test without
// special harness machinery (subprocess-based fixtures, or a
// `runtime_fail` mode switch).  The full-corpus integration tests
// catch the runtime_fail via end-to-end CLI exit-code checks; this
// matrix locks the byte-exact passing arms only.
//
// PR 3 (host_emit code-generation site fixtures) and PR 4 (matrix
// completion) may revisit the bool-rejection coverage if they need a
// dedicated runtime_fail subprocess harness.
