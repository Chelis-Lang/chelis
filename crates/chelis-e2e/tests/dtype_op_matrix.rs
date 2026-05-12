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
    chelis_alloc, chelis_fill_f32, chelis_fill_f64, chelis_fill_i64, chelis_free, chelis_tensor,
    chelis_tensor_to_f64,
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
#[ignore = "blocked on TensorElement trait fix commit (W2 PR 1)"]
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
#[ignore = "blocked on TensorElement trait fix commit (W2 PR 1)"]
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
#[ignore = "blocked on TensorElement trait fix commit (W2 PR 1)"]
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
#[ignore = "blocked on TensorElement trait fix commit (W2 PR 1)"]
fn tensor_to_f64_i32() {
    unsafe {
        let t = alloc_scalar(CHELIS_I32);
        // I32 storage today is 4-byte f32-encoded.  Use f32 fill to
        // write the canonical bit pattern the runtime expects.
        f32::fill(t, 42.0);
        let out = chelis_tensor_to_f64(t);
        assert_eq!(
            out, 42.0,
            "i32 rank-0 tensor (f32-encoded storage) must read back as f64"
        );
        chelis_free(t);
    }
}

#[test]
#[ignore = "blocked on TensorElement trait fix commit (W2 PR 1)"]
fn tensor_to_f64_bool_true() {
    unsafe {
        let t = alloc_scalar(CHELIS_BOOL);
        // Bool storage today is 4-byte f32-encoded (1.0f32 / 0.0f32).
        // Use f32 fill to match the runtime's storage convention.
        f32::fill(t, 1.0);
        let out = chelis_tensor_to_f64(t);
        assert_eq!(
            out, 1.0,
            "bool=true rank-0 tensor (f32-encoded storage) must read back as 1.0"
        );
        chelis_free(t);
    }
}

#[test]
#[ignore = "blocked on TensorElement trait fix commit (W2 PR 1)"]
fn tensor_to_f64_bool_false() {
    unsafe {
        let t = alloc_scalar(CHELIS_BOOL);
        f32::fill(t, 0.0);
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
#[ignore = "blocked on TensorElement trait fix commit (W2 PR 1)"]
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
#[ignore = "blocked on TensorElement trait fix commit (W2 PR 1)"]
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
#[ignore = "blocked on TensorElement trait fix commit (W2 PR 1)"]
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
#[ignore = "blocked on TensorElement trait fix commit (W2 PR 1)"]
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
#[ignore = "blocked on TensorElement trait fix commit (W2 PR 1)"]
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
#[ignore = "blocked on TensorElement trait fix commit (W2 PR 1)"]
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
#[ignore = "blocked on TensorElement trait fix commit (W2 PR 1)"]
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
#[ignore = "blocked on TensorElement trait fix commit (W2 PR 1)"]
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
#[ignore = "blocked on TensorElement trait fix commit (W2 PR 1)"]
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
#[ignore = "blocked on TensorElement trait fix commit (W2 PR 1)"]
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
#[ignore = "blocked on TensorElement trait fix commit (W2 PR 1)"]
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

// ---- TODO: PR 2-4 op coverage ------------------------------------------
//
// Each entry below names the runtime function, source line in
// `crates/chelis-runtime/src/lib.rs`, and the precision domain.
// Agent B's fresh-context PRs (2 runtime sites, 3 host_emit sites, 4
// matrix completion) expand the harness with cross-validation
// fixtures per (precision, op) pair.
//
// PR 2 -- runtime call sites:
//   TODO chelis_pad_sequences         (lib.rs:1659-L1679) -- i32, f32
//   TODO chelis_pad_sequences_to      (lib.rs:1703-L1726) -- i32, f32
//   TODO chelis_list_from_tensor      (lib.rs:1626)       -- any numeric / bool
//   TODO chelis_tensor_concat         (lib.rs:1772)       -- any precision
//   TODO chelis_tensor_split          (lib.rs:1815)       -- any precision
//   TODO chelis_tensor_gather         (lib.rs:1869, 1884) -- any numeric / bool
//   TODO chelis_tensor_cmplt          (lib.rs:1897)       -- numeric -> bool
//   TODO chelis_tensor_scatter        (lib.rs:1952, 1970) -- any numeric / bool
//   TODO chelis_tensor_where          (lib.rs:1992-L1995) -- any precision
//   TODO chelis_tensor_cumsum         (lib.rs:2022-L2023) -- any numeric
//   TODO chelis_tensor_sort           (lib.rs:2051-L2066) -- any numeric
//   TODO chelis_tensor_diagonal       (lib.rs:2131)       -- any precision
//   TODO chelis_tensor_trace          (lib.rs:2167-L2169) -- any numeric
//   TODO chelis_tensor_clamp          (lib.rs:2190-L2199) -- any numeric
//   TODO chelis_tensor_einsum         (lib.rs:2314-L2325) -- any numeric
//   TODO chelis_tensor_to_string      (lib.rs:2582)       -- any numeric / bool
//
// PR 3 -- host_emit code-generation sites:
//   TODO elementwise op emit          (host_emit.rs:1563, 1599, 1623, 1649, 1728, 1732)
//
// PR 4 -- cross-validation matrix completion.
