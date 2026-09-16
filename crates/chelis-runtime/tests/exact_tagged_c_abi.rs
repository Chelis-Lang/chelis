//! Executable contract for the v0.19 exact tagged public C ABI.
//!
//! These tests are derived from `spec/05-risc-primitives.md`
//! [05-OP-31], [05-OP-33], and [05-OP-44]. The child-process cases exercise malformed
//! foreign carriers because the C boundary must reject them before sizing,
//! allocation, access, or observation.

use chelis_runtime::{
    chelis_alloc, chelis_dtype, chelis_dtype_size, chelis_fill_scalar, chelis_option_is_some,
    chelis_option_release, chelis_option_unwrap, chelis_parse_scalar, chelis_scalar,
    chelis_scalar_from_bits, chelis_scalar_tensor, chelis_string_from_cstr, chelis_string_release,
    chelis_tensor_begin_write, chelis_tensor_end_write, chelis_tensor_entry_borrow,
    chelis_tensor_numel, chelis_tensor_rank, chelis_tensor_read_view, chelis_tensor_release,
    chelis_tensor_shape, chelis_tensor_to_scalar, chelis_value, chelis_value_box_scalar,
    chelis_value_unbox_scalar, CHELIS_DTYPE_BF16, CHELIS_DTYPE_BOOL, CHELIS_DTYPE_F16,
    CHELIS_DTYPE_F32, CHELIS_DTYPE_F64, CHELIS_DTYPE_I16, CHELIS_DTYPE_I32, CHELIS_DTYPE_I64,
    CHELIS_DTYPE_I8, CHELIS_VALUE_SCALAR,
};
use std::env;
use std::ffi::CString;
use std::process::Command;
use std::ptr;

const CHILD_ENV: &str = "CHELIS_EXACT_TAGGED_ABI_CHILD";

fn scalar(dtype: chelis_dtype, bits: u64) -> chelis_scalar {
    chelis_scalar_from_bits(dtype, bits)
}

fn parse_scalar(text: &str, dtype: chelis_dtype) -> Option<chelis_scalar> {
    let c_text = CString::new(text).expect("test scalar has no NUL");
    unsafe {
        let runtime_text = chelis_string_from_cstr(c_text.as_ptr());
        let parsed = chelis_parse_scalar(runtime_text, dtype);
        chelis_string_release(runtime_text);
        let value = if chelis_option_is_some(parsed) {
            Some(chelis_value_unbox_scalar(chelis_option_unwrap(parsed)))
        } else {
            None
        };
        chelis_option_release(parsed);
        value
    }
}

#[test]
fn decimal_scalar_parsing_rounds_once_at_each_declared_float_width() {
    let cases = [
        (
            CHELIS_DTYPE_F64,
            "1.000000000000000111022302462515654042363166809082031249999999999999",
            "1.00000000000000011102230246251565404236316680908203125",
            "1.000000000000000111022302462515654042363166809082031250000000000001",
            0x3ff0_0000_0000_0000,
            0x3ff0_0000_0000_0001,
        ),
        (
            CHELIS_DTYPE_F32,
            "1.000000059604644775390624999999",
            "1.000000059604644775390625",
            "1.000000059604644775390625000001",
            0x3f80_0000,
            0x3f80_0001,
        ),
        (
            CHELIS_DTYPE_F16,
            "1.000488281249999999999999999999",
            "1.00048828125",
            "1.000488281250000000000000000001",
            0x3c00,
            0x3c01,
        ),
        (
            CHELIS_DTYPE_BF16,
            "1.003906249999999999999999999999",
            "1.00390625",
            "1.003906250000000000000000000001",
            0x3f80,
            0x3f81,
        ),
    ];
    for (dtype, below, midpoint, above, lower, upper) in cases {
        assert_eq!(
            parse_scalar(below, dtype).map(|value| value.bits),
            Some(lower)
        );
        assert_eq!(
            parse_scalar(midpoint, dtype).map(|value| value.bits),
            Some(lower),
            "exact midpoint must choose the even lower significand for dtype {dtype}"
        );
        assert_eq!(
            parse_scalar(above, dtype).map(|value| value.bits),
            Some(upper)
        );
    }
}

#[test]
fn malformed_and_finite_overflow_decimal_scalars_are_none() {
    for dtype in [
        CHELIS_DTYPE_F16,
        CHELIS_DTYPE_BF16,
        CHELIS_DTYPE_F32,
        CHELIS_DTYPE_F64,
    ] {
        assert_eq!(parse_scalar("1.2.3", dtype), None);
        assert_eq!(parse_scalar("1e10000", dtype), None);
    }
}

#[test]
fn every_active_dtype_round_trips_exact_stored_bits() {
    let cases = [
        (CHELIS_DTYPE_F32, 0x8000_0000),
        (CHELIS_DTYPE_F32, 0x7fc1_2345),
        (CHELIS_DTYPE_F64, 0x8000_0000_0000_0000),
        (CHELIS_DTYPE_F64, 0x7ff8_1234_5678_9abc),
        (CHELIS_DTYPE_I32, 0xffff_ffff),
        (CHELIS_DTYPE_BOOL, 1),
        (CHELIS_DTYPE_I64, (1_u64 << 53) + 17),
        (CHELIS_DTYPE_BF16, 0x7fc1),
        (CHELIS_DTYPE_F16, 0x7e11),
        (CHELIS_DTYPE_I8, 0x80),
        (CHELIS_DTYPE_I16, 0x8000),
    ];

    for (dtype, bits) in cases {
        unsafe {
            let value = scalar(dtype, bits);
            assert_eq!(value.dtype, dtype);
            assert_eq!(value.reserved, [0; 7]);
            assert_eq!(value.bits, bits);

            let boxed = chelis_value_box_scalar(value);
            assert_eq!(boxed.tag, CHELIS_VALUE_SCALAR);
            assert_eq!(boxed.reserved, [0; 7]);
            assert_eq!(chelis_value_unbox_scalar(boxed).bits, bits);

            let tensor = chelis_scalar_tensor(value);
            assert_eq!(chelis_tensor_rank(tensor), 0);
            assert_eq!(chelis_tensor_numel(tensor), 1);
            assert_eq!(chelis_tensor_to_scalar(tensor).bits, bits);
            chelis_tensor_release(tensor);
        }
    }
}

#[test]
fn bool_is_one_byte_and_fill_preserves_the_canonical_bit() {
    unsafe {
        assert_eq!(chelis_dtype_size(CHELIS_DTYPE_BOOL), 1);
        let shape = [4_i64];
        let tensor = chelis_alloc(1, shape.as_ptr(), CHELIS_DTYPE_BOOL);
        let guard = chelis_tensor_begin_write(tensor);
        chelis_fill_scalar(guard, scalar(CHELIS_DTYPE_BOOL, 1));
        chelis_tensor_end_write(guard);
        let view = chelis_tensor_read_view(tensor);
        let bytes = std::slice::from_raw_parts(view.data.cast::<u8>(), view.count as usize);
        assert_eq!(bytes, &[1, 1, 1, 1]);
        chelis_tensor_release(tensor);
    }
}

#[test]
fn dynamic_rank_layout_covers_zero_one_eight_and_greater_than_eight() {
    for rank in [0_i32, 1, 8, 9, 17] {
        let shape = vec![1_i64; rank as usize];
        unsafe {
            let tensor = chelis_alloc(
                rank,
                if rank == 0 {
                    ptr::null()
                } else {
                    shape.as_ptr()
                },
                CHELIS_DTYPE_I16,
            );
            assert_eq!(chelis_tensor_rank(tensor), rank);
            assert_eq!(chelis_tensor_numel(tensor), 1);
            for axis in 0..rank {
                assert_eq!(chelis_tensor_shape(tensor, axis), 1);
            }
            assert_eq!(chelis_tensor_read_view(tensor).count, 1);
            chelis_tensor_release(tensor);
        }
    }
}

#[test]
fn borrowed_view_copies_shape_and_honors_declared_capacity() {
    let mut backing = vec![0_u64; 6];
    let mut shape = [2_i64, 3];
    unsafe {
        let tensor = chelis_tensor_entry_borrow(
            2,
            shape.as_ptr(),
            CHELIS_DTYPE_I64,
            backing.as_mut_ptr().cast(),
            48,
        );
        shape[0] = 99;
        assert_eq!(shape[0], 99, "the caller-side mutation must take effect");
        assert_eq!(chelis_tensor_shape(tensor, 0), 2);
        assert_eq!(chelis_tensor_shape(tensor, 1), 3);
        let view = chelis_tensor_read_view(tensor);
        assert_eq!(view.count, 6);
        assert_eq!(view.data, backing.as_ptr().cast());
        chelis_tensor_release(tensor);
    }
    backing[0] = 7;
    assert_eq!(
        backing[0], 7,
        "freeing a borrowed view must not free its data"
    );
}

/// chelis#889 positive parity for the zero and offset legs of the capacity
/// controls. The negative cases below must reject malformed metadata without
/// also rejecting the legal boundary cases sitting next to it: a genuinely
/// empty view, and a view based at an offset into a larger buffer.
///
/// Both assertions are profile-independent. The runtime sizes with checked
/// int64 arithmetic rather than with debug overflow checks, so a release build
/// must produce these same exact answers.
#[test]
fn zero_size_and_offset_views_are_accepted_at_their_exact_capacity() {
    let mut backing = [1_i64, 2, 3, 4];
    unsafe {
        // Zero elements admit caller storage and excess declared capacity, but
        // the published view canonicalizes that empty range to null data.
        let shape = [0_i64, 4];
        let empty = chelis_tensor_entry_borrow(
            2,
            shape.as_ptr(),
            CHELIS_DTYPE_I64,
            backing.as_ptr().cast(),
            8,
        );
        assert_eq!(chelis_tensor_numel(empty), 0);
        let empty_view = chelis_tensor_read_view(empty);
        assert_eq!(empty_view.count, 0);
        assert!(empty_view.data.is_null());
        chelis_tensor_release(empty);

        // An element-aligned base inside a larger backing buffer, declared at
        // exactly the remaining capacity, is legal and reads from the offset
        // onward rather than from the start of the allocation.
        let base = backing.as_mut_ptr().add(1);
        let shape = [3_i64];
        let view = chelis_tensor_entry_borrow(1, shape.as_ptr(), CHELIS_DTYPE_I64, base.cast(), 24);
        assert_eq!(chelis_tensor_numel(view), 3);
        let read = chelis_tensor_read_view(view);
        assert_eq!(
            std::slice::from_raw_parts(read.data.cast::<i64>(), read.count as usize),
            &[2, 3, 4]
        );
        chelis_tensor_release(view);
    }
    assert_eq!(
        backing,
        [1, 2, 3, 4],
        "freeing a borrowed offset view must not free or disturb its backing"
    );
}

fn run_invalid_case(case: &str) -> ! {
    unsafe {
        match case {
            "dtype" => {
                chelis_dtype_size(255);
            }
            "scalar-high-bits" => {
                chelis_scalar_from_bits(CHELIS_DTYPE_I8, 0x100);
            }
            "scalar-bool" => {
                chelis_scalar_from_bits(CHELIS_DTYPE_BOOL, 2);
            }
            "scalar-reserved" => {
                let malformed = chelis_scalar {
                    dtype: CHELIS_DTYPE_F32,
                    reserved: [0, 0, 0, 0, 0, 0, 1],
                    bits: 0,
                };
                chelis_value_box_scalar(malformed);
            }
            "value-tag" => {
                let malformed: chelis_value = std::mem::zeroed();
                let malformed = chelis_value {
                    tag: chelis_runtime::chelis_value_tag(255),
                    ..malformed
                };
                chelis_value_unbox_scalar(malformed);
            }
            "value-reserved" => {
                let good = chelis_value_box_scalar(scalar(CHELIS_DTYPE_I64, 1));
                let malformed = chelis_value {
                    reserved: [1; 7],
                    ..good
                };
                chelis_value_unbox_scalar(malformed);
            }
            "rank-negative" => {
                chelis_alloc(-1, ptr::null(), CHELIS_DTYPE_F32);
            }
            "shape-null" => {
                chelis_alloc(1, ptr::null(), CHELIS_DTYPE_F32);
            }
            "extent-negative" => {
                let shape = [-1_i64];
                chelis_alloc(1, shape.as_ptr(), CHELIS_DTYPE_F32);
            }
            "shape-overflow" => {
                let shape = [i64::MAX, 2];
                chelis_tensor_entry_borrow(
                    2,
                    shape.as_ptr(),
                    CHELIS_DTYPE_I64,
                    8_usize as *mut _,
                    i64::MAX,
                );
            }
            "byte-overflow" => {
                // The element count fits int64, but the exact I16 byte count
                // does not. This is distinct from the product overflow above
                // and must fail before inspecting the placeholder data.
                let shape = [(i64::MAX / 2) + 1];
                chelis_tensor_entry_borrow(
                    1,
                    shape.as_ptr(),
                    CHELIS_DTYPE_I16,
                    8_usize as *mut _,
                    i64::MAX,
                );
            }
            // chelis#889 owned-allocation leg of the byte-overflow control.
            // The extent product 2^62 is a legal int64 element count; the f32
            // byte size it names, 2^64, is not. `chelis_alloc` sizes its own
            // storage, so this is the path the `byte-overflow` view case
            // above cannot reach.
            "alloc-byte-overflow" => {
                let shape = [1_i64 << 31, 1_i64 << 31];
                chelis_alloc(2, shape.as_ptr(), CHELIS_DTYPE_F32);
            }
            // chelis#889 view leg of the negative-rank control. `rank-negative`
            // above covers `chelis_alloc`; the view entry point must route
            // through the same checked metadata rather than trusting the rank
            // it was handed.
            "view-rank-negative" => {
                chelis_tensor_entry_borrow(-1, ptr::null(), CHELIS_DTYPE_F32, ptr::null_mut(), 0);
            }
            // chelis#889 view leg of the negative-extent control, the
            // counterpart of `extent-negative` on the borrowed-data path.
            "view-extent-negative" => {
                let mut backing = [0_u64; 2];
                let shape = [-1_i64];
                chelis_tensor_entry_borrow(
                    1,
                    shape.as_ptr(),
                    CHELIS_DTYPE_I64,
                    backing.as_mut_ptr().cast(),
                    16,
                );
            }
            // chelis#889 negative leg of the declared-capacity control. A
            // negative capacity is not merely "too small": it is outside the
            // domain, and it must be rejected before the smaller-than-required
            // comparison the `view-capacity` case exercises.
            "view-capacity-negative" => {
                let mut backing = [0_u64; 2];
                let shape = [2_i64];
                chelis_tensor_entry_borrow(
                    1,
                    shape.as_ptr(),
                    CHELIS_DTYPE_I64,
                    backing.as_mut_ptr().cast(),
                    -1,
                );
            }
            "view-capacity" => {
                let mut backing = [0_u64; 2];
                let shape = [2_i64];
                chelis_tensor_entry_borrow(
                    1,
                    shape.as_ptr(),
                    CHELIS_DTYPE_I64,
                    backing.as_mut_ptr().cast(),
                    15,
                );
            }
            "view-null" => {
                let shape = [1_i64];
                chelis_tensor_entry_borrow(1, shape.as_ptr(), CHELIS_DTYPE_I64, ptr::null_mut(), 8);
            }
            "view-alignment" => {
                let mut backing = [0_u64; 3];
                let shape = [2_i64];
                let misaligned = backing.as_mut_ptr().cast::<u8>().add(1);
                chelis_tensor_entry_borrow(
                    1,
                    shape.as_ptr(),
                    CHELIS_DTYPE_I64,
                    misaligned.cast(),
                    16,
                );
            }
            other => panic!("unknown invalid ABI case: {other}"),
        }
    }
    panic!("invalid ABI case `{case}` returned instead of terminating")
}

/// The exact diagnostic each chelis#889 tensor-metadata control owes.
///
/// The loop below already requires *a* typed trap, which a case can satisfy by
/// failing somewhere else for an unrelated reason. Naming the message pins
/// which entry point rejected the input and why, so Phase 1's move onto the
/// checked capacity types cannot quietly relocate a control's meaning. The
/// rows cover the five controls #889's Phase 1 exit names: negative, zero,
/// product overflow, byte overflow, and declared capacity, on both the owned
/// (`chelis_alloc`) and borrowed (`chelis_tensor_entry_borrow`) paths.
const CAPACITY_CONTROL_DIAGNOSTICS: &[(&str, &str)] = &[
    // negative
    ("rank-negative", "Domain: chelis_alloc has negative rank -1"),
    (
        "view-rank-negative",
        "Domain: chelis_tensor_entry_borrow has negative rank -1",
    ),
    (
        "extent-negative",
        "Domain: chelis_alloc has negative extent -1 at axis 0",
    ),
    (
        "view-extent-negative",
        "Domain: chelis_tensor_entry_borrow has negative extent -1 at axis 0",
    ),
    (
        "view-capacity-negative",
        "Domain: chelis_tensor_entry_borrow has negative byte capacity -1",
    ),
    // product overflow
    (
        "shape-overflow",
        "Overflow: chelis_tensor_entry_borrow extent product exceeds i64",
    ),
    // byte overflow
    (
        "byte-overflow",
        "Overflow: chelis_tensor_entry_borrow byte size exceeds i64",
    ),
    (
        "alloc-byte-overflow",
        "Overflow: chelis_alloc byte size exceeds i64",
    ),
    // declared capacity and base pointer
    (
        "view-capacity",
        "Domain: chelis_tensor_entry_borrow byte capacity 15 is smaller than required 16",
    ),
    (
        "view-null",
        "Domain: chelis_tensor_entry_borrow nonempty tensor has null data",
    ),
    (
        "view-alignment",
        "Domain: chelis_tensor_entry_borrow data pointer is not aligned for i64",
    ),
];

#[test]
fn malformed_foreign_carriers_and_tensor_metadata_fail_loudly() {
    if let Ok(case) = env::var(CHILD_ENV) {
        run_invalid_case(&case);
    }
    let test_binary = env::current_exe().expect("current test binary");
    let cases = [
        "dtype",
        "scalar-high-bits",
        "scalar-bool",
        "scalar-reserved",
        "value-tag",
        "value-reserved",
        "rank-negative",
        "shape-null",
        "extent-negative",
        "shape-overflow",
        "byte-overflow",
        "alloc-byte-overflow",
        "view-rank-negative",
        "view-extent-negative",
        "view-capacity-negative",
        "view-capacity",
        "view-null",
        "view-alignment",
    ];
    for case in cases {
        let output = Command::new(&test_binary)
            .env(CHILD_ENV, case)
            .arg("--exact")
            .arg("malformed_foreign_carriers_and_tensor_metadata_fail_loudly")
            .arg("--nocapture")
            .output()
            .expect("run invalid ABI child");
        assert!(
            !output.status.success(),
            "invalid ABI case `{case}` returned success"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("Domain") || stderr.contains("Overflow"),
            "invalid ABI case `{case}` did not report a typed trap:\n{stderr}"
        );
        if let Some((_, expected)) = CAPACITY_CONTROL_DIAGNOSTICS
            .iter()
            .find(|(name, _)| *name == case)
        {
            assert!(
                stderr.contains(expected),
                "chelis#889 control `{case}` must trap with `{expected}`:\n{stderr}"
            );
        }
    }
    for (case, _) in CAPACITY_CONTROL_DIAGNOSTICS {
        assert!(
            cases.contains(case),
            "chelis#889 control `{case}` is named but never run"
        );
    }
}
