//! Executable contract for the v0.19 exact tagged public C ABI.
//!
//! These tests are derived from `spec/05-risc-primitives.md`
//! [05-OP-31] and [05-OP-33].  The child-process cases exercise malformed
//! foreign carriers because the C boundary must reject them before sizing,
//! allocation, access, or observation.

use chelis_runtime::{
    chelis_alloc, chelis_alloc_view, chelis_dtype, chelis_dtype_size, chelis_fill_scalar,
    chelis_free, chelis_parse_scalar, chelis_scalar, chelis_scalar_from_bits, chelis_scalar_tensor,
    chelis_string_from_cstr, chelis_string_release, chelis_tensor_numel, chelis_tensor_rank,
    chelis_tensor_shape, chelis_tensor_to_scalar, chelis_value, chelis_value_as_scalar,
    chelis_value_from_scalar, CHELIS_DTYPE_BF16, CHELIS_DTYPE_BOOL, CHELIS_DTYPE_F16,
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
        (parsed.is_some != 0).then_some(parsed.value)
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

            let boxed = chelis_value_from_scalar(value);
            assert_eq!(boxed.tag, CHELIS_VALUE_SCALAR);
            assert_eq!(boxed.reserved, [0; 7]);
            assert_eq!(chelis_value_as_scalar(boxed).bits, bits);

            let tensor = chelis_scalar_tensor(value);
            assert_eq!(chelis_tensor_rank(tensor), 0);
            assert_eq!(chelis_tensor_numel(tensor), 1);
            assert_eq!(chelis_tensor_to_scalar(tensor).bits, bits);
            chelis_free(tensor);
        }
    }
}

#[test]
fn bool_is_one_byte_and_fill_preserves_the_canonical_bit() {
    unsafe {
        assert_eq!(chelis_dtype_size(CHELIS_DTYPE_BOOL), 1);
        let shape = [4_i64];
        let tensor = chelis_alloc(1, shape.as_ptr(), CHELIS_DTYPE_BOOL);
        chelis_fill_scalar(tensor, scalar(CHELIS_DTYPE_BOOL, 1));
        let bytes = std::slice::from_raw_parts((*tensor).data.cast::<u8>(), 4);
        assert_eq!(bytes, &[1, 1, 1, 1]);
        chelis_free(tensor);
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
                assert_eq!(*(*tensor).strides.add(axis as usize), 1);
            }
            if rank == 0 {
                assert!((*tensor).shape.is_null());
                assert!((*tensor).strides.is_null());
            } else {
                assert!(!(*tensor).shape.is_null());
                assert!(!(*tensor).strides.is_null());
            }
            chelis_free(tensor);
        }
    }
}

#[test]
fn borrowed_view_copies_shape_and_honors_declared_capacity() {
    let mut backing = vec![0_u64; 6];
    let mut shape = [2_i64, 3];
    unsafe {
        let tensor = chelis_alloc_view(
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
        assert_eq!((*tensor).byte_capacity, 48);
        assert_eq!((*tensor).owns_data, 0);
        chelis_free(tensor);
    }
    backing[0] = 7;
    assert_eq!(
        backing[0], 7,
        "freeing a borrowed view must not free its data"
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
                chelis_value_from_scalar(malformed);
            }
            "value-tag" => {
                let malformed: chelis_value = std::mem::zeroed();
                let malformed = chelis_value {
                    tag: chelis_runtime::chelis_value_tag(255),
                    ..malformed
                };
                chelis_value_as_scalar(malformed);
            }
            "value-reserved" => {
                let good = chelis_value_from_scalar(scalar(CHELIS_DTYPE_I64, 1));
                let malformed = chelis_value {
                    reserved: [1; 7],
                    ..good
                };
                chelis_value_as_scalar(malformed);
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
                chelis_alloc_view(
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
                chelis_alloc_view(
                    1,
                    shape.as_ptr(),
                    CHELIS_DTYPE_I16,
                    8_usize as *mut _,
                    i64::MAX,
                );
            }
            "view-capacity" => {
                let mut backing = [0_u64; 2];
                let shape = [2_i64];
                chelis_alloc_view(
                    1,
                    shape.as_ptr(),
                    CHELIS_DTYPE_I64,
                    backing.as_mut_ptr().cast(),
                    15,
                );
            }
            "view-null" => {
                let shape = [1_i64];
                chelis_alloc_view(1, shape.as_ptr(), CHELIS_DTYPE_I64, ptr::null_mut(), 8);
            }
            "view-alignment" => {
                let mut backing = [0_u64; 3];
                let shape = [2_i64];
                let misaligned = backing.as_mut_ptr().cast::<u8>().add(1);
                chelis_alloc_view(1, shape.as_ptr(), CHELIS_DTYPE_I64, misaligned.cast(), 16);
            }
            "zero-view-data" => {
                let mut byte = 0_u8;
                let shape = [0_i64];
                chelis_alloc_view(
                    1,
                    shape.as_ptr(),
                    CHELIS_DTYPE_I8,
                    (&mut byte as *mut u8).cast(),
                    0,
                );
            }
            "zero-view-capacity" => {
                let shape = [0_i64];
                chelis_alloc_view(1, shape.as_ptr(), CHELIS_DTYPE_I8, ptr::null_mut(), 1);
            }
            other => panic!("unknown invalid ABI case: {other}"),
        }
    }
    panic!("invalid ABI case `{case}` returned instead of terminating")
}

#[test]
fn malformed_foreign_carriers_and_tensor_metadata_fail_loudly() {
    if let Ok(case) = env::var(CHILD_ENV) {
        run_invalid_case(&case);
    }
    let test_binary = env::current_exe().expect("current test binary");
    for case in [
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
        "view-capacity",
        "view-null",
        "view-alignment",
        "zero-view-data",
        "zero-view-capacity",
    ] {
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
    }
}
