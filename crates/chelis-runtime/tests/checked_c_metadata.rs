//! [04-SHAPE-1], [05-OP-31/33]: immutable checked metadata at the C boundary.
use chelis_runtime::*;
use std::{env, process::Command, ptr};

const CHILD: &str = "CHELIS_CHECKED_C_METADATA_CHILD";

unsafe fn shape_list(dims: &[i64]) -> *mut chelis_list {
    let values: Vec<_> = dims
        .iter()
        .map(|&dim| chelis_value_box_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_I64, dim as u64)))
        .collect();
    chelis_list_from_values(values.as_ptr(), values.len() as i64)
}

// Keep malformed null/rank cases observable while transporting every readable
// extent through the same canonical tagged scalar that generated C uses.
unsafe fn check_reshape(tensor: *const chelis_tensor, rank: i32, shape: *const i64) {
    let extents: Vec<_> = if shape.is_null() || rank < 0 {
        vec![]
    } else {
        (0..rank as usize)
            .map(|axis| chelis_scalar_from_bits(CHELIS_DTYPE_I64, shape.add(axis).read() as u64))
            .collect()
    };
    chelis_tensor_check_reshape(
        tensor,
        chelis_scalar_from_bits(CHELIS_DTYPE_I64, i64::from(rank) as u64),
        if shape.is_null() {
            ptr::null()
        } else {
            extents.as_ptr()
        },
    );
}

#[test]
fn owned_reshape_preserves_exact_bits_and_independent_storage_for_every_dtype() {
    unsafe {
        for (dtype, bits) in [
            (CHELIS_DTYPE_F32, 0x7fc1_2345),
            (CHELIS_DTYPE_F64, 0x7ff8_1234_5678_9abc),
            (CHELIS_DTYPE_I32, 0x8123_4567),
            (CHELIS_DTYPE_BOOL, 1),
            (CHELIS_DTYPE_I64, 0x8123_4567_89ab_cdef),
            (CHELIS_DTYPE_BF16, 0x7fc1),
            (CHELIS_DTYPE_F16, 0x7e01),
            (CHELIS_DTYPE_I8, 0x81),
            (CHELIS_DTYPE_I16, 0x8123),
        ] {
            let input = chelis_alloc(2, [2, 3].as_ptr(), dtype);
            let guard = chelis_tensor_begin_write(input);
            chelis_fill_scalar(guard, chelis_scalar_from_bits(dtype, bits));
            chelis_tensor_end_write(guard);
            let shape = shape_list(&[3, 2]);
            let output = chelis_tensor_reshape(input, shape);
            chelis_list_release(shape);
            assert_ne!(input, output);
            assert_eq!(chelis_tensor_shape(input, 0), 2);
            assert_eq!(chelis_tensor_shape(output, 0), 3);
            let before = chelis_tensor_read_view(input);
            let after = chelis_tensor_read_view(output);
            assert_eq!(before.dtype, after.dtype);
            assert_eq!(after.count, 6);
            let bytes = chelis_tensor_byte_count(input) as usize;
            assert_eq!(
                std::slice::from_raw_parts(before.data.cast::<u8>(), bytes),
                std::slice::from_raw_parts(after.data.cast::<u8>(), bytes)
            );
            let saved = std::slice::from_raw_parts(before.data.cast::<u8>(), bytes).to_vec();
            let guard = chelis_tensor_begin_write(output);
            chelis_fill_scalar(guard, chelis_scalar_from_bits(dtype, 0));
            chelis_tensor_end_write(guard);
            let input_view = chelis_tensor_read_view(input);
            assert_eq!(
                std::slice::from_raw_parts(input_view.data.cast::<u8>(), bytes),
                saved
            );
            chelis_tensor_release(input);
            chelis_tensor_release(output);
        }
    }
}

#[test]
fn checked_metadata_observations_preserve_dtype_width_and_live_guard_access() {
    unsafe {
        for (dtype, width) in [
            (CHELIS_DTYPE_F32, 4),
            (CHELIS_DTYPE_F64, 8),
            (CHELIS_DTYPE_I32, 4),
            (CHELIS_DTYPE_BOOL, 1),
            (CHELIS_DTYPE_I64, 8),
            (CHELIS_DTYPE_BF16, 2),
            (CHELIS_DTYPE_F16, 2),
            (CHELIS_DTYPE_I8, 1),
            (CHELIS_DTYPE_I16, 2),
        ] {
            let tensor = chelis_alloc(2, [2, 3].as_ptr(), dtype);
            assert_eq!(chelis_tensor_stride(tensor, 0), 3);
            assert_eq!(chelis_tensor_stride(tensor, 1), 1);
            assert_eq!(chelis_tensor_stride(tensor, -2), 3);
            assert_eq!(chelis_tensor_stride(tensor, -1), 1);
            assert_eq!(chelis_tensor_byte_count(tensor), 6 * width);
            let guard = chelis_tensor_begin_write(tensor);
            assert_eq!(chelis_tensor_stride(tensor, 0), 3);
            assert_eq!(chelis_tensor_byte_count(tensor), 6 * width);
            check_reshape(tensor, 2, [3, 2].as_ptr());
            assert_eq!(chelis_tensor_shape(tensor, 0), 2);
            assert_eq!(chelis_tensor_shape(tensor, 1), 3);
            chelis_tensor_end_write(guard);
            chelis_tensor_release(tensor);
        }
    }
}

#[test]
fn rank_zero_and_zero_extents_keep_exact_metadata_without_large_allocations() {
    unsafe {
        let scalar = chelis_alloc(0, ptr::null(), CHELIS_DTYPE_I64);
        assert_eq!(chelis_tensor_byte_count(scalar), 8);
        check_reshape(scalar, 1, [1].as_ptr());
        check_reshape(scalar, 0, ptr::null());
        chelis_tensor_release(scalar);

        for (shape, strides) in [
            ([0, 3, 4], [12, 4, 1]),
            ([2, 0, 4], [0, 4, 1]),
            ([2, 3, 0], [0, 0, 1]),
            ([i64::MAX, i64::MAX, 0], [0, 0, 1]),
        ] {
            let tensor = chelis_alloc(3, shape.as_ptr(), CHELIS_DTYPE_F64);
            assert_eq!(chelis_tensor_numel(tensor), 0);
            assert_eq!(chelis_tensor_byte_count(tensor), 0);
            for (axis, stride) in strides.into_iter().enumerate() {
                assert_eq!(chelis_tensor_stride(tensor, axis as i32), stride);
            }
            check_reshape(tensor, 2, [0, i64::MAX].as_ptr());
            chelis_tensor_release(tensor);
        }
        let wide = chelis_alloc(2, [0, i64::MAX].as_ptr(), CHELIS_DTYPE_I8);
        assert_eq!(chelis_tensor_stride(wide, 0), i64::MAX);
        chelis_tensor_release(wide);
    }
}

#[test]
fn owned_reshape_supports_scalar_empty_and_dynamic_rank() {
    unsafe {
        for dims in [
            vec![],
            vec![1],
            vec![1; 8],
            vec![1; 11],
            vec![i64::MAX, i64::MAX, 0],
        ] {
            let empty = dims.contains(&0);
            let input = chelis_alloc(1, [if empty { 0 } else { 1 }].as_ptr(), CHELIS_DTYPE_I64);
            let shape = shape_list(&dims);
            let output = chelis_tensor_reshape(input, shape);
            assert_eq!(chelis_tensor_rank(output) as usize, dims.len());
            assert_eq!(chelis_tensor_numel(output), if empty { 0 } else { 1 });
            for (axis, &extent) in dims.iter().enumerate() {
                assert_eq!(chelis_tensor_shape(output, axis as i32), extent);
            }
            if empty {
                assert!(chelis_tensor_read_view(output).data.is_null());
            }
            chelis_tensor_release(output);
            chelis_list_release(shape);
            chelis_tensor_release(input);
        }
    }
}

#[test]
fn byte_observation_excludes_spare_foreign_capacity() {
    unsafe {
        let storage = [17_i64; 8];
        let tensor = chelis_tensor_entry_borrow(
            1,
            [2].as_ptr(),
            CHELIS_DTYPE_I64,
            storage.as_ptr().cast(),
            64,
        );
        assert_eq!(chelis_tensor_byte_count(tensor), 16);
        let shape = shape_list(&[1, 2]);
        let output = chelis_tensor_reshape(tensor, shape);
        assert_eq!(chelis_tensor_byte_count(output), 16);
        let view = chelis_tensor_read_view(output);
        assert_eq!(
            std::slice::from_raw_parts(view.data.cast::<i64>(), 2),
            &[17, 17]
        );
        chelis_tensor_release(output);
        chelis_list_release(shape);
        chelis_tensor_release(tensor);
    }
}

fn invalid_case(case: &str) {
    unsafe {
        let tensor = chelis_alloc(2, [2, 3].as_ptr(), CHELIS_DTYPE_F64);
        match case {
            "reshape-rank-unknown-dtype" => {
                let mut rank = chelis_scalar_from_bits(CHELIS_DTYPE_I64, 0);
                rank.dtype = 255;
                chelis_tensor_check_reshape(tensor, rank, ptr::null());
            }
            "reshape-extent-unknown-dtype" => {
                let mut extent = chelis_scalar_from_bits(CHELIS_DTYPE_I64, 6);
                extent.dtype = 255;
                chelis_tensor_check_reshape(
                    tensor,
                    chelis_scalar_from_bits(CHELIS_DTYPE_I64, 1),
                    &extent,
                );
            }

            "reshape-rank-wrong-dtype" => {
                chelis_tensor_check_reshape(
                    tensor,
                    chelis_scalar_from_bits(CHELIS_DTYPE_I32, 0),
                    ptr::null(),
                );
            }
            "reshape-rank-overflow" => {
                chelis_tensor_check_reshape(
                    tensor,
                    chelis_scalar_from_bits(CHELIS_DTYPE_I64, i32::MAX as u64 + 1),
                    ptr::null(),
                );
            }
            "reshape-extent-wrong-dtype" => {
                chelis_tensor_check_reshape(
                    tensor,
                    chelis_scalar_from_bits(CHELIS_DTYPE_I64, 1),
                    &chelis_scalar_from_bits(CHELIS_DTYPE_I32, 6),
                );
            }
            "reshape-extent-noncanonical" => {
                let mut extent = chelis_scalar_from_bits(CHELIS_DTYPE_I64, 6);
                extent.reserved[0] = 1;
                chelis_tensor_check_reshape(
                    tensor,
                    chelis_scalar_from_bits(CHELIS_DTYPE_I64, 1),
                    &extent,
                );
            }

            "owned-reshape-null-shape" => {
                chelis_tensor_reshape(tensor, ptr::null());
            }
            "owned-reshape-wrong-shape-kind" => {
                chelis_tensor_reshape(tensor, tensor.cast());
            }
            "owned-reshape-unit-extent" => {
                let value: chelis_value = std::mem::zeroed();
                chelis_tensor_reshape(tensor, chelis_list_from_values(&value, 1));
            }
            "owned-reshape-negative-extent" => {
                chelis_tensor_reshape(tensor, shape_list(&[0, -1]));
            }
            "owned-reshape-byte-overflow" => {
                chelis_tensor_reshape(tensor, shape_list(&[i64::MAX / 8 + 1]));
            }
            "owned-reshape-stride-overflow" => {
                chelis_tensor_reshape(tensor, shape_list(&[0, i64::MAX, i64::MAX]));
            }

            "stride-axis-high" => {
                chelis_tensor_stride(tensor, 2);
            }
            "stride-axis-low" => {
                chelis_tensor_stride(tensor, -3);
            }
            "stride-axis-min" => {
                chelis_tensor_stride(tensor, i32::MIN);
            }
            "stride-null" => {
                chelis_tensor_stride(ptr::null(), 0);
            }
            "bytes-null" => {
                chelis_tensor_byte_count(ptr::null());
            }
            "reshape-null" => {
                check_reshape(ptr::null(), 0, ptr::null());
            }
            "reshape-rank-negative" => {
                check_reshape(tensor, -1, ptr::null());
            }
            "reshape-shape-null" => {
                check_reshape(tensor, 1, ptr::null());
            }
            "reshape-extent-negative" => {
                check_reshape(tensor, 2, [0, -1].as_ptr());
            }
            "reshape-count-mismatch" => {
                check_reshape(tensor, 1, [5].as_ptr());
            }
            "reshape-count-above-u32" => {
                check_reshape(tensor, 1, [4_294_967_302].as_ptr());
            }
            "reshape-product-overflow" => {
                check_reshape(tensor, 2, [i64::MAX, 2].as_ptr());
            }
            "reshape-byte-overflow" => {
                check_reshape(tensor, 1, [i64::MAX / 8 + 1].as_ptr());
            }
            "reshape-stride-overflow" => {
                check_reshape(tensor, 3, [0, i64::MAX, i64::MAX].as_ptr());
            }
            "owned-reshape-wrong-count" => {
                chelis_tensor_reshape(tensor, shape_list(&[5]));
            }
            "owned-reshape-product-overflow" => {
                chelis_tensor_reshape(tensor, shape_list(&[i64::MAX, 2]));
            }
            "owned-reshape-guarded-input" => {
                chelis_tensor_begin_write(tensor);
                chelis_tensor_reshape(tensor, shape_list(&[3, 2]));
            }
            "owned-reshape-wrong-extent-dtype" => {
                let value = chelis_value_box_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_I32, 6));
                let shape = chelis_list_from_values(&value, 1);
                chelis_tensor_reshape(tensor, shape);
            }
            "stride-scalar" => {
                let scalar = chelis_alloc(0, ptr::null(), CHELIS_DTYPE_I8);
                chelis_tensor_stride(scalar, 0);
            }
            _ => panic!("unknown case {case}"),
        }
        chelis_tensor_release(tensor);
    }
}

#[test]
fn metadata_boundary_rejects_invalid_inputs_before_allocation_or_access() {
    if let Ok(case) = env::var(CHILD) {
        invalid_case(&case);
        panic!("invalid metadata returned normally: {case}");
    }
    for (case, brand) in [
        ("reshape-rank-unknown-dtype", "Domain:"),
        ("reshape-extent-unknown-dtype", "Domain:"),
        ("reshape-rank-wrong-dtype", "Domain:"),
        ("reshape-rank-overflow", "Overflow:"),
        ("reshape-extent-wrong-dtype", "Domain:"),
        ("reshape-extent-noncanonical", "Domain:"),
        ("owned-reshape-null-shape", "Domain:"),
        ("owned-reshape-wrong-shape-kind", "Domain:"),
        ("owned-reshape-unit-extent", "Domain:"),
        ("owned-reshape-negative-extent", "Domain:"),
        ("owned-reshape-byte-overflow", "Overflow:"),
        ("owned-reshape-stride-overflow", "Overflow:"),
        ("stride-axis-high", "Domain:"),
        ("stride-axis-low", "Domain:"),
        ("stride-axis-min", "Domain:"),
        ("stride-null", "Domain:"),
        ("bytes-null", "Domain:"),
        ("reshape-null", "Domain:"),
        ("reshape-rank-negative", "Domain:"),
        ("reshape-shape-null", "Domain:"),
        ("reshape-extent-negative", "Domain:"),
        ("reshape-count-mismatch", "Domain:"),
        ("reshape-count-above-u32", "Domain:"),
        ("stride-scalar", "Domain:"),
        ("reshape-product-overflow", "Overflow:"),
        ("reshape-byte-overflow", "Overflow:"),
        ("reshape-stride-overflow", "Overflow:"),
        ("owned-reshape-wrong-count", "Domain:"),
        ("owned-reshape-product-overflow", "Overflow:"),
        ("owned-reshape-guarded-input", "Domain:"),
        ("owned-reshape-wrong-extent-dtype", "Domain:"),
    ] {
        let output = Command::new(env::current_exe().unwrap())
            .env(CHILD, case)
            .args([
                "--exact",
                "metadata_boundary_rejects_invalid_inputs_before_allocation_or_access",
                "--nocapture",
            ])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{case} returned success");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(brand),
            "{case} must report {brand}: {stderr}"
        );
        assert!(!stderr.contains("panicked at"), "{case} panicked: {stderr}");
    }
}
