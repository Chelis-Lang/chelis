//! [04-SHAPE-1], [05-MOV-1], [05-OP-33]: checked movement coordinates.
use chelis_runtime::*;
use std::{env, process::Command, ptr};

const CHILD: &str = "CHELIS_CHECKED_C_MOVEMENT_CHILD";

fn int(value: i64) -> chelis_scalar {
    chelis_scalar_from_bits(CHELIS_DTYPE_I64, value as u64)
}

unsafe fn permute(input: *const chelis_tensor, shape: &[i64], axes: &[i64]) {
    let shape: Vec<_> = shape.iter().copied().map(int).collect();
    let axes: Vec<_> = axes.iter().copied().map(int).collect();
    chelis_tensor_check_permute(
        input,
        int(shape.len() as i64),
        shape.as_ptr(),
        axes.as_ptr(),
    );
}

unsafe fn expand(input: *const chelis_tensor, shape: &[i64], axis: i32) {
    let shape: Vec<_> = shape.iter().copied().map(int).collect();
    chelis_tensor_check_expand(input, int(shape.len() as i64), shape.as_ptr(), axis);
}

#[test]
fn movement_coordinates_are_exact_for_every_dtype_and_dynamic_rank() {
    unsafe {
        for dtype in [
            CHELIS_DTYPE_F32,
            CHELIS_DTYPE_F64,
            CHELIS_DTYPE_I32,
            CHELIS_DTYPE_BOOL,
            CHELIS_DTYPE_I64,
            CHELIS_DTYPE_BF16,
            CHELIS_DTYPE_F16,
            CHELIS_DTYPE_I8,
            CHELIS_DTYPE_I16,
        ] {
            for shape in [
                vec![],
                vec![3],
                vec![1; 8],
                vec![1; 9],
                vec![1; 32],
                vec![2, 3],
            ] {
                let tensor = chelis_alloc(shape.len() as i32, shape.as_ptr(), dtype);
                let axes: Vec<_> = (0..shape.len() as i64).rev().collect();
                let reversed: Vec<_> = shape.iter().copied().rev().collect();
                permute(tensor, &reversed, &axes);
                for linear in 0..chelis_tensor_numel(tensor) {
                    let mut coordinates = vec![int(-1); shape.len()];
                    chelis_tensor_unravel_index(tensor, int(linear), coordinates.as_mut_ptr());
                    assert!(coordinates
                        .iter()
                        .all(|n| n.dtype == CHELIS_DTYPE_I64 && n.reserved == [0; 7]));
                    assert_eq!(
                        chelis_tensor_flat_index(tensor, coordinates.as_ptr()),
                        linear
                    );
                }
                chelis_tensor_release(tensor);
            }
            let matrix = chelis_alloc(2, [2, 3].as_ptr(), dtype);
            permute(matrix, &[3, 2], &[-1, -2]);
            let mut coordinates = [int(0); 2];
            chelis_tensor_unravel_index(matrix, int(5), coordinates.as_mut_ptr());
            assert_eq!(coordinates.map(|n| n.bits), [1, 2]);
            chelis_tensor_release(matrix);
            let input = chelis_alloc(2, [2, 1].as_ptr(), dtype);
            expand(input, &[2, 3], -1);
            expand(input, &[2, 0], 1);
            expand(input, &[4, 2, 1], 0);
            chelis_tensor_release(input);
            let empty = chelis_alloc(3, [2, 0, 3].as_ptr(), dtype);
            permute(empty, &[3, 2, 0], &[2, 0, 1]);
            expand(empty, &[2, 0, 0, 3], 2);
            chelis_tensor_release(empty);
            let scalar = chelis_alloc(0, ptr::null(), dtype);
            chelis_tensor_check_permute(scalar, int(0), ptr::null(), ptr::null());
            chelis_tensor_unravel_index(scalar, int(0), ptr::null_mut());
            assert_eq!(chelis_tensor_flat_index(scalar, ptr::null()), 0);
            expand(scalar, &[3], 0);
            chelis_tensor_release(scalar);
        }
    }
}

#[test]
fn movement_validation_preserves_large_empty_metadata_and_live_write_guards() {
    unsafe {
        let unit = chelis_alloc(1, [1].as_ptr(), CHELIS_DTYPE_I8);
        for extent in [i32::MAX as i64 + 1, 9_007_199_254_740_993, i64::MAX] {
            expand(unit, &[extent], 0); // Metadata validation does not allocate the result.
            let empty = chelis_alloc(2, [extent, 0].as_ptr(), CHELIS_DTYPE_I8);
            permute(empty, &[0, extent], &[1, 0]);
            chelis_tensor_release(empty);
        }
        chelis_tensor_release(unit);
        let tensor = chelis_alloc(1, [2].as_ptr(), CHELIS_DTYPE_BOOL);
        let guard = chelis_tensor_begin_write(tensor);
        let view = chelis_tensor_write_view(guard);
        view.data.cast::<u8>().write(2);
        permute(tensor, &[2], &[0]);
        expand(tensor, &[3, 2], 0);
        let mut coordinate = int(-1);
        chelis_tensor_unravel_index(tensor, int(1), &mut coordinate);
        assert_eq!(chelis_tensor_flat_index(tensor, &coordinate), 1);
        view.data.cast::<u8>().write(1);
        chelis_tensor_end_write(guard);
        chelis_tensor_release(tensor);
    }
}

unsafe fn invalid(case: &str) {
    let tensor = chelis_alloc(2, [2, 3].as_ptr(), CHELIS_DTYPE_I8);
    match case {
        "duplicate-axes" => permute(tensor, &[2, 2], &[0, 0]),
        "duplicate-normalized-axes" => permute(tensor, &[3, 3], &[1, -1]),
        "axis-range" => permute(tensor, &[2, 3], &[0, 2]),
        "permutation-shape" => permute(tensor, &[2, 3], &[1, 0]),
        "permutation-rank" => permute(tensor, &[6], &[0]),
        "wrong-axis-tag" => chelis_tensor_check_permute(
            tensor,
            int(2),
            [int(3), int(2)].as_ptr(),
            [chelis_scalar_from_bits(CHELIS_DTYPE_F64, 1), int(0)].as_ptr(),
        ),
        "reserved-axis" => {
            let mut axis = int(1);
            axis.reserved[0] = 1;
            chelis_tensor_check_permute(
                tensor,
                int(2),
                [int(3), int(2)].as_ptr(),
                [axis, int(0)].as_ptr(),
            );
        }
        "large-axis" => permute(tensor, &[3, 2], &[i64::MAX, 0]),
        "minimum-axis" => permute(tensor, &[3, 2], &[i64::MIN, 0]),
        "null-axes" => {
            chelis_tensor_check_permute(tensor, int(2), [int(2), int(3)].as_ptr(), ptr::null())
        }
        "expand-nonunit" => expand(tensor, &[2, 4], 1),
        "expand-bystander" => expand(tensor, &[3, 2, 4], 0),
        "expand-rank" => expand(tensor, &[6], 0),
        "expand-axis" => expand(tensor, &[2, 3], -3),
        "empty-mismatch" => {
            let empty = chelis_alloc(2, [0, 2].as_ptr(), CHELIS_DTYPE_I8);
            permute(empty, &[3, 0], &[1, 0]);
        }
        "negative-index" | "index-range" => {
            let mut out = [int(0); 2];
            chelis_tensor_unravel_index(
                tensor,
                int(if case == "negative-index" { -1 } else { 6 }),
                out.as_mut_ptr(),
            );
        }
        "empty-index" => {
            let empty = chelis_alloc(1, [0].as_ptr(), CHELIS_DTYPE_I8);
            chelis_tensor_unravel_index(empty, int(0), &mut int(0));
        }
        "coordinate-range" => {
            chelis_tensor_flat_index(tensor, [int(0), int(3)].as_ptr());
        }
        "negative-coordinate" => {
            chelis_tensor_flat_index(tensor, [int(-1), int(0)].as_ptr());
        }
        "coordinate-tag" | "coordinate-reserved" => {
            let mut coordinate = int(0);
            if case == "coordinate-tag" {
                coordinate.dtype = CHELIS_DTYPE_I32;
            } else {
                coordinate.reserved[6] = 1;
            }
            chelis_tensor_flat_index(tensor, [coordinate, int(0)].as_ptr());
        }
        "index-tag" => {
            chelis_tensor_unravel_index(
                tensor,
                chelis_scalar_from_bits(CHELIS_DTYPE_F32, 0),
                [int(0); 2].as_mut_ptr(),
            );
        }
        "null-coordinates" => {
            chelis_tensor_flat_index(tensor, ptr::null());
        }
        "null-output" => chelis_tensor_unravel_index(tensor, int(0), ptr::null_mut()),
        "null-tensor" => {
            chelis_tensor_flat_index(ptr::null(), ptr::null());
        }
        "wrong-handle" => {
            let list = chelis_list_from_values(ptr::null(), 0);
            chelis_tensor_flat_index(list.cast(), ptr::null());
        }
        "null-shape" => chelis_tensor_check_expand(tensor, int(2), ptr::null(), 0),
        "negative-rank" => chelis_tensor_check_expand(tensor, int(-1), ptr::null(), 0),
        "rank-overflow" => {
            chelis_tensor_check_expand(tensor, int(i32::MAX as i64 + 1), ptr::null(), 0)
        }
        "rank-tag" => chelis_tensor_check_expand(
            tensor,
            chelis_scalar_from_bits(CHELIS_DTYPE_I32, 0),
            ptr::null(),
            0,
        ),
        "extent-tag-after-zero" => chelis_tensor_check_expand(
            tensor,
            int(2),
            [int(0), chelis_scalar_from_bits(CHELIS_DTYPE_I32, 1)].as_ptr(),
            0,
        ),
        "negative-after-zero" => expand(tensor, &[0, -1], 0),
        "product-overflow" => expand(tensor, &[i64::MAX, 2], 0),
        "stride-overflow" => expand(tensor, &[0, i64::MAX, i64::MAX], 0),
        "byte-overflow" => {
            let wide = chelis_alloc(1, [1].as_ptr(), CHELIS_DTYPE_F64);
            expand(wide, &[i64::MAX], 0);
        }
        _ => panic!("unknown case {case}"),
    }
}

#[test]
fn malformed_movement_metadata_and_indices_fail_with_the_owning_reason() {
    if let Ok(case) = env::var(CHILD) {
        unsafe {
            invalid(&case);
        }
        return;
    }
    for case in [
        "duplicate-axes",
        "duplicate-normalized-axes",
        "axis-range",
        "permutation-shape",
        "permutation-rank",
        "null-axes",
        "wrong-axis-tag",
        "reserved-axis",
        "large-axis",
        "minimum-axis",
        "expand-nonunit",
        "expand-bystander",
        "expand-rank",
        "expand-axis",
        "empty-mismatch",
        "negative-index",
        "index-range",
        "empty-index",
        "coordinate-range",
        "negative-coordinate",
        "coordinate-tag",
        "coordinate-reserved",
        "index-tag",
        "null-coordinates",
        "null-output",
        "null-tensor",
        "wrong-handle",
        "null-shape",
        "negative-rank",
        "rank-overflow",
        "rank-tag",
        "extent-tag-after-zero",
        "negative-after-zero",
        "product-overflow",
        "stride-overflow",
        "byte-overflow",
    ] {
        let output = Command::new(env::current_exe().unwrap())
            .env(CHILD, case)
            .args([
                "--exact",
                "malformed_movement_metadata_and_indices_fail_with_the_owning_reason",
                "--nocapture",
            ])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{case} returned success");
        let stderr = String::from_utf8_lossy(&output.stderr);
        let reason = if case.contains("overflow") {
            "Overflow:"
        } else {
            "Domain:"
        };
        assert!(
            stderr.contains(reason),
            "{case}: expected {reason}: {stderr}"
        );
        assert!(!stderr.contains("panicked at"), "{case}: {stderr}");
    }
}
