//! [04-SHAPE-1], [05-OP-33]: checked scalar/identity iteration projections.
use chelis_runtime::*;
use std::{env, process::Command, ptr};

const CHILD: &str = "CHELIS_CHECKED_C_INDEXING_CHILD";

fn int(value: i64) -> chelis_scalar {
    chelis_scalar_from_bits(CHELIS_DTYPE_I64, value as u64)
}

unsafe fn step(input: *const chelis_tensor, shape: &[i64]) -> i64 {
    let extents: Vec<_> = shape.iter().copied().map(int).collect();
    chelis_tensor_elementwise_index_step_for_shape(input, int(shape.len() as i64), extents.as_ptr())
}

#[test]
fn exact_shapes_and_explicit_scalars_cover_dynamic_rank_and_all_dtypes() {
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
            let scalar = chelis_alloc(0, ptr::null(), dtype);
            for shape in [
                vec![],
                vec![3],
                vec![1; 8],
                vec![1; 9],
                vec![1; 32],
                vec![2, 0, 3],
            ] {
                let input = chelis_alloc(shape.len() as i32, shape.as_ptr(), dtype);
                let domain = chelis_alloc(shape.len() as i32, shape.as_ptr(), CHELIS_DTYPE_I8);
                let expected = i64::from(!shape.is_empty());
                assert_eq!(
                    chelis_tensor_elementwise_index_step(input, domain),
                    expected
                );
                assert_eq!(step(input, &shape), expected);
                assert_eq!(chelis_tensor_elementwise_index_step(scalar, domain), 0);
                assert_eq!(step(scalar, &shape), 0);
                chelis_tensor_release(input);
                chelis_tensor_release(domain);
            }
            // Iteration domains owe neither storage bytes nor unused suffix strides.
            for shape in [
                vec![i32::MAX as i64 + 1],
                vec![9_007_199_254_740_993],
                vec![i64::MAX],
                vec![0, i64::MAX, i64::MAX],
                vec![i64::MAX, i64::MAX, 0],
            ] {
                assert_eq!(step(scalar, &shape), 0);
            }
            chelis_tensor_release(scalar);
        }
    }
}

#[test]
fn projections_observe_only_metadata_and_preserve_live_write_guards() {
    unsafe {
        let tensor = chelis_alloc(1, [2].as_ptr(), CHELIS_DTYPE_BOOL);
        let guard = chelis_tensor_begin_write(tensor);
        let view = chelis_tensor_write_view(guard);
        view.data.cast::<u8>().write(2); // In-progress payload is intentionally noncanonical.
        assert_eq!(chelis_tensor_elementwise_index_step(tensor, tensor), 1);
        assert_eq!(step(tensor, &[2]), 1);
        assert_eq!(chelis_tensor_rank(tensor), 1);
        view.data.cast::<u8>().write(1);
        chelis_tensor_end_write(guard);
        assert_eq!(chelis_tensor_read_view(tensor).data.cast::<u8>().read(), 1);
        chelis_tensor_release(tensor);
    }
}

unsafe fn invalid_case(case: &str) {
    let scalar = chelis_alloc(0, ptr::null(), CHELIS_DTYPE_I8);
    let matrix = chelis_alloc(2, [2, 3].as_ptr(), CHELIS_DTYPE_I8);
    match case {
        "equal-count-shape" => {
            step(matrix, &[3, 2]);
        }
        "equal-count-rank" => {
            step(matrix, &[6]);
        }
        "nonscalar-into-scalar" => {
            step(matrix, &[]);
        }
        "singleton-is-not-scalar" => {
            let singleton = chelis_alloc(1, [1].as_ptr(), CHELIS_DTYPE_I8);
            step(singleton, &[3]);
        }
        "tensor-shape" => {
            let domain = chelis_alloc(2, [3, 2].as_ptr(), CHELIS_DTYPE_I8);
            chelis_tensor_elementwise_index_step(matrix, domain);
        }
        "empty-shape" => {
            let empty = chelis_alloc(2, [0, 3].as_ptr(), CHELIS_DTYPE_I8);
            step(empty, &[0, 4]);
        }
        "null-input" => {
            step(ptr::null(), &[]);
        }
        "null-domain" => {
            chelis_tensor_elementwise_index_step(scalar, ptr::null());
        }
        "wrong-kind-input" | "wrong-kind-domain" => {
            let list = chelis_list_from_values(ptr::null(), 0);
            if case == "wrong-kind-input" {
                step(list.cast(), &[]);
            } else {
                chelis_tensor_elementwise_index_step(scalar, list.cast());
            }
        }
        "null-shape" => {
            chelis_tensor_elementwise_index_step_for_shape(scalar, int(1), ptr::null());
        }
        "negative-rank" => {
            chelis_tensor_elementwise_index_step_for_shape(scalar, int(-1), ptr::null());
        }
        "rank-overflow" => {
            chelis_tensor_elementwise_index_step_for_shape(
                scalar,
                int(i32::MAX as i64 + 1),
                ptr::null(),
            );
        }
        "wrong-rank-tag" | "unknown-rank-tag" => {
            let mut rank = int(0);
            rank.dtype = if case == "wrong-rank-tag" {
                CHELIS_DTYPE_I32
            } else {
                255
            };
            chelis_tensor_elementwise_index_step_for_shape(scalar, rank, ptr::null());
        }
        "wrong-extent-tag" | "unknown-extent-tag" => {
            let mut extent = int(0);
            extent.dtype = if case == "wrong-extent-tag" {
                CHELIS_DTYPE_I32
            } else {
                255
            };
            chelis_tensor_elementwise_index_step_for_shape(scalar, int(1), &extent);
        }
        "rank-reserved-bits" => {
            let mut rank = int(0);
            rank.reserved[6] = 1;
            chelis_tensor_elementwise_index_step_for_shape(scalar, rank, ptr::null());
        }
        "extent-reserved-after-zero" => {
            let mut extents = [int(0), int(1)];
            extents[1].reserved[0] = 1;
            chelis_tensor_elementwise_index_step_for_shape(scalar, int(2), extents.as_ptr());
        }
        "negative-extent" => {
            step(scalar, &[-1]);
        }
        "negative-after-zero" => {
            step(scalar, &[0, -1]);
        }
        "negative-before-zero" => {
            step(scalar, &[-1, 0]);
        }
        "count-overflow" => {
            step(scalar, &[i64::MAX, 2]);
        }
        "exact-count-overflow" => {
            step(scalar, &[3_074_457_345_618_258_603, 3]);
        }
        "validate-before-repurpose" => {
            step(matrix, &[3, 2]);
            // If validation was bypassed, repurpose would erase the witness.
            let shape = [int(3), int(2)];
            chelis_tensor_repurpose(matrix, int(2), shape.as_ptr());
        }
        _ => panic!("unknown case {case}"),
    }
}

#[test]
fn invalid_domains_fail_before_scalar_empty_or_repurpose_shortcuts() {
    if let Ok(case) = env::var(CHILD) {
        unsafe {
            invalid_case(&case);
        }
        return;
    }
    for case in [
        "equal-count-shape",
        "equal-count-rank",
        "nonscalar-into-scalar",
        "singleton-is-not-scalar",
        "tensor-shape",
        "empty-shape",
        "null-input",
        "null-domain",
        "wrong-kind-input",
        "wrong-kind-domain",
        "null-shape",
        "negative-rank",
        "rank-overflow",
        "wrong-rank-tag",
        "unknown-rank-tag",
        "wrong-extent-tag",
        "unknown-extent-tag",
        "negative-extent",
        "negative-after-zero",
        "negative-before-zero",
        "count-overflow",
        "exact-count-overflow",
        "validate-before-repurpose",
        "rank-reserved-bits",
        "extent-reserved-after-zero",
    ] {
        let output = Command::new(env::current_exe().unwrap())
            .env(CHILD, case)
            .args([
                "--exact",
                "invalid_domains_fail_before_scalar_empty_or_repurpose_shortcuts",
                "--nocapture",
            ])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{case} returned success");
        let stderr = String::from_utf8_lossy(&output.stderr);
        let brand = if case.contains("overflow") {
            "Overflow:"
        } else {
            "Domain:"
        };
        assert!(
            stderr.contains(brand),
            "{case} must report {brand}: {stderr}"
        );
        assert!(!stderr.contains("panicked at"), "{case} panicked: {stderr}");
    }
}
