//! [05-MOV-1]/[05-OP-33]: checked affine movement metadata, in both profiles.
use chelis_runtime::*;
use std::{env, process::Command, ptr};

fn int(value: i64) -> chelis_scalar {
    chelis_scalar_from_bits(CHELIS_DTYPE_I64, value as u64)
}

#[test]
fn affine_shapes_and_coordinates_preserve_exact_metadata() {
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
            for rank in [0, 1, 2, 8, 9, 32] {
                let shape = vec![2; rank];
                // Keep high-rank cases small while still exercising every axis.
                let shape = if rank > 2 { vec![1; rank] } else { shape };
                let tensor = chelis_alloc(rank as i32, shape.as_ptr(), dtype);
                let guard = chelis_tensor_begin_write(tensor);
                let zeros = vec![int(0); rank];
                let ones = vec![int(1); rank];
                let ends: Vec<_> = shape.iter().copied().map(int).collect();
                let mut out = vec![int(-1); rank];
                chelis_tensor_pad_shape(
                    tensor,
                    int(rank as i64),
                    zeros.as_ptr(),
                    ones.as_ptr(),
                    out.as_mut_ptr(),
                );
                assert_eq!(
                    out.iter().map(|n| n.bits as i64).collect::<Vec<_>>(),
                    shape.iter().map(|n| n + 1).collect::<Vec<_>>()
                );
                chelis_tensor_shrink_shape(
                    tensor,
                    int(rank as i64),
                    zeros.as_ptr(),
                    ends.as_ptr(),
                    out.as_mut_ptr(),
                );
                assert_eq!(out.iter().map(|n| n.bits as i64).collect::<Vec<_>>(), shape);
                chelis_tensor_stride_shape(
                    tensor,
                    int(rank as i64),
                    ones.as_ptr(),
                    out.as_mut_ptr(),
                );
                assert_eq!(out.iter().map(|n| n.bits as i64).collect::<Vec<_>>(), shape);
                assert_eq!(
                    chelis_tensor_affine_index(
                        tensor,
                        zeros.as_ptr(),
                        zeros.as_ptr(),
                        ones.as_ptr()
                    ),
                    0
                );
                assert!(out
                    .iter()
                    .all(|n| n.dtype == CHELIS_DTYPE_I64 && n.reserved == [0; 7]));
                chelis_tensor_end_write(guard);
                chelis_tensor_release(tensor);
            }
        }
        let tensor = chelis_alloc(2, [3, 5].as_ptr(), CHELIS_DTYPE_I8);
        assert_eq!(
            chelis_tensor_affine_index(
                tensor,
                [int(1), int(1)].as_ptr(),
                [int(0), int(1)].as_ptr(),
                [int(2), int(3)].as_ptr()
            ),
            14
        );
        chelis_tensor_release(tensor);
        let empty = chelis_alloc(2, [i64::MAX, 0].as_ptr(), CHELIS_DTYPE_I8);
        let mut out = [int(-1); 2];
        chelis_tensor_stride_shape(
            empty,
            int(2),
            [int(i64::MAX), int(1)].as_ptr(),
            out.as_mut_ptr(),
        );
        assert_eq!(out.map(|n| n.bits), [1, 0]);
        chelis_tensor_shrink_shape(
            empty,
            int(2),
            [int(0), int(0)].as_ptr(),
            [int(i64::MAX), int(0)].as_ptr(),
            out.as_mut_ptr(),
        );
        assert_eq!(out.map(|n| n.bits), [i64::MAX as u64, 0]);
        chelis_tensor_release(empty);
    }
}

#[test]
fn malformed_affine_metadata_traps_before_returning() {
    const CHILD: &str = "CHELIS_AFFINE_METADATA_CHILD";
    if let Ok(case) = env::var(CHILD) {
        unsafe {
            let tensor = chelis_alloc(2, [2, 3].as_ptr(), CHELIS_DTYPE_I8);
            let mut out = [int(-1); 2];
            let zeros = [int(0); 2];
            let ones = [int(1); 2];
            match case.as_str() {
                "rank-overflow" => chelis_tensor_stride_shape(
                    tensor,
                    int(i32::MAX as i64 + 1),
                    ones.as_ptr(),
                    out.as_mut_ptr(),
                ),
                "rank-negative" => {
                    chelis_tensor_stride_shape(tensor, int(-1), ones.as_ptr(), out.as_mut_ptr())
                }
                "byte-overflow" => {
                    let wide = chelis_alloc(1, [1].as_ptr(), CHELIS_DTYPE_F64);
                    chelis_tensor_pad_shape(
                        wide,
                        int(1),
                        [int(i64::MAX / 8)].as_ptr(),
                        zeros.as_ptr(),
                        out.as_mut_ptr(),
                    );
                }
                "empty-negative" => {
                    let empty = chelis_alloc(2, [0, 3].as_ptr(), CHELIS_DTYPE_I8);
                    chelis_tensor_pad_shape(
                        empty,
                        int(2),
                        [int(0), int(-1)].as_ptr(),
                        zeros.as_ptr(),
                        out.as_mut_ptr(),
                    );
                }
                "negative-pad" => chelis_tensor_pad_shape(
                    tensor,
                    int(2),
                    [int(-1), int(0)].as_ptr(),
                    zeros.as_ptr(),
                    out.as_mut_ptr(),
                ),
                "pad-overflow" => chelis_tensor_pad_shape(
                    tensor,
                    int(2),
                    [int(i64::MAX), int(0)].as_ptr(),
                    zeros.as_ptr(),
                    out.as_mut_ptr(),
                ),
                "product-overflow" => chelis_tensor_pad_shape(
                    tensor,
                    int(2),
                    [int(i64::MAX / 2), int(0)].as_ptr(),
                    zeros.as_ptr(),
                    out.as_mut_ptr(),
                ),
                "shrink-overshoot" => chelis_tensor_shrink_shape(
                    tensor,
                    int(2),
                    zeros.as_ptr(),
                    [int(3), int(3)].as_ptr(),
                    out.as_mut_ptr(),
                ),
                "shrink-inverted" => chelis_tensor_shrink_shape(
                    tensor,
                    int(2),
                    ones.as_ptr(),
                    zeros.as_ptr(),
                    out.as_mut_ptr(),
                ),
                "stride-zero" => {
                    chelis_tensor_stride_shape(tensor, int(2), zeros.as_ptr(), out.as_mut_ptr())
                }
                "stride-negative" => chelis_tensor_stride_shape(
                    tensor,
                    int(2),
                    [int(1), int(-1)].as_ptr(),
                    out.as_mut_ptr(),
                ),
                "rank" => {
                    chelis_tensor_stride_shape(tensor, int(1), ones.as_ptr(), out.as_mut_ptr())
                }
                "null" => chelis_tensor_stride_shape(tensor, int(2), ptr::null(), out.as_mut_ptr()),
                "null-output" => {
                    chelis_tensor_stride_shape(tensor, int(2), ones.as_ptr(), ptr::null_mut())
                }
                "tag" => chelis_tensor_stride_shape(
                    tensor,
                    int(2),
                    [chelis_scalar_from_bits(CHELIS_DTYPE_I32, 1), int(1)].as_ptr(),
                    out.as_mut_ptr(),
                ),
                "reserved" => {
                    let mut bad = ones;
                    bad[1].reserved[0] = 1;
                    chelis_tensor_stride_shape(tensor, int(2), bad.as_ptr(), out.as_mut_ptr());
                }
                "coordinate" => {
                    chelis_tensor_affine_index(
                        tensor,
                        [int(2), int(0)].as_ptr(),
                        zeros.as_ptr(),
                        ones.as_ptr(),
                    );
                }
                "coordinate-overflow" => {
                    chelis_tensor_affine_index(
                        tensor,
                        [int(i64::MAX), int(0)].as_ptr(),
                        ones.as_ptr(),
                        [int(2), int(1)].as_ptr(),
                    );
                }
                _ => panic!("unknown child"),
            }
        }
        panic!("invalid metadata returned");
    }
    for (case, class) in [
        ("rank-overflow", "overflow"),
        ("rank-negative", "domain"),
        ("byte-overflow", "overflow"),
        ("empty-negative", "domain"),
        ("negative-pad", "domain"),
        ("pad-overflow", "overflow"),
        ("product-overflow", "overflow"),
        ("shrink-overshoot", "domain"),
        ("shrink-inverted", "domain"),
        ("stride-zero", "domain"),
        ("stride-negative", "domain"),
        ("rank", "domain"),
        ("null", "domain"),
        ("null-output", "domain"),
        ("tag", "domain"),
        ("reserved", "domain"),
        ("coordinate", "domain"),
        ("coordinate-overflow", "overflow"),
    ] {
        let output = Command::new(env::current_exe().unwrap())
            .args([
                "--exact",
                "malformed_affine_metadata_traps_before_returning",
                "--nocapture",
            ])
            .env(CHILD, case)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1), "{case}: {output:?}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(&format!("numeric trap: {class} in ")),
            "{case}: {stderr}"
        );
        let op = if case.starts_with("coordinate") {
            "affine_index"
        } else if case.starts_with("shrink") {
            "shrink"
        } else if case.contains("pad")
            || case == "product-overflow"
            || case == "byte-overflow"
            || case == "empty-negative"
        {
            "pad"
        } else {
            "stride"
        };
        assert!(
            stderr.ends_with(&format!("numeric trap: {class} in {op} at int64\n")),
            "{case}: {stderr}"
        );
    }
}
