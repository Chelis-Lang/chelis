//! [05-OP-33]: reduction domains are checked before allocation or access.
use chelis_runtime::*;
use std::{env, process::Command, ptr};

fn int(n: i64) -> chelis_scalar {
    chelis_scalar_from_bits(CHELIS_DTYPE_I64, n as u64)
}

unsafe fn plan(shape: &[i64], axes: &[i64]) -> *mut chelis_reduction_plan {
    let shape: Vec<_> = shape.iter().copied().map(int).collect();
    let axes: Vec<_> = axes.iter().copied().map(int).collect();
    chelis_shape_reduction_plan(
        int(shape.len() as i64),
        shape.as_ptr(),
        int(axes.len() as i64),
        axes.as_ptr(),
        chelis_scalar_from_bits(CHELIS_DTYPE_I64, 0),
        CHELIS_REDUCE_COUNT,
    )
}

#[test]
fn nontrailing_and_multiple_axes_have_exact_row_major_indices() {
    unsafe {
        let p = plan(&[2, 3, 2], &[1]);
        assert_eq!(chelis_reduction_count(p), 3);
        assert_eq!(chelis_reduction_extent(p, int(0)), 2);
        assert_eq!(chelis_reduction_extent(p, int(1)), 2);
        for (outer, expected) in [[0, 2, 4], [1, 3, 5], [6, 8, 10], [7, 9, 11]]
            .iter()
            .enumerate()
        {
            for (leaf, expected) in expected.iter().enumerate() {
                assert_eq!(
                    chelis_reduction_index(p, int(outer as i64), int(leaf as i64)),
                    *expected
                );
            }
        }
        chelis_reduction_check_target(p, int(2), [int(2), int(2)].as_ptr());
        chelis_reduction_plan_release(p);
        let p = plan(&[2, 3, 2], &[-1, -3]);
        assert_eq!(chelis_reduction_count(p), 4);
        assert_eq!(chelis_reduction_extent(p, int(-1)), 3);
        for (outer, expected) in [[0, 1, 6, 7], [2, 3, 8, 9], [4, 5, 10, 11]]
            .iter()
            .enumerate()
        {
            for (leaf, expected) in expected.iter().enumerate() {
                assert_eq!(
                    chelis_reduction_index(p, int(outer as i64), int(leaf as i64)),
                    *expected
                );
            }
        }
        chelis_reduction_plan_release(p);
    }
}

#[test]
fn empty_domains_and_high_rank_do_not_invent_storage_or_scratch() {
    unsafe {
        let p = plan(&[0], &[0]);
        assert_eq!(chelis_reduction_count(p), 0);
        chelis_reduction_check_target(p, int(0), ptr::null());
        chelis_reduction_check_scratch(p, chelis_scalar_from_bits(CHELIS_DTYPE_I64, 0));
        chelis_reduction_plan_release(p);
        let p = plan(&[0, i64::MAX, i64::MAX], &[2, 1]);
        assert_eq!(chelis_reduction_count(p), 0);
        assert_eq!(chelis_reduction_extent(p, int(0)), 0);
        chelis_reduction_plan_release(p);
        for rank in [1, 8, 9, 32] {
            let p = plan(&vec![1; rank], &[rank as i64 - 1]);
            assert_eq!(chelis_reduction_count(p), 1);
            assert_eq!(chelis_reduction_index(p, int(0), int(0)), 0);
            chelis_reduction_plan_release(p);
        }
        // A virtual domain has no stored payload, including above the i32 range.
        let p = plan(&[i32::MAX as i64 + 1], &[0]);
        assert_eq!(chelis_reduction_count(p), i32::MAX as i64 + 1);
        assert_eq!(
            chelis_reduction_index(p, int(0), int(i32::MAX as i64)),
            i32::MAX as i64
        );
        chelis_reduction_plan_release(p);
    }
}

#[test]
fn tensor_plan_snapshots_metadata_and_remains_read_only_during_writes() {
    unsafe {
        let t = chelis_alloc(2, [2, 3].as_ptr(), CHELIS_DTYPE_BOOL);
        let guard = chelis_tensor_begin_write(t);
        let p =
            chelis_tensor_reduction_plan(t, int(1), [int(1)].as_ptr(), int(0), CHELIS_REDUCE_COUNT);
        assert_eq!(chelis_reduction_count(p), 3);
        assert_eq!(chelis_reduction_index(p, int(1), int(2)), 5);
        chelis_fill_scalar(guard, chelis_scalar_from_bits(CHELIS_DTYPE_BOOL, 1));
        chelis_tensor_end_write(guard);
        chelis_tensor_release(t);
        assert_eq!(chelis_reduction_index(p, int(1), int(2)), 5);
        chelis_reduction_plan_release(p);
    }
}

#[test]
fn tagged_result_and_scratch_representations_cover_every_runtime_dtype() {
    unsafe {
        for dtype in [
            CHELIS_DTYPE_F32,
            CHELIS_DTYPE_F64,
            CHELIS_DTYPE_I64,
            CHELIS_DTYPE_I32,
            CHELIS_DTYPE_I16,
            CHELIS_DTYPE_I8,
            CHELIS_DTYPE_F16,
            CHELIS_DTYPE_BF16,
            CHELIS_DTYPE_BOOL,
        ] {
            let exemplar = chelis_scalar_from_bits(dtype, 0);
            let p = chelis_shape_reduction_plan(
                int(2),
                [int(2), int(3)].as_ptr(),
                int(1),
                [int(1)].as_ptr(),
                exemplar,
                CHELIS_REDUCE_SUM,
            );
            assert_eq!(chelis_reduction_count(p), 3);
            assert_eq!(chelis_reduction_extent(p, int(0)), 2);
            chelis_reduction_check_target(p, int(1), [int(2)].as_ptr());
            chelis_reduction_check_scratch(p, exemplar);
            chelis_reduction_plan_release(p);
        }
    }
}

const CHILD: &str = "CHELIS_CHECKED_REDUCTION_CHILD";

#[test]
fn invalid_metadata_child() {
    let Ok(case) = env::var(CHILD) else { return };
    unsafe {
        match case.as_str() {
            "negative" => {
                plan(&[-1, 0], &[0]);
            }
            "product" => {
                plan(&[i64::MAX, 2], &[1]);
            }
            "result-bytes" => {
                plan(&[i64::MAX, 0], &[1]);
            }
            "result-stride" => {
                plan(&[0, i64::MAX, i64::MAX, 0], &[3]);
            }
            "duplicate-axis" => {
                plan(&[2, 3], &[1, 1]);
            }
            "axis-order" => {
                plan(&[2, 3], &[0, 1]);
            }
            "axis-range" => {
                plan(&[2, 3], &[2]);
            }
            "no-axes" => {
                plan(&[2, 3], &[]);
            }
            "negative-axis-range" => {
                plan(&[2, 3], &[-3]);
            }
            "normalized-duplicate" => {
                plan(&[2, 3], &[1, -1]);
            }
            "axis-count" => {
                plan(&[2], &[0, 0]);
            }
            "result-axis" => {
                let p = plan(&[2, 3], &[1]);
                chelis_reduction_extent(p, int(-2));
            }
            "exemplar-reserved" | "exemplar-bits" | "exemplar-bool" | "exemplar-dtype" => {
                let mut exemplar = chelis_scalar_from_bits(CHELIS_DTYPE_I8, 0);
                match case.as_str() {
                    "exemplar-reserved" => exemplar.reserved[0] = 1,
                    "exemplar-bits" => exemplar.bits = 256,
                    "exemplar-bool" => {
                        exemplar.dtype = CHELIS_DTYPE_BOOL;
                        exemplar.bits = 2;
                    }
                    "exemplar-dtype" => exemplar.dtype = 255,
                    _ => unreachable!(),
                }
                chelis_shape_reduction_plan(
                    int(1),
                    [int(2)].as_ptr(),
                    int(1),
                    [int(0)].as_ptr(),
                    exemplar,
                    CHELIS_REDUCE_COUNT,
                );
            }
            "diagnostic" => {
                let op = env::var("CHELIS_REDUCTION_OP").unwrap().parse().unwrap();
                chelis_shape_reduction_plan(
                    int(1),
                    [int(-1)].as_ptr(),
                    int(1),
                    [int(0)].as_ptr(),
                    int(0),
                    op,
                );
            }
            "empty-index" => {
                let p = plan(&[0], &[0]);
                chelis_reduction_index(p, int(0), int(0));
            }
            "outer-index" => {
                let p = plan(&[2, 3], &[1]);
                chelis_reduction_index(p, int(2), int(0));
            }
            "leaf-index" => {
                let p = plan(&[2, 3], &[1]);
                chelis_reduction_index(p, int(0), int(3));
            }
            "target" => {
                let p = plan(&[2, 3, 2], &[1]);
                chelis_reduction_check_target(p, int(2), [int(1), int(4)].as_ptr());
            }
            "scratch-bytes" => {
                let p = plan(&[i64::MAX], &[0]);
                chelis_reduction_check_scratch(p, int(0));
            }
            "null-shape" => {
                chelis_shape_reduction_plan(
                    int(1),
                    ptr::null(),
                    int(1),
                    [int(0)].as_ptr(),
                    int(0),
                    CHELIS_REDUCE_COUNT,
                );
            }
            "scalar-tag" => {
                let p = plan(&[2], &[0]);
                chelis_reduction_index(p, chelis_scalar_from_bits(CHELIS_DTYPE_F32, 0), int(0));
            }
            _ => panic!("unknown case"),
        }
    }
    panic!("invalid reduction metadata was accepted");
}

#[test]
fn malformed_and_overflowing_domains_fail_with_the_owning_numeric_trap() {
    for (case, class) in [
        ("negative", "domain"),
        ("product", "overflow"),
        ("result-bytes", "overflow"),
        ("result-stride", "overflow"),
        ("duplicate-axis", "domain"),
        ("axis-order", "domain"),
        ("axis-range", "domain"),
        ("no-axes", "domain"),
        ("negative-axis-range", "domain"),
        ("normalized-duplicate", "domain"),
        ("axis-count", "domain"),
        ("result-axis", "domain"),
        ("exemplar-reserved", "domain"),
        ("exemplar-bits", "domain"),
        ("exemplar-bool", "domain"),
        ("exemplar-dtype", "domain"),
        ("empty-index", "domain"),
        ("outer-index", "domain"),
        ("leaf-index", "domain"),
        ("target", "domain"),
        ("scratch-bytes", "overflow"),
        ("null-shape", "domain"),
        ("scalar-tag", "domain"),
    ] {
        let result = Command::new(env::current_exe().unwrap())
            .args(["--exact", "invalid_metadata_child", "--nocapture"])
            .env(CHILD, case)
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(1), "{case}: {result:?}");
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(
            stderr
                .lines()
                .any(|line| line == format!("numeric trap: {class} in count at int64")),
            "{case}: {stderr}"
        );
    }
}

#[test]
fn every_reduction_preserves_its_canonical_diagnostic_identity() {
    for (op, name) in [
        (CHELIS_REDUCE_SUM, "sum"),
        (CHELIS_REDUCE_COUNT, "count"),
        (CHELIS_REDUCE_MAX, "max_reduce"),
        (CHELIS_REDUCE_MIN, "min_reduce"),
        (CHELIS_REDUCE_PROD, "prod_reduce"),
        (CHELIS_REDUCE_ARGMAX, "argmax_reduce"),
        (CHELIS_REDUCE_ARGMIN, "argmin_reduce"),
    ] {
        let result = Command::new(env::current_exe().unwrap())
            .args(["--exact", "invalid_metadata_child", "--nocapture"])
            .env(CHILD, "diagnostic")
            .env("CHELIS_REDUCTION_OP", op.to_string())
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(1), "{name}: {result:?}");
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(
            stderr
                .lines()
                .any(|line| line == format!("numeric trap: domain in {name} at int64")),
            "{name}: {stderr}"
        );
    }
}
