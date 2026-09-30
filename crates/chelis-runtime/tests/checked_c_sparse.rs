//! [05-OP-33]: sparse domains validate complete shapes before submission.
use chelis_runtime::*;
use std::{env, process::Command, ptr};
fn int(n: i64) -> chelis_scalar {
    chelis_scalar_from_bits(CHELIS_DTYPE_I64, n as u64)
}

#[test]
fn sparse_plans_preserve_exact_slots_shapes_and_input_lifetimes() {
    unsafe {
        for index_dtype in [
            CHELIS_DTYPE_I8,
            CHELIS_DTYPE_I16,
            CHELIS_DTYPE_I32,
            CHELIS_DTYPE_I64,
        ] {
            let base = chelis_alloc(3, [2, 3, 2].as_ptr(), CHELIS_DTYPE_F64);
            let indices = chelis_alloc(2, [2, 2].as_ptr(), index_dtype);
            let p = chelis_tensor_sparse_plan(
                base,
                indices,
                ptr::null(),
                int(-2),
                CHELIS_SPARSE_GATHER,
            );
            assert_eq!(chelis_sparse_count(p), 16);
            assert_eq!(chelis_sparse_extent(p, int(-1)), 2);
            chelis_sparse_check_target(p, int(4), [int(2); 4].as_ptr());
            let guard = chelis_tensor_begin_write(base);
            assert_eq!(chelis_sparse_data_index(p, int(15), int(2)), 11);
            chelis_tensor_end_write(guard);
            chelis_tensor_release(base);
            chelis_tensor_release(indices);
            assert_eq!(chelis_sparse_index_slot(p, int(15)), 3);
            chelis_sparse_plan_release(p);
        }
        for dtype in [
            CHELIS_DTYPE_F32,
            CHELIS_DTYPE_F64,
            CHELIS_DTYPE_F16,
            CHELIS_DTYPE_BF16,
            CHELIS_DTYPE_I8,
            CHELIS_DTYPE_I16,
            CHELIS_DTYPE_I32,
            CHELIS_DTYPE_I64,
            CHELIS_DTYPE_BOOL,
        ] {
            let base = chelis_alloc(2, [2, 3].as_ptr(), dtype);
            let indices = chelis_alloc(2, [2, 2].as_ptr(), CHELIS_DTYPE_I64);
            let updates = chelis_alloc(2, [2, 2].as_ptr(), dtype);
            let p =
                chelis_tensor_sparse_plan(base, indices, updates, int(1), CHELIS_SPARSE_ELEMENTS);
            assert_eq!(chelis_sparse_count(p), 4);
            chelis_sparse_check_target(p, int(2), [int(2), int(3)].as_ptr());
            assert_eq!(chelis_sparse_data_index(p, int(2), int(1)), 4);
            chelis_sparse_plan_release(p);
            chelis_tensor_release(base);
            chelis_tensor_release(indices);
            chelis_tensor_release(updates);
        }
    }
}

#[test]
fn paired_sparse_plan_reads_indices_from_the_matching_batch_row() {
    unsafe {
        let base = chelis_alloc(3, [2, 3, 4].as_ptr(), CHELIS_DTYPE_F32);
        let indices = chelis_alloc(2, [2, 2].as_ptr(), CHELIS_DTYPE_I64);
        let updates = chelis_alloc(3, [2, 3, 2].as_ptr(), CHELIS_DTYPE_F32);
        for operation in [
            CHELIS_SPARSE_GATHER,
            CHELIS_SPARSE_ADD,
            CHELIS_SPARSE_REPLACE,
        ] {
            let plan = chelis_tensor_sparse_plan(
                base,
                indices,
                if operation == CHELIS_SPARSE_GATHER {
                    ptr::null()
                } else {
                    updates
                },
                int(2),
                operation + 4,
            );
            assert_eq!(chelis_sparse_count(plan), 12);
            assert_eq!(chelis_sparse_extent(plan, int(0)), 2);
            assert_eq!(chelis_sparse_extent(plan, int(1)), 3);
            assert_eq!(
                chelis_sparse_extent(plan, int(2)),
                if operation == CHELIS_SPARSE_GATHER {
                    2
                } else {
                    4
                }
            );
            assert_eq!(chelis_sparse_index_slot(plan, int(0)), 0);
            assert_eq!(chelis_sparse_index_slot(plan, int(5)), 1);
            assert_eq!(chelis_sparse_index_slot(plan, int(6)), 2);
            assert_eq!(chelis_sparse_index_slot(plan, int(11)), 3);
            assert_eq!(chelis_sparse_data_index(plan, int(6), int(2)), 14);
            chelis_sparse_plan_release(plan);
        }
        chelis_tensor_release(base);
        chelis_tensor_release(indices);
        chelis_tensor_release(updates);
    }
}

#[test]
fn sparse_empty_domains_keep_exact_shapes() {
    unsafe {
        let base = chelis_alloc(3, [0, 3, 2].as_ptr(), CHELIS_DTYPE_I8);
        let indices = chelis_alloc(1, [4].as_ptr(), CHELIS_DTYPE_I64);
        let updates = chelis_alloc(3, [0, 4, 2].as_ptr(), CHELIS_DTYPE_I8);
        for op in [
            CHELIS_SPARSE_GATHER,
            CHELIS_SPARSE_ADD,
            CHELIS_SPARSE_REPLACE,
        ] {
            let p = chelis_tensor_sparse_plan(
                base,
                indices,
                if op == CHELIS_SPARSE_GATHER {
                    ptr::null()
                } else {
                    updates
                },
                int(1),
                op,
            );
            assert_eq!(chelis_sparse_count(p), 0);
            chelis_sparse_check_target(
                p,
                int(3),
                [
                    int(0),
                    int(if op == CHELIS_SPARSE_GATHER { 4 } else { 3 }),
                    int(2),
                ]
                .as_ptr(),
            );
            chelis_sparse_plan_release(p);
        }
        chelis_tensor_release(base);
        chelis_tensor_release(indices);
        chelis_tensor_release(updates);
    }
}

#[test]
fn invalid_sparse_child() {
    let Ok(case) = env::var("CHELIS_SPARSE_CHILD") else {
        return;
    };
    unsafe {
        let base = chelis_alloc(2, [2, 3].as_ptr(), CHELIS_DTYPE_I8);
        let indices = chelis_alloc(
            1,
            [2].as_ptr(),
            if case == "index-dtype" {
                CHELIS_DTYPE_F32
            } else {
                CHELIS_DTYPE_I64
            },
        );
        let updates = chelis_alloc(
            2,
            if case == "updates-shape" {
                [1, 4]
            } else {
                [2, 2]
            }
            .as_ptr(),
            if case == "updates-dtype" {
                CHELIS_DTYPE_I32
            } else {
                CHELIS_DTYPE_I8
            },
        );
        let op = env::var("CHELIS_SPARSE_OP")
            .unwrap_or_else(|_| CHELIS_SPARSE_REPLACE.to_string())
            .parse()
            .unwrap();
        let p = chelis_tensor_sparse_plan(
            base,
            indices,
            if case == "null-updates" {
                ptr::null()
            } else {
                updates
            },
            if case == "axis" { int(-3) } else { int(1) },
            op,
        );
        match case.as_str() {
            "target" => chelis_sparse_check_target(p, int(2), [int(1), int(6)].as_ptr()),
            "slot" => {
                chelis_sparse_index_slot(p, int(4));
            }
            "negative-slot" => {
                chelis_sparse_index_slot(p, int(-1));
            }
            "selected" => {
                chelis_sparse_data_index(p, int(0), int(3));
            }
            "negative-selected" => {
                chelis_sparse_data_index(p, int(0), int(-1));
            }
            "scalar" => {
                chelis_sparse_data_index(p, chelis_scalar_from_bits(CHELIS_DTYPE_F32, 0), int(0));
            }
            "extent" => {
                chelis_sparse_extent(p, int(-3));
            }
            _ => (),
        }
    }
    panic!("invalid sparse metadata accepted");
}

#[test]
fn sparse_shape_and_index_failures_keep_the_owning_diagnostic() {
    for case in [
        "updates-shape",
        "updates-dtype",
        "index-dtype",
        "null-updates",
        "axis",
        "target",
        "slot",
        "negative-slot",
        "selected",
        "negative-selected",
        "scalar",
        "extent",
    ] {
        let result = Command::new(env::current_exe().unwrap())
            .args(["--exact", "invalid_sparse_child", "--nocapture"])
            .env("CHELIS_SPARSE_CHILD", case)
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(1), "{case}: {result:?}");
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(
            stderr
                .lines()
                .any(|l| l == "numeric trap: domain in scatter_replace at i64"),
            "{case}: {stderr}"
        );
    }
    for (op, name) in [
        (CHELIS_SPARSE_GATHER, "gather"),
        (CHELIS_SPARSE_ADD, "scatter"),
        (CHELIS_SPARSE_ELEMENTS, "scatter_elements"),
    ] {
        let result = Command::new(env::current_exe().unwrap())
            .args(["--exact", "invalid_sparse_child", "--nocapture"])
            .env("CHELIS_SPARSE_CHILD", "axis")
            .env("CHELIS_SPARSE_OP", op.to_string())
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(1), "{name}: {result:?}");
        assert!(String::from_utf8_lossy(&result.stderr)
            .lines()
            .any(|l| l == format!("numeric trap: domain in {name} at i64")));
    }
}
