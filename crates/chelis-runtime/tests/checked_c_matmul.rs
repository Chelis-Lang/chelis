//! [05-OP-33]: matrix plans validate complete domains before submission.
use chelis_runtime::*;
use std::{env, process::Command};
fn int(n: i64) -> chelis_scalar {
    chelis_scalar_from_bits(CHELIS_DTYPE_I64, n as u64)
}

#[test]
fn matmul_plans_preserve_matrix_shapes_and_survive_source_release() {
    unsafe {
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
            let a = chelis_alloc(3, [2, 3, 4].as_ptr(), dtype);
            let b = chelis_alloc(3, [2, 4, 5].as_ptr(), dtype);
            let g = chelis_tensor_begin_write(a);
            let p = chelis_tensor_matmul_plan(a, b, chelis_scalar_from_bits(dtype, 0));
            chelis_tensor_end_write(g);
            chelis_tensor_release(a);
            chelis_tensor_release(b);
            assert_eq!(chelis_matmul_extent(p, int(-1)), 5);
            assert_eq!(chelis_matmul_dimension(p, CHELIS_MATMUL_ROWS), 3);
            assert_eq!(chelis_matmul_dimension(p, CHELIS_MATMUL_COLUMNS), 5);
            assert_eq!(chelis_matmul_dimension(p, CHELIS_MATMUL_REDUCTION), 4);
            assert_eq!(chelis_matmul_batch_count(p), 2);
            chelis_matmul_check_target(p, int(3), [int(2), int(3), int(5)].as_ptr());
            for (part, n) in [
                (CHELIS_MATMUL_LEFT, 12),
                (CHELIS_MATMUL_RIGHT, 20),
                (CHELIS_MATMUL_RESULT, 15),
            ] {
                assert_eq!(chelis_matmul_matrix_count(p, part), n);
                assert_eq!(chelis_matmul_index(p, part, int(1), int(n - 1)), 2 * n - 1);
                chelis_matmul_check_scratch(p, part, chelis_scalar_from_bits(CHELIS_DTYPE_F64, 0));
            }
            chelis_matmul_check_vendor(p, int(i64::from(i32::MAX)));
            chelis_matmul_plan_release(p);
        }
    }
}

#[test]
fn invalid_matmul_child() {
    let Ok(case) = env::var("CHELIS_MATMUL_CHILD") else {
        return;
    };
    unsafe {
        let a = chelis_alloc(2, [3, 4].as_ptr(), CHELIS_DTYPE_F32);
        let b = chelis_alloc(
            2,
            if case == "contraction" {
                [6, 5]
            } else {
                [4, 5]
            }
            .as_ptr(),
            if case == "dtype" {
                CHELIS_DTYPE_F64
            } else {
                CHELIS_DTYPE_F32
            },
        );
        let exemplar = if case == "exemplar" {
            {
                let mut value = chelis_scalar_from_bits(CHELIS_DTYPE_BOOL, 0);
                value.bits = 2;
                value
            }
        } else {
            chelis_scalar_from_bits(CHELIS_DTYPE_F32, 0)
        };
        let p = chelis_tensor_matmul_plan(a, b, exemplar);
        match case.as_str() {
            "target" => chelis_matmul_check_target(p, int(2), [int(5), int(3)].as_ptr()),
            "extent" => {
                chelis_matmul_extent(p, int(2));
            }
            "dimension" => {
                chelis_matmul_dimension(p, 99);
            }
            "part" => {
                chelis_matmul_matrix_count(p, 99);
            }
            "batch" => {
                chelis_matmul_index(p, CHELIS_MATMUL_LEFT, int(1), int(0));
            }
            "element" => {
                chelis_matmul_index(p, CHELIS_MATMUL_LEFT, int(0), int(12));
            }
            "negative" => {
                chelis_matmul_index(p, CHELIS_MATMUL_LEFT, int(0), int(-1));
            }
            "scalar" => {
                chelis_matmul_index(
                    p,
                    CHELIS_MATMUL_LEFT,
                    chelis_scalar_from_bits(CHELIS_DTYPE_F32, 0),
                    int(0),
                );
            }
            "vendor-domain" => chelis_matmul_check_vendor(p, int(0)),
            "vendor-overflow" => chelis_matmul_check_vendor(p, int(4)),
            "contraction" | "dtype" | "exemplar" => {}
            other => panic!("unknown child {other}"),
        }
        panic!("invalid matrix plan returned");
    }
}

#[test]
fn invalid_matmul_domains_keep_canonical_failure_classes() {
    for case in [
        "target",
        "extent",
        "dimension",
        "part",
        "batch",
        "element",
        "negative",
        "scalar",
        "vendor-domain",
        "vendor-overflow",
        "contraction",
        "dtype",
        "exemplar",
    ] {
        let result = Command::new(env::current_exe().unwrap())
            .args(["--exact", "invalid_matmul_child", "--nocapture"])
            .env("CHELIS_MATMUL_CHILD", case)
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(1), "{case}: {result:?}");
        let class = if case == "vendor-overflow" {
            "overflow"
        } else {
            "domain"
        };
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(
            stderr
                .lines()
                .any(|line| line == format!("numeric trap: {class} in matmul at i64")),
            "{case}: {stderr}"
        );
    }
}
