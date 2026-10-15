//! [05-OP-33]: logical metadata composes independently of tensor payloads.
use chelis_runtime::*;
use std::{env, process::Command};

fn int(value: i64) -> chelis_scalar {
    chelis_scalar_from_bits(CHELIS_DTYPE_I64, value as u64)
}

#[test]
fn logical_movement_plans_compose_without_replicated_payloads() {
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
            let tensor = chelis_alloc(2, [2, 3].as_ptr(), dtype);
            let metadata = chelis_metadata_plan_tensor(tensor);
            chelis_tensor_release(tensor);
            for context in [CHELIS_PRODUCT_MUL, CHELIS_PRODUCT_MATMUL] {
                let product = chelis_metadata_product_plan(metadata, metadata, context);
                assert_eq!(chelis_metadata_plan_count(product), 6);
                assert_eq!(chelis_metadata_plan_dtype(product), dtype);
                chelis_metadata_plan_release(product);
            }
            let expanded =
                chelis_metadata_expand_plan(metadata, int(2), int(5), CHELIS_MOVEMENT_INSERT);
            let expansion_metadata = chelis_movement_result_metadata(expanded);
            chelis_movement_plan_release(expanded);
            chelis_metadata_plan_release(metadata);
            let permuted = chelis_metadata_permute_plan(
                expansion_metadata,
                int(3),
                [int(0), int(2), int(1)].as_ptr(),
            );
            chelis_movement_check_target(permuted, int(3), [int(2), int(5), int(3)].as_ptr());
            assert_eq!(chelis_movement_index(permuted, int(29)), 29);
            let permute_metadata = chelis_movement_result_metadata(permuted);
            let reshaped =
                chelis_metadata_plan_reshape(permute_metadata, int(2), [int(10), int(3)].as_ptr());
            assert_eq!(chelis_metadata_plan_count(reshaped), 30);
            assert_eq!(chelis_metadata_plan_dtype(reshaped), dtype);
            for plan in [expansion_metadata, permute_metadata, reshaped] {
                chelis_metadata_plan_release(plan);
            }
            chelis_movement_plan_release(permuted);
        }
        let empty = chelis_metadata_plan_new(
            int(3),
            [int(1 << 40), int(1 << 40), int(0)].as_ptr(),
            chelis_scalar_from_bits(CHELIS_DTYPE_F32, 0),
        );
        let moved = chelis_metadata_permute_plan(empty, int(3), [int(1), int(0), int(2)].as_ptr());
        assert_eq!(chelis_movement_count(moved), 0);
        chelis_movement_plan_release(moved);
        chelis_metadata_plan_release(empty);
    }
}

#[test]
fn malformed_logical_movement_child() {
    let Ok(case) = env::var("CHELIS_LOGICAL_MOVEMENT_CHILD") else {
        return;
    };
    unsafe {
        let input = chelis_metadata_plan_new(
            int(2),
            [int(2), int(3)].as_ptr(),
            chelis_scalar_from_bits(CHELIS_DTYPE_F32, 0),
        );
        match case.as_str() {
            "product-strided" => {
                let strided = chelis_metadata_plan_view(
                    int(1),
                    [int(2)].as_ptr(),
                    [int(8)].as_ptr(),
                    chelis_scalar_from_bits(CHELIS_DTYPE_F32, 0),
                    int(16),
                );
                chelis_metadata_product_plan(strided, input, CHELIS_PRODUCT_MUL);
            }
            "product-context" => {
                chelis_metadata_product_plan(input, input, 99);
            }
            "product-null" => {
                chelis_metadata_product_plan(std::ptr::null(), input, CHELIS_PRODUCT_MUL);
            }
            "product-mul" | "product-matmul" => {
                let other = chelis_metadata_plan_new(
                    int(2),
                    [int(2), int(4)].as_ptr(),
                    chelis_scalar_from_bits(CHELIS_DTYPE_F32, 0),
                );
                chelis_metadata_product_plan(
                    input,
                    other,
                    if case == "product-matmul" {
                        CHELIS_PRODUCT_MATMUL
                    } else {
                        CHELIS_PRODUCT_MUL
                    },
                );
            }
            "null" => {
                chelis_metadata_expand_plan(
                    std::ptr::null(),
                    int(0),
                    int(2),
                    CHELIS_MOVEMENT_INSERT,
                );
            }
            "rank" => {
                chelis_metadata_permute_plan(input, int(1), [int(0)].as_ptr());
            }
            "axes-null" => {
                chelis_metadata_permute_plan(input, int(2), std::ptr::null());
            }
            "axes-duplicate" => {
                chelis_metadata_permute_plan(input, int(2), [int(0), int(0)].as_ptr());
            }
            "selector" => {
                chelis_metadata_expand_plan(input, int(0), int(2), 99);
            }
            "size-tag" => {
                chelis_metadata_expand_plan(
                    input,
                    int(0),
                    chelis_scalar_from_bits(CHELIS_DTYPE_I32, 2),
                    CHELIS_MOVEMENT_INSERT,
                );
            }
            "reshape" => {
                chelis_metadata_plan_reshape(input, int(1), [int(5)].as_ptr());
            }
            "strided" => {
                let strided = chelis_metadata_plan_view(
                    int(1),
                    [int(2)].as_ptr(),
                    [int(8)].as_ptr(),
                    chelis_scalar_from_bits(CHELIS_DTYPE_F32, 0),
                    int(16),
                );
                chelis_metadata_expand_plan(strided, int(0), int(2), CHELIS_MOVEMENT_INSERT);
            }
            _ => panic!("unknown case"),
        }
        panic!("malformed logical movement returned");
    }
}

#[test]
fn malformed_logical_movements_keep_checked_owner_failures() {
    for case in [
        "product-strided",
        "product-context",
        "product-null",
        "product-mul",
        "product-matmul",
        "null",
        "rank",
        "axes-null",
        "axes-duplicate",
        "selector",
        "size-tag",
        "reshape",
        "strided",
    ] {
        let output = Command::new(env::current_exe().unwrap())
            .args(["--exact", "malformed_logical_movement_child", "--nocapture"])
            .env("CHELIS_LOGICAL_MOVEMENT_CHILD", case)
            .output()
            .unwrap();
        assert!(!output.status.success(), "{case} returned");
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(
            error.contains("Domain") || error.contains("numeric trap: domain"),
            "{case}: {error}"
        );
        if case == "product-matmul" {
            assert!(
                error.contains(&chelis_abi::failure::matmul_product_disagreement(
                    &[2, 3],
                    &[2, 4]
                )),
                "{error}"
            );
        } else if case == "product-mul" {
            assert!(
                error.contains(&chelis_abi::failure::operand_shape_disagreement(
                    "mul",
                    &[2, 3],
                    &[2, 4]
                )),
                "{error}"
            );
        }
        assert!(
            !error.contains("malformed logical movement returned"),
            "{error}"
        );
    }
}
