//! [05-OP-33]: immutable movement plans validate before allocation and access.
use chelis_runtime::*;
use std::{env, process::Command};
fn int(n: i64) -> chelis_scalar {
    chelis_scalar_from_bits(CHELIS_DTYPE_I64, n as u64)
}

#[test]
fn movement_plan_c_boundary_preserves_shapes_indices_and_independent_lifetime() {
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
            let input = chelis_alloc(2, [2, 3].as_ptr(), dtype);
            let guard = chelis_tensor_begin_write(input);
            let permute = chelis_tensor_permute_plan(input, int(2), [int(-1), int(0)].as_ptr());
            let insert = chelis_tensor_expand_plan(input, int(1), int(4), CHELIS_MOVEMENT_INSERT);
            let pad = chelis_tensor_affine_plan(
                input,
                int(2),
                [int(1), int(2)].as_ptr(),
                [int(0), int(1)].as_ptr(),
                CHELIS_MOVEMENT_PAD,
            );
            chelis_tensor_end_write(guard);
            chelis_tensor_repurpose(input, int(1), [int(6)].as_ptr());
            chelis_tensor_release(input);
            chelis_movement_check_target(permute, int(2), [int(3), int(2)].as_ptr());
            chelis_movement_check_target(insert, int(3), [int(2), int(4), int(3)].as_ptr());
            chelis_movement_check_target(pad, int(2), [int(3), int(6)].as_ptr());
            assert_eq!(
                chelis_movement_extent(permute, CHELIS_MOVEMENT_SOURCE, int(-1)),
                3
            );
            assert_eq!(
                chelis_movement_extent(insert, CHELIS_MOVEMENT_RESULT, int(-2)),
                4
            );
            assert_eq!(chelis_movement_count(permute), 6);
            assert_eq!(chelis_movement_count(insert), 24);
            assert_eq!(chelis_movement_count(pad), 6);
            assert_eq!(chelis_movement_index(permute, int(4)), 2);
            assert_eq!(chelis_movement_index(insert, int(23)), 5);
            assert_eq!(chelis_movement_index(pad, int(5)), 16);
            for plan in [permute, insert, pad] {
                chelis_movement_plan_release(plan);
            }
            for rank in [0, 1, 8, 9] {
                let input = chelis_alloc(rank, vec![1; rank as usize].as_ptr(), dtype);
                let axes: Vec<_> = (0..rank).rev().map(|n| int(i64::from(n))).collect();
                let p = chelis_tensor_permute_plan(input, int(i64::from(rank)), axes.as_ptr());
                chelis_tensor_release(input);
                assert_eq!(chelis_movement_index(p, int(0)), 0);
                chelis_movement_plan_release(p);
            }
        }
    }
}

#[test]
fn invalid_movement_plan_child() {
    let Ok(case) = env::var("CHELIS_MOVEMENT_PLAN_CHILD") else {
        return;
    };
    unsafe {
        let input = chelis_alloc(2, [2, 3].as_ptr(), CHELIS_DTYPE_I8);
        let p = match case.as_str() {
            "rank" => chelis_tensor_permute_plan(input, int(1), [int(0)].as_ptr()),
            "rank-tag" => chelis_tensor_permute_plan(
                input,
                chelis_scalar_from_bits(CHELIS_DTYPE_F32, 0),
                [int(0)].as_ptr(),
            ),
            "axes-null" => chelis_tensor_permute_plan(input, int(2), std::ptr::null()),
            "bijection" => chelis_tensor_permute_plan(input, int(2), [int(0), int(0)].as_ptr()),
            "axis-tag" => chelis_tensor_permute_plan(
                input,
                int(2),
                [int(0), chelis_scalar_from_bits(CHELIS_DTYPE_I32, 1)].as_ptr(),
            ),
            "expand" => chelis_tensor_expand_plan(input, int(0), int(3), CHELIS_MOVEMENT_EXPAND),
            "insert" => chelis_tensor_expand_plan(input, int(3), int(3), CHELIS_MOVEMENT_INSERT),
            "operation" => chelis_tensor_expand_plan(input, int(0), int(1), 99),
            "pad" => chelis_tensor_affine_plan(
                input,
                int(2),
                [int(-1), int(0)].as_ptr(),
                [int(0), int(0)].as_ptr(),
                CHELIS_MOVEMENT_PAD,
            ),
            "shrink" => chelis_tensor_affine_plan(
                input,
                int(2),
                [int(0), int(0)].as_ptr(),
                [int(3), int(3)].as_ptr(),
                CHELIS_MOVEMENT_SHRINK,
            ),
            "stride" => chelis_tensor_affine_plan(
                input,
                int(2),
                [int(0), int(1)].as_ptr(),
                std::ptr::null(),
                CHELIS_MOVEMENT_STRIDE,
            ),
            _ => chelis_tensor_permute_plan(input, int(2), [int(1), int(0)].as_ptr()),
        };
        match case.as_str() {
            "index" => {
                chelis_movement_index(p, int(6));
            }
            "side" => {
                chelis_movement_extent(p, 99, int(0));
            }
            "target" => chelis_movement_check_target(p, int(2), [int(2), int(3)].as_ptr()),
            "null-plan" => {
                chelis_movement_count(std::ptr::null());
            }
            _ => panic!("invalid constructor returned"),
        }
        panic!("invalid observation returned");
    }
}

#[test]
fn invalid_movement_plan_metadata_fails_with_its_operation_identity() {
    for (case, op) in [
        ("rank", "permute"),
        ("rank-tag", "permute"),
        ("axes-null", "permute"),
        ("bijection", "permute"),
        ("axis-tag", "permute"),
        ("expand", "expand"),
        ("insert", "insert"),
        ("operation", "movement"),
        ("pad", "pad"),
        ("shrink", "shrink"),
        ("stride", "stride"),
        ("index", "permute"),
        ("side", "permute"),
        ("target", "permute"),
        ("null-plan", "movement"),
    ] {
        let output = Command::new(env::current_exe().unwrap())
            .args(["--exact", "invalid_movement_plan_child", "--nocapture"])
            .env("CHELIS_MOVEMENT_PLAN_CHILD", case)
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(1),
            "{case}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .lines()
                .any(|line| line == format!("numeric trap: domain in {op} at i64")),
            "{case}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
