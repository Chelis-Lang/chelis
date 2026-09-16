//! [05-OP-33]/[05-RWIN-1]: checked window metadata precedes storage submission.
use chelis_runtime::*;
use std::{env, process::Command};
fn int(n: i64) -> chelis_scalar {
    chelis_scalar_from_bits(CHELIS_DTYPE_I64, n as u64)
}

#[test]
fn window_plans_keep_shapes_indices_and_lifetime_independent_of_payload() {
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
            for op in [
                CHELIS_WINDOW_SUM,
                CHELIS_WINDOW_MEAN,
                CHELIS_WINDOW_MAX,
                CHELIS_WINDOW_MIN,
                CHELIS_WINDOW_GRAD,
            ] {
                let input = chelis_alloc(3, [2, 5, 6].as_ptr(), dtype);
                let output = chelis_alloc(3, [2, 2, 2].as_ptr(), dtype);
                let guard = chelis_tensor_begin_write(input);
                let plan = chelis_tensor_window_plan(
                    input,
                    int(2),
                    [int(2), int(3)].as_ptr(),
                    [int(2), int(2)].as_ptr(),
                    op,
                );
                chelis_window_check_tensor(plan, input, CHELIS_WINDOW_SOURCE);
                chelis_window_check_tensor(plan, output, CHELIS_WINDOW_RESULT);
                chelis_tensor_end_write(guard);
                chelis_tensor_repurpose(input, int(1), [int(60)].as_ptr());
                chelis_tensor_release(input);
                chelis_tensor_release(output);
                chelis_window_check_target(
                    plan,
                    CHELIS_WINDOW_SOURCE,
                    int(3),
                    [int(2), int(5), int(6)].as_ptr(),
                );
                chelis_window_check_target(
                    plan,
                    CHELIS_WINDOW_RESULT,
                    int(3),
                    [int(2), int(2), int(2)].as_ptr(),
                );
                assert_eq!(chelis_window_extent(plan, CHELIS_WINDOW_RESULT, int(-1)), 2);
                assert_eq!(chelis_window_count(plan), 6);
                assert_eq!(chelis_window_index(plan, int(7), int(5)), 52);
                chelis_window_plan_release(plan);
            }
        }
    }
}
#[test]
fn invalid_window_child() {
    let Ok(case) = env::var("CHELIS_WINDOW_CHILD") else {
        return;
    };
    unsafe {
        let input = chelis_alloc(2, [4, 5].as_ptr(), CHELIS_DTYPE_F32);
        let window = if case == "window" { 6 } else { 2 };
        let step = if case == "step" { 0 } else { 1 };
        let count = if case == "arity" { 0 } else { 1 };
        let op = if case == "operation" {
            99
        } else {
            CHELIS_WINDOW_GRAD
        };
        let p = chelis_tensor_window_plan(
            input,
            int(count),
            [int(window)].as_ptr(),
            [int(step)].as_ptr(),
            op,
        );
        match case.as_str() {
            "arity" | "window" | "step" | "operation" => panic!("invalid constructor returned"),
            "side" => {
                chelis_window_extent(p, 99, int(0));
            }
            "axis" => {
                chelis_window_extent(p, CHELIS_WINDOW_RESULT, int(2));
            }
            "group" => {
                chelis_window_index(p, int(16), int(0));
            }
            "leaf" => {
                chelis_window_index(p, int(0), int(2));
            }
            "negative" => {
                chelis_window_index(p, int(0), int(-1));
            }
            "scalar" => {
                chelis_window_index(p, chelis_scalar_from_bits(CHELIS_DTYPE_F32, 0), int(0));
            }
            "shape" => {
                chelis_window_check_target(
                    p,
                    CHELIS_WINDOW_RESULT,
                    int(2),
                    [int(2), int(8)].as_ptr(),
                );
            }
            "dtype" => {
                let tensor = chelis_alloc(2, [4, 4].as_ptr(), CHELIS_DTYPE_I32);
                chelis_window_check_tensor(p, tensor, CHELIS_WINDOW_RESULT);
            }
            "tensor-shape" => {
                let tensor = chelis_alloc(2, [2, 8].as_ptr(), CHELIS_DTYPE_F32);
                chelis_window_check_tensor(p, tensor, CHELIS_WINDOW_RESULT);
            }
            _ => panic!("unknown child"),
        }
        panic!("invalid window metadata returned");
    }
}
#[test]
fn invalid_window_metadata_retains_selected_canonical_diagnostics() {
    for case in [
        "arity",
        "window",
        "step",
        "operation",
        "side",
        "axis",
        "group",
        "leaf",
        "negative",
        "scalar",
        "shape",
        "dtype",
        "tensor-shape",
    ] {
        let result = Command::new(env::current_exe().unwrap())
            .args(["--exact", "invalid_window_child", "--nocapture"])
            .env("CHELIS_WINDOW_CHILD", case)
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(1), "{case}: {result:?}");
        let op = if case == "operation" {
            "reduce_window"
        } else {
            "reduce_window_grad"
        };
        assert!(
            String::from_utf8_lossy(&result.stderr)
                .lines()
                .any(|line| line == format!("numeric trap: domain in {op} at i64")),
            "{case}: {result:?}"
        );
    }
}
