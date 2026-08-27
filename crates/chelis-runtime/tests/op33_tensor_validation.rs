//! Fail-closed operation-entry coverage for the public tensor runtime.
//!
//! These cases derive from `spec/05-risc-primitives.md` [05-OP-33]. Every
//! public operation must validate each tensor carrier before inspecting shape
//! metadata, and operation-specific dtype domains must reject forbidden
//! carriers without narrowing the signed-integer/float families they admit.

use chelis_runtime::{
    chelis_alloc, chelis_dims, chelis_free, chelis_list_from_values, chelis_string_from_cstr,
    chelis_tensor, chelis_tensor_clamp, chelis_tensor_cmplt, chelis_tensor_concat,
    chelis_tensor_cumsum, chelis_tensor_einsum, chelis_tensor_gather, chelis_tensor_scatter_add,
    chelis_tensor_scatter_replace, chelis_tensor_sort, chelis_tensor_trace, chelis_tensor_where,
    chelis_value_from_tensor, CHELIS_DTYPE_BOOL, CHELIS_DTYPE_F32, CHELIS_DTYPE_I16,
    CHELIS_DTYPE_I32, CHELIS_DTYPE_I64, CHELIS_DTYPE_I8,
};
use std::env;
use std::ffi::CString;
use std::process::Command;
use std::ptr;

const CHILD_ENV: &str = "CHELIS_OP33_VALIDATION_CHILD";

unsafe fn tensor(dtype: u8, shape: &[i64]) -> *mut chelis_tensor {
    chelis_alloc(shape.len() as i32, shape.as_ptr(), dtype)
}

unsafe fn malformed_positive_rank(dtype: u8, data: *mut u8) -> chelis_tensor {
    chelis_tensor {
        data,
        shape: chelis_dims(ptr::null()),
        strides: chelis_dims(ptr::null()),
        size: 1,
        byte_capacity: 8,
        rank: 1,
        dtype,
        owns_data: 0,
        reserved: [0; 2],
    }
}

fn run_invalid_case(case: &str) -> ! {
    unsafe {
        let shape = [1_i64];
        match case {
            "cmplt-null" => {
                chelis_tensor_cmplt(ptr::null(), ptr::null());
            }
            "cmplt-malformed-shape" => {
                let mut data = 0_i64;
                let malformed =
                    malformed_positive_rank(CHELIS_DTYPE_I64, (&mut data as *mut i64).cast::<u8>());
                let rhs = tensor(CHELIS_DTYPE_I64, &shape);
                chelis_tensor_cmplt(&malformed, rhs);
            }
            "where-null" => {
                chelis_tensor_where(ptr::null(), ptr::null(), ptr::null());
            }
            "where-malformed-shape" => {
                let mut data = 0_u8;
                let malformed = malformed_positive_rank(CHELIS_DTYPE_BOOL, &mut data);
                let branch = tensor(CHELIS_DTYPE_I32, &shape);
                chelis_tensor_where(&malformed, branch, branch);
            }
            "clamp-null" => {
                chelis_tensor_clamp(ptr::null(), ptr::null(), ptr::null());
            }
            "clamp-malformed-bound" => {
                let input = tensor(CHELIS_DTYPE_I64, &shape);
                let mut data = 0_i64;
                let malformed =
                    malformed_positive_rank(CHELIS_DTYPE_I64, (&mut data as *mut i64).cast::<u8>());
                chelis_tensor_clamp(input, &malformed, input);
            }
            "scatter-malformed-updates" => {
                let base = tensor(CHELIS_DTYPE_I32, &shape);
                let indices = tensor(CHELIS_DTYPE_I8, &shape);
                let mut data = 0_i32;
                let malformed =
                    malformed_positive_rank(CHELIS_DTYPE_I32, (&mut data as *mut i32).cast::<u8>());
                chelis_tensor_scatter_replace(base, indices, &malformed, 0);
            }
            "concat-malformed-second-part" => {
                let first = tensor(CHELIS_DTYPE_I32, &shape);
                let second = tensor(CHELIS_DTYPE_I32, &shape);
                let parts = [
                    chelis_value_from_tensor(first),
                    chelis_value_from_tensor(second),
                ];
                let list = chelis_list_from_values(parts.as_ptr(), parts.len() as i64);
                (*second).rank = 2;
                (*second).shape = chelis_dims(ptr::null());
                (*second).strides = chelis_dims(ptr::null());
                chelis_tensor_concat(list, 0);
            }
            "cmplt-bool" => {
                let lhs = tensor(CHELIS_DTYPE_BOOL, &shape);
                let rhs = tensor(CHELIS_DTYPE_BOOL, &shape);
                chelis_tensor_cmplt(lhs, rhs);
            }
            "where-f32-condition" => {
                let condition = tensor(CHELIS_DTYPE_F32, &shape);
                let then_tensor = tensor(CHELIS_DTYPE_I32, &shape);
                let else_tensor = tensor(CHELIS_DTYPE_I32, &shape);
                chelis_tensor_where(condition, then_tensor, else_tensor);
            }
            "gather-f32-indices" => {
                let base = tensor(CHELIS_DTYPE_I32, &shape);
                let indices = tensor(CHELIS_DTYPE_F32, &shape);
                *(indices
                    .cast::<chelis_tensor>()
                    .as_mut()
                    .unwrap()
                    .data
                    .cast::<f32>()) = 0.0;
                chelis_tensor_gather(base, indices, 0);
            }
            "scatter-replace-bool-indices" => {
                let base = tensor(CHELIS_DTYPE_I32, &shape);
                let indices = tensor(CHELIS_DTYPE_BOOL, &shape);
                let updates = tensor(CHELIS_DTYPE_I32, &shape);
                chelis_tensor_scatter_replace(base, indices, updates, 0);
            }
            "scatter-add-f64-indices" => {
                let base = tensor(CHELIS_DTYPE_I32, &shape);
                let indices = tensor(chelis_runtime::CHELIS_DTYPE_F64, &shape);
                let updates = tensor(CHELIS_DTYPE_I32, &shape);
                *(*indices).data.cast::<f64>() = 0.0;
                chelis_tensor_scatter_add(base, indices, updates, 0);
            }
            "scatter-add-bool-base" => {
                let base = tensor(CHELIS_DTYPE_BOOL, &shape);
                let indices = tensor(CHELIS_DTYPE_I16, &shape);
                let updates = tensor(CHELIS_DTYPE_BOOL, &shape);
                chelis_tensor_scatter_add(base, indices, updates, 0);
            }
            "cumsum-bool" => {
                let input = tensor(CHELIS_DTYPE_BOOL, &shape);
                chelis_tensor_cumsum(input, 0);
            }
            "sort-bool" => {
                let input = tensor(CHELIS_DTYPE_BOOL, &shape);
                chelis_tensor_sort(input, 0);
            }
            "trace-bool" => {
                let matrix_shape = [1_i64, 1];
                let input = tensor(CHELIS_DTYPE_BOOL, &matrix_shape);
                chelis_tensor_trace(input, 0, 1);
            }
            "clamp-bool" => {
                let input = tensor(CHELIS_DTYPE_BOOL, &shape);
                let lower = tensor(CHELIS_DTYPE_BOOL, &[]);
                let upper = tensor(CHELIS_DTYPE_BOOL, &[]);
                chelis_tensor_clamp(input, lower, upper);
            }
            "einsum-bool" => {
                let input = tensor(CHELIS_DTYPE_BOOL, &shape);
                let text = CString::new("i,i->").unwrap();
                let equation = chelis_string_from_cstr(text.as_ptr());
                chelis_tensor_einsum(equation, input, input, CHELIS_DTYPE_BOOL);
            }
            other => panic!("unknown invalid OP33 case: {other}"),
        }
    }
    panic!("invalid OP33 case `{case}` returned instead of terminating")
}

#[test]
fn malformed_carriers_and_forbidden_dtypes_trap_domain_at_operation_entry() {
    if let Ok(case) = env::var(CHILD_ENV) {
        run_invalid_case(&case);
    }
    let test_binary = env::current_exe().expect("current test binary");
    for case in [
        "cmplt-null",
        "cmplt-malformed-shape",
        "where-null",
        "where-malformed-shape",
        "clamp-null",
        "clamp-malformed-bound",
        "scatter-malformed-updates",
        "concat-malformed-second-part",
        "cmplt-bool",
        "where-f32-condition",
        "gather-f32-indices",
        "scatter-replace-bool-indices",
        "scatter-add-f64-indices",
        "scatter-add-bool-base",
        "cumsum-bool",
        "sort-bool",
        "trace-bool",
        "clamp-bool",
        "einsum-bool",
    ] {
        let output = Command::new(&test_binary)
            .env(CHILD_ENV, case)
            .arg("--exact")
            .arg("malformed_carriers_and_forbidden_dtypes_trap_domain_at_operation_entry")
            .arg("--nocapture")
            .output()
            .expect("run invalid OP33 child");
        assert!(
            !output.status.success(),
            "invalid OP33 case `{case}` returned success"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("Domain:"),
            "invalid OP33 case `{case}` did not report Domain before access:\n{stderr}"
        );
    }
}

#[test]
fn admitted_dtypes_still_execute_across_comparison_selection_and_sparse_ops() {
    unsafe {
        let pair = [2_i64];

        let lhs = tensor(CHELIS_DTYPE_I64, &pair);
        let rhs = tensor(CHELIS_DTYPE_I64, &pair);
        (*lhs).data.cast::<i64>().copy_from([1_i64, 4].as_ptr(), 2);
        (*rhs).data.cast::<i64>().copy_from([2_i64, 3].as_ptr(), 2);
        let compared = chelis_tensor_cmplt(lhs, rhs);
        assert_eq!(std::slice::from_raw_parts((*compared).data, 2), &[1, 0]);

        let condition = tensor(CHELIS_DTYPE_BOOL, &pair);
        (*condition).data.copy_from([1_u8, 0].as_ptr(), 2);
        let then_tensor = tensor(CHELIS_DTYPE_I16, &pair);
        let else_tensor = tensor(CHELIS_DTYPE_I16, &pair);
        (*then_tensor)
            .data
            .cast::<i16>()
            .copy_from([11_i16, 12].as_ptr(), 2);
        (*else_tensor)
            .data
            .cast::<i16>()
            .copy_from([21_i16, 22].as_ptr(), 2);
        let selected = chelis_tensor_where(condition, then_tensor, else_tensor);
        assert_eq!(
            std::slice::from_raw_parts((*selected).data.cast::<i16>(), 2),
            &[11, 22]
        );

        let base_shape = [3_i64];
        let base = tensor(CHELIS_DTYPE_I32, &base_shape);
        (*base)
            .data
            .cast::<i32>()
            .copy_from([10_i32, 20, 30].as_ptr(), 3);
        let indices = tensor(CHELIS_DTYPE_I8, &pair);
        (*indices)
            .data
            .cast::<i8>()
            .copy_from([2_i8, 0].as_ptr(), 2);
        let gathered = chelis_tensor_gather(base, indices, 0);
        assert_eq!(
            std::slice::from_raw_parts((*gathered).data.cast::<i32>(), 2),
            &[30, 10]
        );

        let updates = tensor(CHELIS_DTYPE_I32, &pair);
        (*updates)
            .data
            .cast::<i32>()
            .copy_from([7_i32, 8].as_ptr(), 2);
        let replaced = chelis_tensor_scatter_replace(base, indices, updates, 0);
        assert_eq!(
            std::slice::from_raw_parts((*replaced).data.cast::<i32>(), 3),
            &[8, 20, 7]
        );

        let indices_i16 = tensor(CHELIS_DTYPE_I16, &pair);
        (*indices_i16)
            .data
            .cast::<i16>()
            .copy_from([1_i16, 1].as_ptr(), 2);
        let added = chelis_tensor_scatter_add(base, indices_i16, updates, 0);
        assert_eq!(
            std::slice::from_raw_parts((*added).data.cast::<i32>(), 3),
            &[10, 35, 30]
        );

        for tensor in [
            compared,
            selected,
            gathered,
            replaced,
            added,
            lhs,
            rhs,
            condition,
            then_tensor,
            else_tensor,
            base,
            indices,
            updates,
            indices_i16,
        ] {
            chelis_free(tensor);
        }
    }
}
