use std::process::Command;
use std::ptr;

use chelis_runtime::{
    chelis_alloc, chelis_alloc_view, chelis_dtype_size, chelis_tensor,
    chelis_tensor_from_value_list_typed, chelis_tensor_to_f64,
};

const CHILD_CASE_ENV: &str = "CHELIS_RUNTIME_DTYPE_INVALID_CHILD_CASE";
const INVALID_DTYPE: i32 = 9;

#[test]
fn invalid_dtype_child() {
    let Ok(case) = std::env::var(CHILD_CASE_ENV) else {
        return;
    };
    match case.as_str() {
        "alloc" => unsafe {
            chelis_alloc(0, ptr::null(), INVALID_DTYPE);
        },
        "alloc_view" => unsafe {
            chelis_alloc_view(0, ptr::null(), INVALID_DTYPE, ptr::null_mut());
        },
        "dtype_size" => {
            chelis_dtype_size(INVALID_DTYPE);
        }
        "value_list_typed" => unsafe {
            chelis_tensor_from_value_list_typed(ptr::null(), INVALID_DTYPE);
        },
        "tensor_field_read" => unsafe {
            let tensor = chelis_tensor {
                data: ptr::null_mut(),
                shape: [0; 8],
                strides: [0; 8],
                ndim: 0,
                dtype: INVALID_DTYPE,
                size: 1,
                owns_data: 0,
            };
            chelis_tensor_to_f64(&tensor);
        },
        other => panic!("unknown child case {other}"),
    }
    panic!("invalid dtype case `{case}` returned instead of terminating");
}

#[test]
fn every_raw_dtype_ffi_boundary_rejects_before_returning_a_value() {
    let test_binary = std::env::current_exe().expect("current test binary");
    for case in [
        "alloc",
        "alloc_view",
        "dtype_size",
        "value_list_typed",
        "tensor_field_read",
    ] {
        let output = Command::new(&test_binary)
            .args(["--exact", "invalid_dtype_child", "--nocapture"])
            .env(CHILD_CASE_ENV, case)
            .output()
            .unwrap_or_else(|error| panic!("run invalid-dtype child `{case}`: {error}"));
        assert!(
            !output.status.success(),
            "invalid dtype case `{case}` returned success"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("invalid Chelis runtime dtype id: 9"),
            "invalid dtype case `{case}` lost the raw ID diagnostic:\n{stderr}"
        );
    }
}
