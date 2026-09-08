use std::process::Command;
use std::ptr;

use chelis_runtime::{
    chelis_alloc, chelis_dtype, chelis_dtype_size, chelis_tensor_entry_borrow,
    chelis_tensor_from_values,
};

const CHILD_CASE_ENV: &str = "CHELIS_RUNTIME_DTYPE_INVALID_CHILD_CASE";
const INVALID_DTYPE: chelis_dtype = 9;

#[test]
fn invalid_dtype_child() {
    let Ok(case) = std::env::var(CHILD_CASE_ENV) else {
        return;
    };
    match case.as_str() {
        "alloc" => unsafe {
            chelis_alloc(0, ptr::null(), INVALID_DTYPE);
        },
        "entry_borrow" => unsafe {
            chelis_tensor_entry_borrow(0, ptr::null(), INVALID_DTYPE, ptr::null_mut(), 0);
        },
        "dtype_size" => {
            chelis_dtype_size(INVALID_DTYPE);
        }
        "tensor_from_values" => unsafe {
            chelis_tensor_from_values(ptr::null(), INVALID_DTYPE);
        },
        other => panic!("unknown child case {other}"),
    }
    panic!("invalid dtype case `{case}` returned instead of terminating");
}

#[test]
fn every_raw_dtype_ffi_boundary_rejects_before_returning_a_value() {
    let test_binary = std::env::current_exe().expect("current test binary");
    for case in ["alloc", "entry_borrow", "dtype_size", "tensor_from_values"] {
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
