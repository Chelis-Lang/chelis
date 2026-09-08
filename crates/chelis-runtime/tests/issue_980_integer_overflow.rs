//! chelis#980: ordinary integer tensor arithmetic traps identically in
//! every build profile instead of wrapping, panicking, or relying on C UB.

use std::ffi::c_int;
use std::process::Command;
use std::ptr;

use chelis_runtime::{
    chelis_alloc, chelis_string_from_cstr, chelis_tensor, chelis_tensor_cumsum,
    chelis_tensor_einsum, chelis_tensor_scatter_add, chelis_tensor_trace, TensorElement,
    CHELIS_DTYPE_I16, CHELIS_DTYPE_I32, CHELIS_DTYPE_I64, CHELIS_DTYPE_I8,
};

const CHILD_CASE_ENV: &str = "CHELIS_ISSUE_980_CHILD_CASE";

unsafe fn integer_tensor_i32(shape: &[i64], values: &[i32]) -> *mut chelis_tensor {
    unsafe {
        let shape_ptr = if shape.is_empty() {
            ptr::null()
        } else {
            shape.as_ptr()
        };
        let tensor = chelis_alloc(shape.len() as c_int, shape_ptr, CHELIS_DTYPE_I32);
        let data = i32::data_ptr_unchecked(tensor);
        for (index, value) in values.iter().copied().enumerate() {
            *data.add(index) = value;
        }
        tensor
    }
}

unsafe fn integer_tensor_i64(shape: &[i64], values: &[i64]) -> *mut chelis_tensor {
    unsafe {
        let shape_ptr = if shape.is_empty() {
            ptr::null()
        } else {
            shape.as_ptr()
        };
        let tensor = chelis_alloc(shape.len() as c_int, shape_ptr, CHELIS_DTYPE_I64);
        let data = i64::data_ptr_unchecked(tensor);
        for (index, value) in values.iter().copied().enumerate() {
            *data.add(index) = value;
        }
        tensor
    }
}

unsafe fn integer_tensor_i8(shape: &[i64], values: &[i8]) -> *mut chelis_tensor {
    unsafe {
        let tensor = chelis_alloc(shape.len() as c_int, shape.as_ptr(), CHELIS_DTYPE_I8);
        let data = i8::data_ptr_unchecked(tensor);
        for (index, value) in values.iter().copied().enumerate() {
            *data.add(index) = value;
        }
        tensor
    }
}

unsafe fn integer_tensor_i16(shape: &[i64], values: &[i16]) -> *mut chelis_tensor {
    unsafe {
        let tensor = chelis_alloc(shape.len() as c_int, shape.as_ptr(), CHELIS_DTYPE_I16);
        let data = i16::data_ptr_unchecked(tensor);
        for (index, value) in values.iter().copied().enumerate() {
            *data.add(index) = value;
        }
        tensor
    }
}

#[test]
fn integer_overflow_child() {
    let Ok(case) = std::env::var(CHILD_CASE_ENV) else {
        return;
    };
    unsafe {
        match case.as_str() {
            "scatter_i8" => {
                chelis_tensor_scatter_add(
                    integer_tensor_i8(&[1], &[i8::MAX]),
                    integer_tensor_i32(&[1], &[0]),
                    integer_tensor_i8(&[1], &[1]),
                    0,
                );
            }
            "scatter_i16" => {
                chelis_tensor_scatter_add(
                    integer_tensor_i16(&[1], &[i16::MAX]),
                    integer_tensor_i32(&[1], &[0]),
                    integer_tensor_i16(&[1], &[1]),
                    0,
                );
            }
            "einsum_i16_acc_i32" => {
                chelis_tensor_einsum(
                    chelis_string_from_cstr(c"i,i->".as_ptr()),
                    integer_tensor_i16(&[3], &[i16::MAX; 3]),
                    integer_tensor_i16(&[3], &[i16::MAX; 3]),
                    CHELIS_DTYPE_I32,
                );
            }
            _ => {}
        }
        if matches!(
            case.as_str(),
            "scatter_i8" | "scatter_i16" | "einsum_i16_acc_i32"
        ) {
            panic!("overflow child case `{case}` returned instead of trapping");
        }
        let is_i64 = case.ends_with("_i64");
        let op = case.strip_suffix("_i64").unwrap_or(&case);
        match (op, is_i64) {
            ("scatter", false) => {
                let base = integer_tensor_i32(&[1], &[i32::MAX]);
                let indices = integer_tensor_i32(&[1], &[0]);
                let updates = integer_tensor_i32(&[1], &[1]);
                chelis_tensor_scatter_add(base, indices, updates, 0);
            }
            ("scatter", true) => {
                let base = integer_tensor_i64(&[1], &[i64::MAX]);
                let indices = integer_tensor_i32(&[1], &[0]);
                let updates = integer_tensor_i64(&[1], &[1]);
                chelis_tensor_scatter_add(base, indices, updates, 0);
            }
            ("cumsum", false) => {
                chelis_tensor_cumsum(integer_tensor_i32(&[2], &[i32::MAX, 1]), 0);
            }
            ("cumsum", true) => {
                chelis_tensor_cumsum(integer_tensor_i64(&[2], &[i64::MAX, 1]), 0);
            }
            ("trace", false) => {
                chelis_tensor_trace(integer_tensor_i32(&[2, 2], &[i32::MAX, 0, 0, 1]), 0, 1);
            }
            ("trace", true) => {
                chelis_tensor_trace(integer_tensor_i64(&[2, 2], &[i64::MAX, 0, 0, 1]), 0, 1);
            }
            ("einsum", false) => {
                chelis_tensor_einsum(
                    chelis_string_from_cstr(c"i,i->".as_ptr()),
                    integer_tensor_i32(&[1], &[i32::MAX]),
                    integer_tensor_i32(&[1], &[2]),
                    CHELIS_DTYPE_I32,
                );
            }
            ("einsum", true) => {
                chelis_tensor_einsum(
                    chelis_string_from_cstr(c"i,i->".as_ptr()),
                    integer_tensor_i64(&[1], &[i64::MAX]),
                    integer_tensor_i64(&[1], &[2]),
                    CHELIS_DTYPE_I64,
                );
            }
            ("einsum_add", false) => {
                chelis_tensor_einsum(
                    chelis_string_from_cstr(c"i,i->".as_ptr()),
                    integer_tensor_i32(&[2], &[i32::MAX, 1]),
                    integer_tensor_i32(&[2], &[1, 1]),
                    CHELIS_DTYPE_I32,
                );
            }
            ("einsum_add", true) => {
                chelis_tensor_einsum(
                    chelis_string_from_cstr(c"i,i->".as_ptr()),
                    integer_tensor_i64(&[2], &[i64::MAX, 1]),
                    integer_tensor_i64(&[2], &[1, 1]),
                    CHELIS_DTYPE_I64,
                );
            }
            _ => panic!("unknown overflow child case `{case}`"),
        }
    }
    panic!("overflow child case `{case}` returned instead of trapping");
}

#[test]
fn integer_overflow_traps_are_branded_for_every_runtime_op_and_width() {
    let test_binary = std::env::current_exe().expect("current test binary");
    for (case_op, diagnostic_op) in [
        ("scatter", "scatter"),
        ("cumsum", "cumsum"),
        ("trace", "trace"),
        ("einsum", "einsum"),
        ("einsum_add", "einsum"),
    ] {
        for (suffix, dtype) in [("", "int32"), ("_i64", "int64")] {
            let case = format!("{case_op}{suffix}");
            let output = Command::new(&test_binary)
                .args(["--exact", "integer_overflow_child", "--nocapture"])
                .env(CHILD_CASE_ENV, &case)
                .output()
                .unwrap_or_else(|error| panic!("run overflow child `{case}`: {error}"));
            assert!(
                !output.status.success(),
                "overflow child `{case}` returned success instead of trapping"
            );
            let stderr = String::from_utf8_lossy(&output.stderr);
            let diagnostic = format!("numeric trap: overflow in {diagnostic_op} at {dtype}");
            assert!(
                stderr.contains(&diagnostic),
                "overflow child `{case}` did not emit `{diagnostic}`:\n{stderr}"
            );
        }
    }
    for (case, diagnostic) in [
        ("scatter_i8", "numeric trap: overflow in scatter at int8"),
        ("scatter_i16", "numeric trap: overflow in scatter at int16"),
        (
            "einsum_i16_acc_i32",
            "numeric trap: overflow in einsum at int32",
        ),
    ] {
        let output = Command::new(&test_binary)
            .args(["--exact", "integer_overflow_child", "--nocapture"])
            .env(CHILD_CASE_ENV, case)
            .output()
            .unwrap_or_else(|error| panic!("run overflow child `{case}`: {error}"));
        assert!(
            !output.status.success(),
            "overflow child `{case}` returned success instead of trapping"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(diagnostic),
            "overflow child `{case}` did not emit `{diagnostic}`:\n{stderr}"
        );
    }
}
