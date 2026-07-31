//! Regression tests for native int32 element access.
//!
//! Each fixture stores `CHELIS_I32` values through `i32` pointers. The tested
//! operations must not reinterpret those bytes as IEEE binary32 values.

use std::ffi::c_int;
use std::ptr;

use chelis_runtime::{
    chelis_alloc, chelis_free, chelis_string_from_cstr, chelis_tensor, chelis_tensor_clamp,
    chelis_tensor_cmplt, chelis_tensor_cumsum, chelis_tensor_einsum, chelis_tensor_scatter,
    chelis_tensor_trace, chelis_tensor_where, data_as_f32, data_as_f32_const, TensorElement,
    CHELIS_BOOL, CHELIS_F32, CHELIS_I32,
};

unsafe fn i32_tensor(shape: &[c_int], values: &[i32]) -> *mut chelis_tensor {
    unsafe {
        let shape_ptr = if shape.is_empty() {
            ptr::null()
        } else {
            shape.as_ptr()
        };
        let tensor = chelis_alloc(shape.len() as c_int, shape_ptr, CHELIS_I32);
        assert_eq!((*tensor).size as usize, values.len());
        let data = i32::data_ptr_unchecked(tensor);
        for (index, value) in values.iter().copied().enumerate() {
            *data.add(index) = value;
        }
        tensor
    }
}

unsafe fn read_i32(tensor: *mut chelis_tensor) -> Vec<i32> {
    unsafe {
        let data = i32::data_ptr_unchecked(tensor);
        (0..(*tensor).size as usize)
            .map(|index| *data.add(index))
            .collect()
    }
}

unsafe fn read_bool_payload(tensor: *mut chelis_tensor) -> Vec<bool> {
    unsafe {
        let data = data_as_f32_const(tensor);
        (0..(*tensor).size as usize)
            .map(|index| *data.add(index) != 0.0)
            .collect()
    }
}

#[test]
fn cmplt_orders_negative_and_non_negative_int32_values() {
    unsafe {
        let lhs = i32_tensor(&[3], &[-1, 0, 1]);
        let rhs = i32_tensor(&[3], &[0, 0, 0]);
        let out = chelis_tensor_cmplt(lhs, rhs);

        assert_eq!(read_bool_payload(out), vec![true, false, false]);

        chelis_free(out);
        chelis_free(rhs);
        chelis_free(lhs);
    }
}

#[test]
fn where_treats_int32_minimum_as_nonzero() {
    unsafe {
        let cond = i32_tensor(&[2], &[i32::MIN, 0]);
        let then_tensor = i32_tensor(&[2], &[10, 20]);
        let else_tensor = i32_tensor(&[2], &[-10, -20]);
        let out = chelis_tensor_where(cond, then_tensor, else_tensor);

        assert_eq!(read_i32(out), vec![10, -20]);

        chelis_free(out);
        chelis_free(else_tensor);
        chelis_free(then_tensor);
        chelis_free(cond);
    }
}

#[test]
fn scatter_add_accumulates_native_int32_updates() {
    unsafe {
        let base = i32_tensor(&[1], &[0]);
        let indices = i32_tensor(&[2], &[0, 0]);
        let updates = i32_tensor(&[2], &[1_000_000_000, -999_999_999]);
        let mode = chelis_string_from_cstr(c"add".as_ptr());
        let out = chelis_tensor_scatter(base, indices, updates, 0, mode);

        assert_eq!(read_i32(out), vec![1]);

        chelis_free(out);
        chelis_free(updates);
        chelis_free(indices);
        chelis_free(base);
    }
}

#[test]
fn cumsum_preserves_exact_int32_prefixes() {
    unsafe {
        let input = i32_tensor(&[2], &[1_000_000_000, -999_999_999]);
        let out = chelis_tensor_cumsum(input, 0);

        assert_eq!(read_i32(out), vec![1_000_000_000, 1]);

        chelis_free(out);
        chelis_free(input);
    }
}

#[test]
fn trace_accumulates_native_int32_diagonal_values() {
    unsafe {
        let matrix = i32_tensor(&[2, 2], &[1_000_000_000, 0, 0, -999_999_999]);
        let out = chelis_tensor_trace(matrix, 0, 1);

        assert_eq!(read_i32(out), vec![1]);

        chelis_free(out);
        chelis_free(matrix);
    }
}

#[test]
fn clamp_compares_signed_int32_values() {
    unsafe {
        let input = i32_tensor(&[3], &[-1_000_000_000, 7, 1_000_000_000]);
        let lower = i32_tensor(&[], &[-8]);
        let upper = i32_tensor(&[], &[8]);
        let out = chelis_tensor_clamp(input, lower, upper);

        assert_eq!(read_i32(out), vec![-8, 7, 8]);

        chelis_free(out);
        chelis_free(upper);
        chelis_free(lower);
        chelis_free(input);
    }
}

#[test]
fn einsum_multiplies_native_int32_values() {
    unsafe {
        let lhs = i32_tensor(&[1], &[2]);
        let rhs = i32_tensor(&[1], &[3]);
        let equation = chelis_string_from_cstr(c"i,i->".as_ptr());
        let out = chelis_tensor_einsum(equation, lhs, rhs);

        assert_eq!(read_i32(out), vec![6]);

        chelis_free(out);
        chelis_free(rhs);
        chelis_free(lhs);
    }
}

#[test]
fn f32_boundary_accepts_f32_and_current_bool_payloads() {
    unsafe {
        let f32_tensor = chelis_alloc(0, ptr::null(), CHELIS_F32);
        let bool_tensor = chelis_alloc(0, ptr::null(), CHELIS_BOOL);

        assert_eq!(data_as_f32(f32_tensor), (*f32_tensor).data.cast::<f32>());
        assert_eq!(
            data_as_f32_const(f32_tensor),
            (*f32_tensor).data.cast::<f32>()
        );
        assert_eq!(data_as_f32(bool_tensor), (*bool_tensor).data.cast::<f32>());
        assert_eq!(
            data_as_f32_const(bool_tensor),
            (*bool_tensor).data.cast::<f32>()
        );

        chelis_free(bool_tensor);
        chelis_free(f32_tensor);
    }
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "data_as_f32 requires an IEEE binary32-compatible representation")]
fn mutable_f32_boundary_rejects_int32() {
    unsafe {
        let tensor = i32_tensor(&[], &[1]);
        let _ = data_as_f32(tensor);
    }
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "data_as_f32_const requires an IEEE binary32-compatible representation")]
fn const_f32_boundary_rejects_int32() {
    unsafe {
        let tensor = i32_tensor(&[], &[1]);
        let _ = data_as_f32_const(tensor);
    }
}
