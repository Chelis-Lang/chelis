//! Regression tests for native int32 element access.
//!
//! Each fixture stores `CHELIS_DTYPE_I32` values through `i32` pointers. The tested
//! operations must not reinterpret those bytes as IEEE binary32 values.

use std::ffi::c_int;
use std::ptr;

use chelis_runtime::{
    chelis_alloc, chelis_string_from_cstr, chelis_tensor, chelis_tensor_begin_write,
    chelis_tensor_clamp, chelis_tensor_cmplt, chelis_tensor_cumsum, chelis_tensor_einsum,
    chelis_tensor_end_write, chelis_tensor_read_view, chelis_tensor_release,
    chelis_tensor_scatter_add, chelis_tensor_trace, chelis_tensor_where, chelis_tensor_write_view,
    data_as_f32, data_as_f32_const, Bool8, TensorElement, CHELIS_DTYPE_BOOL, CHELIS_DTYPE_F32,
    CHELIS_DTYPE_I32,
};

unsafe fn i32_tensor(shape: &[i64], values: &[i32]) -> *mut chelis_tensor {
    unsafe {
        let shape_ptr = if shape.is_empty() {
            ptr::null()
        } else {
            shape.as_ptr()
        };
        let tensor = chelis_alloc(shape.len() as c_int, shape_ptr, CHELIS_DTYPE_I32);
        let guard = chelis_tensor_begin_write(tensor);
        let view = chelis_tensor_write_view(guard);
        assert_eq!(view.count as usize, values.len());
        view.data
            .cast::<i32>()
            .copy_from(values.as_ptr(), values.len());
        chelis_tensor_end_write(guard);
        tensor
    }
}

unsafe fn read_i32(tensor: *mut chelis_tensor) -> Vec<i32> {
    unsafe {
        let view = chelis_tensor_read_view(tensor);
        let data = view.data.cast::<i32>();
        (0..view.count as usize)
            .map(|index| *data.add(index))
            .collect()
    }
}

unsafe fn read_bool_payload(tensor: *mut chelis_tensor) -> Vec<bool> {
    unsafe {
        let view = chelis_tensor_read_view(tensor);
        let data = view.data.cast::<Bool8>();
        (0..view.count as usize)
            .map(|index| (*data.add(index)).get())
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

        chelis_tensor_release(out);
        chelis_tensor_release(rhs);
        chelis_tensor_release(lhs);
    }
}

#[test]
fn where_selects_int32_branches_with_exact_bool8_condition() {
    unsafe {
        let shape = [2_i64];
        let cond = chelis_alloc(1, shape.as_ptr(), CHELIS_DTYPE_BOOL);
        let guard = chelis_tensor_begin_write(cond);
        let write = chelis_tensor_write_view(guard);
        let cond_data = write.data.cast::<Bool8>();
        *cond_data = Bool8::new(true);
        *cond_data.add(1) = Bool8::new(false);
        chelis_tensor_end_write(guard);
        let then_tensor = i32_tensor(&[2], &[10, 20]);
        let else_tensor = i32_tensor(&[2], &[-10, -20]);
        let out = chelis_tensor_where(cond, then_tensor, else_tensor);

        assert_eq!(read_i32(out), vec![10, -20]);

        chelis_tensor_release(out);
        chelis_tensor_release(else_tensor);
        chelis_tensor_release(then_tensor);
        chelis_tensor_release(cond);
    }
}

#[test]
fn scatter_add_accumulates_native_int32_updates() {
    unsafe {
        let base = i32_tensor(&[1], &[0]);
        let indices = i32_tensor(&[2], &[0, 0]);
        let updates = i32_tensor(&[2], &[1_000_000_000, -999_999_999]);
        let out = chelis_tensor_scatter_add(base, indices, updates, 0);

        assert_eq!(read_i32(out), vec![1]);

        chelis_tensor_release(out);
        chelis_tensor_release(updates);
        chelis_tensor_release(indices);
        chelis_tensor_release(base);
    }
}

#[test]
fn cumsum_preserves_exact_int32_prefixes() {
    unsafe {
        let input = i32_tensor(&[2], &[1_000_000_000, -999_999_999]);
        let out = chelis_tensor_cumsum(input, 0);

        assert_eq!(read_i32(out), vec![1_000_000_000, 1]);

        chelis_tensor_release(out);
        chelis_tensor_release(input);
    }
}

#[test]
fn trace_accumulates_native_int32_diagonal_values() {
    unsafe {
        let matrix = i32_tensor(&[2, 2], &[1_000_000_000, 0, 0, -999_999_999]);
        let out = chelis_tensor_trace(matrix, 0, 1);

        assert_eq!(read_i32(out), vec![1]);

        chelis_tensor_release(out);
        chelis_tensor_release(matrix);
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

        chelis_tensor_release(out);
        chelis_tensor_release(upper);
        chelis_tensor_release(lower);
        chelis_tensor_release(input);
    }
}

#[test]
fn einsum_multiplies_native_int32_values() {
    unsafe {
        let lhs = i32_tensor(&[1], &[2]);
        let rhs = i32_tensor(&[1], &[3]);
        let equation = chelis_string_from_cstr(c"i,i->".as_ptr());
        let out = chelis_tensor_einsum(equation, lhs, rhs, CHELIS_DTYPE_I32);

        assert_eq!(read_i32(out), vec![6]);

        chelis_tensor_release(out);
        chelis_tensor_release(rhs);
        chelis_tensor_release(lhs);
    }
}

#[test]
fn typed_boundaries_separate_f32_and_bool8_payloads() {
    unsafe {
        let f32_tensor = chelis_alloc(0, ptr::null(), CHELIS_DTYPE_F32);
        let bool_tensor = chelis_alloc(0, ptr::null(), CHELIS_DTYPE_BOOL);
        let f32_view = chelis_tensor_read_view(f32_tensor);
        let bool_view = chelis_tensor_read_view(bool_tensor);

        assert_eq!(
            data_as_f32(f32_tensor),
            f32_view.data.cast_mut().cast::<f32>()
        );
        assert_eq!(data_as_f32_const(f32_tensor), f32_view.data.cast::<f32>());
        assert_eq!(
            Bool8::data_ptr(bool_tensor).expect("bool tensor uses Bool8 storage"),
            bool_view.data.cast_mut().cast::<Bool8>()
        );

        chelis_tensor_release(bool_tensor);
        chelis_tensor_release(f32_tensor);
    }
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "data_as_f32 requires f32 storage")]
fn mutable_f32_boundary_rejects_int32() {
    unsafe {
        let tensor = i32_tensor(&[], &[1]);
        let _ = data_as_f32(tensor);
    }
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "data_as_f32_const requires f32 storage")]
fn const_f32_boundary_rejects_int32() {
    unsafe {
        let tensor = i32_tensor(&[], &[1]);
        let _ = data_as_f32_const(tensor);
    }
}
