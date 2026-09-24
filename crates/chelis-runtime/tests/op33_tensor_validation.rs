//! Fail-closed operation-entry coverage for the public tensor runtime.
//!
//! These cases derive from `spec/05-risc-primitives.md` [05-OP-33]. Every
//! public operation must validate each tensor carrier before inspecting shape
//! metadata, and operation-specific dtype domains must reject forbidden
//! carriers without narrowing the signed-integer/float families they admit.

use chelis_runtime::{
    chelis_alloc, chelis_contiguous, chelis_dict_from_pairs, chelis_dict_get_scalar, chelis_list,
    chelis_list_from_values, chelis_metadata_plan_new, chelis_print_list, chelis_scalar,
    chelis_scalar_from_bits, chelis_string_from_cstr, chelis_tensor, chelis_tensor_alloc_like,
    chelis_tensor_begin_write, chelis_tensor_clamp, chelis_tensor_cmplt, chelis_tensor_concat,
    chelis_tensor_cumsum, chelis_tensor_diagonal, chelis_tensor_einsum, chelis_tensor_elements,
    chelis_tensor_end_write, chelis_tensor_from_values, chelis_tensor_gather,
    chelis_tensor_read_view, chelis_tensor_release, chelis_tensor_reshape,
    chelis_tensor_scatter_add, chelis_tensor_scatter_replace, chelis_tensor_sort,
    chelis_tensor_split, chelis_tensor_trace, chelis_tensor_where, chelis_tensor_write_literal,
    chelis_tensor_write_view, chelis_value_box_scalar, chelis_value_take_tensor, CHELIS_DTYPE_BOOL,
    CHELIS_DTYPE_F32, CHELIS_DTYPE_I16, CHELIS_DTYPE_I32, CHELIS_DTYPE_I64, CHELIS_DTYPE_I8,
    CHELIS_DTYPE_KEY,
};
use std::env;
use std::ffi::CString;
use std::process::Command;
use std::ptr;

const CHILD_ENV: &str = "CHELIS_OP33_VALIDATION_CHILD";

unsafe fn tensor(dtype: u8, shape: &[i64]) -> *mut chelis_tensor {
    chelis_alloc(shape.len() as i32, shape.as_ptr(), dtype)
}

unsafe fn write<T: Copy>(tensor: *mut chelis_tensor, values: &[T]) {
    let guard = chelis_tensor_begin_write(tensor);
    let view = chelis_tensor_write_view(guard);
    assert_eq!(view.count as usize, values.len());
    view.data
        .cast::<T>()
        .copy_from(values.as_ptr(), values.len());
    chelis_tensor_end_write(guard);
}

unsafe fn read<T: Copy>(tensor: *const chelis_tensor) -> Vec<T> {
    let view = chelis_tensor_read_view(tensor);
    std::slice::from_raw_parts(view.data.cast::<T>(), view.count as usize).to_vec()
}

fn run_invalid_case(case: &str) -> ! {
    unsafe {
        let shape = [1_i64];
        match case {
            "cmplt-null" => {
                chelis_tensor_cmplt(ptr::null(), ptr::null());
            }
            "where-null" => {
                chelis_tensor_where(ptr::null(), ptr::null(), ptr::null());
            }
            "clamp-null" => {
                chelis_tensor_clamp(ptr::null(), ptr::null(), ptr::null());
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
                write(indices, &[0.0_f32]);
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
                write(indices, &[0.0_f64]);
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
        "where-null",
        "clamp-null",
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
        write(lhs, &[1_i64, 4]);
        write(rhs, &[2_i64, 3]);
        let compared = chelis_tensor_cmplt(lhs, rhs);
        assert_eq!(read::<u8>(compared), [1, 0]);

        let condition = tensor(CHELIS_DTYPE_BOOL, &pair);
        write(condition, &[1_u8, 0]);
        let then_tensor = tensor(CHELIS_DTYPE_I16, &pair);
        let else_tensor = tensor(CHELIS_DTYPE_I16, &pair);
        write(then_tensor, &[11_i16, 12]);
        write(else_tensor, &[21_i16, 22]);
        let selected = chelis_tensor_where(condition, then_tensor, else_tensor);
        assert_eq!(read::<i16>(selected), [11, 22]);

        let base_shape = [3_i64];
        let base = tensor(CHELIS_DTYPE_I32, &base_shape);
        write(base, &[10_i32, 20, 30]);
        let indices = tensor(CHELIS_DTYPE_I8, &pair);
        write(indices, &[2_i8, 0]);
        let gathered = chelis_tensor_gather(base, indices, 0);
        assert_eq!(read::<i32>(gathered), [30, 10]);

        let updates = tensor(CHELIS_DTYPE_I32, &pair);
        write(updates, &[7_i32, 8]);
        let replaced = chelis_tensor_scatter_replace(base, indices, updates, 0);
        assert_eq!(read::<i32>(replaced), [8, 20, 7]);

        let indices_i16 = tensor(CHELIS_DTYPE_I16, &pair);
        write(indices_i16, &[1_i16, 1]);
        let added = chelis_tensor_scatter_add(base, indices_i16, updates, 0);
        assert_eq!(read::<i32>(added), [10, 35, 30]);

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
            chelis_tensor_release(tensor);
        }
    }
}

const KEY_CHILD_ENV: &str = "CHELIS_OP33_KEY_CHILD";

unsafe fn keys(shape: &[i64]) -> *mut chelis_tensor {
    let keys = tensor(CHELIS_DTYPE_KEY, shape);
    let count = shape.iter().product::<i64>() as u64;
    write(
        keys,
        &(0..count).map(|index| 0xaaaa + index).collect::<Vec<_>>(),
    );
    keys
}

/// A key-tagged scalar spelled directly, as a foreign caller could.
fn key_scalar() -> chelis_scalar {
    chelis_scalar {
        dtype: CHELIS_DTYPE_KEY,
        reserved: [0; 7],
        bits: 0,
    }
}

unsafe fn i64_list(values: &[i64]) -> *mut chelis_list {
    let values: Vec<_> = values
        .iter()
        .map(|&value| {
            chelis_value_box_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_I64, value as u64))
        })
        .collect();
    chelis_list_from_values(values.as_ptr(), values.len() as i64)
}

fn run_key_case(case: &str) -> ! {
    unsafe {
        let pair = [2_i64];
        match case {
            "where" => {
                let condition = tensor(CHELIS_DTYPE_BOOL, &pair);
                write(condition, &[1_u8, 0]);
                chelis_tensor_where(condition, keys(&pair), keys(&pair));
            }
            "gather" => {
                let indices = tensor(CHELIS_DTYPE_I64, &pair);
                write(indices, &[1_i64, 1]);
                chelis_tensor_gather(keys(&pair), indices, 0);
            }
            "scatter-replace" => {
                let indices = tensor(CHELIS_DTYPE_I64, &pair);
                write(indices, &[1_i64, 0]);
                chelis_tensor_scatter_replace(keys(&pair), indices, keys(&pair), 0);
            }
            "concat" => {
                let parts = [
                    chelis_value_take_tensor(keys(&pair)),
                    chelis_value_take_tensor(keys(&pair)),
                ];
                chelis_tensor_concat(chelis_list_from_values(parts.as_ptr(), 2), 0);
            }
            "split" => {
                chelis_tensor_split(keys(&pair), 0, i64_list(&[1, 1]));
            }
            "diagonal" => {
                chelis_tensor_diagonal(keys(&[2, 2]), 0, 1);
            }
            "reshape" => {
                chelis_tensor_reshape(keys(&pair), i64_list(&[2, 1]));
            }
            "from-values" => {
                chelis_tensor_from_values(i64_list(&[]), CHELIS_DTYPE_KEY);
            }
            "cmplt" => {
                chelis_tensor_cmplt(keys(&pair), keys(&pair));
            }
            "sort" => {
                chelis_tensor_sort(keys(&pair), 0);
            }
            // [05-OP-31]: a `chelis_scalar` never carries a key, so no
            // exemplar or scalar selector names one.
            "alloc-like-exemplar" => {
                chelis_tensor_alloc_like(keys(&pair), key_scalar());
            }
            "metadata-plan-exemplar" => {
                let extent = chelis_scalar_from_bits(CHELIS_DTYPE_I64, 2);
                chelis_metadata_plan_new(
                    chelis_scalar_from_bits(CHELIS_DTYPE_I64, 1),
                    &extent,
                    key_scalar(),
                );
            }
            "dict-scalar-dtype" => {
                chelis_dict_get_scalar(
                    chelis_dict_from_pairs(ptr::null()),
                    chelis_value_box_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_I64, 1)),
                    CHELIS_DTYPE_KEY,
                );
            }
            // An empty key tensor has no element for a per-element check to
            // reach, so each of these rejects the dtype at entry.
            "elements-empty" => {
                chelis_tensor_elements(tensor(CHELIS_DTYPE_KEY, &[0]));
            }
            "format-empty" => {
                let parts = [chelis_value_take_tensor(tensor(CHELIS_DTYPE_KEY, &[0]))];
                chelis_print_list(chelis_list_from_values(parts.as_ptr(), 1));
            }
            "write-literal-empty" => {
                let guard = chelis_tensor_begin_write(tensor(CHELIS_DTYPE_KEY, &[0]));
                chelis_tensor_write_literal(
                    guard,
                    chelis_scalar_from_bits(CHELIS_DTYPE_I64, 0),
                    ptr::null(),
                );
            }
            other => panic!("unknown key OP33 case: {other}"),
        }
    }
    panic!("key OP33 case `{case}` returned instead of terminating")
}

/// spec/04 §1.1: an operation admits `key` elements only where its own atom
/// names `key`, and no [05-OP-33] data operation names it. A key tensor is
/// therefore a forbidden carrier at the entry of each one, and the byte-copy
/// operations, which admit every other dtype by width, trap before they copy
/// or duplicate a key.
#[test]
fn key_tensors_are_forbidden_carriers_of_every_data_operation() {
    if let Ok(case) = env::var(KEY_CHILD_ENV) {
        run_key_case(&case);
    }
    let test_binary = env::current_exe().expect("current test binary");
    for case in [
        "where",
        "gather",
        "scatter-replace",
        "concat",
        "split",
        "diagonal",
        "reshape",
        "from-values",
        "cmplt",
        "sort",
        "alloc-like-exemplar",
        "metadata-plan-exemplar",
        "dict-scalar-dtype",
        "elements-empty",
        "format-empty",
        "write-literal-empty",
    ] {
        let output = Command::new(&test_binary)
            .env(KEY_CHILD_ENV, case)
            .arg("--exact")
            .arg("key_tensors_are_forbidden_carriers_of_every_data_operation")
            .arg("--nocapture")
            .output()
            .expect("run key OP33 child");
        assert!(
            !output.status.success(),
            "key OP33 case `{case}` returned success"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        let rejected = match case {
            "alloc-like-exemplar" | "metadata-plan-exemplar" => {
                stderr.contains("a key is not a scalar carrier")
                    && stderr.contains("numeric trap: domain in")
            }
            "write-literal-empty" => {
                stderr.contains("a key tensor has no literal")
                    && stderr.contains("numeric trap: domain in const")
            }
            _ => {
                stderr.contains("Domain:")
                    && stderr.contains("key is not an active data element dtype")
            }
        };
        assert!(
            rejected,
            "key OP33 case `{case}` did not reject the key carrier:\n{stderr}"
        );
    }
}

/// [05-OP-31] names `key` as a tensor dtype, so allocation, the views and
/// the owned-output copy of a key `Load` root still carry a key tensor's
/// words exactly.
#[test]
fn key_tensors_keep_their_storage_callables() {
    unsafe {
        let pair = [2_i64];
        let source = keys(&pair);
        let copied = chelis_contiguous(source);
        assert_eq!(read::<u64>(copied), [0xaaaa, 0xaaab]);
        assert_eq!(chelis_tensor_read_view(copied).dtype, CHELIS_DTYPE_KEY);
        chelis_tensor_release(copied);
        chelis_tensor_release(source);
    }
}
