//! Executable legal-domain matrix for the arithmetic-bearing public tensor ABI.
//!
//! `spec/05-risc-primitives.md` [05-OP-33] admits every active signed-integer
//! and float dtype for these seven operations. Each child asserts the exact
//! result dtype and stored output, so a row cannot pass merely by avoiding an
//! abort. `op33_tensor_validation.rs` supplies the paired null, malformed, and
//! forbidden-domain controls.

use chelis_runtime::{
    chelis_alloc, chelis_free, chelis_string_from_cstr, chelis_tensor, chelis_tensor_clamp,
    chelis_tensor_cmplt, chelis_tensor_cumsum, chelis_tensor_einsum, chelis_tensor_scatter_add,
    chelis_tensor_sort, chelis_tensor_trace, chelis_tuple_get, chelis_tuple_release,
    chelis_value_as_tensor, chelis_value_release, CHELIS_DTYPE_BF16, CHELIS_DTYPE_BOOL,
    CHELIS_DTYPE_F16, CHELIS_DTYPE_F32, CHELIS_DTYPE_F64, CHELIS_DTYPE_I16, CHELIS_DTYPE_I32,
    CHELIS_DTYPE_I64, CHELIS_DTYPE_I8,
};
use std::env;
use std::process::Command;
use std::ptr;

const CHILD_CASE_ENV: &str = "CHELIS_OP33_LEGAL_DOMAIN_CHILD";

#[derive(Clone, Copy)]
struct DtypeCase {
    name: &'static str,
    dtype: u8,
}

const DTYPES: [DtypeCase; 8] = [
    DtypeCase {
        name: "int8",
        dtype: CHELIS_DTYPE_I8,
    },
    DtypeCase {
        name: "int16",
        dtype: CHELIS_DTYPE_I16,
    },
    DtypeCase {
        name: "int32",
        dtype: CHELIS_DTYPE_I32,
    },
    DtypeCase {
        name: "int64",
        dtype: CHELIS_DTYPE_I64,
    },
    DtypeCase {
        name: "f16",
        dtype: CHELIS_DTYPE_F16,
    },
    DtypeCase {
        name: "bf16",
        dtype: CHELIS_DTYPE_BF16,
    },
    DtypeCase {
        name: "f32",
        dtype: CHELIS_DTYPE_F32,
    },
    DtypeCase {
        name: "f64",
        dtype: CHELIS_DTYPE_F64,
    },
];

const OPS: [&str; 7] = [
    "cmplt",
    "cumsum",
    "sort",
    "trace",
    "clamp",
    "einsum",
    "scatter_add",
];

const ADDITIONAL_EINSUM_ACCUMULATORS: [(&str, u8, u8); 6] = [
    ("int8", CHELIS_DTYPE_I64, CHELIS_DTYPE_I64),
    ("int16", CHELIS_DTYPE_I64, CHELIS_DTYPE_I64),
    ("int32", CHELIS_DTYPE_I64, CHELIS_DTYPE_I64),
    ("f16", CHELIS_DTYPE_F64, CHELIS_DTYPE_F16),
    ("bf16", CHELIS_DTYPE_F64, CHELIS_DTYPE_BF16),
    ("f32", CHELIS_DTYPE_F64, CHELIS_DTYPE_F64),
];

fn default_sum_dtype(dtype: u8) -> u8 {
    match dtype {
        CHELIS_DTYPE_I8 | CHELIS_DTYPE_I16 => CHELIS_DTYPE_I32,
        _ => dtype,
    }
}

fn default_accumulator_dtype(dtype: u8) -> u8 {
    match dtype {
        CHELIS_DTYPE_I8 | CHELIS_DTYPE_I16 => CHELIS_DTYPE_I32,
        CHELIS_DTYPE_F16 | CHELIS_DTYPE_BF16 => CHELIS_DTYPE_F32,
        _ => dtype,
    }
}

fn reduced_float_accumulation_probe(dtype: u8) -> Option<(i64, i64)> {
    match dtype {
        // One is below one ULP at `large`, but two survives when the
        // continuing accumulator stays at the required f32 width.
        CHELIS_DTYPE_F16 => Some((2048, 2050)),
        CHELIS_DTYPE_BF16 => Some((256, 258)),
        _ => None,
    }
}

unsafe fn tensor(dtype: u8, shape: &[i64], values: &[i64]) -> *mut chelis_tensor {
    let shape_ptr = if shape.is_empty() {
        ptr::null()
    } else {
        shape.as_ptr()
    };
    let tensor = chelis_alloc(shape.len() as i32, shape_ptr, dtype);
    assert_eq!((*tensor).size as usize, values.len());
    match dtype {
        CHELIS_DTYPE_I8 => write_values::<i8>(tensor, values, |value| value as i8),
        CHELIS_DTYPE_I16 => write_values::<i16>(tensor, values, |value| value as i16),
        CHELIS_DTYPE_I32 => write_values::<i32>(tensor, values, |value| value as i32),
        CHELIS_DTYPE_I64 => write_values::<i64>(tensor, values, |value| value),
        CHELIS_DTYPE_F16 => write_values::<u16>(tensor, values, |value| {
            half::f16::from_f32(value as f32).to_bits()
        }),
        CHELIS_DTYPE_BF16 => write_values::<u16>(tensor, values, |value| {
            half::bf16::from_f32(value as f32).to_bits()
        }),
        CHELIS_DTYPE_F32 => write_values::<f32>(tensor, values, |value| value as f32),
        CHELIS_DTYPE_F64 => write_values::<f64>(tensor, values, |value| value as f64),
        other => panic!("unexpected legal matrix dtype {other}"),
    }
    tensor
}

unsafe fn write_values<T: Copy>(
    tensor: *mut chelis_tensor,
    values: &[i64],
    convert: impl Fn(i64) -> T,
) {
    let data = (*tensor).data.cast::<T>();
    for (index, value) in values.iter().copied().enumerate() {
        *data.add(index) = convert(value);
    }
}

unsafe fn assert_values(tensor: *const chelis_tensor, dtype: u8, expected: &[i64]) {
    assert_eq!((*tensor).dtype, dtype, "unexpected result dtype");
    assert_eq!((*tensor).size as usize, expected.len());
    match dtype {
        CHELIS_DTYPE_I8 => assert_storage::<i8>(tensor, expected, |value| value as i8),
        CHELIS_DTYPE_I16 => assert_storage::<i16>(tensor, expected, |value| value as i16),
        CHELIS_DTYPE_I32 => assert_storage::<i32>(tensor, expected, |value| value as i32),
        CHELIS_DTYPE_I64 => assert_storage::<i64>(tensor, expected, |value| value),
        CHELIS_DTYPE_F16 => assert_storage::<u16>(tensor, expected, |value| {
            half::f16::from_f32(value as f32).to_bits()
        }),
        CHELIS_DTYPE_BF16 => assert_storage::<u16>(tensor, expected, |value| {
            half::bf16::from_f32(value as f32).to_bits()
        }),
        CHELIS_DTYPE_F32 => {
            assert_storage::<u32>(tensor, expected, |value| (value as f32).to_bits())
        }
        CHELIS_DTYPE_F64 => {
            assert_storage::<u64>(tensor, expected, |value| (value as f64).to_bits())
        }
        other => panic!("unexpected result dtype {other}"),
    }
}

unsafe fn assert_storage<T: Copy + std::fmt::Debug + PartialEq>(
    tensor: *const chelis_tensor,
    expected: &[i64],
    convert: impl Fn(i64) -> T,
) {
    let data = (*tensor).data.cast::<T>();
    let actual = (0..expected.len())
        .map(|index| *data.add(index))
        .collect::<Vec<_>>();
    let expected = expected.iter().copied().map(convert).collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

unsafe fn run_case(op: &str, dtype: u8) {
    match op {
        "cmplt" => {
            let lhs = tensor(dtype, &[3], &[-2, 1, 3]);
            let rhs = tensor(dtype, &[3], &[-1, 1, 2]);
            let output = chelis_tensor_cmplt(lhs, rhs);
            assert_eq!((*output).dtype, CHELIS_DTYPE_BOOL);
            assert_eq!(std::slice::from_raw_parts((*output).data, 3), &[1, 0, 0]);
            chelis_free(output);
            chelis_free(rhs);
            chelis_free(lhs);
        }
        "cumsum" => {
            let (input_values, expected) = reduced_float_accumulation_probe(dtype)
                .map_or((vec![1, 2, -1], vec![1, 3, 2]), |(large, final_sum)| {
                    (vec![large, 1, 1], vec![large, large, final_sum])
                });
            let input = tensor(dtype, &[3], &input_values);
            let output = chelis_tensor_cumsum(input, 0);
            assert_values(output, default_sum_dtype(dtype), &expected);
            chelis_free(output);
            chelis_free(input);
        }
        "sort" => {
            let input = tensor(dtype, &[4], &[3, -1, 2, -1]);
            let output = chelis_tensor_sort(input, 0);
            let values_value = chelis_tuple_get(output, 0);
            let indices_value = chelis_tuple_get(output, 1);
            let values = chelis_value_as_tensor(values_value);
            let indices = chelis_value_as_tensor(indices_value);
            assert_values(values, dtype, &[-1, -1, 2, 3]);
            assert_values(indices, CHELIS_DTYPE_I64, &[1, 3, 2, 0]);
            chelis_value_release(indices_value);
            chelis_value_release(values_value);
            chelis_tuple_release(output);
            chelis_free(input);
        }
        "trace" => {
            let (shape, values, expected) = reduced_float_accumulation_probe(dtype)
                .map_or((vec![2, 2], vec![1, 9, -9, 2], 3), |(large, final_sum)| {
                    (vec![3, 3], vec![large, 0, 0, 0, 1, 0, 0, 0, 1], final_sum)
                });
            let input = tensor(dtype, &shape, &values);
            let output = chelis_tensor_trace(input, 0, 1);
            assert_values(output, default_sum_dtype(dtype), &[expected]);
            chelis_free(output);
            chelis_free(input);
        }
        "clamp" => {
            let input = tensor(dtype, &[3], &[-3, 0, 4]);
            let lower = tensor(dtype, &[], &[-1]);
            let upper = tensor(dtype, &[], &[2]);
            let output = chelis_tensor_clamp(input, lower, upper);
            assert_values(output, dtype, &[-1, 0, 2]);
            chelis_free(output);
            chelis_free(upper);
            chelis_free(lower);
            chelis_free(input);
        }
        "einsum" => {
            let (lhs_values, rhs_values, expected) = reduced_float_accumulation_probe(dtype)
                .map_or((vec![1, 2], vec![3, 4], 11), |(large, final_sum)| {
                    (vec![large, 1, 1], vec![1, 1, 1], final_sum)
                });
            let lhs = tensor(dtype, &[lhs_values.len() as i64], &lhs_values);
            let rhs = tensor(dtype, &[rhs_values.len() as i64], &rhs_values);
            let output = chelis_tensor_einsum(
                chelis_string_from_cstr(c"i,i->".as_ptr()),
                lhs,
                rhs,
                default_accumulator_dtype(dtype),
            );
            assert_values(output, default_sum_dtype(dtype), &[expected]);
            chelis_free(output);
            chelis_free(rhs);
            chelis_free(lhs);
        }
        "scatter_add" => {
            let (base_values, index_values, update_values, expected) =
                reduced_float_accumulation_probe(dtype).map_or(
                    (vec![10, 20, 30], vec![1, 1], vec![2, 3], vec![10, 25, 30]),
                    |(large, _)| (vec![large], vec![0, 0], vec![1, 1], vec![large]),
                );
            let base = tensor(dtype, &[base_values.len() as i64], &base_values);
            let indices = tensor(CHELIS_DTYPE_I8, &[index_values.len() as i64], &index_values);
            let updates = tensor(dtype, &[update_values.len() as i64], &update_values);
            let output = chelis_tensor_scatter_add(base, indices, updates, 0);
            assert_values(output, dtype, &expected);
            chelis_free(output);
            chelis_free(updates);
            chelis_free(indices);
            chelis_free(base);
        }
        other => panic!("unknown OP33 legal-domain operation {other}"),
    }
}

unsafe fn run_einsum_accumulator_case(dtype: u8, accumulator: u8, result_dtype: u8) {
    let lhs = tensor(dtype, &[2], &[1, 2]);
    let rhs = tensor(dtype, &[2], &[3, 4]);
    let output = chelis_tensor_einsum(
        chelis_string_from_cstr(c"i,i->".as_ptr()),
        lhs,
        rhs,
        accumulator,
    );
    assert_values(output, result_dtype, &[11]);
    chelis_free(output);
    chelis_free(rhs);
    chelis_free(lhs);
}

#[test]
fn every_active_numeric_dtype_has_exact_outputs_across_op33_arithmetic() {
    if let Ok(case) = env::var(CHILD_CASE_ENV) {
        let (op, dtype_name) = case.split_once(':').expect("operation:dtype child case");
        let dtype = DTYPES
            .iter()
            .find(|candidate| candidate.name == dtype_name)
            .unwrap_or_else(|| panic!("unknown child dtype {dtype_name}"));
        unsafe { run_case(op, dtype.dtype) };
        return;
    }

    let test_binary = env::current_exe().expect("current test binary");
    let mut failures = Vec::new();
    for op in OPS {
        for dtype in DTYPES {
            let case = format!("{op}:{}", dtype.name);
            let output = Command::new(&test_binary)
                .args([
                    "--exact",
                    "every_active_numeric_dtype_has_exact_outputs_across_op33_arithmetic",
                    "--nocapture",
                ])
                .env(CHILD_CASE_ENV, &case)
                .output()
                .unwrap_or_else(|error| panic!("run legal OP33 child {case}: {error}"));
            if !output.status.success() {
                failures.push(format!(
                    "{case}: {}{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{}/{} legal OP33 cells failed:\n{}",
        failures.len(),
        OPS.len() * DTYPES.len(),
        failures.join("\n")
    );
}

#[test]
fn every_additional_permitted_einsum_accumulator_has_the_exact_result_dtype() {
    if let Ok(case) = env::var(CHILD_CASE_ENV) {
        let dtype_name = case
            .strip_prefix("einsum-acc:")
            .expect("einsum-acc child case");
        let (name, accumulator, result_dtype) = ADDITIONAL_EINSUM_ACCUMULATORS
            .iter()
            .copied()
            .find(|(name, _, _)| *name == dtype_name)
            .unwrap_or_else(|| panic!("unknown accumulator child dtype {dtype_name}"));
        let dtype = DTYPES
            .iter()
            .find(|candidate| candidate.name == name)
            .expect("operand dtype");
        unsafe { run_einsum_accumulator_case(dtype.dtype, accumulator, result_dtype) };
        return;
    }

    let test_binary = env::current_exe().expect("current test binary");
    let mut failures = Vec::new();
    for (dtype_name, _, _) in ADDITIONAL_EINSUM_ACCUMULATORS {
        let case = format!("einsum-acc:{dtype_name}");
        let output = Command::new(&test_binary)
            .args([
                "--exact",
                "every_additional_permitted_einsum_accumulator_has_the_exact_result_dtype",
                "--nocapture",
            ])
            .env(CHILD_CASE_ENV, &case)
            .output()
            .unwrap_or_else(|error| panic!("run additional einsum child {case}: {error}"));
        if !output.status.success() {
            failures.push(format!(
                "{case}: {}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{}/{} additional legal einsum accumulator cells failed:\n{}",
        failures.len(),
        ADDITIONAL_EINSUM_ACCUMULATORS.len(),
        failures.join("\n")
    );
}
