//! Exact regressions for the public tensor operations governed by [05-OP-33].
//!
//! These lock the canonical adjacent-pair trace tree, [04-NUM-2] arithmetic
//! NaN finalization without weakening bit-moving payload preservation, and
//! the complete two-operand explicit-output einsum grammar and size checks.

use chelis_runtime::{
    chelis_alloc, chelis_free, chelis_string_from_cstr, chelis_string_release, chelis_tensor,
    chelis_tensor_cumsum, chelis_tensor_einsum, chelis_tensor_scatter_add, chelis_tensor_trace,
    chelis_tensor_where, CHELIS_DTYPE_BOOL, CHELIS_DTYPE_F32, CHELIS_DTYPE_F64, CHELIS_DTYPE_I32,
    CHELIS_DTYPE_I64, CHELIS_DTYPE_I8,
};
use std::env;
use std::ffi::CString;
use std::process::{Command, Output};

const CHILD_ENV: &str = "CHELIS_OP33_EXACTNESS_CHILD";

unsafe fn tensor(dtype: u8, shape: &[i64]) -> *mut chelis_tensor {
    chelis_alloc(shape.len() as i32, shape.as_ptr(), dtype)
}

unsafe fn f32_tensor(shape: &[i64], bits: &[u32]) -> *mut chelis_tensor {
    unsafe {
        let value = tensor(CHELIS_DTYPE_F32, shape);
        (*value)
            .data
            .cast::<u32>()
            .copy_from(bits.as_ptr(), bits.len());
        value
    }
}

unsafe fn f64_tensor(shape: &[i64], bits: &[u64]) -> *mut chelis_tensor {
    unsafe {
        let value = tensor(CHELIS_DTYPE_F64, shape);
        (*value)
            .data
            .cast::<u64>()
            .copy_from(bits.as_ptr(), bits.len());
        value
    }
}

unsafe fn i32_tensor(shape: &[i64], values: &[i32]) -> *mut chelis_tensor {
    unsafe {
        let value = tensor(CHELIS_DTYPE_I32, shape);
        (*value)
            .data
            .cast::<i32>()
            .copy_from(values.as_ptr(), values.len());
        value
    }
}

unsafe fn i64_tensor(shape: &[i64], values: &[i64]) -> *mut chelis_tensor {
    unsafe {
        let value = tensor(CHELIS_DTYPE_I64, shape);
        (*value)
            .data
            .cast::<i64>()
            .copy_from(values.as_ptr(), values.len());
        value
    }
}

unsafe fn call_einsum(
    equation: &str,
    lhs: *const chelis_tensor,
    rhs: *const chelis_tensor,
    accumulator: u8,
) -> *mut chelis_tensor {
    unsafe {
        let equation_text = CString::new(equation).expect("einsum test equation has no NUL");
        let equation_value = chelis_string_from_cstr(equation_text.as_ptr());
        let output = chelis_tensor_einsum(equation_value, lhs, rhs, accumulator);
        chelis_string_release(equation_value);
        output
    }
}

fn child_output(test_name: &str, case: &str) -> Output {
    Command::new(env::current_exe().expect("current test binary"))
        .args(["--exact", test_name, "--nocapture"])
        .env(CHILD_ENV, case)
        .output()
        .unwrap_or_else(|error| panic!("run exactness child `{case}`: {error}"))
}

#[test]
fn f32_trace_uses_the_canonical_adjacent_pair_tree() {
    unsafe {
        let shape = [5_i64, 5];
        let matrix = tensor(CHELIS_DTYPE_F32, &shape);
        let data = (*matrix).data.cast::<f32>();
        for (index, value) in [1e20_f32, 1.0, -1e20_f32, 1.0, 1.0]
            .iter()
            .copied()
            .enumerate()
        {
            *data.add(index * 5 + index) = value;
        }
        let output = chelis_tensor_trace(matrix, 0, 1);
        assert_eq!((*output).dtype, CHELIS_DTYPE_F32);
        assert_eq!((*(*output).data.cast::<f32>()).to_bits(), 1.0_f32.to_bits());
        chelis_free(output);
        chelis_free(matrix);
    }
}

#[test]
fn f64_trace_uses_the_canonical_adjacent_pair_tree() {
    unsafe {
        let shape = [5_i64, 5];
        let matrix = tensor(CHELIS_DTYPE_F64, &shape);
        let data = (*matrix).data.cast::<f64>();
        for (index, value) in [1e300_f64, 1.0, -1e300_f64, 1.0, 1.0]
            .iter()
            .copied()
            .enumerate()
        {
            *data.add(index * 5 + index) = value;
        }
        let output = chelis_tensor_trace(matrix, 0, 1);
        assert_eq!((*output).dtype, CHELIS_DTYPE_F64);
        assert_eq!((*(*output).data.cast::<f64>()).to_bits(), 1.0_f64.to_bits());
        chelis_free(output);
        chelis_free(matrix);
    }
}

#[test]
fn integer_trace_balancing_avoids_spurious_overflow_but_traps_true_overflow() {
    if let Ok(case) = env::var(CHILD_ENV) {
        unsafe {
            match case.as_str() {
                "trace-i32-balanced" => {
                    let mut matrix = vec![0_i32; 16];
                    for (index, value) in [i32::MAX, -1, 2, -2].iter().copied().enumerate() {
                        matrix[index * 4 + index] = value;
                    }
                    let input = i32_tensor(&[4, 4], &matrix);
                    let output = chelis_tensor_trace(input, 0, 1);
                    assert_eq!(*(*output).data.cast::<i32>(), i32::MAX - 1);
                    chelis_free(output);
                    chelis_free(input);
                }
                "trace-i64-balanced" => {
                    let mut matrix = vec![0_i64; 16];
                    for (index, value) in [i64::MAX, -1, 2, -2].iter().copied().enumerate() {
                        matrix[index * 4 + index] = value;
                    }
                    let input = i64_tensor(&[4, 4], &matrix);
                    let output = chelis_tensor_trace(input, 0, 1);
                    assert_eq!(*(*output).data.cast::<i64>(), i64::MAX - 1);
                    chelis_free(output);
                    chelis_free(input);
                }
                "trace-i32-overflow" => {
                    let input = i32_tensor(&[2, 2], &[i32::MAX, 0, 0, 1]);
                    chelis_tensor_trace(input, 0, 1);
                    panic!("i32 overflow case returned");
                }
                "trace-i64-overflow" => {
                    let input = i64_tensor(&[2, 2], &[i64::MAX, 0, 0, 1]);
                    chelis_tensor_trace(input, 0, 1);
                    panic!("i64 overflow case returned");
                }
                other => panic!("unknown integer trace child `{other}`"),
            }
        }
        return;
    }

    let test_name = "integer_trace_balancing_avoids_spurious_overflow_but_traps_true_overflow";
    let mut failures = Vec::new();
    for case in ["trace-i32-balanced", "trace-i64-balanced"] {
        let output = child_output(test_name, case);
        if !output.status.success() {
            failures.push(format!(
                "{case}: {}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
    }
    for (case, diagnostic) in [
        (
            "trace-i32-overflow",
            "numeric trap: overflow in trace at int32",
        ),
        (
            "trace-i64-overflow",
            "numeric trap: overflow in trace at int64",
        ),
    ] {
        let output = child_output(test_name, case);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if output.status.success() || !stderr.contains(diagnostic) {
            failures.push(format!(
                "{case}: expected `{diagnostic}` from a failing child, got status {}:\n{stderr}",
                output.status
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{}/4 canonical integer trace cells failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

unsafe fn run_nan_arithmetic_case(case: &str) {
    const F32_PAYLOAD: u32 = 0x7fc1_2345;
    const F64_PAYLOAD: u64 = 0x7ff8_0000_0000_1234;
    unsafe {
        match case {
            "f32-cumsum" => {
                let input = f32_tensor(&[1], &[F32_PAYLOAD]);
                let output = chelis_tensor_cumsum(input, 0);
                assert_eq!(*(*output).data.cast::<u32>(), 0x7fc0_0000);
                chelis_free(output);
                chelis_free(input);
            }
            "f64-cumsum" => {
                let input = f64_tensor(&[1], &[F64_PAYLOAD]);
                let output = chelis_tensor_cumsum(input, 0);
                assert_eq!(*(*output).data.cast::<u64>(), 0x7ff8_0000_0000_0000);
                chelis_free(output);
                chelis_free(input);
            }
            "f32-scatter" => {
                let base = f32_tensor(&[1], &[F32_PAYLOAD]);
                let indices = tensor(CHELIS_DTYPE_I8, &[1]);
                *(*indices).data.cast::<i8>() = 0;
                let updates = f32_tensor(&[1], &[0.0_f32.to_bits()]);
                let output = chelis_tensor_scatter_add(base, indices, updates, 0);
                assert_eq!(*(*output).data.cast::<u32>(), 0x7fc0_0000);
                chelis_free(output);
                chelis_free(updates);
                chelis_free(indices);
                chelis_free(base);
            }
            "f64-scatter" => {
                let base = f64_tensor(&[1], &[F64_PAYLOAD]);
                let indices = tensor(CHELIS_DTYPE_I8, &[1]);
                *(*indices).data.cast::<i8>() = 0;
                let updates = f64_tensor(&[1], &[0.0_f64.to_bits()]);
                let output = chelis_tensor_scatter_add(base, indices, updates, 0);
                assert_eq!(*(*output).data.cast::<u64>(), 0x7ff8_0000_0000_0000);
                chelis_free(output);
                chelis_free(updates);
                chelis_free(indices);
                chelis_free(base);
            }
            "f32-einsum" => {
                let lhs = f32_tensor(&[1], &[F32_PAYLOAD]);
                let rhs = f32_tensor(&[1], &[1.0_f32.to_bits()]);
                let output = call_einsum("i,i->", lhs, rhs, CHELIS_DTYPE_F32);
                assert_eq!(*(*output).data.cast::<u32>(), 0x7fc0_0000);
                chelis_free(output);
                chelis_free(rhs);
                chelis_free(lhs);
            }
            "f64-einsum" => {
                let lhs = f64_tensor(&[1], &[F64_PAYLOAD]);
                let rhs = f64_tensor(&[1], &[1.0_f64.to_bits()]);
                let output = call_einsum("i,i->", lhs, rhs, CHELIS_DTYPE_F64);
                assert_eq!(*(*output).data.cast::<u64>(), 0x7ff8_0000_0000_0000);
                chelis_free(output);
                chelis_free(rhs);
                chelis_free(lhs);
            }
            "f32-trace" => {
                let input = f32_tensor(
                    &[2, 2],
                    &[
                        F32_PAYLOAD,
                        0.0_f32.to_bits(),
                        0.0_f32.to_bits(),
                        0.0_f32.to_bits(),
                    ],
                );
                let output = chelis_tensor_trace(input, 0, 1);
                assert_eq!(*(*output).data.cast::<u32>(), 0x7fc0_0000);
                chelis_free(output);
                chelis_free(input);
            }
            "f64-trace" => {
                let input = f64_tensor(
                    &[2, 2],
                    &[
                        F64_PAYLOAD,
                        0.0_f64.to_bits(),
                        0.0_f64.to_bits(),
                        0.0_f64.to_bits(),
                    ],
                );
                let output = chelis_tensor_trace(input, 0, 1);
                assert_eq!(*(*output).data.cast::<u64>(), 0x7ff8_0000_0000_0000);
                chelis_free(output);
                chelis_free(input);
            }
            other => panic!("unknown arithmetic NaN child `{other}`"),
        }
    }
}

#[test]
fn every_shared_runtime_arithmetic_consumer_canonicalizes_f32_f64_nan() {
    if let Ok(case) = env::var(CHILD_ENV) {
        unsafe { run_nan_arithmetic_case(&case) };
        return;
    }
    let test_name = "every_shared_runtime_arithmetic_consumer_canonicalizes_f32_f64_nan";
    let mut failures = Vec::new();
    for case in [
        "f32-cumsum",
        "f64-cumsum",
        "f32-scatter",
        "f64-scatter",
        "f32-einsum",
        "f64-einsum",
        "f32-trace",
        "f64-trace",
    ] {
        let output = child_output(test_name, case);
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
        "{}/8 arithmetic NaN cells failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn where_selection_preserves_f32_f64_nan_payload_bits() {
    unsafe {
        let condition = tensor(CHELIS_DTYPE_BOOL, &[1]);
        *(*condition).data = 1;

        let then_f32 = f32_tensor(&[1], &[0x7fc1_2345]);
        let else_f32 = f32_tensor(&[1], &[0x7fc5_4321]);
        let selected_f32 = chelis_tensor_where(condition, then_f32, else_f32);
        assert_eq!(*(*selected_f32).data.cast::<u32>(), 0x7fc1_2345);

        let then_f64 = f64_tensor(&[1], &[0x7ff8_0000_0000_1234]);
        let else_f64 = f64_tensor(&[1], &[0x7ff8_0000_0000_4321]);
        let selected_f64 = chelis_tensor_where(condition, then_f64, else_f64);
        assert_eq!(*(*selected_f64).data.cast::<u64>(), 0x7ff8_0000_0000_1234);

        for value in [
            selected_f64,
            else_f64,
            then_f64,
            selected_f32,
            else_f32,
            then_f32,
            condition,
        ] {
            chelis_free(value);
        }
    }
}

unsafe fn run_einsum_grammar_case(case: &str) {
    unsafe {
        match case {
            "legal-rank-zero" => {
                let lhs = f32_tensor(&[], &[2.0_f32.to_bits()]);
                let rhs = f32_tensor(&[], &[3.0_f32.to_bits()]);
                let output = call_einsum(",->", lhs, rhs, CHELIS_DTYPE_F32);
                assert_eq!((*output).rank, 0);
                assert_eq!(*(*output).data.cast::<f32>(), 6.0);
                chelis_free(output);
                chelis_free(rhs);
                chelis_free(lhs);
            }
            "legal-scalar-vector" => {
                let lhs = f32_tensor(&[], &[2.0_f32.to_bits()]);
                let rhs = f32_tensor(&[2], &[3.0_f32.to_bits(), 4.0_f32.to_bits()]);
                let output = call_einsum(",i->i", lhs, rhs, CHELIS_DTYPE_F32);
                assert_eq!((*output).rank, 1);
                assert_eq!(*(*output).shape.as_ptr(), 2);
                assert_eq!(
                    std::slice::from_raw_parts((*output).data.cast::<f32>(), 2),
                    &[6.0, 8.0]
                );
                chelis_free(output);
                chelis_free(rhs);
                chelis_free(lhs);
            }
            "legal-repeated-input" => {
                let lhs = f32_tensor(
                    &[2, 2],
                    &[
                        1.0_f32.to_bits(),
                        0.0_f32.to_bits(),
                        0.0_f32.to_bits(),
                        2.0_f32.to_bits(),
                    ],
                );
                let rhs = f32_tensor(&[2], &[3.0_f32.to_bits(), 4.0_f32.to_bits()]);
                let output = call_einsum("ii,i->", lhs, rhs, CHELIS_DTYPE_F32);
                assert_eq!(*(*output).data.cast::<f32>(), 11.0);
                chelis_free(output);
                chelis_free(rhs);
                chelis_free(lhs);
            }
            "invalid-uppercase" => {
                let lhs = f32_tensor(&[1], &[1.0_f32.to_bits()]);
                let rhs = f32_tensor(&[1], &[1.0_f32.to_bits()]);
                call_einsum("I,I->", lhs, rhs, CHELIS_DTYPE_F32);
                panic!("uppercase einsum returned");
            }
            "invalid-nonletter" => {
                let lhs = f32_tensor(&[1], &[1.0_f32.to_bits()]);
                let rhs = f32_tensor(&[1], &[1.0_f32.to_bits()]);
                call_einsum("_,i->", lhs, rhs, CHELIS_DTYPE_F32);
                panic!("nonletter einsum returned");
            }
            "invalid-duplicate-output" => {
                let lhs = f32_tensor(&[1], &[1.0_f32.to_bits()]);
                let rhs = f32_tensor(&[1], &[1.0_f32.to_bits()]);
                call_einsum("i,i->ii", lhs, rhs, CHELIS_DTYPE_F32);
                panic!("duplicate-output einsum returned");
            }
            "invalid-output-missing" => {
                let lhs = f32_tensor(&[1], &[1.0_f32.to_bits()]);
                let rhs = f32_tensor(&[1], &[1.0_f32.to_bits()]);
                call_einsum("i,i->z", lhs, rhs, CHELIS_DTYPE_F32);
                panic!("missing-output-label einsum returned");
            }
            "invalid-rank" => {
                let lhs = f32_tensor(&[1], &[1.0_f32.to_bits()]);
                let rhs = f32_tensor(&[1], &[1.0_f32.to_bits()]);
                call_einsum("ij,i->", lhs, rhs, CHELIS_DTYPE_F32);
                panic!("rank-mismatch einsum returned");
            }
            "invalid-repeated-extent" => {
                let lhs = f32_tensor(&[2, 3], &[0; 6]);
                let rhs = f32_tensor(&[2], &[0; 2]);
                call_einsum("ii,i->", lhs, rhs, CHELIS_DTYPE_F32);
                panic!("repeated-extent einsum returned");
            }
            "invalid-whitespace" => {
                let lhs = f32_tensor(&[1], &[1.0_f32.to_bits()]);
                let rhs = f32_tensor(&[1], &[1.0_f32.to_bits()]);
                call_einsum("i, i->", lhs, rhs, CHELIS_DTYPE_F32);
                panic!("whitespace einsum returned");
            }
            "invalid-third-operand" => {
                let lhs = f32_tensor(&[1], &[1.0_f32.to_bits()]);
                let rhs = f32_tensor(&[1], &[1.0_f32.to_bits()]);
                call_einsum("i,i,i->", lhs, rhs, CHELIS_DTYPE_F32);
                panic!("three-operand einsum returned");
            }
            "invalid-implicit-output" => {
                let lhs = f32_tensor(&[1], &[1.0_f32.to_bits()]);
                let rhs = f32_tensor(&[1], &[1.0_f32.to_bits()]);
                call_einsum("i,i", lhs, rhs, CHELIS_DTYPE_F32);
                panic!("implicit-output einsum returned");
            }
            "invalid-ellipsis" => {
                let lhs = f32_tensor(&[1], &[1.0_f32.to_bits()]);
                let rhs = f32_tensor(&[1], &[1.0_f32.to_bits()]);
                call_einsum("...,i->i", lhs, rhs, CHELIS_DTYPE_F32);
                panic!("ellipsis einsum returned");
            }
            "overflow-output-product" => {
                let lhs = f32_tensor(&[5_000_000_000, 0], &[]);
                let rhs = f32_tensor(&[5_000_000_000, 0], &[]);
                call_einsum("za,wa->zw", lhs, rhs, CHELIS_DTYPE_F32);
                panic!("output-product overflow einsum returned");
            }
            "overflow-reduction-product" => {
                let lhs = f32_tensor(&[5_000_000_000, 0], &[]);
                let rhs = f32_tensor(&[5_000_000_000, 0], &[]);
                call_einsum("az,cw->zw", lhs, rhs, CHELIS_DTYPE_F32);
                panic!("reduction-product overflow einsum returned");
            }
            other => panic!("unknown einsum grammar child `{other}`"),
        }
    }
}

#[test]
fn einsum_implements_the_complete_legal_grammar_and_checked_products() {
    if let Ok(case) = env::var(CHILD_ENV) {
        unsafe { run_einsum_grammar_case(&case) };
        return;
    }
    let test_name = "einsum_implements_the_complete_legal_grammar_and_checked_products";
    let mut failures = Vec::new();
    for case in [
        "legal-rank-zero",
        "legal-scalar-vector",
        "legal-repeated-input",
    ] {
        let output = child_output(test_name, case);
        if !output.status.success() {
            failures.push(format!(
                "{case}: {}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
    }
    for case in [
        "invalid-uppercase",
        "invalid-nonletter",
        "invalid-duplicate-output",
        "invalid-output-missing",
        "invalid-rank",
        "invalid-repeated-extent",
        "invalid-whitespace",
        "invalid-third-operand",
        "invalid-implicit-output",
        "invalid-ellipsis",
    ] {
        let output = child_output(test_name, case);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if output.status.success() || !stderr.contains("Domain:") {
            failures.push(format!(
                "{case}: expected branded Domain from a failing child, got status {}:\n{stderr}",
                output.status
            ));
        }
    }
    for case in ["overflow-output-product", "overflow-reduction-product"] {
        let output = child_output(test_name, case);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if output.status.success() || !stderr.contains("Overflow:") {
            failures.push(format!(
                "{case}: expected branded Overflow from a failing child, got status {}:\n{stderr}",
                output.status
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{}/15 einsum grammar/product cells failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
