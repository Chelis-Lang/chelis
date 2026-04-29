//! Tests for chelis_simd.h SIMD reduction functions.
//!
//! Each test generates a standalone C program that includes chelis_simd.h
//! directly (without the rest of the runtime), compiles it with gcc -mavx2,
//! runs it, and verifies the result matches naive scalar computation.

use std::path::PathBuf;
use std::process::Command;

fn simd_include_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-runtime/include")
}

fn gcc_available() -> bool {
    // These tests require real GNU gcc with `-mavx2` and `-fopenmp` support,
    // which only makes sense on x86_64. On other architectures (e.g. Apple
    // Silicon, where `/usr/bin/gcc` is a symlink to Apple's clang that
    // doesn't ship libomp and doesn't understand `-mavx2`), skip cleanly.
    if !cfg!(target_arch = "x86_64") {
        return false;
    }
    Command::new("gcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Compile and run a standalone C program that includes chelis_simd.h.
/// Returns stdout on success, or panics on compile/run failure.
fn compile_and_run(test_name: &str, c_src: &str) -> String {
    let dir = std::env::temp_dir().join(format!("chelis_simd_{test_name}"));
    std::fs::create_dir_all(&dir).unwrap();

    let src_path = dir.join("test.c");
    let bin_path = dir.join("test_bin");

    std::fs::write(&src_path, c_src).unwrap();

    let include_dir = simd_include_dir();

    let compile = Command::new("gcc")
        .args([
            "-O2",
            "-mavx2",
            "-fopenmp",
            "-std=c11",
            "-I",
            include_dir.to_str().unwrap(),
            src_path.to_str().unwrap(),
            "-o",
            bin_path.to_str().unwrap(),
            "-lm",
        ])
        .output()
        .expect("failed to invoke gcc");

    if !compile.status.success() {
        let stderr = String::from_utf8_lossy(&compile.stderr);
        panic!("Compilation failed for {test_name}:\n{stderr}\nSource:\n{c_src}");
    }

    let run = Command::new(&bin_path)
        .output()
        .expect("failed to run compiled binary");

    if !run.status.success() {
        let stderr = String::from_utf8_lossy(&run.stderr);
        panic!("Binary exited non-zero for {test_name}:\n{stderr}");
    }

    String::from_utf8_lossy(&run.stdout).into_owned()
}

// ---------------------------------------------------------------------------
// Naive scalar reference implementations (in Rust, for expected-value
// computation in tests).
// ---------------------------------------------------------------------------

fn naive_sum(data: &[f32]) -> f32 {
    // Use f64 accumulator to avoid large-array precision issues.
    data.iter().map(|&x| x as f64).sum::<f64>() as f32
}

fn naive_max(data: &[f32]) -> f32 {
    data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

fn naive_min(data: &[f32]) -> f32 {
    data.iter().copied().fold(f32::INFINITY, f32::min)
}

fn naive_argmax(data: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in data.iter().enumerate().skip(1) {
        if v > data[best] {
            best = i;
        }
    }
    best
}

fn naive_argmin(data: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in data.iter().enumerate().skip(1) {
        if v < data[best] {
            best = i;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// C program templates.
// ---------------------------------------------------------------------------

/// Generate a C program that calls a float-returning SIMD function and
/// prints the result with %f. Uses malloc for large arrays to avoid stack
/// overflow.
fn make_c_program_float(fn_name: &str, data: &[f32]) -> String {
    let n = data.len();
    let init = to_c_float_list(data);
    format!(
        r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chelis_simd.h"

static float g_data[{n}] = {{ {init} }};

int main(void) {{
    float result = {fn_name}(g_data, {n});
    printf("%.8g\n", (double)result);
    return 0;
}}
"#
    )
}

/// Generate a C program that calls an int-returning SIMD function and
/// prints the result with %d.
fn make_c_program_int(fn_name: &str, data: &[f32]) -> String {
    let n = data.len();
    let init = to_c_float_list(data);
    format!(
        r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chelis_simd.h"

static float g_data[{n}] = {{ {init} }};

int main(void) {{
    int result = {fn_name}(g_data, {n});
    printf("%d\n", result);
    return 0;
}}
"#
    )
}

/// Build a comma-separated C float literal list from a slice.
fn to_c_float_list(data: &[f32]) -> String {
    data.iter()
        .map(|v| {
            if v.is_nan() {
                "0.0f".to_string()
            } else if *v == f32::INFINITY {
                "__builtin_inff()".to_string()
            } else if *v == f32::NEG_INFINITY {
                "-__builtin_inff()".to_string()
            } else {
                format!("{v:.8}f")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

// ---------------------------------------------------------------------------
// Generate test data for each size.
// ---------------------------------------------------------------------------

/// Deterministic test data in the range [-4.0, 4.0] with no NaN/Inf values.
/// Uses a simple LCG.
fn test_data(n: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(n);
    let mut x: u64 = 0x1234_5678_ABCD_EF01;
    for _ in 0..n {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map high 32 bits to [0, 1) then to [-4, 4)
        let unit = (x >> 32) as f32 / u32::MAX as f32; // [0, 1]
        data.push(unit * 8.0 - 4.0);
    }
    data
}

// ---------------------------------------------------------------------------
// chelis_sum_f32 tests
// ---------------------------------------------------------------------------

fn run_sum_test(n: usize) {
    let data = test_data(n);
    let expected = naive_sum(&data);
    let src = make_c_program_float("chelis_sum_f32", &data);
    let out = compile_and_run(&format!("sum_{n}"), &src);
    let got: f32 = out
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("sum n={n}: could not parse output {:?}", out.trim()));
    // Tolerate 1e-3 relative error; SIMD horizontal adds can reorder ops.
    let tol = (expected.abs() * 1e-3).max(1e-2);
    assert!(
        (got - expected).abs() <= tol,
        "sum n={n}: expected {expected}, got {got} (tol {tol})"
    );
}

#[test]
fn simd_sum_n1() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_sum_test(1);
}

#[test]
fn simd_sum_n7() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_sum_test(7);
}

#[test]
fn simd_sum_n8() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_sum_test(8);
}

#[test]
fn simd_sum_n9() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_sum_test(9);
}

#[test]
fn simd_sum_n1024() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_sum_test(1024);
}

#[test]
fn simd_sum_n100003() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_sum_test(100003);
}

// ---------------------------------------------------------------------------
// chelis_max_f32 tests
// ---------------------------------------------------------------------------

fn run_max_test(n: usize) {
    let data = test_data(n);
    let expected = naive_max(&data);
    let src = make_c_program_float("chelis_max_f32", &data);
    let out = compile_and_run(&format!("max_{n}"), &src);
    let got: f32 = out
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("max n={n}: could not parse output {:?}", out.trim()));
    assert!(
        (got - expected).abs() <= 1e-5,
        "max n={n}: expected {expected}, got {got}"
    );
}

#[test]
fn simd_max_n1() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_max_test(1);
}

#[test]
fn simd_max_n7() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_max_test(7);
}

#[test]
fn simd_max_n8() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_max_test(8);
}

#[test]
fn simd_max_n9() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_max_test(9);
}

#[test]
fn simd_max_n1024() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_max_test(1024);
}

#[test]
fn simd_max_n100003() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_max_test(100003);
}

// ---------------------------------------------------------------------------
// chelis_min_f32 tests
// ---------------------------------------------------------------------------

fn run_min_test(n: usize) {
    let data = test_data(n);
    let expected = naive_min(&data);
    let src = make_c_program_float("chelis_min_f32", &data);
    let out = compile_and_run(&format!("min_{n}"), &src);
    let got: f32 = out
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("min n={n}: could not parse output {:?}", out.trim()));
    assert!(
        (got - expected).abs() <= 1e-5,
        "min n={n}: expected {expected}, got {got}"
    );
}

#[test]
fn simd_min_n1() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_min_test(1);
}

#[test]
fn simd_min_n7() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_min_test(7);
}

#[test]
fn simd_min_n8() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_min_test(8);
}

#[test]
fn simd_min_n9() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_min_test(9);
}

#[test]
fn simd_min_n1024() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_min_test(1024);
}

#[test]
fn simd_min_n100003() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_min_test(100003);
}

// ---------------------------------------------------------------------------
// chelis_argmax_f32 tests
// ---------------------------------------------------------------------------

fn run_argmax_test(n: usize) {
    let data = test_data(n);
    let expected = naive_argmax(&data);
    let src = make_c_program_int("chelis_argmax_f32", &data);
    let out = compile_and_run(&format!("argmax_{n}"), &src);
    let got: usize = out
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("argmax n={n}: could not parse output {:?}", out.trim()));
    assert_eq!(
        got, expected,
        "argmax n={n}: expected {expected}, got {got}"
    );
}

#[test]
fn simd_argmax_n1() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_argmax_test(1);
}

#[test]
fn simd_argmax_n7() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_argmax_test(7);
}

#[test]
fn simd_argmax_n8() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_argmax_test(8);
}

#[test]
fn simd_argmax_n9() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_argmax_test(9);
}

#[test]
fn simd_argmax_n1024() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_argmax_test(1024);
}

#[test]
fn simd_argmax_n100003() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_argmax_test(100003);
}

// ---------------------------------------------------------------------------
// chelis_argmin_f32 tests
// ---------------------------------------------------------------------------

fn run_argmin_test(n: usize) {
    let data = test_data(n);
    let expected = naive_argmin(&data);
    let src = make_c_program_int("chelis_argmin_f32", &data);
    let out = compile_and_run(&format!("argmin_{n}"), &src);
    let got: usize = out
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("argmin n={n}: could not parse output {:?}", out.trim()));
    assert_eq!(
        got, expected,
        "argmin n={n}: expected {expected}, got {got}"
    );
}

#[test]
fn simd_argmin_n1() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_argmin_test(1);
}

#[test]
fn simd_argmin_n7() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_argmin_test(7);
}

#[test]
fn simd_argmin_n8() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_argmin_test(8);
}

#[test]
fn simd_argmin_n9() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_argmin_test(9);
}

#[test]
fn simd_argmin_n1024() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_argmin_test(1024);
}

#[test]
fn simd_argmin_n100003() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    run_argmin_test(100003);
}
