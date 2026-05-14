//! Tests for chelis_simd.h SIMD reduction functions.
//!
//! Each test generates a standalone C program that includes chelis_simd.h
//! directly (without the rest of the runtime), compiles it with gcc -mavx2,
//! runs it, and verifies the result matches naive scalar computation.
//!
//! There is one test per reduction op. Each test emits a single C program
//! that exercises every size in `SIZES` internally, so the AVX2 lane-boundary
//! sweep still runs at sizes 1, 7, 8, 9, 1024, and 100003 while keeping gcc
//! invocations to one per op.

use std::path::PathBuf;
use std::process::Command;

/// Size sweep probing every AVX2 lane-boundary case: below one lane (1),
/// just under a full lane (7), exactly one lane (8), one past a lane (9),
/// many full lanes (1024), and a large non-lane-multiple (100003).
const SIZES: &[usize] = &[1, 7, 8, 9, 1024, 100003];

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

/// Emit `static float g_data_<n>[<n>] = { ... };` declarations for every
/// size, returning the joined declaration block.
fn emit_data_arrays() -> String {
    SIZES
        .iter()
        .map(|&n| {
            let init = to_c_float_list(&test_data(n));
            format!("static float g_data_{n}[{n}] = {{ {init} }};")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Generate a C program that calls a float-returning SIMD function once per
/// size and prints `n result` lines (one per size).
fn make_c_program_float_sweep(fn_name: &str) -> String {
    let arrays = emit_data_arrays();
    let calls = SIZES
        .iter()
        .map(|&n| format!("    printf(\"{n} %.8g\\n\", (double){fn_name}(g_data_{n}, {n}));"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chelis_simd.h"

{arrays}

int main(void) {{
{calls}
    return 0;
}}
"#
    )
}

/// Generate a C program that calls an int-returning SIMD function once per
/// size and prints `n result` lines (one per size).
fn make_c_program_int_sweep(fn_name: &str) -> String {
    let arrays = emit_data_arrays();
    let calls = SIZES
        .iter()
        .map(|&n| format!("    printf(\"{n} %d\\n\", {fn_name}(g_data_{n}, {n}));"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chelis_simd.h"

{arrays}

int main(void) {{
{calls}
    return 0;
}}
"#
    )
}

/// Parse `n value` lines from sweep-program stdout into a map from size to
/// the raw value string.
fn parse_sweep_output(out: &str) -> std::collections::HashMap<usize, String> {
    let mut map = std::collections::HashMap::new();
    for line in out.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let n: usize = parts
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("malformed sweep line: {line:?}"));
        let value = parts
            .next()
            .unwrap_or_else(|| panic!("malformed sweep line: {line:?}"))
            .to_string();
        map.insert(n, value);
    }
    map
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
// One test per reduction op. Each emits a single C program that sweeps every
// size in SIZES, so scalar-vs-SIMD agreement is checked at every AVX2
// lane-boundary case with a single gcc invocation.
// ---------------------------------------------------------------------------

#[test]
fn simd_sum_f32_all_sizes() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    let src = make_c_program_float_sweep("chelis_sum_f32");
    let out = compile_and_run("sum_sweep", &src);
    let results = parse_sweep_output(&out);
    for &n in SIZES {
        let data = test_data(n);
        let expected = naive_sum(&data);
        let got: f32 = results
            .get(&n)
            .unwrap_or_else(|| panic!("sum: no output for n={n}"))
            .parse()
            .unwrap_or_else(|_| panic!("sum n={n}: could not parse output {:?}", results.get(&n)));
        // Tolerate 1e-3 relative error; SIMD horizontal adds can reorder ops.
        let tol = (expected.abs() * 1e-3).max(1e-2);
        assert!(
            (got - expected).abs() <= tol,
            "sum n={n}: expected {expected}, got {got} (tol {tol})"
        );
    }
}

#[test]
fn simd_max_f32_all_sizes() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    let src = make_c_program_float_sweep("chelis_max_f32");
    let out = compile_and_run("max_sweep", &src);
    let results = parse_sweep_output(&out);
    for &n in SIZES {
        let data = test_data(n);
        let expected = naive_max(&data);
        let got: f32 = results
            .get(&n)
            .unwrap_or_else(|| panic!("max: no output for n={n}"))
            .parse()
            .unwrap_or_else(|_| panic!("max n={n}: could not parse output {:?}", results.get(&n)));
        assert!(
            (got - expected).abs() <= 1e-5,
            "max n={n}: expected {expected}, got {got}"
        );
    }
}

#[test]
fn simd_min_f32_all_sizes() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    let src = make_c_program_float_sweep("chelis_min_f32");
    let out = compile_and_run("min_sweep", &src);
    let results = parse_sweep_output(&out);
    for &n in SIZES {
        let data = test_data(n);
        let expected = naive_min(&data);
        let got: f32 = results
            .get(&n)
            .unwrap_or_else(|| panic!("min: no output for n={n}"))
            .parse()
            .unwrap_or_else(|_| panic!("min n={n}: could not parse output {:?}", results.get(&n)));
        assert!(
            (got - expected).abs() <= 1e-5,
            "min n={n}: expected {expected}, got {got}"
        );
    }
}

#[test]
fn simd_argmax_f32_all_sizes() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    let src = make_c_program_int_sweep("chelis_argmax_f32");
    let out = compile_and_run("argmax_sweep", &src);
    let results = parse_sweep_output(&out);
    for &n in SIZES {
        let data = test_data(n);
        let expected = naive_argmax(&data);
        let got: usize = results
            .get(&n)
            .unwrap_or_else(|| panic!("argmax: no output for n={n}"))
            .parse()
            .unwrap_or_else(|_| {
                panic!("argmax n={n}: could not parse output {:?}", results.get(&n))
            });
        assert_eq!(
            got, expected,
            "argmax n={n}: expected {expected}, got {got}"
        );
    }
}

#[test]
fn simd_argmin_f32_all_sizes() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping");
        return;
    }
    let src = make_c_program_int_sweep("chelis_argmin_f32");
    let out = compile_and_run("argmin_sweep", &src);
    let results = parse_sweep_output(&out);
    for &n in SIZES {
        let data = test_data(n);
        let expected = naive_argmin(&data);
        let got: usize = results
            .get(&n)
            .unwrap_or_else(|| panic!("argmin: no output for n={n}"))
            .parse()
            .unwrap_or_else(|_| {
                panic!("argmin n={n}: could not parse output {:?}", results.get(&n))
            });
        assert_eq!(
            got, expected,
            "argmin n={n}: expected {expected}, got {got}"
        );
    }
}
