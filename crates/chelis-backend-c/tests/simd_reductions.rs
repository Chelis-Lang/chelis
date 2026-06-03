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

/// Reference stride-4 ILP cascade in pure Rust f32 — mirrors the shape
/// the C backend's `chelis_sum_f32` emits (see `chelis_simd.h:45` and
/// `crates/chelis-runtime/include/chelis_simd.h`). Used to bit-exact-
/// compare against the SIMD helper for sizes ≤ 16, where the helper
/// has no SIMD body to execute and reduces to scalar stride-4.
///
/// Algorithm: four f32 accumulator lanes loaded in round-robin via
/// `acc[i & 3] += value[i]`, combined as `(acc0 + acc1) + (acc2 +
/// acc3)`. For n < 4 the tail handles all elements in their natural
/// lane assignment (acc0 = x[0], acc1 = x[1], acc2 = x[2]); the
/// combine is then `(acc0 + acc1) + (acc2 + 0.0)` which equals the
/// natural left-fold for n ≤ 3.
fn stride4_sum_f32(data: &[f32]) -> f32 {
    let mut acc = [0.0_f32; 4];
    for (i, &v) in data.iter().enumerate() {
        acc[i & 3] += v;
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3])
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

/// Render a finite f32 as an exact C99 hexadecimal-float constant
/// expression (e.g. `0x1.fffffep+1f`). Unlike a `{:.8}f` decimal literal,
/// a hex float carries every mantissa bit verbatim, so the `static float`
/// array these literals initialize is byte-identical to the Rust source
/// data the bit-exact reductions below compare against. A hex float is
/// also a constant expression, so it is legal in a `static` initializer
/// (a `chelis_f32_from_bits(...)` call would not be). Sibling of the #189
/// / #248 / #250 / #251 / #252 lossy-float-emission class.
fn f32_to_c_hex_literal(v: f32) -> String {
    let bits = v.to_bits();
    let sign = if bits >> 31 == 1 { "-" } else { "" };
    let exp_field = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;
    if exp_field == 0 && mantissa == 0 {
        // Signed zero.
        return format!("{sign}0x0p+0f");
    }
    // 23-bit mantissa, left-aligned to 24 bits (6 hex digits) so each
    // nibble is a clean hex digit. `mantissa / 2^23 == (mantissa << 1) / 2^24`.
    let frac = format!("{:06x}", mantissa << 1);
    if exp_field == 0 {
        // Subnormal: implicit leading digit is 0, fixed exponent -126.
        format!("{sign}0x0.{frac}p-126f")
    } else {
        // Normal: implicit leading digit is 1, unbiased exponent.
        let unbiased = exp_field - 127;
        format!("{sign}0x1.{frac}p{unbiased:+}f")
    }
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
                f32_to_c_hex_literal(*v)
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
        let got: f32 = results
            .get(&n)
            .unwrap_or_else(|| panic!("sum: no output for n={n}"))
            .parse()
            .unwrap_or_else(|_| panic!("sum n={n}: could not parse output {:?}", results.get(&n)));
        if n <= 16 {
            // For n ≤ 16 the SIMD helper executes no SIMD body (the
            // AVX2 path requires n ≥ 32) — it reduces to scalar
            // stride-4 cascade. That contract is the whole point of
            // PR #168 (issue #163), so the test must enforce
            // bit-exact agreement here. A loose tolerance would
            // silently mask a regression to left-fold or a different
            // tree shape.
            let expected_stride4 = stride4_sum_f32(&data);
            assert_eq!(
                got.to_bits(),
                expected_stride4.to_bits(),
                "sum n={n}: expected stride-4 bit-exact {expected_stride4} \
                 (bits 0x{:08x}), got {got} (bits 0x{:08x})",
                expected_stride4.to_bits(),
                got.to_bits(),
            );
        } else {
            // For n > 16 the AVX2 SIMD body executes and the
            // horizontal-add at the end of the SIMD register can
            // produce a different rounding tree than stride-4
            // scalar. Bound the error tightly to ~1 ULP scaled by
            // sqrt(n) (the standard pairwise-reduction error model),
            // not the 1e-3 relative bound the test previously used.
            // f32::EPSILON ≈ 1.19e-7; for n=100003 sqrt(n)≈316, so
            // tol ≈ 3.8e-5 of |expected| — tight enough to catch a
            // real regression while permitting normal SIMD reordering.
            let expected = naive_sum(&data);
            let tol = (expected.abs() * f32::EPSILON * (n as f32).sqrt()).max(1e-6);
            assert!(
                (got - expected).abs() <= tol,
                "sum n={n}: expected {expected} ± {tol}, got {got} (delta {})",
                (got - expected).abs()
            );
        }
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

/// Reference decoder for the C99 hex-float strings `f32_to_c_hex_literal`
/// emits, mirroring how a C compiler reads them. Used only by the
/// round-trip test below; deliberately independent of `f32::from_str`
/// (which cannot parse C hex floats). Returns the f32 the literal denotes.
fn parse_c_hex_literal(s: &str) -> f32 {
    let (sign, rest) = match s.strip_prefix('-') {
        Some(r) => (-1.0_f64, r),
        None => (1.0_f64, s),
    };
    let body = rest
        .strip_suffix('f')
        .and_then(|r| r.strip_prefix("0x"))
        .unwrap_or_else(|| panic!("unexpected hex-float shape: {s}"));
    let (mantissa_part, exp_part) = body
        .split_once('p')
        .unwrap_or_else(|| panic!("missing exponent in {s}"));
    let exp: i32 = exp_part.parse().expect("exponent");
    let (int_part, frac_part) = match mantissa_part.split_once('.') {
        Some((i, f)) => (i, f),
        None => (mantissa_part, ""),
    };
    let mut mantissa = u64::from_str_radix(int_part, 16).expect("int digit") as f64;
    for (i, c) in frac_part.chars().enumerate() {
        let digit = c.to_digit(16).expect("hex frac digit") as f64;
        mantissa += digit * 16f64.powi(-(i as i32 + 1));
    }
    (sign * mantissa * 2f64.powi(exp)) as f32
}

/// Sibling of the #189 / #248 lossy-float-emission fix: every finite f32
/// `f32_to_c_hex_literal` emits must round-trip to its exact source bits.
/// The pre-fix `{:.8}f` decimal form silently lost mantissa bits, which
/// would have broken the `simd_sum_f32_all_sizes` bit-exact (`n <= 16`)
/// assertion for any input whose 8-place decimal does not reparse to the
/// same f32. Pure string/bit oracle: no gcc or run needed.
#[test]
fn hex_float_literal_round_trips_exact_f32_bits() {
    let mut cases: Vec<f32> = vec![
        0.1_f32,
        (1.0_f64 / 3.0_f64) as f32,
        1e-40_f32, // subnormal: `%.8` -> `0.00000000f`
        f32::from_bits(0x1234_5678), // #189 small-magnitude reproducer
        0.0_f32,
        -0.0_f32,
        -4.0_f32,
        3.999_999_8_f32,
        f32::MIN_POSITIVE, // smallest normal
        f32::from_bits(1), // smallest subnormal
        f32::MAX,
        f32::MIN,
    ];
    // Sweep the actual PRNG corpus the reductions feed in.
    for &n in SIZES {
        cases.extend(test_data(n));
    }
    for value in cases {
        let lit = f32_to_c_hex_literal(value);
        let back = parse_c_hex_literal(&lit);
        assert_eq!(
            back.to_bits(),
            value.to_bits(),
            "hex literal {lit} for {value:e} (bits {:#010x}) did not round-trip; got bits {:#010x}",
            value.to_bits(),
            back.to_bits()
        );
        // Negative parity: the emitted literal must be the hex form, not
        // a lossy `{:.8}f` decimal.
        assert!(
            lit.starts_with("0x") || lit.starts_with("-0x"),
            "literal for {value:e} must be a C hex float, not decimal: {lit}"
        );
    }
    // The subnormal reproducer is the sharpest: `%.8` collapses it to
    // `0.00000000f` (bits 0), but the bit pattern is nonzero and survives.
    let denormal = 1e-40_f32;
    assert_ne!(denormal.to_bits(), 0);
    assert_eq!(
        parse_c_hex_literal(&f32_to_c_hex_literal(denormal)).to_bits(),
        denormal.to_bits()
    );
}
