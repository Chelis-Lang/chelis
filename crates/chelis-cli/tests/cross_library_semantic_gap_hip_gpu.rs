//! Wave 3 W3-A — M5(a): HIP summary consumption GPU manual gate.
//!
//! This is the numeric oracle for the structural tests in
//! `crates/chelis-cli/tests/cross_library_semantic_gap.rs`. Where the
//! structural tests assert exact-substring `chelis_hipblas_sgemm_row_major(`
//! emission for the four matmul forms on `--target hip`, this file goes one
//! step further: it builds a user-`def` matmul wrapper through the CLI,
//! compiles the emitted HIP `.cpp` with `hipcc`, runs the binary on the
//! local GPU through hipBLAS, and verifies the result against a hand-rolled
//! CPU reference (no autograd, no chelis-eval dependency — just a row-major
//! 8x16 @ 16x4 sgemm reference).
//!
//! Acceptance per the W3-A brief: "Add at least one new
//! `g*_user_def_matmul_helper_hits_hipblas_numeric` GPU test that exercises
//! the user-`def` form and asserts numeric agreement vs CPU."
//!
//! Manual gate per AGENTS.md: not part of default CI. To run:
//!
//! ```sh
//! HSA_OVERRIDE_GFX_VERSION=11.5.1 \
//! LD_LIBRARY_PATH=$WHEEL_CORE/lib:$WHEEL_GFX/lib \
//! HIPCC_COMPILE_FLAGS_APPEND="-isystem $WHEEL_CORE/include -L$WHEEL_GFX/lib" \
//! cargo test -p chelis-cli --test cross_library_semantic_gap_hip_gpu -- \
//!     --ignored --test-threads=1
//! ```
//!
//! The structural test pins the dispatch shape (row-major sgemm at 8x16 @
//! 16x4); this gate proves the dispatched call is numerically correct.

use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;

use assert_cmd::cargo::CommandCargoExt;
use tempfile::tempdir;

/// Row-major reference matmul. Locked at f32 since the user-`def` fixture
/// below is `tensor[8, 16, f32]` @ `tensor[16, 4, f32]`.
fn reference_matmul_row_major(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn require_hipcc() {
    let output = StdCommand::new("hipcc")
        .arg("--version")
        .output()
        .unwrap_or_else(|err| panic!("failed to probe hipcc: {err}"));
    assert!(
        output.status.success(),
        "hipcc is required for the M5(a) HIP GPU manual gate"
    );
}

fn write_harness_main_cpp(harness_path: &Path, hip_entry_symbol: &str, a: &[f32], b: &[f32]) {
    // Allocate 8x16 and 16x4 f32 tensors with the supplied data, call the
    // generated HIP entry, and print the 8x4 output as space-separated f32
    // values on one line. The harness deliberately uses the same
    // `chelis_tensor*` ABI that `chelis build --target hip` emits at
    // `crates/chelis-backend-hip/src/lib.rs:65-69` so this test stays
    // ABI-locked to the HIP backend's published function shape.
    let mut lines = Vec::new();
    lines.push(r#"#include "chelis_runtime.h""#.to_string());
    lines.push(format!(
        r#"extern "C" void {hip_entry_symbol}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);"#
    ));
    lines.push(String::new());
    lines.push("int main(void) {".to_string());
    // 8x16 input `a`
    lines.push("    int a_shape[2] = { 8, 16 };".to_string());
    lines.push("    chelis_tensor *a_t = chelis_alloc(2, a_shape, CHELIS_F32);".to_string());
    for (idx, value) in a.iter().enumerate() {
        lines.push(format!("    a_t->data[{idx}] = {value:.8}f;"));
    }
    // 16x4 input `b`
    lines.push("    int b_shape[2] = { 16, 4 };".to_string());
    lines.push("    chelis_tensor *b_t = chelis_alloc(2, b_shape, CHELIS_F32);".to_string());
    for (idx, value) in b.iter().enumerate() {
        lines.push(format!("    b_t->data[{idx}] = {value:.8}f;"));
    }
    lines.push("    chelis_tensor *inputs[2] = { a_t, b_t };".to_string());
    lines.push("    chelis_tensor *outputs[1] = { NULL };".to_string());
    lines.push(format!("    {hip_entry_symbol}(inputs, 2, outputs, 1);"));
    lines.push("    for (int i = 0; i < outputs[0]->size; i++) {".to_string());
    lines.push("        if (i > 0) printf(\" \");".to_string());
    lines.push("        printf(\"%.6f\", outputs[0]->data[i]);".to_string());
    lines.push("    }".to_string());
    lines.push("    printf(\"\\n\");".to_string());
    lines.push("    chelis_free(a_t);".to_string());
    lines.push("    chelis_free(b_t);".to_string());
    lines.push("    chelis_free(outputs[0]);".to_string());
    lines.push("    return 0;".to_string());
    lines.push("}".to_string());

    fs::write(harness_path, lines.join("\n")).expect("write main.cpp");
}

fn parse_output_line(line: &str) -> Vec<f32> {
    line.split_whitespace()
        .map(|token| {
            token
                .parse::<f32>()
                .unwrap_or_else(|err| panic!("parse output token `{token}`: {err}"))
        })
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tag: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{tag}: output length mismatch — actual={} expected={}",
        actual.len(),
        expected.len()
    );
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= 1e-3,
            "{tag}: numerical mismatch at index {idx}: actual={a}, expected={e}"
        );
    }
}

/// Build + compile + run a `.ch` source on HIP. The `.ch` source must
/// produce a tensor[8, 4, f32] output named `<expected_entry>` from inputs
/// `a: tensor[8, 16, f32]` and `b: tensor[16, 4, f32]`. Returns the parsed
/// f32 vector printed by the generated binary.
fn build_run_user_def_matmul_hip(source: &str, name: &str, a: &[f32], b: &[f32]) -> Vec<f32> {
    require_hipcc();

    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}_out"));
    fs::write(&src_path, source).expect("write source");

    let build = StdCommand::cargo_bin("chelis")
        .expect("chelis binary")
        .args([
            "build",
            src_path.to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("chelis build should run");
    assert!(
        build.status.success(),
        "chelis build --target hip failed for {name}: {}",
        String::from_utf8_lossy(&build.stderr)
    );

    // Locking precondition: the generated HIP source must dispatch
    // hipBLAS — this is what the W3-A structural tests already assert,
    // but we re-check here so a regression in summary consumption shows
    // up before we run the GPU binary.
    let cpp_path = out_dir.join(format!("{name}_hip.cpp"));
    let cpp = fs::read_to_string(&cpp_path).expect("read generated hip cpp");
    let row_major_calls = cpp.matches("chelis_hipblas_sgemm_row_major(").count();
    assert_eq!(
        row_major_calls, 2,
        "M5(a) precondition: user-`def` matmul wrapper on HIP must emit \
         exactly two `chelis_hipblas_sgemm_row_major(` calls. got={row_major_calls}"
    );

    // Write a main.cpp that allocates 8x16 / 16x4 f32 inputs, calls the
    // emitted entry, and prints the 8x4 output. The entry symbol name
    // follows the `chelis build` convention of `<file_stem>`.
    let main_path = out_dir.join("main.cpp");
    write_harness_main_cpp(&main_path, name, a, b);

    // Compile through hipcc using the same link-flag conventions
    // `crates/chelis-cli/src/main.rs::cmd_build_hip` prints. -lhipblas /
    // -lhiprtc come from `chelis_backend_hip::codegen_hip`.
    let bin_path = out_dir.join(format!("{name}_bin"));
    let cpp_source_path = out_dir.join(format!("{name}_hip.cpp"));
    let compile = StdCommand::new("hipcc")
        .current_dir(&out_dir)
        .arg("-O2")
        .arg(main_path.file_name().unwrap())
        .arg(cpp_source_path.file_name().unwrap())
        .arg("-L.")
        .arg("-lchelis_runtime")
        .arg("-lhipblas")
        .arg("-lhiprtc")
        .arg("-lpthread")
        .arg("-ldl")
        .arg("-o")
        .arg(&bin_path)
        .output()
        .expect("run hipcc");
    assert!(
        compile.status.success(),
        "hipcc failed for {name}:\nstderr:\n{}\nstdout:\n{}",
        String::from_utf8_lossy(&compile.stderr),
        String::from_utf8_lossy(&compile.stdout),
    );

    let run = StdCommand::new(&bin_path)
        .output()
        .expect("run generated GPU binary");
    assert!(
        run.status.success(),
        "GPU binary failed for {name}:\nstderr:\n{}\nstdout:\n{}",
        String::from_utf8_lossy(&run.stderr),
        String::from_utf8_lossy(&run.stdout),
    );

    let stdout = String::from_utf8(run.stdout).expect("utf8 stdout");
    let line = stdout
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or_else(|| panic!("{name} produced no non-empty output"));
    parse_output_line(line)
}

const FIXTURE_A: [f32; 8 * 16] = [
    0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60,
    -0.10, -0.20, -0.30, -0.40, -0.50, -0.60, -0.70, -0.80, -0.90, -1.00, -1.10, -1.20, -1.30,
    -1.40, -1.50, -1.60, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15,
    1.25, 1.35, 1.45, 1.55, -0.05, -0.15, -0.25, -0.35, -0.45, -0.55, -0.65, -0.75, -0.85, -0.95,
    -1.05, -1.15, -1.25, -1.35, -1.45, -1.55, 0.02, 0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72, 0.82,
    0.92, 1.02, 1.12, 1.22, 1.32, 1.42, 1.52, -0.02, -0.12, -0.22, -0.32, -0.42, -0.52, -0.62,
    -0.72, -0.82, -0.92, -1.02, -1.12, -1.22, -1.32, -1.42, -1.52, 0.08, 0.18, 0.28, 0.38, 0.48,
    0.58, 0.68, 0.78, 0.88, 0.98, 1.08, 1.18, 1.28, 1.38, 1.48, 1.58, -0.08, -0.18, -0.28, -0.38,
    -0.48, -0.58, -0.68, -0.78, -0.88, -0.98, -1.08, -1.18, -1.28, -1.38, -1.48, -1.58,
];

const FIXTURE_B: [f32; 16 * 4] = [
    0.10, 0.20, 0.30, 0.40, -0.10, -0.20, -0.30, -0.40, 0.50, 0.60, 0.70, 0.80, -0.50, -0.60,
    -0.70, -0.80, 0.90, 1.00, 1.10, 1.20, -0.90, -1.00, -1.10, -1.20, 1.30, 1.40, 1.50, 1.60,
    -1.30, -1.40, -1.50, -1.60, 0.05, 0.15, 0.25, 0.35, -0.05, -0.15, -0.25, -0.35, 0.45, 0.55,
    0.65, 0.75, -0.45, -0.55, -0.65, -0.75, 0.85, 0.95, 1.05, 1.15, -0.85, -0.95, -1.05, -1.15,
    1.25, 1.35, 1.45, 1.55, -1.25, -1.35, -1.45, -1.55,
];

#[test]
#[ignore = "manual gate: requires HIP-capable GPU, hipcc, and hipBLAS"]
fn g15_user_def_matmul_helper_hits_hipblas_numeric() {
    // Acceptance: `def my_mm(a, b) = matmul(a, b); def f(a, b) = my_mm(a, b)`
    // on HIP must produce numerically correct results through hipBLAS.
    // The CLI lowers `f`'s body via `lower_named_tensor_entry_dag` with
    // `program_defs` populated, which inlines `my_mm` and yields the same
    // BLAS-eligible DAG that `f(a, b) = matmul(a, b)` produces, so HIP
    // codegen dispatches `chelis_hipblas_sgemm_row_major`.
    let source = "def my_mm(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                  -> tensor[8, 4, f32] = matmul(a, b)\n\
                  def hip_user_def_mm(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                  -> tensor[8, 4, f32] = my_mm(a, b)\n";
    let actual = build_run_user_def_matmul_hip(source, "hip_user_def_mm", &FIXTURE_A, &FIXTURE_B);
    let expected = reference_matmul_row_major(&FIXTURE_A, &FIXTURE_B, 8, 4, 16);
    assert_close(
        &actual,
        &expected,
        "g15_user_def_matmul_helper_hits_hipblas",
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU, hipcc, and hipBLAS"]
fn g15_user_def_manual_matmul_helper_hits_hipblas_numeric() {
    // Same shape as above, but the helper body is hand-written
    // expand+mul+sum (the Einstein-form Tier 2 matmul). The Tier 2 pattern
    // matcher must specialize it back to a BLAS matmul before HIP emit, so
    // the binary still hits hipBLAS numerically.
    let source = "def my_mm(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                  -> tensor[8, 4, f32] = {\n  \
                    ae = expand(a, 2, 4)\n  \
                    be = expand(b, 0, 8)\n  \
                    sum(mul(ae, be), 1)\n\
                  }\n\
                  def hip_user_def_manual_mm(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                  -> tensor[8, 4, f32] = my_mm(a, b)\n";
    let actual =
        build_run_user_def_matmul_hip(source, "hip_user_def_manual_mm", &FIXTURE_A, &FIXTURE_B);
    let expected = reference_matmul_row_major(&FIXTURE_A, &FIXTURE_B, 8, 4, 16);
    assert_close(
        &actual,
        &expected,
        "g15_user_def_manual_matmul_helper_hits_hipblas",
    );
}
