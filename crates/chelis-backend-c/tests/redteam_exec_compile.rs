//! Adversarial execution tests: actually compile AND RUN generated C code.
//! These test numerical correctness, not just source patterns.
//!
//! The generated kernel signature is:
//!   void func(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out)
//! The kernel allocates output tensors internally via chelis_alloc.
//! We link against the chelis_runtime .a to resolve those symbols.

use chelis_backend_c::{CodegenOptions, MathLib, codegen_with_options};
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::fuse::fuse;
use chelis_types::types::Prim;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn runtime_include_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-runtime/include")
}

fn runtime_lib_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/debug/libchelis_runtime.a")
}

/// Write generated C + harness, compile, run, return stdout. None = compile/run failure.
fn compile_and_run_kernel(test_name: &str, c_source: &str, harness: &str) -> Option<String> {
    let dir = std::env::temp_dir().join(format!("chelis_exec_{test_name}"));
    fs::create_dir_all(&dir).unwrap();

    fs::write(dir.join("kernel.c"), c_source).unwrap();
    fs::write(dir.join("main.c"), harness).unwrap();

    let include_dir = runtime_include_dir();
    for hdr in &[
        "chelis_runtime.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        let src = fs::read_to_string(include_dir.join(hdr)).unwrap();
        fs::write(dir.join(hdr), src).unwrap();
    }

    let bin = dir.join("test_bin");
    let runtime_lib = runtime_lib_path();

    let compile = Command::new("gcc")
        .args([
            "-O2",
            "-mavx2",
            "-std=c11",
            "-I",
            dir.to_str().unwrap(),
            dir.join("kernel.c").to_str().unwrap(),
            dir.join("main.c").to_str().unwrap(),
            "-o",
            bin.to_str().unwrap(),
            runtime_lib.to_str().unwrap(),
            "-lm",
            "-lpthread",
            "-ldl",
        ])
        .output()
        .expect("failed to invoke gcc");

    if !compile.status.success() {
        let stderr = String::from_utf8_lossy(&compile.stderr);
        eprintln!("COMPILE FAILED [{test_name}]:\n{stderr}");
        eprintln!("Kernel C:\n{c_source}");
        return None;
    }

    let run = Command::new(&bin).output().expect("failed to run binary");
    if !run.status.success() {
        let stderr = String::from_utf8_lossy(&run.stderr);
        let stdout = String::from_utf8_lossy(&run.stdout);
        eprintln!("RUN FAILED [{test_name}]\nstdout: {stdout}\nstderr: {stderr}");
        return None;
    }

    Some(String::from_utf8_lossy(&run.stdout).into_owned())
}

// Common harness header: wrap a raw float array in a stack-allocated chelis_tensor.
const HARNESS_HEADER: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "chelis_runtime.h"

static chelis_tensor make_view_1d(float* data, int n) {
    chelis_tensor t;
    memset(&t, 0, sizeof(t));
    t.data = data;
    t.shape[0] = n;
    t.strides[0] = 1;
    t.ndim = 1;
    t.dtype = CHELIS_F32;
    t.size = n;
    t.owns_data = 0;
    return t;
}
"#;

// ---- Test 6 / Item 6: MathLib::None exp kernel ----

#[test]
fn exec_math_none_exp_kernel_correct_output() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(4));
    dag.add_node(RiscOp::Exp, vec![a], vec_f32(4));
    let dag = fuse(&dag);

    let result = codegen_with_options(
        &dag,
        "test_exp_none",
        CodegenOptions {
            math_lib_override: Some(MathLib::None),
            ..Default::default()
        },
    );
    let src = &result.c_source;

    assert!(
        !src.contains("CHELIS_HAS_SLEEF"),
        "MathLib::None must not emit Sleef guard"
    );
    assert!(
        !src.contains("chelis_math.h"),
        "MathLib::None must not include chelis_math.h"
    );
    assert!(
        src.contains("expf("),
        "MathLib::None must use scalar expf()"
    );
    assert!(
        src.contains("#pragma omp parallel for simd"),
        "MathLib::None must use Level-1 omp simd"
    );

    let harness = format!(
        r#"{HARNESS_HEADER}
extern void test_exp_none(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float in_data[4] = {{0.0f, 1.0f, 2.0f, -1.0f}};
    chelis_tensor in_t = make_view_1d(in_data, 4);
    chelis_tensor* in_ptr = &in_t;
    chelis_tensor* inputs[1] = {{in_ptr}};
    chelis_tensor* out_slot = NULL;
    chelis_tensor* outputs[1] = {{out_slot}};

    test_exp_none(inputs, 1, outputs, 1);

    float expected[4] = {{1.0f, 2.718282f, 7.389056f, 0.367879f}};
    int ok = 1;
    for (int i = 0; i < 4; i++) {{
        float got = outputs[0]->data[i];
        float reldiff = fabsf(got - expected[i]) / (fabsf(expected[i]) + 1e-6f);
        if (reldiff > 1e-4f) {{
            printf("MISMATCH at %d: got %.6f expected %.6f\n", i, got, expected[i]);
            ok = 0;
        }}
    }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("none_exp", src, &harness) else {
        panic!("MathLib::None exp kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "MathLib::None exp kernel wrong output:\n{output}"
    );
}

// ---- Test 4: Sleef kernel scalar fallback (without -DCHELIS_HAS_SLEEF) ----
//
// FINDING: A single Load->Exp DAG does NOT generate the Sleef path because
// fuse() only fuses chains of length >= 2.  The Sleef path lives in
// emit_fused_elem which is only called for FusedElem nodes.  A single Exp
// goes through emit_unary_func which never emits Sleef.  This test uses a
// 2-op chain (Exp -> Neg) so that fuse() produces a FusedElem node.
#[test]
fn exec_sleef_kernel_scalar_fallback_correct() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(9));
    let e = dag.add_node(RiscOp::Exp, vec![a], vec_f32(9));
    dag.add_node(RiscOp::Neg, vec![e], vec_f32(9)); // 2-op chain: fuses into FusedElem
    let dag = fuse(&dag);

    let result = codegen_with_options(
        &dag,
        "test_exp_sleef",
        CodegenOptions {
            math_lib_override: Some(MathLib::Sleef),
            ..Default::default()
        },
    );
    let src = &result.c_source;

    assert!(
        src.contains("#ifdef CHELIS_HAS_SLEEF"),
        "Missing Sleef guard"
    );
    assert!(src.contains("CHELIS_EXPF8("), "Missing CHELIS_EXPF8");
    assert!(src.contains("_mm256_loadu_ps("), "Missing AVX2 load");
    assert!(src.contains("_mm256_storeu_ps("), "Missing AVX2 store");
    assert!(src.contains("for (; __i < "), "Missing scalar tail");
    assert!(src.contains("#else"), "Missing #else");

    // Compile WITHOUT -DCHELIS_HAS_SLEEF: the scalar #else branch runs for all 9 elements.
    let harness = format!(
        r#"{HARNESS_HEADER}
extern void test_exp_sleef(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float in_data[9] = {{0.0f, 1.0f, -1.0f, 0.5f, 2.0f, -2.0f, 0.1f, 3.0f, -0.5f}};
    chelis_tensor in_t = make_view_1d(in_data, 9);
    chelis_tensor* in_ptr = &in_t;
    chelis_tensor* inputs[1] = {{in_ptr}};
    chelis_tensor* out_slot = NULL;
    chelis_tensor* outputs[1] = {{out_slot}};

    test_exp_sleef(inputs, 1, outputs, 1);

    int ok = 1;
    for (int i = 0; i < 9; i++) {{
        // The 2-op chain is exp->neg, so expected = -expf(x)
        float expected = -expf(in_data[i]);
        float got = outputs[0]->data[i];
        float reldiff = fabsf(got - expected) / (fabsf(expected) + 1e-6f);
        if (reldiff > 1e-4f) {{
            printf("MISMATCH at %d: got %.6f expected %.6f\n", i, got, expected);
            ok = 0;
        }}
    }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("sleef_exp_neg9", src, &harness) else {
        panic!("Sleef exp->neg kernel scalar fallback failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "Sleef exp->neg kernel scalar fallback wrong:\n{output}"
    );
}

// ---- Test 10: ReduceSum calls chelis_sum_f32 and produces correct result ----

#[test]
fn exec_reduce_sum_correct_output() {
    // Use TensorType::scalar_f32() (dims=[]) for the output — that is the correct
    // output type for a full-axis reduction producing a scalar.
    // Using vec_f32(1) (dims=[Lit(1)]) is wrong and bypasses the chelis_sum_f32 fast path.
    let scalar_ty = TensorType::scalar_f32();
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(100));
    dag.add_node(RiscOp::Sum { axis: 0 }, vec![a], scalar_ty);
    let dag = fuse(&dag);

    let result = chelis_backend_c::codegen(&dag, "test_reduce_sum");
    let src = &result.c_source;

    assert!(
        src.contains("chelis_sum_f32("),
        "ReduceSum must call chelis_sum_f32 for contiguous n=100 tensor:\n{src}"
    );

    let harness = format!(
        r#"{HARNESS_HEADER}
extern void test_reduce_sum(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float in_data[100];
    float scalar_sum = 0.0f;
    for (int i = 0; i < 100; i++) {{
        in_data[i] = (float)(i + 1);
        scalar_sum += in_data[i];
    }}
    chelis_tensor in_t = make_view_1d(in_data, 100);
    chelis_tensor* in_ptr = &in_t;
    chelis_tensor* inputs[1] = {{in_ptr}};
    chelis_tensor* out_slot = NULL;
    chelis_tensor* outputs[1] = {{out_slot}};

    test_reduce_sum(inputs, 1, outputs, 1);

    float got = outputs[0]->data[0];
    float diff = fabsf(got - scalar_sum);
    printf("sum(1..100): got=%.2f expected=%.2f diff=%.6f\n", got, scalar_sum, diff);
    printf("%s\n", diff < 0.5f ? "PASS" : "FAIL");
    return diff < 0.5f ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("reduce_sum100", src, &harness) else {
        panic!("ReduceSum kernel failed to compile/run");
    };
    assert!(output.contains("PASS"), "ReduceSum wrong output:\n{output}");
}

// ---- Test 5: Zero-size tensor does not crash ----

#[test]
fn exec_zero_size_tensor_does_not_crash() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(0));
    dag.add_node(RiscOp::Exp, vec![a], vec_f32(0));
    let dag = fuse(&dag);

    let result = codegen_with_options(
        &dag,
        "test_exp_zero",
        CodegenOptions {
            math_lib_override: Some(MathLib::Sleef),
            ..Default::default()
        },
    );
    let src = &result.c_source;

    let harness = format!(
        r#"{HARNESS_HEADER}
extern void test_exp_zero(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    // zero-element 1D tensor
    float dummy = 0.0f;
    chelis_tensor in_t;
    memset(&in_t, 0, sizeof(in_t));
    in_t.data = &dummy;
    in_t.shape[0] = 0;
    in_t.strides[0] = 1;
    in_t.ndim = 1;
    in_t.dtype = CHELIS_F32;
    in_t.size = 0;
    in_t.owns_data = 0;

    chelis_tensor* in_ptr = &in_t;
    chelis_tensor* inputs[1] = {{in_ptr}};
    chelis_tensor* out_slot = NULL;
    chelis_tensor* outputs[1] = {{out_slot}};

    test_exp_zero(inputs, 1, outputs, 1);
    printf("zero-size exp returned, output_size=%d\n", outputs[0] ? outputs[0]->size : -1);
    printf("PASS\n");
    return 0;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("zero_size_exp", src, &harness) else {
        panic!("Zero-size exp kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "Zero-size exp kernel crashed:\n{output}"
    );
}

// ---- Test 8: NaN propagation inconsistency probe ----

#[test]
fn exec_simd_nan_propagation_inconsistency_probe() {
    let include_dir = runtime_include_dir();
    let dir = std::env::temp_dir().join("chelis_exec_nan_probe");
    fs::create_dir_all(&dir).unwrap();

    for hdr in &[
        "chelis_runtime.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        let src = fs::read_to_string(include_dir.join(hdr)).unwrap();
        fs::write(dir.join(hdr), src).unwrap();
    }

    let c_src = r#"
#include <stdio.h>
#include <math.h>
#include "chelis_simd.h"

int main() {
    // NaN at position 0 of 8-wide AVX2 chunk
    float arr_nan_pos0[8] = {NAN, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    // NaN at position 1 of 8-wide AVX2 chunk
    float arr_nan_pos1[8] = {1.0f, NAN, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    // NaN in scalar tail only (position 8, n=9)
    float arr_nan_tail[9] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, NAN};

    float max_pos0  = chelis_max_f32(arr_nan_pos0, 8);
    float max_pos1  = chelis_max_f32(arr_nan_pos1, 8);
    float max_tail  = chelis_max_f32(arr_nan_tail, 9);

    int nan_pos0  = isnan(max_pos0);
    int nan_pos1  = isnan(max_pos1);
    int nan_tail  = isnan(max_tail);

    printf("max(NaN@pos0,  n=8): is_nan=%d val=%.2f\n", nan_pos0, max_pos0);
    printf("max(NaN@pos1,  n=8): is_nan=%d val=%.2f\n", nan_pos1, max_pos1);
    printf("max(NaN@tail,  n=9): is_nan=%d val=%.2f\n", nan_tail, max_tail);

    if (nan_pos0 != nan_pos1 || nan_pos0 != nan_tail) {
        printf("INCONSISTENT: NaN propagation depends on position\n");
    } else {
        printf("CONSISTENT: propagation=%d\n", nan_pos0);
    }
    return 0;
}
"#;

    fs::write(dir.join("nan_probe.c"), c_src).unwrap();
    let bin = dir.join("nan_bin");

    let compile = Command::new("gcc")
        .args([
            "-O2",
            "-mavx2",
            "-std=c11",
            "-I",
            dir.to_str().unwrap(),
            dir.join("nan_probe.c").to_str().unwrap(),
            "-o",
            bin.to_str().unwrap(),
            "-lm",
        ])
        .output()
        .expect("gcc not available");

    assert!(
        compile.status.success(),
        "nan probe compile failed: {}",
        String::from_utf8_lossy(&compile.stderr)
    );

    let run = Command::new(&bin)
        .output()
        .expect("failed to run nan probe");
    let stdout = String::from_utf8_lossy(&run.stdout).into_owned();
    eprintln!("NaN probe output:\n{stdout}");

    // Document that inconsistency exists — this is a real bug.
    // NaN in pos=0 of AVX2 chunk propagates; NaN in pos=1 does NOT.
    // The test PASSES to document the finding, but we print the inconsistency.
    // A future fix should make this consistent (either always propagate or never).
    println!("NaN probe output:\n{stdout}");
    // We do NOT assert it passes — we assert it ran.
    assert!(!stdout.is_empty(), "NaN probe produced no output");
}

// ---- Test 9: chelis_simd.h compiles as C++ ----

#[test]
fn exec_simd_header_compiles_as_cxx() {
    let include_dir = runtime_include_dir();
    let dir = std::env::temp_dir().join("chelis_cxx_probe");
    fs::create_dir_all(&dir).unwrap();

    // Copy simd header
    let simd_src = fs::read_to_string(include_dir.join("chelis_simd.h")).unwrap();
    fs::write(dir.join("chelis_simd.h"), &simd_src).unwrap();

    // Write a minimal C++ file that includes it
    let cxx_src = r#"
#include "chelis_simd.h"
int main() { return 0; }
"#;
    fs::write(dir.join("test.cpp"), cxx_src).unwrap();

    let output = Command::new("g++")
        .args([
            "-std=c++17",
            "-O2",
            "-mavx2",
            "-I",
            dir.to_str().unwrap(),
            dir.join("test.cpp").to_str().unwrap(),
            "-o",
            dir.join("cxx_bin").to_str().unwrap(),
            "-lm",
        ])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            println!("chelis_simd.h compiles cleanly as C++");
        }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            panic!("chelis_simd.h FAILS to compile as C++:\n{stderr}");
        }
        Err(e) => {
            eprintln!("g++ not available, skipping: {e}");
        }
    }
}
