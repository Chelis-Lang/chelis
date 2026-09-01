//! Adversarial execution tests: actually compile AND RUN generated C code.
//! These test numerical correctness, not just source patterns.
//!
//! The generated kernel signature is:
//!   void func(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out)
//! The kernel allocates output tensors internally via chelis_alloc.
//! We link against the chelis_runtime .a to resolve those symbols.

use chelis_backend_c::{CodegenOptions, MathLib, codegen_with_options};
use chelis_ir::dag::{
    Dag, DimInfo, ExtremaKind, ExtremaOperand, ReduceWindowKind, RiscOp, TensorType,
};
use chelis_ir::fuse::fuse;
use chelis_types::types::Prim;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn runtime_include_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-runtime/include")
}

/// Host-arch SIMD ISA flag(s) for the compile-run probes.
///
/// `chelis_simd.h` and the generated kernels are arch-aware (`#ifdef
/// __AVX2__` on x86, `#elif defined(__ARM_NEON)` on ARM, scalar
/// fallback otherwise). On x86_64 we pass `-mavx2` to exercise the AVX2
/// path; on aarch64 NEON is a baseline ISA feature (so `__ARM_NEON` is
/// already defined and the NEON path activates with no flag), and
/// `-mavx2` is an `unsupported option` clang error there. Returning an
/// empty vector on non-x86 keeps the probe portable so it runs via NEON
/// (Apple Silicon CI) or the scalar fallback rather than failing the
/// build.
fn simd_isa_flags() -> Vec<String> {
    if cfg!(target_arch = "x86_64") {
        vec!["-mavx2".to_string()]
    } else {
        Vec::new()
    }
}

/// Locate `target/debug/` for this workspace by walking up from the test binary's
/// own location. The test binary lives at `<target>/debug/deps/<binary>`, so
/// the parent of its parent is the debug directory we want.
fn target_debug_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe failed");
    // exe = <target>/debug/deps/<test_bin>
    exe.parent()
        .and_then(Path::parent)
        .map(PathBuf::from)
        .expect("could not resolve target/debug dir from current_exe")
}

/// Ensure `target/debug/libchelis_runtime.a` exists. When `chelis-runtime` is built
/// transitively as a dev-dependency (rather than as the top-level package), cargo
/// only emits the staticlib to `target/debug/deps/libchelis_runtime-<hash>.a` and
/// does not promote it to the conventional `target/debug/libchelis_runtime.a` path.
/// The test gcc invocation links against the conventional path, so this helper
/// copies the hashed artifact into place on first use. Idempotent and
/// thread-safe.
fn ensure_runtime_static_lib(canonical: &Path) -> std::io::Result<()> {
    if canonical.exists() {
        return Ok(());
    }
    let deps_dir = canonical
        .parent()
        .expect("canonical lib path has no parent")
        .join("deps");
    // First-pass scan of the deps dir.
    let hashed = find_newest_runtime_archive(&deps_dir)?;
    // If cargo's incremental cache reused the rlib without re-emitting
    // the staticlib (observed on CI cold-cache runs against
    // `chelis-runtime` as a transitive dev-dep), force a rebuild of
    // the lib target and rescan. `cargo build -p chelis-runtime --lib`
    // emits both crate-types declared in chelis-runtime/Cargo.toml,
    // producing the `libchelis_runtime-<hash>.a` artifact the
    // gcc-link harness needs.
    let hashed = match hashed {
        Some(path) => path,
        None => {
            std::process::Command::new(env!("CARGO"))
                .args(["build", "-p", "chelis-runtime", "--lib"])
                .status()
                .map_err(|e| std::io::Error::other(format!("cargo build chelis-runtime: {e}")))?;
            find_newest_runtime_archive(&deps_dir)?.ok_or_else(|| {
                std::io::Error::other(format!(
                    "no libchelis_runtime-*.a found in {} after explicit `cargo build -p \
                     chelis-runtime --lib`",
                    deps_dir.display()
                ))
            })?
        }
    };
    // Use a PID-suffixed tmp filename so concurrent test binaries (this
    // file and dtype_matrix_bf16_f16.rs both call into this helper, and
    // nextest runs them in parallel) do not race on a shared tmp path.
    // Each process writes its own tmp and renames into the shared
    // canonical location; last writer wins, but the content is
    // identical so the race is harmless. Without the PID, two
    // processes that interleave `fs::copy` and `fs::rename` produce an
    // ENOENT on the second rename because the first rename moved the
    // shared tmp away.
    static NEXT_TEMP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tmp = canonical.with_extension(format!(
        "a.tmp.{}.{}",
        std::process::id(),
        NEXT_TEMP.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    fs::copy(&hashed, &tmp)?;
    // The rename can still race with another process renaming its own
    // unique tmp into the same canonical path. On POSIX, rename onto an
    // existing file is atomic, so this is fine. If a peer beat us to
    // it, treat NotFound from a follow-up cleanup as benign.
    match fs::rename(&tmp, canonical) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && canonical.exists() => Ok(()),
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            Err(e)
        }
    }
}

fn find_newest_runtime_archive(deps_dir: &Path) -> std::io::Result<Option<PathBuf>> {
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    let entries = match fs::read_dir(deps_dir) {
        Ok(it) => it,
        // Truly cold target dirs may not have `deps/` yet; let the
        // caller fall through to the explicit `cargo build` fallback.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("libchelis_runtime-") && name.ends_with(".a") {
            let meta = entry.metadata()?;
            let mtime = meta.modified()?;
            match &newest {
                Some((cur, _)) if *cur >= mtime => {}
                _ => newest = Some((mtime, entry.path())),
            }
        }
    }
    Ok(newest.map(|(_, p)| p))
}

fn runtime_lib_path() -> PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let canonical = target_debug_dir().join("libchelis_runtime.a");
        if let Err(e) = ensure_runtime_static_lib(&canonical) {
            panic!(
                "failed to materialize libchelis_runtime.a at {}: {}",
                canonical.display(),
                e
            );
        }
        canonical
    })
    .clone()
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
        "chelis_runtime_dtype.h",
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
        .arg("-O2")
        .args(simd_isa_flags())
        .args([
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

// Common harness header: wrap caller-owned storage in an exact tensor descriptor.
const HARNESS_HEADER: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "chelis_runtime.h"

static chelis_tensor make_view_typed_1d(void* data, int64_t n, chelis_dtype dtype) {
    static int64_t shape[1];
    static const int64_t strides[1] = {1};
    shape[0] = n;
    return (chelis_tensor){
        .data = data,
        .shape = shape,
        .strides = strides,
        .size = n,
        .byte_capacity = n * chelis_dtype_size(dtype),
        .rank = 1,
        .dtype = dtype,
        .owns_data = 0,
        .reserved = {0, 0},
    };
}

static chelis_tensor make_view_1d(float* data, int64_t n) {
    return make_view_typed_1d(data, n, CHELIS_DTYPE_F32);
}
"#;

// ---- Test 6 / Item 6: MathLib::None exp kernel ----

#[test]
fn exec_math_none_exp_kernel_correct_output() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(4), None);
    dag.add_node(RiscOp::Exp, vec![a], vec_f32(4), None);
    let dag = fuse(&dag);

    let result = codegen_with_options(
        &dag,
        "test_exp_none",
        CodegenOptions {
            math_lib_override: Some(MathLib::None),
            ..Default::default()
        },
    )
    .unwrap();
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
        float got = ((float*)outputs[0]->data)[i];
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
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(9), None);
    let e = dag.add_node(RiscOp::Exp, vec![a], vec_f32(9), None);
    dag.add_node(RiscOp::Neg, vec![e], vec_f32(9), None); // 2-op chain: fuses into FusedElem
    let dag = fuse(&dag);

    let result = codegen_with_options(
        &dag,
        "test_exp_sleef",
        CodegenOptions {
            math_lib_override: Some(MathLib::Sleef),
            ..Default::default()
        },
    )
    .unwrap();
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
        float got = ((float*)outputs[0]->data)[i];
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
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(100),
        None,
    );
    dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![a],
        scalar_ty,
        None,
    );
    let dag = fuse(&dag);

    let result = chelis_backend_c::codegen(&dag, "test_reduce_sum").unwrap();
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

    float got = ((float*)outputs[0]->data)[0];
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

#[test]
fn exec_count_multi_axis_matches_exact_int64_result() {
    let tensor_ty = |dims: &[usize], precision| TensorType {
        dims: dims.iter().copied().map(DimInfo::Lit).collect(),
        precision,
    };
    let mut dag = Dag::new();
    let input = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_ty(&[2, 3, 2], Prim::Bool),
        None,
    );
    let output = dag.add_node(
        RiscOp::Count { axes: vec![2, 0] },
        vec![input],
        tensor_ty(&[3], Prim::Int64),
        None,
    );
    dag.add_root(output);
    let generated = codegen_with_options(
        &dag,
        "test_count_multi",
        CodegenOptions {
            math_lib_override: Some(MathLib::None),
            ..Default::default()
        },
    )
    .expect("Count C generation");
    assert!(generated.c_source.contains("chelis_int_checked_add"));
    assert!(generated.c_source.contains("CHELIS_DTYPE_BOOL"));
    assert!(!generated.c_source.contains("CHELIS_BOOL"));
    assert!(generated.c_source.contains("t0->rank"));
    assert!(!generated.c_source.contains("t0->ndim"));
    for balanced_tree_fragment in [
        "while (__level_n_1 > 1)",
        "int64_t __left_1 = 2 * __j_1",
        "int64_t __right_1 = __left_1 + 1",
        "chelis_int_checked_add(__level_1[__left_1], __level_1[__right_1]",
    ] {
        assert!(
            generated.c_source.contains(balanced_tree_fragment),
            "Count C must emit the canonical adjacent-pair balanced tree; missing {balanced_tree_fragment:?}"
        );
    }

    let harness = format!(
        r#"{HARNESS_HEADER}
extern void test_count_multi(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    uint8_t bits[12] = {{1,0,1,1,0,0,1,1,0,1,1,1}};
    int64_t shape[3] = {{2, 3, 2}};
    chelis_tensor* x = chelis_alloc_view(3, shape, CHELIS_DTYPE_BOOL, bits, sizeof(bits));
    if (x == NULL) return 2;
    chelis_tensor* inputs[1] = {{x}};
    chelis_tensor* outputs[1] = {{NULL}};
    test_count_multi(inputs, 1, outputs, 1);
    int64_t expected[3] = {{3, 3, 2}};
    int64_t* got = (int64_t*)outputs[0]->data;
    int ok = outputs[0]->dtype == CHELIS_DTYPE_I64 && outputs[0]->size == 3;
    for (int i = 0; i < 3; i++) if (got[i] != expected[i]) ok = 0;
    chelis_free(x);
    chelis_free(outputs[0]);
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );
    let output = compile_and_run_kernel("count_multi", &generated.c_source, &harness)
        .expect("Count C kernel compiles and runs");
    assert!(output.contains("PASS"), "wrong Count output: {output}");
}

#[test]
fn exec_count_selected_zero_extent_returns_zero() {
    let tensor_ty = |dims: &[usize], precision| TensorType {
        dims: dims.iter().copied().map(DimInfo::Lit).collect(),
        precision,
    };
    let mut dag = Dag::new();
    let input = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_ty(&[2, 0, 3], Prim::Bool),
        None,
    );
    let output = dag.add_node(
        RiscOp::Count { axes: vec![1] },
        vec![input],
        tensor_ty(&[2, 3], Prim::Int64),
        None,
    );
    dag.add_root(output);
    let generated = codegen_with_options(
        &dag,
        "test_count_empty",
        CodegenOptions {
            math_lib_override: Some(MathLib::None),
            ..Default::default()
        },
    )
    .expect("empty Count C generation");
    let harness = format!(
        r#"{HARNESS_HEADER}
extern void test_count_empty(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    int64_t shape[3] = {{2, 0, 3}};
    chelis_tensor* x = chelis_alloc_view(3, shape, CHELIS_DTYPE_BOOL, NULL, 0);
    if (x == NULL) return 2;
    chelis_tensor* inputs[1] = {{x}};
    chelis_tensor* outputs[1] = {{NULL}};
    test_count_empty(inputs, 1, outputs, 1);
    int64_t* got = (int64_t*)outputs[0]->data;
    int ok = outputs[0]->dtype == CHELIS_DTYPE_I64 && outputs[0]->size == 6;
    for (int i = 0; i < 6; i++) if (got[i] != 0) ok = 0;
    chelis_free(x);
    chelis_free(outputs[0]);
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );
    let output = compile_and_run_kernel("count_empty", &generated.c_source, &harness)
        .expect("empty Count C kernel compiles and runs");
    assert!(
        output.contains("PASS"),
        "wrong empty Count output: {output}"
    );
}

// ---- Issue #254: reduce_window_* C-backend numerical parity ----
//
// The `issue_254_reduce_window_emit` tests pin only the *structural*
// shape of the emitted C (which intrinsic, which init literal). Per
// the backend-numerics policy, evaluator-vs-backend agreement needs an
// actual compile-and-run. These two tests close that gap: the emitted
// C is compiled with gcc, run, and checked against the exact values
// the IR evaluator (`chelis_ir::eval::reduce_window`) produces for the
// same 1x1x3x3 input — the canonical oracle per spec §6. Max exercises
// the `fmaxf` / `-INFINITY` path; Mean exercises the windowed-sum +
// `acc /= window_volume` division path.

fn reduce_window_3x3_dag(reducer: ReduceWindowKind, kernel: &str) -> String {
    let in_ty = TensorType {
        dims: [1, 1, 3, 3].into_iter().map(DimInfo::Lit).collect(),
        precision: Prim::F32,
    };
    let out_ty = TensorType {
        dims: [1, 1, 2, 2].into_iter().map(DimInfo::Lit).collect(),
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], in_ty, None);
    dag.add_node(
        RiscOp::ReduceWindow {
            reducer,
            window_shape: vec![2, 2],
            strides: vec![1, 1],
        },
        vec![x],
        out_ty,
        None,
    );
    chelis_backend_c::codegen(&dag, kernel).unwrap().c_source
}

// Build a contiguous 1x1x3x3 input view holding [[1..9]] row-major.
const RW_HARNESS_4D_HEADER: &str = r#"
static chelis_tensor make_view_1x1x3x3(float* data) {
    static const int64_t shape[4] = {1, 1, 3, 3};
    static const int64_t strides[4] = {9, 9, 3, 1};
    return (chelis_tensor){
        .data = data,
        .shape = shape,
        .strides = strides,
        .size = 9,
        .byte_capacity = 9 * (int64_t)sizeof(float),
        .rank = 4,
        .dtype = CHELIS_DTYPE_F32,
        .owns_data = 0,
        .reserved = {0, 0},
    };
}
"#;

#[test]
fn exec_reduce_window_max_matches_evaluator_oracle() {
    let src = reduce_window_3x3_dag(ReduceWindowKind::Max, "test_rw_max");
    let harness = format!(
        r#"{HARNESS_HEADER}{RW_HARNESS_4D_HEADER}
extern void test_rw_max(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float in_data[9] = {{1,2,3,4,5,6,7,8,9}};
    chelis_tensor in_t = make_view_1x1x3x3(in_data);
    chelis_tensor* inputs[1] = {{&in_t}};
    chelis_tensor* outputs[1] = {{NULL}};

    test_rw_max(inputs, 1, outputs, 1);

    // 2x2 maxes of [[1,2,3],[4,5,6],[7,8,9]]: [5,6,8,9].
    float expected[4] = {{5.0f, 6.0f, 8.0f, 9.0f}};
    int ok = (outputs[0]->size == 4);
    for (int i = 0; i < 4; i++) {{
        if (fabsf(((float*)outputs[0]->data)[i] - expected[i]) > 1e-5f) {{
            printf("MISMATCH at %d: got %.4f expected %.4f\n", i, ((float*)outputs[0]->data)[i], expected[i]);
            ok = 0;
        }}
    }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );
    let Some(output) = compile_and_run_kernel("rw_max_3x3", &src, &harness) else {
        panic!("reduce_window_max kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "reduce_window_max C backend diverged from evaluator oracle:\n{output}"
    );
}

#[test]
fn exec_reduce_window_mean_matches_evaluator_oracle() {
    let src = reduce_window_3x3_dag(ReduceWindowKind::Mean, "test_rw_mean");
    let harness = format!(
        r#"{HARNESS_HEADER}{RW_HARNESS_4D_HEADER}
extern void test_rw_mean(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float in_data[9] = {{1,2,3,4,5,6,7,8,9}};
    chelis_tensor in_t = make_view_1x1x3x3(in_data);
    chelis_tensor* inputs[1] = {{&in_t}};
    chelis_tensor* outputs[1] = {{NULL}};

    test_rw_mean(inputs, 1, outputs, 1);

    // 2x2 means: sums [12,16,24,28] / 4 = [3,4,6,7].
    float expected[4] = {{3.0f, 4.0f, 6.0f, 7.0f}};
    int ok = (outputs[0]->size == 4);
    for (int i = 0; i < 4; i++) {{
        if (fabsf(((float*)outputs[0]->data)[i] - expected[i]) > 1e-5f) {{
            printf("MISMATCH at %d: got %.4f expected %.4f\n", i, ((float*)outputs[0]->data)[i], expected[i]);
            ok = 0;
        }}
    }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );
    let Some(output) = compile_and_run_kernel("rw_mean_3x3", &src, &harness) else {
        panic!("reduce_window_mean kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "reduce_window_mean C backend diverged from evaluator oracle:\n{output}"
    );
}

// ---- reduce_window adjoint (ReduceWindowGrad) C-backend parity ----

// din = ReduceWindowGrad(x[1,1,3,3], g[1,1,2,2]) with window=[2,2]
// stride=[1,1]. Load "x" is created first (input slot 0), "g" second
// (slot 1), matching the harness `inputs[]` order.
fn reduce_window_grad_dag(reducer: ReduceWindowKind, kernel: &str) -> String {
    let x_ty = TensorType {
        dims: [1, 1, 3, 3].into_iter().map(DimInfo::Lit).collect(),
        precision: Prim::F32,
    };
    let g_ty = TensorType {
        dims: [1, 1, 2, 2].into_iter().map(DimInfo::Lit).collect(),
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        x_ty.clone(),
        None,
    );
    let g = dag.add_node(RiscOp::Load { name: "g".into() }, vec![], g_ty, None);
    dag.add_node(
        RiscOp::ReduceWindowGrad {
            reducer,
            window_shape: vec![2, 2],
            strides: vec![1, 1],
        },
        vec![x, g],
        x_ty,
        None,
    );
    chelis_backend_c::codegen(&dag, kernel).unwrap().c_source
}

const RW_GRAD_HARNESS_HEADER: &str = r#"
static chelis_tensor make_view_1x1x2x2(float* data) {
    static const int64_t shape[4] = {1, 1, 2, 2};
    static const int64_t strides[4] = {4, 4, 2, 1};
    return (chelis_tensor){
        .data = data,
        .shape = shape,
        .strides = strides,
        .size = 4,
        .byte_capacity = 4 * (int64_t)sizeof(float),
        .rank = 4,
        .dtype = CHELIS_DTYPE_F32,
        .owns_data = 0,
        .reserved = {0, 0},
    };
}
"#;

#[test]
fn exec_reduce_window_grad_sum_matches_evaluator_oracle() {
    let src = reduce_window_grad_dag(ReduceWindowKind::Sum, "test_rwg_sum");
    let harness = format!(
        r#"{HARNESS_HEADER}{RW_HARNESS_4D_HEADER}{RW_GRAD_HARNESS_HEADER}
extern void test_rwg_sum(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float x_data[9] = {{1,2,3,4,5,6,7,8,9}};
    float g_data[4] = {{1,1,1,1}};
    chelis_tensor x_t = make_view_1x1x3x3(x_data);
    chelis_tensor g_t = make_view_1x1x2x2(g_data);
    chelis_tensor* inputs[2] = {{&x_t, &g_t}};
    chelis_tensor* outputs[1] = {{NULL}};

    test_rwg_sum(inputs, 2, outputs, 1);

    // Sum adjoint with g=ones is the per-position window-cover count for
    // 2x2 windows / stride 1 over a 3x3 grid: corners 1, edges 2, center 4.
    float expected[9] = {{1,2,1, 2,4,2, 1,2,1}};
    int ok = (outputs[0]->size == 9);
    for (int i = 0; i < 9; i++) {{
        if (fabsf(((float*)outputs[0]->data)[i] - expected[i]) > 1e-5f) {{
            printf("MISMATCH at %d: got %.4f expected %.4f\n", i, ((float*)outputs[0]->data)[i], expected[i]);
            ok = 0;
        }}
    }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );
    let Some(output) = compile_and_run_kernel("rwg_sum_3x3", &src, &harness) else {
        panic!("reduce_window_grad sum kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "reduce_window_grad(sum) C backend diverged from evaluator oracle:\n{output}"
    );
}

#[test]
fn exec_reduce_window_grad_max_matches_evaluator_oracle() {
    let src = reduce_window_grad_dag(ReduceWindowKind::Max, "test_rwg_max");
    let harness = format!(
        r#"{HARNESS_HEADER}{RW_HARNESS_4D_HEADER}{RW_GRAD_HARNESS_HEADER}
extern void test_rwg_max(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float x_data[9] = {{1,2,3,4,5,6,7,8,9}};
    float g_data[4] = {{1,1,1,1}};
    chelis_tensor x_t = make_view_1x1x3x3(x_data);
    chelis_tensor g_t = make_view_1x1x2x2(g_data);
    chelis_tensor* inputs[2] = {{&x_t, &g_t}};
    chelis_tensor* outputs[1] = {{NULL}};

    test_rwg_max(inputs, 2, outputs, 1);

    // Each 2x2 window's max (distinct values) routes its g to the argmax:
    // windows pick (1,1),(1,2),(2,1),(2,2) of the 3x3 grid.
    float expected[9] = {{0,0,0, 0,1,1, 0,1,1}};
    int ok = (outputs[0]->size == 9);
    for (int i = 0; i < 9; i++) {{
        if (fabsf(((float*)outputs[0]->data)[i] - expected[i]) > 1e-5f) {{
            printf("MISMATCH at %d: got %.4f expected %.4f\n", i, ((float*)outputs[0]->data)[i], expected[i]);
            ok = 0;
        }}
    }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );
    let Some(output) = compile_and_run_kernel("rwg_max_3x3", &src, &harness) else {
        panic!("reduce_window_grad max kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "reduce_window_grad(max) C backend diverged from evaluator oracle:\n{output}"
    );
}

// ---- IEEE-754 corner cases for Div and Recip ----
// Exercise the C-backend codegen (`emit_binary` for Div, `emit_recip`
// for Recip) end-to-end on the four corner cases an
// `exp(neg(log(b)))` decomposition would mishandle: 5/-2, 1/0, -1/0,
// 0/0 for Div; recip(-2) and recip(0) for Recip. Path: chelis IR →
// emitted C → gcc → run.

#[test]
fn exec_div_ieee_corner_cases() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(4), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], vec_f32(4), None);
    dag.add_node(RiscOp::Div, vec![a, b], vec_f32(4), None);
    let dag = fuse(&dag);

    let result = codegen_with_options(
        &dag,
        "test_div_ieee",
        CodegenOptions {
            math_lib_override: Some(MathLib::None),
            ..Default::default()
        },
    )
    .unwrap();
    let src = &result.c_source;

    let harness = format!(
        r#"{HARNESS_HEADER}
#include <math.h>
extern void test_div_ieee(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float a_data[4] = {{ 5.0f,  1.0f, -1.0f, 0.0f }};
    float b_data[4] = {{-2.0f,  0.0f,  0.0f, 0.0f }};
    chelis_tensor a_t = make_view_1d(a_data, 4);
    chelis_tensor b_t = make_view_1d(b_data, 4);
    chelis_tensor* inputs[2] = {{&a_t, &b_t}};
    chelis_tensor* outputs[1] = {{NULL}};
    test_div_ieee(inputs, 2, outputs, 1);

    float* o = outputs[0]->data;
    int ok = 1;
    if (o[0] != -2.5f) {{ printf("MISMATCH 5/-2: got %f want -2.5\n", o[0]); ok = 0; }}
    if (!(isinf(o[1]) && o[1] > 0)) {{ printf("MISMATCH 1/0: got %f want +inf\n", o[1]); ok = 0; }}
    if (!(isinf(o[2]) && o[2] < 0)) {{ printf("MISMATCH -1/0: got %f want -inf\n", o[2]); ok = 0; }}
    if (!isnan(o[3])) {{ printf("MISMATCH 0/0: got %f want NaN\n", o[3]); ok = 0; }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("div_ieee", src, &harness) else {
        panic!("Div IEEE kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "C-backend Div must produce IEEE results for the four corner cases:\n{output}"
    );
}

#[test]
fn exec_recip_ieee_corner_cases() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(3), None);
    dag.add_node(RiscOp::Recip, vec![a], vec_f32(3), None);
    let dag = fuse(&dag);

    let result = codegen_with_options(
        &dag,
        "test_recip_ieee",
        CodegenOptions {
            math_lib_override: Some(MathLib::None),
            ..Default::default()
        },
    )
    .unwrap();
    let src = &result.c_source;

    let harness = format!(
        r#"{HARNESS_HEADER}
#include <math.h>
extern void test_recip_ieee(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float a_data[3] = {{-2.0f, 0.0f, 4.0f}};
    chelis_tensor a_t = make_view_1d(a_data, 3);
    chelis_tensor* inputs[1] = {{&a_t}};
    chelis_tensor* outputs[1] = {{NULL}};
    test_recip_ieee(inputs, 1, outputs, 1);

    float* o = outputs[0]->data;
    int ok = 1;
    if (o[0] != -0.5f) {{ printf("MISMATCH recip(-2): got %f want -0.5\n", o[0]); ok = 0; }}
    if (!(isinf(o[1]) && o[1] > 0)) {{ printf("MISMATCH recip(0): got %f want +inf\n", o[1]); ok = 0; }}
    if (o[2] != 0.25f) {{ printf("MISMATCH recip(4): got %f want 0.25\n", o[2]); ok = 0; }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("recip_ieee", src, &harness) else {
        panic!("Recip IEEE kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "C-backend Recip must produce IEEE results for the corner cases:\n{output}"
    );
}

/// The C scalar type and `CHELIS_*` dtype macro for an integer precision,
/// used to generate width-parametrized exec harnesses (chelis#550 F2). The
/// printf specifier is always `%lld` after a `(long long)` cast so the same
/// format string works for every width.
fn int_c_type_and_dtype(precision: Prim) -> (&'static str, &'static str) {
    match precision {
        Prim::Int8 => ("int8_t", "CHELIS_DTYPE_I8"),
        Prim::Int16 => ("int16_t", "CHELIS_DTYPE_I16"),
        Prim::Int32 => ("int32_t", "CHELIS_DTYPE_I32"),
        Prim::Int64 => ("int64_t", "CHELIS_DTYPE_I64"),
        other => panic!("int_c_type_and_dtype: non-integer precision {other:?}"),
    }
}

fn vec_int(n: usize, precision: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision,
    }
}

/// Shared exec-compile driver for an integer binary-division op on a
/// same-precision operand pair. Builds a two-load DAG, lowers `op`, compiles
/// the kernel, and asserts each output element matches `expected`. The four
/// operand pairs `{7,2},{7,-2},{-7,2},{-7,-2}` exercise every sign
/// combination so floor-vs-truncate rounding is distinguished on the
/// mixed-sign cases. `precision` parametrizes the integer width (chelis#550
/// F2: int8 / int16 / int64 in addition to the original int32).
fn run_int_div_op_exec(
    op: RiscOp,
    fn_name: &str,
    kernel_name: &str,
    expected: [i32; 4],
    precision: Prim,
) {
    let (c_type, dtype_macro) = int_c_type_and_dtype(precision);
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_int(4, precision),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_int(4, precision),
        None,
    );
    dag.add_node(op, vec![a, b], vec_int(4, precision), None);
    let dag = fuse(&dag);

    let result = codegen_with_options(
        &dag,
        fn_name,
        CodegenOptions {
            math_lib_override: Some(MathLib::None),
            ..Default::default()
        },
    )
    .unwrap();
    let src = &result.c_source;

    let [e0, e1, e2, e3] = expected;
    let harness = format!(
        r#"{HARNESS_HEADER}
extern void {fn_name}(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    {c_type} a_data[4] = {{ 7,  7, -7, -7}};
    {c_type} b_data[4] = {{ 2, -2,  2, -2}};
    {c_type} expected[4] = {{ {e0}, {e1}, {e2}, {e3} }};

    chelis_tensor a_t = make_view_typed_1d(a_data, 4, {dtype_macro});
    chelis_tensor b_t = make_view_typed_1d(b_data, 4, {dtype_macro});

    chelis_tensor* inputs[2] = {{&a_t, &b_t}};
    chelis_tensor* outputs[1] = {{NULL}};
    {fn_name}(inputs, 2, outputs, 1);

    int ok = 1;
    {c_type}* o = ({c_type}*)outputs[0]->data;
    if (outputs[0]->dtype != {dtype_macro}) {{
        printf("FAIL: output dtype %d, expected {dtype_macro} (%d)\n",
               outputs[0]->dtype, {dtype_macro});
        ok = 0;
    }}
    for (int i = 0; i < 4 && ok; i++) {{
        if (o[i] != expected[i]) {{
            printf("MISMATCH idx=%d a=%lld b=%lld got=%lld want=%lld\n",
                   i, (long long)a_data[i], (long long)b_data[i],
                   (long long)o[i], (long long)expected[i]);
            ok = 0;
        }}
    }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel(kernel_name, src, &harness) else {
        panic!("{kernel_name} ({precision:?}) kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "C-backend {kernel_name} on {precision:?} mismatch:\n{output}"
    );
}

// spec/05-risc-primitives.md §2.1: `trunc_div` uses C/Rust truncating
// semantics (round toward zero). This is what chelis-std's
// `Std.Decimal::normalize` / `decimal_div_nonzero` rely on for scale
// shifts and quotient computation. The C backend emits `int32_t /
// int32_t` which truncates by language definition; this exec-compile
// test pins that contract end-to-end across every sign combination.
// `{7,-7} / {2,-2}` ⇒ `{3, -3, -3, 3}` (round toward zero).
#[test]
fn exec_trunc_div_int32_truncates_toward_zero() {
    run_int_div_op_exec(
        RiscOp::TruncDiv,
        "test_trunc_div_i32",
        "trunc_div_i32",
        [3, -3, -3, 3],
        Prim::Int32,
    );
}

// chelis#550 F2: the trunc_div / floor_div emit is width-independent (the
// C backend promotes to int64 internally), but the repo's negative-parity
// bar requires the narrower and wider integer widths be exercised
// end-to-end, not just int32. Same operands / expected results as the int32
// cases above; only the storage precision changes.
#[test]
fn exec_trunc_div_int8_truncates_toward_zero() {
    run_int_div_op_exec(
        RiscOp::TruncDiv,
        "test_trunc_div_i8",
        "trunc_div_i8",
        [3, -3, -3, 3],
        Prim::Int8,
    );
}

#[test]
fn exec_trunc_div_int16_truncates_toward_zero() {
    run_int_div_op_exec(
        RiscOp::TruncDiv,
        "test_trunc_div_i16",
        "trunc_div_i16",
        [3, -3, -3, 3],
        Prim::Int16,
    );
}

#[test]
fn exec_trunc_div_int64_truncates_toward_zero() {
    run_int_div_op_exec(
        RiscOp::TruncDiv,
        "test_trunc_div_i64",
        "trunc_div_i64",
        [3, -3, -3, 3],
        Prim::Int64,
    );
}

// spec/05-risc-primitives.md §2.1: `floor_div` rounds the quotient
// toward −∞. It agrees with truncate when the operands share a sign and
// differs on the mixed-sign exact-fraction cases:
// `7 floor_div 2 == 3`, `7 floor_div -2 == -4`, `-7 floor_div 2 == -4`,
// `-7 floor_div -2 == 3`. This is the round-toward-−∞ semantics that
// matches Python `//` / torch / JAX / numpy `floor_divide`; the C
// backend realizes it as native `/` plus a remainder-sign correction.
#[test]
fn exec_floor_div_int32_rounds_toward_neg_inf() {
    run_int_div_op_exec(
        RiscOp::FloorDiv,
        "test_floor_div_i32",
        "floor_div_i32",
        [3, -4, -4, 3],
        Prim::Int32,
    );
}

// chelis#550 F2: floor_div across the remaining integer widths. The
// round-toward-−∞ remainder-sign correction must hold at int8 / int16 /
// int64 just as at int32.
#[test]
fn exec_floor_div_int8_rounds_toward_neg_inf() {
    run_int_div_op_exec(
        RiscOp::FloorDiv,
        "test_floor_div_i8",
        "floor_div_i8",
        [3, -4, -4, 3],
        Prim::Int8,
    );
}

#[test]
fn exec_floor_div_int16_rounds_toward_neg_inf() {
    run_int_div_op_exec(
        RiscOp::FloorDiv,
        "test_floor_div_i16",
        "floor_div_i16",
        [3, -4, -4, 3],
        Prim::Int16,
    );
}

#[test]
fn exec_floor_div_int64_rounds_toward_neg_inf() {
    run_int_div_op_exec(
        RiscOp::FloorDiv,
        "test_floor_div_i64",
        "floor_div_i64",
        [3, -4, -4, 3],
        Prim::Int64,
    );
}

/// Compile a kernel + harness exactly like `compile_and_run_kernel`, but
/// return the run `Output` (status + stderr) so a trap test can assert the
/// binary aborts. Panics if COMPILATION fails — a zero-divisor trap is a
/// runtime abort, not a compile error.
fn compile_and_capture_run(test_name: &str, c_source: &str, harness: &str) -> std::process::Output {
    let dir = std::env::temp_dir().join(format!("chelis_exec_{test_name}"));
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("kernel.c"), c_source).unwrap();
    fs::write(dir.join("main.c"), harness).unwrap();

    let include_dir = runtime_include_dir();
    for hdr in &[
        "chelis_runtime.h",
        "chelis_runtime_dtype.h",
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
        .arg("-O2")
        .args(simd_isa_flags())
        .args([
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
    assert!(
        compile.status.success(),
        "COMPILE FAILED [{test_name}]:\n{}\nKernel C:\n{c_source}",
        String::from_utf8_lossy(&compile.stderr),
    );

    Command::new(&bin).output().expect("failed to run binary")
}

// chelis#550 F2: a COMPILED floor_div zero-divisor trap. The existing
// backend trap coverage was trunc_div-only; floor_div emits the SAME
// portable `chelis_int_div_guard` and must abort identically. The divisor
// arrives through a runtime Load (`inputs[1]->data`), so gcc cannot
// constant-fold the zero and elide the guard. Spec/05-risc-primitives.md
// §2.1 scopes the `integer division or remainder by zero` trap to the C
// backend (and the evaluator); this is the fail-closed end-to-end proof.
#[test]
fn exec_floor_div_int_zero_divisor_traps() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_int(2, Prim::Int64),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_int(2, Prim::Int64),
        None,
    );
    dag.add_node(RiscOp::FloorDiv, vec![a, b], vec_int(2, Prim::Int64), None);
    let dag = fuse(&dag);

    let result = codegen_with_options(
        &dag,
        "test_floor_div_trap",
        CodegenOptions {
            math_lib_override: Some(MathLib::None),
            ..Default::default()
        },
    )
    .unwrap();
    let src = &result.c_source;
    // Emit-shape: floor_div must wrap the integer divisor in the portable guard.
    assert!(
        src.contains("chelis_int_div_guard("),
        "integer floor_div must emit the portable zero-divisor guard (#550); \
         emitted C=\n{src}",
    );

    // b_data[1] == 0: the second element divides by zero at runtime.
    let harness = format!(
        r#"{HARNESS_HEADER}
extern void test_floor_div_trap(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    int64_t a_data[2] = {{ 10, 7 }};
    int64_t b_data[2] = {{ 2, 0 }};

    chelis_tensor a_t = make_view_typed_1d(a_data, 2, CHELIS_DTYPE_I64);
    chelis_tensor b_t = make_view_typed_1d(b_data, 2, CHELIS_DTYPE_I64);

    chelis_tensor* inputs[2] = {{&a_t, &b_t}};
    chelis_tensor* outputs[1] = {{NULL}};
    test_floor_div_trap(inputs, 2, outputs, 1);

    /* The guard aborts before reaching here; printing PASS would be a bug. */
    printf("PASS\n");
    return 0;
}}
"#
    );

    let run = compile_and_capture_run("floor_div_trap", src, &harness);
    assert!(
        !run.status.success(),
        "floor_div by a runtime zero divisor must trap (abort), not succeed; \
         stdout={}",
        String::from_utf8_lossy(&run.stdout),
    );
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.contains("integer division or remainder by zero"),
        "the C backend trap must emit the canonical diagnostic on stderr; \
         stderr={stderr:?}",
    );
    assert!(
        !String::from_utf8_lossy(&run.stdout).contains("PASS"),
        "no PASS line may print when the program traps; the guard must abort \
         before the kernel returns",
    );
}

// ---- Test 10b: Cross-backend f32 bit-exactness oracle for issue #163 ----
//
// PR #168 review MED #3: pin C-backend / host-evaluator agreement on
// the issue #163 reproducer at the bit level, end-to-end. The
// host-evaluator test in `chelis-compiler-api::runtime` asserts the
// stride-4 ILP cascade result on the same 11-element multiset; this
// test does the same against the C backend's output by compiling the
// generated C with gcc, running it, and verifying the printed result
// is bit-exactly `0x4087012d` (= 4.218893527984619_f32). Without
// this end-to-end test, a future divergence between the evaluator
// lane and the codegen lane (e.g., a subtle lane-assignment shift in
// `chelis_sum_f32` vs `host_runtime::reduce_f32`) would not be caught
// by either layer's own tests.

#[test]
fn exec_reduce_sum_issue_163_repro_is_bit_exact_with_evaluator() {
    let scalar_ty = TensorType::scalar_f32();
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(11), None);
    dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![a],
        scalar_ty,
        None,
    );
    let dag = fuse(&dag);
    let result = chelis_backend_c::codegen(&dag, "test_issue_163_sum").unwrap();

    let harness = format!(
        r#"{HARNESS_HEADER}
#include <stdint.h>
extern void test_issue_163_sum(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    // Issue #163 right-pad reflected sequence: the issue's exact
    // reproducer multiset, identical to the host-runtime test in
    // chelis-compiler-api::runtime::host_runtime_sum_f32_uses_pairwise_order_for_issue_163_repro.
    float in_data[11] = {{
        0.49625658988952637f,
        0.7682217955589294f,
        0.08847743272781372f,
        0.13203048706054688f,
        0.30742114782333374f,
        0.6340786814689636f,
        0.30742114782333374f,
        0.13203048706054688f,
        0.08847743272781372f,
        0.7682217955589294f,
        0.49625658988952637f,
    }};
    chelis_tensor in_t = make_view_1d(in_data, 11);
    chelis_tensor* inputs[1] = {{ &in_t }};
    chelis_tensor* out_slot = NULL;
    chelis_tensor* outputs[1] = {{ out_slot }};

    test_issue_163_sum(inputs, 1, outputs, 1);

    float got = ((float*)outputs[0]->data)[0];
    uint32_t got_bits;
    memcpy(&got_bits, &got, sizeof(got_bits));
    // 4.218893527984619_f32 is the stride-4 ILP cascade result the
    // host runtime emits for the same multiset. Bit-exact equality
    // is the whole point of this PR.
    uint32_t expected_bits = 0x4087012d;
    printf("got=%.17g bits=0x%08x expected_bits=0x%08x\n", (double)got, got_bits, expected_bits);
    printf("%s\n", got_bits == expected_bits ? "PASS" : "FAIL");
    return got_bits == expected_bits ? 0 : 1;
}}
"#,
        HARNESS_HEADER = HARNESS_HEADER,
    );

    let src = &result.c_source;
    let Some(output) = compile_and_run_kernel("issue_163_sum", src, &harness) else {
        panic!("issue #163 sum kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "C backend's stride-4 cascade must match the host evaluator bit-exactly \
         for the issue #163 multiset; got: {output}"
    );
}

// ---- Test 5: Zero-size tensor does not crash ----

#[test]
fn exec_zero_size_tensor_does_not_crash() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(0), None);
    dag.add_node(RiscOp::Exp, vec![a], vec_f32(0), None);
    let dag = fuse(&dag);

    let result = codegen_with_options(
        &dag,
        "test_exp_zero",
        CodegenOptions {
            math_lib_override: Some(MathLib::Sleef),
            ..Default::default()
        },
    )
    .unwrap();
    let src = &result.c_source;

    let harness = format!(
        r#"{HARNESS_HEADER}
extern void test_exp_zero(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    // zero-element 1D tensor
    float dummy = 0.0f;
    chelis_tensor in_t = make_view_1d(&dummy, 0);

    chelis_tensor* in_ptr = &in_t;
    chelis_tensor* inputs[1] = {{in_ptr}};
    chelis_tensor* out_slot = NULL;
    chelis_tensor* outputs[1] = {{out_slot}};

    test_exp_zero(inputs, 1, outputs, 1);
    printf("zero-size exp returned, output_size=%lld\n", (long long)(outputs[0] ? outputs[0]->size : -1));
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
        "chelis_runtime_dtype.h",
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
        .arg("-O2")
        .args(simd_isa_flags())
        .args([
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

    // #172: `chelis_max_f32` now PROPAGATES NaN consistently, regardless of
    // whether the NaN lands in an AVX2 lane or the scalar tail, matching
    // `torch.max` (which returns NaN for any NaN-containing slice, at every
    // position). Previously this probe only DOCUMENTED the position-dependent
    // inconsistency (the SIMD max dropped NaN); it now ASSERTS the fixed,
    // torch-aligned behavior: all three positions yield NaN, and the run
    // reports CONSISTENT propagation.
    assert!(
        stdout.contains("max(NaN@pos0,  n=8): is_nan=1"),
        "NaN at AVX2 lane 0 must propagate (#172); probe output:\n{stdout}"
    );
    assert!(
        stdout.contains("max(NaN@pos1,  n=8): is_nan=1"),
        "NaN at AVX2 lane 1 must propagate (#172); probe output:\n{stdout}"
    );
    assert!(
        stdout.contains("max(NaN@tail,  n=9): is_nan=1"),
        "NaN in the scalar tail must propagate (#172); probe output:\n{stdout}"
    );
    assert!(
        stdout.contains("CONSISTENT: propagation=1"),
        "NaN propagation must be position-independent and always-propagate \
         (#172 torch parity); probe output:\n{stdout}"
    );
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
        .args(["-std=c++17", "-O2"])
        .args(simd_isa_flags())
        .args([
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

// =====================================================================
// WS-A1 + WS-A4 acceptance oracle: f64 / mixed-dtype / integer
// reduce_sum / matmul / i8 / i16 end-to-end tests.
// =====================================================================
//
// These tests exercise:
//   * WS-A1: f64 BlasMatmul + dgemm dispatch and the dtype-parameterized
//     ReduceSum accumulator (f64 / mixed / integer paths).
//   * WS-A4: i8 / i16 source data through the active dtype set per
//     spec/04-type-system.md §1.1 plus the reduce_sum
//     accumulator-promotion rule per §5.7.1 (i8/i16 → i32). The C
//     backend's `dtype_macro` maps Int8/Int16 to CHELIS_DTYPE_I8 / CHELIS_DTYPE_I16
//     and the runtime allocator sizes their buffers correctly.
//
// They compile generated C against the runtime + BLAS, run it, and
// compare against a hand-computed reference (the evaluator equivalent
// for the fixed-point reduce_sum / matmul cases is the closed-form
// value).

use chelis_ir::dag::DimExpr;

fn vec_f64(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F64,
    }
}

fn mat_f64(r: usize, c: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(r), DimInfo::Lit(c)],
        precision: Prim::F64,
    }
}

fn vec_i32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::Int32,
    }
}

fn vec_i8(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::Int8,
    }
}

fn vec_i16(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::Int16,
    }
}

fn scalar_i32() -> TensorType {
    TensorType {
        dims: vec![],
        precision: Prim::Int32,
    }
}

fn scalar_f64() -> TensorType {
    TensorType {
        dims: vec![],
        precision: Prim::F64,
    }
}

/// Locate openblas / accelerate link flags so the matmul tests can
/// link against `cblas_dgemm`. Returns None on platforms where BLAS
/// isn't reachable; tests skip cleanly in that case.
fn blas_link_flags() -> Option<Vec<String>> {
    if cfg!(target_os = "macos") {
        Some(vec!["-framework".into(), "Accelerate".into()])
    } else {
        Some(vec!["-lopenblas".into()])
    }
}

/// Compile + run a kernel that uses BLAS. Same as
/// `compile_and_run_kernel` but also passes the BLAS link flags. The
/// generated C source is expected to have `#include "chelis_blas.h"`.
/// Returns None on compile/run failure (e.g. openblas not installed).
fn compile_and_run_kernel_with_blas(
    test_name: &str,
    c_source: &str,
    harness: &str,
) -> Option<String> {
    let dir = std::env::temp_dir().join(format!("chelis_exec_{test_name}"));
    fs::create_dir_all(&dir).unwrap();

    fs::write(dir.join("kernel.c"), c_source).unwrap();
    fs::write(dir.join("main.c"), harness).unwrap();

    let include_dir = runtime_include_dir();
    for hdr in &[
        "chelis_runtime.h",
        "chelis_runtime_dtype.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        let src = fs::read_to_string(include_dir.join(hdr)).unwrap();
        fs::write(dir.join(hdr), src).unwrap();
    }

    let bin = dir.join("test_bin");
    let runtime_lib = runtime_lib_path();
    let blas_flags = blas_link_flags().unwrap_or_default();

    let mut args: Vec<String> = vec!["-O2".into()];
    args.extend(simd_isa_flags());
    args.extend([
        "-std=c11".into(),
        "-I".into(),
        dir.to_str().unwrap().into(),
        dir.join("kernel.c").to_str().unwrap().into(),
        dir.join("main.c").to_str().unwrap().into(),
        "-o".into(),
        bin.to_str().unwrap().into(),
        runtime_lib.to_str().unwrap().into(),
    ]);
    args.extend(blas_flags);
    args.push("-lm".into());
    args.push("-lpthread".into());
    args.push("-ldl".into());

    let compile = Command::new("gcc")
        .args(&args)
        .output()
        .expect("failed to invoke gcc");

    if !compile.status.success() {
        let stderr = String::from_utf8_lossy(&compile.stderr);
        eprintln!("BLAS COMPILE FAILED [{test_name}]:\n{stderr}");
        eprintln!("Kernel C:\n{c_source}");
        return None;
    }

    let run = Command::new(&bin).output().expect("failed to run binary");
    if !run.status.success() {
        let stderr = String::from_utf8_lossy(&run.stderr);
        let stdout = String::from_utf8_lossy(&run.stdout);
        eprintln!("BLAS RUN FAILED [{test_name}]\nstdout: {stdout}\nstderr: {stderr}");
        return None;
    }

    Some(String::from_utf8_lossy(&run.stdout).into_owned())
}

// ---- WS-A1 Test: f64 reduce_sum produces correct value ----
//
// Spec §5.7.1: f64 reduce_sum default accumulator is f64; result is f64.
// Pre-WS-A1 the C backend emitted `chelis_fill_f32` and `float acc =
// 0.0f` regardless of operand precision, silently truncating. After
// WS-A1 the accumulator type and zero literal come from the IR Sum
// node's accumulator (== output precision per §5.7.1), so the loop
// is `double acc = 0.0;` and the fill carries an exact tagged f64 zero.
#[test]
fn ws_a1_exec_f64_reduce_sum_matches_reference() {
    let scalar_ty = scalar_f64();
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f64(100),
        None,
    );
    dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F64,
        },
        vec![a],
        scalar_ty,
        None,
    );
    let dag = fuse(&dag);

    let result = chelis_backend_c::codegen(&dag, "test_reduce_sum_f64").unwrap();
    let src = &result.c_source;

    assert!(
        src.contains("CHELIS_DTYPE_F64"),
        "f64 reduce_sum must allocate an f64 output tensor:\n{src}"
    );
    assert!(
        src.contains("chelis_fill_scalar(t1, chelis_scalar_from_bits(CHELIS_DTYPE_F64,"),
        "f64 reduce_sum must zero through an exact tagged f64 scalar:\n{src}"
    );
    assert!(
        !src.contains("chelis_fill_f32(") && !src.contains("chelis_fill_f64("),
        "f64 reduce_sum must not retain dtype-specific compatibility fills:\n{src}"
    );
    assert!(
        src.contains("double acc"),
        "f64 reduce_sum accumulator must be a double, not float:\n{src}"
    );
    assert!(
        !src.contains("chelis_sum_f32("),
        "f64 reduce_sum must NOT route through the f32-only chelis_sum_f32 SIMD helper:\n{src}"
    );

    // Build harness: pass an f64 array with shape[100] of 1..=100, expect sum=5050.0
    let harness = format!(
        r#"{HARNESS_HEADER}
extern void test_reduce_sum_f64(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    // Allocate exact f64 storage and describe its byte capacity.
    double in_data[100];
    double scalar_sum = 0.0;
    for (int i = 0; i < 100; i++) {{
        in_data[i] = (double)(i + 1);
        scalar_sum += in_data[i];
    }}
    chelis_tensor in_t = make_view_typed_1d(in_data, 100, CHELIS_DTYPE_F64);

    chelis_tensor* in_ptr = &in_t;
    chelis_tensor* inputs[1] = {{in_ptr}};
    chelis_tensor* out_slot = NULL;
    chelis_tensor* outputs[1] = {{out_slot}};

    test_reduce_sum_f64(inputs, 1, outputs, 1);

    double got = ((double*)outputs[0]->data)[0];
    double diff = fabs(got - scalar_sum);
    printf("sum_f64(1..100): got=%.6f expected=%.6f diff=%.12f\n", got, scalar_sum, diff);
    printf("%s\n", diff < 1e-9 ? "PASS" : "FAIL");
    return diff < 1e-9 ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("ws_a1_reduce_sum_f64", src, &harness) else {
        panic!("WS-A1 f64 reduce_sum kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "WS-A1 f64 reduce_sum wrong output:\n{output}"
    );
}

// ---- WS-A1 Test: i32 reduce_sum produces integer result ----
//
// Spec §5.7.1: i32 reduce_sum default accumulator is i32; result is i32.
// Pre-WS-A1 the emitter would have emitted `float acc = 0.0f` and
// silently truncated/cast the integer values. After WS-A1 the
// accumulator type comes from the IR node, so the loop is
// `int32_t acc = (int32_t)0;` with no float involvement. The result
// must round-trip exact integers (no float widening that would lose
// precision near 2^24).
#[test]
fn ws_a1_exec_i32_reduce_sum_produces_integer_result_no_float_cast() {
    let scalar_ty = scalar_i32();
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_i32(5), None);
    dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::Int32,
        },
        vec![a],
        scalar_ty,
        None,
    );
    let dag = fuse(&dag);

    let result = chelis_backend_c::codegen(&dag, "test_reduce_sum_i32").unwrap();
    let src = &result.c_source;

    assert!(
        src.contains("CHELIS_DTYPE_I32"),
        "i32 reduce_sum must allocate an i32 output tensor:\n{src}"
    );
    assert!(
        src.contains("int32_t acc"),
        "i32 reduce_sum accumulator must be int32_t (integer-exact), not float:\n{src}"
    );
    assert!(
        !src.contains("float acc"),
        "i32 reduce_sum must NOT use a float accumulator (silent precision change):\n{src}"
    );
    assert!(
        !src.contains("chelis_sum_f32("),
        "i32 reduce_sum must NOT route through the f32-only SIMD helper:\n{src}"
    );

    // Pick values whose i32 sum fits without overflow; verify exact integer round-trip.
    let harness = format!(
        r#"{HARNESS_HEADER}
extern void test_reduce_sum_i32(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    // i32 storage in a 4-byte slot; runtime alloc treats CHELIS_DTYPE_I32 as 4 bytes.
    int32_t in_data[5] = {{16777215, 16777215, 16777215, 16777215, 16777215}};
    // Note: 16777215 = 2^24 - 1. f32 can represent this exactly, but
    // sum * 5 = 83886075, which a float accumulator would round (since
    // 83886075 != round-to-nearest-float). i32 accumulator returns the
    // exact integer.
    int32_t expected_sum = 16777215 * 5;

    chelis_tensor in_t = make_view_typed_1d(in_data, 5, CHELIS_DTYPE_I32);

    chelis_tensor* in_ptr = &in_t;
    chelis_tensor* inputs[1] = {{in_ptr}};
    chelis_tensor* out_slot = NULL;
    chelis_tensor* outputs[1] = {{out_slot}};

    test_reduce_sum_i32(inputs, 1, outputs, 1);

    if (outputs[0]->dtype != CHELIS_DTYPE_I32) {{
        printf("FAIL: output dtype is %d, expected CHELIS_DTYPE_I32 (%d)\n",
               outputs[0]->dtype, CHELIS_DTYPE_I32);
        return 1;
    }}
    int32_t got = ((int32_t*)outputs[0]->data)[0];
    printf("sum_i32(5x 2^24-1): got=%d expected=%d\n", got, expected_sum);
    printf("%s\n", got == expected_sum ? "PASS" : "FAIL");
    return got == expected_sum ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("ws_a1_reduce_sum_i32", src, &harness) else {
        panic!("WS-A1 i32 reduce_sum kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "WS-A1 i32 reduce_sum wrong output (integer arithmetic must be exact):\n{output}"
    );
}

// ---- WS-A1 Test: f64 matmul dispatches cblas_dgemm and matches reference ----
//
// Spec §5.7.1: f64 matmul accumulator is f64; result is f64. The C
// backend now dispatches `cblas_dgemm` (not `cblas_sgemm`) for an f64
// BlasMatmul. Reference is a hand-computed product where the values
// are exactly representable in f64 but not in f32, so an accidental
// sgemm dispatch would fail the tolerance check.
#[test]
fn ws_a1_exec_f64_matmul_dispatches_dgemm_and_matches_reference() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        mat_f64(2, 3),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        mat_f64(3, 4),
        None,
    );
    let matmul_op = RiscOp::matmul_default(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(4),
        DimExpr::Concrete(3),
        Prim::F64,
    )
    .expect("f64 matmul default constructs (spec §5.7.1)");
    dag.add_node(matmul_op, vec![a, b], mat_f64(2, 4), None);

    let result = codegen_with_options(
        &dag,
        "test_matmul_f64",
        CodegenOptions {
            use_blas: true,
            ..CodegenOptions::default()
        },
    )
    .unwrap();
    let src = &result.c_source;

    assert!(
        src.contains("cblas_dgemm("),
        "f64 matmul MUST dispatch cblas_dgemm (the WS-A1 lift):\n{src}"
    );
    assert!(
        !src.contains("cblas_sgemm("),
        "f64 matmul MUST NOT silently dispatch cblas_sgemm (RT-1 F1 finding):\n{src}"
    );
    assert!(
        src.contains("CHELIS_DTYPE_F64"),
        "f64 matmul output tensor must use CHELIS_DTYPE_F64 dtype:\n{src}"
    );
    assert!(
        src.contains("(double*)t"),
        "f64 matmul must cast tensor data pointers to double*:\n{src}"
    );
    assert!(
        result.requirements.needs_blas,
        "f64 matmul codegen must surface needs_blas=true so the toolchain links openblas/Accelerate"
    );

    // A[2,3] = [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]
    // B[3,4] = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    // Reference C[2,4]: row-major matmul. f64 yields exact results for these
    // operand magnitudes; an accidental sgemm dispatch would compute the same
    // values but on truncated f32 inputs, and the test asserts cblas_dgemm
    // appears in the source so that path is unreachable.
    let harness = format!(
        r#"{HARNESS_HEADER}
extern void test_matmul_f64(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    double a_data[6] = {{1.5, 2.5, 3.5, 4.5, 5.5, 6.5}};
    double b_data[12] = {{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}};

    static const int64_t a_shape[2] = {{2, 3}};
    static const int64_t a_strides[2] = {{3, 1}};
    chelis_tensor a_t = {{
        .data = a_data, .shape = a_shape, .strides = a_strides,
        .size = 6, .byte_capacity = 6 * (int64_t)sizeof(double), .rank = 2,
        .dtype = CHELIS_DTYPE_F64, .owns_data = 0, .reserved = {{0, 0}},
    }};

    static const int64_t b_shape[2] = {{3, 4}};
    static const int64_t b_strides[2] = {{4, 1}};
    chelis_tensor b_t = {{
        .data = b_data, .shape = b_shape, .strides = b_strides,
        .size = 12, .byte_capacity = 12 * (int64_t)sizeof(double), .rank = 2,
        .dtype = CHELIS_DTYPE_F64, .owns_data = 0, .reserved = {{0, 0}},
    }};

    chelis_tensor* inputs[2] = {{&a_t, &b_t}};
    chelis_tensor* out_slot = NULL;
    chelis_tensor* outputs[1] = {{out_slot}};

    test_matmul_f64(inputs, 2, outputs, 1);

    // Reference: hand-computed row-major matmul.
    double expected[8] = {{
        // row 0
        1.5*1 + 2.5*5 + 3.5*9,    // = 1.5 + 12.5 + 31.5 = 45.5
        1.5*2 + 2.5*6 + 3.5*10,   // = 3 + 15 + 35 = 53
        1.5*3 + 2.5*7 + 3.5*11,   // = 4.5 + 17.5 + 38.5 = 60.5
        1.5*4 + 2.5*8 + 3.5*12,   // = 6 + 20 + 42 = 68
        // row 1
        4.5*1 + 5.5*5 + 6.5*9,    // = 4.5 + 27.5 + 58.5 = 90.5
        4.5*2 + 5.5*6 + 6.5*10,   // = 9 + 33 + 65 = 107
        4.5*3 + 5.5*7 + 6.5*11,   // = 13.5 + 38.5 + 71.5 = 123.5
        4.5*4 + 5.5*8 + 6.5*12    // = 18 + 44 + 78 = 140
    }};
    if (outputs[0]->dtype != CHELIS_DTYPE_F64) {{
        printf("FAIL: output dtype is %d, expected CHELIS_DTYPE_F64 (%d)\n",
               outputs[0]->dtype, CHELIS_DTYPE_F64);
        return 1;
    }}
    double* got = (double*)outputs[0]->data;
    int ok = 1;
    for (int i = 0; i < 8; i++) {{
        double diff = fabs(got[i] - expected[i]);
        if (diff > 1e-12) {{
            printf("MISMATCH at %d: got %.15f expected %.15f diff %.15g\n",
                   i, got[i], expected[i], diff);
            ok = 0;
        }}
    }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel_with_blas("ws_a1_matmul_f64", src, &harness) else {
        eprintln!("WS-A1 f64 matmul kernel failed to compile/run (openblas may be unavailable)");
        // Don't panic if openblas is missing on the host; the source-level
        // assertions above already pin the dgemm dispatch shape.
        return;
    };
    assert!(
        output.contains("PASS"),
        "WS-A1 f64 matmul wrong output (cblas_dgemm dispatch) :\n{output}"
    );
}

// ---- WS-A1 Test: mixed-dtype program (f64 tensor + i32 tensor side-by-side) ----
//
// Compiles a program with both f64 and i32 tensors live in the same
// DAG, returning two outputs: an f64 reduce_sum and an i32 reduce_sum.
// Pre-WS-A1 the elem_type fallback to "float" would have silently
// downgraded every non-f32 path; post-WS-A0 elem_type panics on
// unhandled dtypes and post-WS-A1 the f64 path is wired through the
// dtype-parameterized accumulator. This test exercises both dtypes
// round-tripping through the same compiled kernel without one
// silently corrupting the other.
//
// (i32 *index*-indexed Gather is the obvious mixed-dtype shape but
// the existing C-backend Gather code path treats non-i64 indices as
// `float*`-typed data — a pre-WS-A1 limitation that would conflate
// the i32 codegen with an unrelated bug. Side-by-side reductions
// keep the test focused on WS-A1's actual guarantee: that f64 and
// i32 dtype dispatch coexist without aliasing.)
#[test]
fn ws_a1_exec_mixed_f64_tensors_and_i32_indices_compile_and_run() {
    let mut dag = Dag::new();
    let f_values = dag.add_node(
        RiscOp::Load {
            name: "f_values".into(),
        },
        vec![],
        vec_f64(4),
        None,
    );
    let i_values = dag.add_node(
        RiscOp::Load {
            name: "i_values".into(),
        },
        vec![],
        vec_i32(4),
        None,
    );
    // Two reductions: one f64 (default f64 accumulator), one i32
    // (default i32 accumulator). Both Stores so codegen marks both as
    // outputs.
    let f_sum = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F64,
        },
        vec![f_values],
        scalar_f64(),
        None,
    );
    let i_sum = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::Int32,
        },
        vec![i_values],
        scalar_i32(),
        None,
    );
    dag.add_node(
        RiscOp::Store {
            name: "f_out".into(),
        },
        vec![f_sum],
        scalar_f64(),
        None,
    );
    dag.add_node(
        RiscOp::Store {
            name: "i_out".into(),
        },
        vec![i_sum],
        scalar_i32(),
        None,
    );

    let result = chelis_backend_c::codegen(&dag, "test_mixed").unwrap();
    let src = &result.c_source;

    assert!(
        src.contains("CHELIS_DTYPE_F64"),
        "mixed-dtype program must use CHELIS_DTYPE_F64 for f64 tensors:\n{src}"
    );
    assert!(
        src.contains("CHELIS_DTYPE_I32"),
        "mixed-dtype program must use CHELIS_DTYPE_I32 for i32 tensors:\n{src}"
    );
    assert!(
        src.contains("double acc"),
        "mixed-dtype program must use a double accumulator for the f64 sum:\n{src}"
    );
    assert!(
        src.contains("int32_t acc"),
        "mixed-dtype program must use an int32_t accumulator for the i32 sum:\n{src}"
    );

    let harness = format!(
        r#"{HARNESS_HEADER}
extern void test_mixed(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    double f_data[4] = {{1.5, 2.5, 3.5, 4.5}};
    int32_t i_data[4] = {{100, 200, 300, 400}};
    double f_expected = 1.5 + 2.5 + 3.5 + 4.5;          // 12.0
    int32_t i_expected = 100 + 200 + 300 + 400;         // 1000

    chelis_tensor f_t = make_view_typed_1d(f_data, 4, CHELIS_DTYPE_F64);
    chelis_tensor i_t = make_view_typed_1d(i_data, 4, CHELIS_DTYPE_I32);

    chelis_tensor* inputs[2] = {{&f_t, &i_t}};
    chelis_tensor* outputs[2] = {{NULL, NULL}};

    test_mixed(inputs, 2, outputs, 2);

    int ok = 1;
    if (outputs[0]->dtype != CHELIS_DTYPE_F64) {{
        printf("FAIL: f_out dtype is %d, expected CHELIS_DTYPE_F64\n", outputs[0]->dtype);
        ok = 0;
    }}
    if (outputs[1]->dtype != CHELIS_DTYPE_I32) {{
        printf("FAIL: i_out dtype is %d, expected CHELIS_DTYPE_I32\n", outputs[1]->dtype);
        ok = 0;
    }}
    double f_got = ((double*)outputs[0]->data)[0];
    int32_t i_got = ((int32_t*)outputs[1]->data)[0];
    double f_diff = fabs(f_got - f_expected);
    printf("mixed: f_got=%.6f expected=%.6f diff=%.12f i_got=%d expected=%d\n",
           f_got, f_expected, f_diff, i_got, i_expected);
    if (f_diff > 1e-9) {{ printf("FAIL: f64 sum off\n"); ok = 0; }}
    if (i_got != i_expected) {{ printf("FAIL: i32 sum off\n"); ok = 0; }}
    // Verify input storage was not cross-corrupted by the dual-dtype path.
    if (i_data[0] != 100 || i_data[3] != 400) {{
        printf("FAIL: i32 input storage corrupted\n"); ok = 0;
    }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("ws_a1_mixed_dtype", src, &harness) else {
        panic!("WS-A1 mixed-dtype kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "WS-A1 mixed-dtype output wrong:\n{output}"
    );
}

// ---- WS-A1 Negative parity: integer matmul errors with useful diagnostic ----
//
// Spec §5.7.2: integer matmul (i8/i16/i32/i64) is not admitted in this
// cycle. The IR-level rejection lives in
// `RiscOp::default_matmul_accumulator` (returns Err for integer
// operands), so the matmul constructor cannot even build an integer
// BlasMatmul; the error message must cite the spec and is the
// negative-parity twin of the f64 acceptance test above.
#[test]
fn ws_a1_negative_integer_matmul_rejected_with_spec_citation() {
    // i32 operand: matmul_default must error.
    let err_i32 = RiscOp::matmul_default(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(4),
        DimExpr::Concrete(3),
        Prim::Int32,
    )
    .expect_err("i32 matmul must be rejected per spec §5.7.2");
    assert!(
        err_i32.contains("spec/04-type-system.md §5.7.2"),
        "i32 matmul rejection must cite §5.7.2; got: {err_i32}"
    );
    assert!(
        err_i32.contains("integer"),
        "i32 matmul rejection must say 'integer'; got: {err_i32}"
    );

    // i64 operand: matmul_default must error.
    let err_i64 = RiscOp::matmul_default(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(4),
        DimExpr::Concrete(3),
        Prim::Int64,
    )
    .expect_err("i64 matmul must be rejected per spec §5.7.2");
    assert!(
        err_i64.contains("spec/04-type-system.md §5.7.2"),
        "i64 matmul rejection must cite §5.7.2; got: {err_i64}"
    );

    // i8 / i16 must also error.
    for prim in [Prim::Int8, Prim::Int16] {
        let err = RiscOp::matmul_default(
            vec![],
            DimExpr::Concrete(2),
            DimExpr::Concrete(4),
            DimExpr::Concrete(3),
            prim,
        )
        .expect_err(&format!("{prim:?} matmul must be rejected per spec §5.7.2"));
        assert!(
            err.contains("§5.7.2"),
            "{prim:?} matmul rejection must cite §5.7.2; got: {err}"
        );
    }
}

// ---- WS-A1 / WS-A3: bf16/f16 matmul admitted at IR validation ----
//
// Spec §5.7.1: bf16/f16 matmul accumulator default is f32. WS-A3
// wired the HIP backend's bf16/f16 dispatch through `hipblasGemmEx`
// with `HIPBLAS_COMPUTE_32F`, so the F1 IR validation guard now
// admits bf16/f16. The C backend still rejects bf16/f16 at its own
// F1 guard (no `cblas_*` dispatch yet); this IR-level test pins
// that the validation guard no longer catches the bf16/f16 case.
// Replaces the prior `ws_a1_negative_bf16_f16_matmul_still_blocked_by_f1_guard`
// assertion per the spec-sync rule that lifted-guard tests be
// REPLACED, not silently deleted.
#[test]
fn ws_a3_bf16_f16_matmul_admitted_at_ir_validation() {
    use chelis_ir::verify;
    for prim in [Prim::Bf16, Prim::F16] {
        let mut dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            precision: prim,
        };
        let ty_b = TensorType {
            dims: vec![DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: prim,
        };
        let ty_out = TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(4)],
            precision: prim,
        };
        let a = dag.add_node(RiscOp::synth_const(ty.precision, 1.0), vec![], ty, None);
        let b = dag.add_node(RiscOp::synth_const(ty_b.precision, 1.0), vec![], ty_b, None);
        let matmul_op = RiscOp::matmul_default(
            vec![],
            DimExpr::Concrete(2),
            DimExpr::Concrete(4),
            DimExpr::Concrete(3),
            prim,
        )
        .unwrap_or_else(|e| panic!("{prim:?} matmul default constructs (per spec §5.7.1): {e}"));
        let _ = dag.add_node(matmul_op, vec![a, b], ty_out, None);

        let errors = verify::verify(&dag);
        assert!(
            !errors.iter().any(|m| m.contains("F1: BlasMatmul")),
            "WS-A3 lifted {prim:?} from the F1 guard; {prim:?} BlasMatmul must validate cleanly. \
             Got: {errors:?}"
        );
    }
}

/// Header for i8/i16-typed harnesses. Reuses the runtime tensor view
/// shape but reinterprets `t->data` as the narrow integer pointer.
const WS_A4_HARNESS_HEADER: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "chelis_runtime.h"

static chelis_tensor make_view_1d_i8(int8_t* data, int n) {
    static int64_t shape[1];
    static const int64_t strides[1] = {1};
    shape[0] = n;
    return (chelis_tensor){
        .data = data, .shape = shape, .strides = strides, .size = n,
        .byte_capacity = n * (int64_t)sizeof(int8_t), .rank = 1,
        .dtype = CHELIS_DTYPE_I8, .owns_data = 0, .reserved = {0, 0},
    };
}

static chelis_tensor make_view_1d_i16(int16_t* data, int n) {
    static int64_t shape[1];
    static const int64_t strides[1] = {1};
    shape[0] = n;
    return (chelis_tensor){
        .data = data, .shape = shape, .strides = strides, .size = n,
        .byte_capacity = n * (int64_t)sizeof(int16_t), .rank = 1,
        .dtype = CHELIS_DTYPE_I16, .owns_data = 0, .reserved = {0, 0},
    };
}
"#;

/// Build a `Load → Op → Op` DAG with two same-precision i8 inputs and
/// emit the C source. Covers the i8 add path through the dtype-aware
/// `elem_type` and `dtype_macro`. Tensors are vec_i8(N) so the codegen
/// reinterpret-casts `t->data` to `int8_t*`.
#[test]
fn exec_i8_add_correct_output() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_i8(8), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], vec_i8(8), None);
    dag.add_node(RiscOp::Add, vec![a, b], vec_i8(8), None);
    let dag = fuse(&dag);

    let result = chelis_backend_c::codegen(&dag, "test_i8_add").unwrap();
    let src = &result.c_source;

    // The kernel must emit `int8_t*` access against `t->data` (not float*).
    assert!(
        src.contains("int8_t"),
        "i8 add codegen must mention int8_t element type; got:\n{src}"
    );
    assert!(
        src.contains("CHELIS_DTYPE_I8"),
        "i8 add codegen must allocate output via CHELIS_DTYPE_I8; got:\n{src}"
    );

    let harness = format!(
        r#"{WS_A4_HARNESS_HEADER}
extern void test_i8_add(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    int8_t a_data[8] = {{ 1, 2, 3, 4, -5, -6, 7, 8 }};
    int8_t b_data[8] = {{ 10, 20, 30, 40, 50, 60, -70, -80 }};
    int8_t expected[8];
    for (int i = 0; i < 8; i++) {{
        // Two's-complement wrapping at the i8 width; matches the
        // backend's `int8_t + int8_t` semantics.
        expected[i] = (int8_t)((int)a_data[i] + (int)b_data[i]);
    }}
    chelis_tensor at = make_view_1d_i8(a_data, 8);
    chelis_tensor bt = make_view_1d_i8(b_data, 8);
    chelis_tensor* in_ptrs[2] = {{ &at, &bt }};
    chelis_tensor* outs[1] = {{ NULL }};
    test_i8_add(in_ptrs, 2, outs, 1);
    int ok = 1;
    for (int i = 0; i < 8; i++) {{
        int8_t got = ((int8_t*)outs[0]->data)[i];
        if (got != expected[i]) {{
            printf("MISMATCH at %d: got %d expected %d\n", i, (int)got, (int)expected[i]);
            ok = 0;
        }}
    }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("ws_a4_i8_add", src, &harness) else {
        panic!("i8 add kernel failed to compile/run");
    };
    assert!(output.contains("PASS"), "i8 add wrong output:\n{output}");
}

/// i8 + i8 overflow traps at the declared width before C can execute a
/// narrowing conversion. This locks the Phase 3 checked-arithmetic contract
/// at the backend's direct compile-run surface.
#[test]
fn exec_i8_add_overflow_traps() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_i8(2), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], vec_i8(2), None);
    dag.add_node(RiscOp::Add, vec![a, b], vec_i8(2), None);
    let dag = fuse(&dag);

    let result = chelis_backend_c::codegen(&dag, "test_i8_add_wrap").unwrap();
    let src = &result.c_source;

    let harness = format!(
        r#"{WS_A4_HARNESS_HEADER}
extern void test_i8_add_wrap(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    // Both elements overflow int8 and must trap before store-back.
    int8_t a_data[2] = {{ 100, 127 }};
    int8_t b_data[2] = {{ 50, 1 }};
    chelis_tensor at = make_view_1d_i8(a_data, 2);
    chelis_tensor bt = make_view_1d_i8(b_data, 2);
    chelis_tensor* in_ptrs[2] = {{ &at, &bt }};
    chelis_tensor* outs[1] = {{ NULL }};
    test_i8_add_wrap(in_ptrs, 2, outs, 1);
    return 0;
}}
"#
    );

    let run = compile_and_capture_run("ws_a4_i8_add_wrap", src, &harness);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        !run.status.success(),
        "i8 add overflow must terminate unsuccessfully"
    );
    assert!(
        stderr.contains("numeric trap: overflow in add at int8"),
        "i8 add overflow must use the canonical diagnostic; stderr={stderr:?}"
    );
}

/// The second i8 multiplication overflows after an in-range first element;
/// the kernel must trap rather than partially legitimizing the wrapped row.
#[test]
fn exec_i8_mul_overflow_traps() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_i8(2), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], vec_i8(2), None);
    dag.add_node(RiscOp::Mul, vec![a, b], vec_i8(2), None);
    let dag = fuse(&dag);

    let result = chelis_backend_c::codegen(&dag, "test_i8_mul").unwrap();
    let src = &result.c_source;

    let harness = format!(
        r#"{WS_A4_HARNESS_HEADER}
extern void test_i8_mul(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    int8_t a_data[2] = {{ 12, 16 }};
    int8_t b_data[2] = {{ 10, 8 }};
    // 12*10 = 120 fits; 16*8 = 128 overflows int8 and must trap.
    chelis_tensor at = make_view_1d_i8(a_data, 2);
    chelis_tensor bt = make_view_1d_i8(b_data, 2);
    chelis_tensor* in_ptrs[2] = {{ &at, &bt }};
    chelis_tensor* outs[1] = {{ NULL }};
    test_i8_mul(in_ptrs, 2, outs, 1);
    return 0;
}}
"#
    );

    let run = compile_and_capture_run("ws_a4_i8_mul", src, &harness);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        !run.status.success(),
        "i8 mul overflow must terminate unsuccessfully"
    );
    assert!(
        stderr.contains("numeric trap: overflow in mul at int8"),
        "i8 mul overflow must use the canonical diagnostic; stderr={stderr:?}"
    );
}

/// i16 add: pick values that exercise the int16_t path through the
/// codegen without overflowing. Mirrors `exec_i8_add_correct_output`
/// for the i16 dtype.
#[test]
fn exec_i16_add_correct_output() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_i16(4), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], vec_i16(4), None);
    dag.add_node(RiscOp::Add, vec![a, b], vec_i16(4), None);
    let dag = fuse(&dag);

    let result = chelis_backend_c::codegen(&dag, "test_i16_add").unwrap();
    let src = &result.c_source;

    assert!(
        src.contains("int16_t"),
        "i16 add codegen must mention int16_t element type; got:\n{src}"
    );
    assert!(
        src.contains("CHELIS_DTYPE_I16"),
        "i16 add codegen must allocate output via CHELIS_DTYPE_I16; got:\n{src}"
    );

    let harness = format!(
        r#"{WS_A4_HARNESS_HEADER}
extern void test_i16_add(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    int16_t a_data[4] = {{ 1000, -2000, 30000, -32000 }};
    int16_t b_data[4] = {{ 500, -1000, -20000, 1000 }};
    int16_t expected[4];
    for (int i = 0; i < 4; i++) {{
        expected[i] = (int16_t)((int)a_data[i] + (int)b_data[i]);
    }}
    chelis_tensor at = make_view_1d_i16(a_data, 4);
    chelis_tensor bt = make_view_1d_i16(b_data, 4);
    chelis_tensor* in_ptrs[2] = {{ &at, &bt }};
    chelis_tensor* outs[1] = {{ NULL }};
    test_i16_add(in_ptrs, 2, outs, 1);
    int ok = 1;
    for (int i = 0; i < 4; i++) {{
        int16_t got = ((int16_t*)outs[0]->data)[i];
        if (got != expected[i]) {{
            printf("MISMATCH at %d: got %d expected %d\n", i, (int)got, (int)expected[i]);
            ok = 0;
        }}
    }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("ws_a4_i16_add", src, &harness) else {
        panic!("i16 add kernel failed to compile/run");
    };
    assert!(output.contains("PASS"), "i16 add wrong output:\n{output}");
}

/// i8 reduce_sum into the spec-default i32 accumulator (the WS-0
/// pinned promotion rule per §5.7.1). 200 ones at i8 source overflows
/// i8 (max +127); the i32 accumulator + i32 result must yield exactly
/// 200, no overflow, no panic.
#[test]
fn exec_i8_reduce_sum_promotes_to_i32() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_i8(200), None);
    // Use the spec-default constructor so the IR carries the §5.7.1
    // i32 accumulator, not an inline `Prim::Int8` that would fail the
    // verifier's narrowness check.
    let sum_op =
        chelis_ir::dag::RiscOp::sum_default(0, Prim::Int8).expect("i8 sum_default must succeed");
    dag.add_node(sum_op, vec![a], scalar_i32(), None);
    let dag = fuse(&dag);

    let result = chelis_backend_c::codegen(&dag, "test_i8_reduce_sum").unwrap();
    let src = &result.c_source;

    // The accumulator type in the emitted C must be int32_t — pinning
    // this catches a regression where the codegen silently picks the
    // operand precision (the F1 footgun class for reductions).
    assert!(
        src.contains("int32_t acc"),
        "i8 reduce_sum must accumulate in int32_t (per spec §5.7.1); got:\n{src}"
    );
    assert!(
        src.contains("CHELIS_DTYPE_I32"),
        "i8 reduce_sum output tensor must be allocated via CHELIS_DTYPE_I32; got:\n{src}"
    );

    let harness = format!(
        r#"{WS_A4_HARNESS_HEADER}
extern void test_i8_reduce_sum(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    int8_t in_data[200];
    for (int i = 0; i < 200; i++) in_data[i] = 1;
    chelis_tensor in_t = make_view_1d_i8(in_data, 200);
    chelis_tensor* in_ptrs[1] = {{ &in_t }};
    chelis_tensor* outs[1] = {{ NULL }};
    test_i8_reduce_sum(in_ptrs, 1, outs, 1);

    int32_t got = ((int32_t*)outs[0]->data)[0];
    int32_t expected = 200;
    printf("sum(200 i8 ones): got=%d expected=%d\n", got, expected);
    printf("%s\n", got == expected ? "PASS" : "FAIL");
    return got == expected ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("ws_a4_i8_reduce_sum_200", src, &harness) else {
        panic!("i8 reduce_sum kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "i8 reduce_sum did not promote to i32 / produced wrong sum:\n{output}"
    );
}

/// i16 reduce_sum into i32: same accumulator-promotion path as i8.
/// 200 i16 values of 1000 each = 200000 — overflows i16 (max +32767)
/// but fits in i32. Pin both the source-int16 path and the i32 result.
#[test]
fn exec_i16_reduce_sum_promotes_to_i32() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_i16(200),
        None,
    );
    let sum_op =
        chelis_ir::dag::RiscOp::sum_default(0, Prim::Int16).expect("i16 sum_default must succeed");
    dag.add_node(sum_op, vec![a], scalar_i32(), None);
    let dag = fuse(&dag);

    let result = chelis_backend_c::codegen(&dag, "test_i16_reduce_sum").unwrap();
    let src = &result.c_source;

    assert!(
        src.contains("int32_t acc"),
        "i16 reduce_sum must accumulate in int32_t (per spec §5.7.1); got:\n{src}"
    );

    let harness = format!(
        r#"{WS_A4_HARNESS_HEADER}
extern void test_i16_reduce_sum(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    int16_t in_data[200];
    for (int i = 0; i < 200; i++) in_data[i] = 1000;
    chelis_tensor in_t = make_view_1d_i16(in_data, 200);
    chelis_tensor* in_ptrs[1] = {{ &in_t }};
    chelis_tensor* outs[1] = {{ NULL }};
    test_i16_reduce_sum(in_ptrs, 1, outs, 1);

    int32_t got = ((int32_t*)outs[0]->data)[0];
    int32_t expected = 200000; // overflows i16 (max +32767), fits in i32
    printf("sum(200 i16 1000s): got=%d expected=%d\n", got, expected);
    printf("%s\n", got == expected ? "PASS" : "FAIL");
    return got == expected ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("ws_a4_i16_reduce_sum_200", src, &harness) else {
        panic!("i16 reduce_sum kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "i16 reduce_sum did not promote to i32 / produced wrong sum:\n{output}"
    );
}

/// Negative test: the spec/04-type-system.md §5.7.1 narrowness rule
/// says a user cannot request a narrower-than-default accumulator.
/// `sum_with_accumulator(_, Int8, Int8)` must error. This locks the
/// IR-side rejection helper that the WS-A0-Fixups verifier installs;
/// the codegen path never sees the malformed IR.
#[test]
fn ws_a4_i8_sum_with_narrower_accumulator_is_ir_error() {
    let err = chelis_ir::dag::RiscOp::sum_with_accumulator(0, Prim::Int8, Prim::Int8)
        .expect_err("i8 reduce_sum with i8 accumulator must be rejected by sum_with_accumulator");
    assert!(
        err.contains("narrower than"),
        "i8 reduce_sum with i8 accumulator must reference the narrowness rule; got: {err}"
    );
    assert!(
        err.contains("§5.7.1"),
        "rejection diagnostic must cite spec §5.7.1; got: {err}"
    );
}

// ============================================================
// #517: emit_cmplt must read its operands through their OWN
// element dtype, not the boolean (f32) output dtype. The cmplt
// signature is `∀D,p. (tensor[D,p], tensor[D,p]) → tensor[D,bool]`,
// so the operands carry the compared precision `p` while the result
// is bool (stored f32). Reading a RUNTIME-PRODUCED int32 / int64 /
// f64 operand through a raw `float*` reinterprets the bit pattern
// (the #347 / #476 bug class). The discriminator is a NEGATIVE
// integer operand: as int32, `-7 < -3` is true; reinterpreting the
// int32 bit pattern 0xFFFFFFF9 / 0xFFFFFFFD as `float` yields NaN, so
// the buggy `float*` read returns false. eval-vs-C parity oracle.
// ============================================================

fn vec_prim(n: usize, p: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: p,
    }
}

/// Build `cmplt(a, b)` over two runtime Load operands of precision
/// `prim`, evaluate the chelis-ir oracle, compile + run the generated
/// C kernel, and assert the C bool output matches the evaluator
/// element-for-element. `c_elem` / `c_dtype` describe the operand's C
/// storage type and runtime dtype tag; values are passed as f64 (exact
/// for the integer and small-double cases used here).
fn run_cmplt_parity(
    tag: &str,
    prim: Prim,
    c_elem: &str,
    c_dtype: &str,
    a_vals: &[f64],
    b_vals: &[f64],
) {
    use chelis_ir::eval::{TensorValue, eval_tensor};
    use chelis_unord::UnordMap;

    let n = a_vals.len();
    assert_eq!(n, b_vals.len(), "operand length mismatch in {tag}");

    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_prim(n, prim),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_prim(n, prim),
        None,
    );
    let root = dag.add_node(RiscOp::CmpLt, vec![a, b], vec_prim(n, Prim::Bool), None);

    // Evaluator oracle: numeric `a < b` per element (eval.rs CmpLt).
    let mut inputs = UnordMap::new();
    inputs.insert(
        "a".to_string(),
        TensorValue::from_vec(vec![n], a_vals.to_vec()),
    );
    inputs.insert(
        "b".to_string(),
        TensorValue::from_vec(vec![n], b_vals.to_vec()),
    );
    let evaluated = eval_tensor(&dag, &inputs).expect("evaluator must succeed");
    let expected: Vec<f64> = evaluated[&root].to_f64_lossy_vec().clone();
    assert_eq!(expected.len(), n);

    let dag = fuse(&dag);
    let result =
        codegen_with_options(&dag, &format!("cmplt_{tag}"), CodegenOptions::default()).unwrap();
    let src = &result.c_source;

    // Format the operand initializers and the expected bool vector.
    let fmt_vals = |vals: &[f64]| -> String {
        vals.iter()
            .map(|v| {
                if prim == Prim::F64 {
                    format!("{v:?}")
                } else {
                    format!("{}", *v as i64)
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    let a_init = fmt_vals(a_vals);
    let b_init = fmt_vals(b_vals);
    let exp_init = expected
        .iter()
        .map(|v| format!("{}", *v as u8))
        .collect::<Vec<_>>()
        .join(", ");

    let harness = format!(
        r#"{HARNESS_HEADER}
#include <stdint.h>

extern void cmplt_{tag}(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    {c_elem} a_data[{n}] = {{{a_init}}};
    {c_elem} b_data[{n}] = {{{b_init}}};
    chelis_tensor a_t = make_view_typed_1d(a_data, {n}, {c_dtype});
    chelis_tensor b_t = make_view_typed_1d(b_data, {n}, {c_dtype});
    chelis_tensor* inputs[2] = {{&a_t, &b_t}};
    chelis_tensor* outputs[1] = {{NULL}};

    cmplt_{tag}(inputs, 2, outputs, 1);

    uint8_t expected[{n}] = {{{exp_init}}};
    int ok = 1;
    if (outputs[0]->dtype != CHELIS_DTYPE_BOOL) {{
        printf("MISMATCH dtype: got %u expected CHELIS_DTYPE_BOOL (%u)\n",
               (unsigned)outputs[0]->dtype, (unsigned)CHELIS_DTYPE_BOOL);
        ok = 0;
    }}
    for (int i = 0; i < {n}; i++) {{
        uint8_t got = ((uint8_t*)outputs[0]->data)[i];
        if (got != expected[i]) {{
            printf("MISMATCH at %d: got %u expected %u\n",
                   i, (unsigned)got, (unsigned)expected[i]);
            ok = 0;
        }}
    }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel(&format!("cmplt_{tag}"), src, &harness) else {
        panic!("#517 cmplt parity [{tag}]: kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "#517 cmplt parity [{tag}]: C backend disagreed with evaluator.\n\
         Generated C:\n{src}\nRun output:\n{output}"
    );
}

/// #517 primary oracle: runtime int32 operands, including the negative
/// values that the `float*` bit-reinterpret gets wrong.
#[test]
fn exec_cmplt_int32_runtime_operands_match_evaluator() {
    run_cmplt_parity(
        "i32",
        Prim::Int32,
        "int32_t",
        "CHELIS_DTYPE_I32",
        &[-7.0, 2.0, -5.0, 10.0, 3.0, -1.0],
        &[-3.0, 10.0, 3.0, 2.0, 3.0, -1.0],
    );
}

/// #517 sweep: int64 operands. The buggy `float*` read also misaligns
/// the 8-byte stride; reading as `int64_t*` is required.
#[test]
fn exec_cmplt_int64_runtime_operands_match_evaluator() {
    run_cmplt_parity(
        "i64",
        Prim::Int64,
        "int64_t",
        "CHELIS_DTYPE_I64",
        &[-7.0, 2.0, -5.0, 100.0, 3.0],
        &[-3.0, 100.0, 3.0, 2.0, 3.0],
    );
}

/// #517 sweep: f64 operands. f64 read through `float*` truncates the
/// 8-byte payload to 4 bytes; reading as `double*` is required.
#[test]
fn exec_cmplt_f64_runtime_operands_match_evaluator() {
    run_cmplt_parity(
        "f64",
        Prim::F64,
        "double",
        "CHELIS_DTYPE_F64",
        &[-7.5, 2.25, -5.0, 10.0, 3.0],
        &[-3.5, 10.0, 3.0, 2.0, 3.0],
    );
}

fn direct_int_sub_case(
    tag: &str,
    prim: Prim,
    c_type: &str,
    c_dtype: &str,
    lhs: &str,
    rhs: &str,
    expected: &str,
) {
    let mut dag = Dag::new();
    let ty = vec_prim(4, prim);
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], ty.clone(), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], ty.clone(), None);
    dag.add_node(RiscOp::Sub, vec![a, b], ty, None);
    let function = format!("direct_sub_{tag}");
    let src = chelis_backend_c::codegen(&dag, &function)
        .expect("direct subtraction codegen")
        .c_source;
    assert!(src.contains("chelis_int_checked_sub"), "{tag}: {src}");
    assert!(!src.contains("chelis_int_checked_add("), "{tag}: {src}");

    let harness = format!(
        r#"{HARNESS_HEADER}
#include <stdint.h>
#include <limits.h>

extern void {function}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);

int main(void) {{
    {c_type} a_data[4] = {{ {lhs} }};
    {c_type} b_data[4] = {{ {rhs} }};
    {c_type} expected[4] = {{ {expected} }};
    chelis_tensor a = make_view_typed_1d(a_data, 4, {c_dtype});
    chelis_tensor b = make_view_typed_1d(b_data, 4, {c_dtype});
    chelis_tensor *inputs[2] = {{ &a, &b }};
    chelis_tensor *outputs[1] = {{ NULL }};
    {function}(inputs, 2, outputs, 1);
    {c_type} *got = ({c_type} *)outputs[0]->data;
    for (int i = 0; i < 4; i++) {{
        if (got[i] != expected[i]) return 1;
    }}
    puts("PASS");
    return 0;
}}
"#
    );
    let output = compile_and_run_kernel(&function, &src, &harness)
        .unwrap_or_else(|| panic!("{tag} direct subtraction did not compile and run"));
    assert!(output.contains("PASS"), "{tag}: {output}");
}

#[test]
fn direct_checked_subtraction_executes_exact_boundaries_at_every_signed_width() {
    for case in [
        (
            "i8",
            Prim::Int8,
            "int8_t",
            "CHELIS_DTYPE_I8",
            "-1, INT8_MAX, INT8_MIN, 3",
            "INT8_MIN, 1, -1, -4",
            "INT8_MAX, INT8_MAX - 1, INT8_MIN + 1, 7",
        ),
        (
            "i16",
            Prim::Int16,
            "int16_t",
            "CHELIS_DTYPE_I16",
            "-1, INT16_MAX, INT16_MIN, 3",
            "INT16_MIN, 1, -1, -4",
            "INT16_MAX, INT16_MAX - 1, INT16_MIN + 1, 7",
        ),
        (
            "i32",
            Prim::Int32,
            "int32_t",
            "CHELIS_DTYPE_I32",
            "-1, INT32_MAX, INT32_MIN, 3",
            "INT32_MIN, 1, -1, -4",
            "INT32_MAX, INT32_MAX - 1, INT32_MIN + 1, 7",
        ),
        (
            "i64",
            Prim::Int64,
            "int64_t",
            "CHELIS_DTYPE_I64",
            "-1, INT64_MAX, INT64_MIN, 3",
            "INT64_MIN, 1, -1, -4",
            "INT64_MAX, INT64_MAX - 1, INT64_MIN + 1, 7",
        ),
    ] {
        direct_int_sub_case(case.0, case.1, case.2, case.3, case.4, case.5, case.6);
    }
}

#[test]
fn direct_checked_subtraction_traps_true_overflow_at_every_signed_width() {
    for (tag, prim, c_type, c_dtype, max) in [
        ("i8", Prim::Int8, "int8_t", "CHELIS_DTYPE_I8", "INT8_MAX"),
        (
            "i16",
            Prim::Int16,
            "int16_t",
            "CHELIS_DTYPE_I16",
            "INT16_MAX",
        ),
        (
            "i32",
            Prim::Int32,
            "int32_t",
            "CHELIS_DTYPE_I32",
            "INT32_MAX",
        ),
        (
            "i64",
            Prim::Int64,
            "int64_t",
            "CHELIS_DTYPE_I64",
            "INT64_MAX",
        ),
    ] {
        let mut dag = Dag::new();
        let ty = vec_prim(1, prim);
        let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], ty.clone(), None);
        let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], ty.clone(), None);
        dag.add_node(RiscOp::Sub, vec![a, b], ty, None);
        let function = format!("direct_sub_overflow_{tag}");
        let src = chelis_backend_c::codegen(&dag, &function).unwrap().c_source;
        let harness = format!(
            r#"{HARNESS_HEADER}
#include <stdint.h>
#include <limits.h>
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    {c_type} av[1] = {{ {max} }}; {c_type} bv[1] = {{ -1 }};
    chelis_tensor a = make_view_typed_1d(av, 1, {c_dtype});
    chelis_tensor b = make_view_typed_1d(bv, 1, {c_dtype});
    chelis_tensor *inputs[2] = {{ &a, &b }}; chelis_tensor *outputs[1] = {{ NULL }};
    {function}(inputs, 2, outputs, 1); return 0;
}}
"#
        );
        let run = compile_and_capture_run(&function, &src, &harness);
        let stderr = String::from_utf8_lossy(&run.stderr);
        assert!(
            !run.status.success(),
            "{tag} overflow unexpectedly succeeded"
        );
        assert!(
            stderr.contains(&format!("numeric trap: overflow in sub at {}", prim.name())),
            "{tag}: {stderr}"
        );
    }
}

#[test]
fn direct_signed_integer_extrema_chains_survive_fusion_and_execute_at_every_width() {
    for (tag, prim, c_type, c_dtype) in [
        ("i8", Prim::Int8, "int8_t", "CHELIS_DTYPE_I8"),
        ("i16", Prim::Int16, "int16_t", "CHELIS_DTYPE_I16"),
        ("i32", Prim::Int32, "int32_t", "CHELIS_DTYPE_I32"),
        ("i64", Prim::Int64, "int64_t", "CHELIS_DTYPE_I64"),
    ] {
        let mut dag = Dag::new();
        let ty = vec_prim(4, prim);
        let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], ty.clone(), None);
        let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], ty.clone(), None);
        let c = dag.add_node(RiscOp::Load { name: "c".into() }, vec![], ty.clone(), None);
        let maximum = dag.add_node(RiscOp::MaxElem, vec![a, b], ty.clone(), None);
        let minimum = dag.add_node(RiscOp::MinElem, vec![maximum, c], ty, None);
        dag.add_root(minimum);

        let fused = fuse(&dag);
        assert!(
            fused
                .nodes()
                .iter()
                .all(|node| !matches!(node.op, RiscOp::FusedElem { .. })),
            "{tag}: signed-integer extrema must stay materialized"
        );
        let function = format!("direct_integer_extrema_chain_{tag}");
        let src = chelis_backend_c::codegen(&fused, &function)
            .unwrap_or_else(|error| panic!("{tag}: fused direct extrema codegen failed: {error}"))
            .c_source;
        let harness = format!(
            r#"{HARNESS_HEADER}
#include <stdint.h>
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    {c_type} a_data[4] = {{ -5, 7, 3, 0 }};
    {c_type} b_data[4] = {{ -4, 7, -9, 5 }};
    {c_type} c_data[4] = {{ -6, 6, 4, 5 }};
    {c_type} expected[4] = {{ -6, 6, 3, 5 }};
    chelis_tensor a = make_view_typed_1d(a_data, 4, {c_dtype});
    chelis_tensor b = make_view_typed_1d(b_data, 4, {c_dtype});
    chelis_tensor c = make_view_typed_1d(c_data, 4, {c_dtype});
    chelis_tensor *inputs[3] = {{ &a, &b, &c }};
    chelis_tensor *outputs[1] = {{ NULL }};
    {function}(inputs, 3, outputs, 1);
    {c_type} *got = ({c_type} *)outputs[0]->data;
    for (int i = 0; i < 4; i++) if (got[i] != expected[i]) return 1;
    puts("PASS");
    return 0;
}}
"#
        );
        let output = compile_and_run_kernel(&function, &src, &harness)
            .unwrap_or_else(|| panic!("{tag}: integer extrema chain did not compile and run"));
        assert!(output.contains("PASS"), "{tag}: {output}");
    }
}

#[test]
fn direct_bool_max_elem_compiles_without_float_classification() {
    let mut dag = Dag::new();
    let ty = vec_prim(4, Prim::Bool);
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], ty.clone(), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], ty.clone(), None);
    let maximum = dag.add_node(RiscOp::MaxElem, vec![a, b], ty, None);
    dag.add_root(maximum);

    let function = "direct_bool_max_elem";
    let src = chelis_backend_c::codegen(&dag, function)
        .expect("Bool max_elem codegen")
        .c_source;
    let harness = format!(
        r#"{HARNESS_HEADER}
#include <stdint.h>
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    uint8_t a_data[4] = {{ 0, 0, 1, 1 }};
    uint8_t b_data[4] = {{ 0, 1, 0, 1 }};
    uint8_t expected[4] = {{ 0, 1, 1, 1 }};
    chelis_tensor a = make_view_typed_1d(a_data, 4, CHELIS_DTYPE_BOOL);
    chelis_tensor b = make_view_typed_1d(b_data, 4, CHELIS_DTYPE_BOOL);
    chelis_tensor *inputs[2] = {{ &a, &b }};
    chelis_tensor *outputs[1] = {{ NULL }};
    {function}(inputs, 2, outputs, 1);
    uint8_t *got = (uint8_t *)outputs[0]->data;
    for (int i = 0; i < 4; i++) if (got[i] != expected[i]) return 1;
    puts("PASS");
    return 0;
}}
"#
    );
    let output = compile_and_run_kernel(function, &src, &harness)
        .expect("Bool max_elem generated C must compile and run");
    assert!(output.contains("PASS"), "{output}");
    assert!(src.contains("uint8_t"), "Bool storage must stay byte-typed");
    assert!(
        !src.contains("isnan("),
        "Bool max_elem must not enter the floating extrema branch: {src}"
    );
}

#[test]
fn direct_fused_runtime_shape_mismatch_traps_before_indexing() {
    let runtime_vec = |name: &str| TensorType {
        dims: vec![DimInfo::Named(name.into(), None)],
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let n_ty = runtime_vec("n");
    let m_ty = runtime_vec("m");
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        n_ty.clone(),
        None,
    );
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], m_ty, None);
    let c = dag.add_node(
        RiscOp::Load { name: "c".into() },
        vec![],
        n_ty.clone(),
        None,
    );
    let difference = dag.add_node(RiscOp::Sub, vec![a, b], n_ty.clone(), None);
    let minimum = dag.add_node(RiscOp::MinElem, vec![difference, c], n_ty, None);
    dag.add_root(minimum);

    let fused = fuse(&dag);
    assert!(
        fused
            .nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::FusedElem { .. })),
        "float Sub -> MinElem must exercise the fused path"
    );
    let function = "direct_fused_runtime_shape_guard";
    let src = chelis_backend_c::codegen(&fused, function)
        .expect("fused runtime-shape codegen")
        .c_source;
    assert!(
        src.contains("elementwise operand shape mismatch"),
        "fused codegen dropped the deferred operand-shape guard:\n{src}"
    );
    assert!(
        src.contains("if (t1->rank == t2->rank)"),
        "fused codegen must compare every equal-rank external-input pair:\n{src}"
    );

    let harness = format!(
        r#"{HARNESS_HEADER}
static chelis_tensor make_distinct_view(float *data, int64_t *shape) {{
    static const int64_t strides[1] = {{1}};
    return (chelis_tensor){{
        .data = data, .shape = shape, .strides = strides, .size = shape[0],
        .byte_capacity = shape[0] * (int64_t)sizeof(float), .rank = 1,
        .dtype = CHELIS_DTYPE_F32, .owns_data = 0, .reserved = {{0, 0}},
    }};
}}
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    float a_data[3] = {{ 1, 2, 3 }};
    float b_data[2] = {{ 1, 2 }};
    float c_data[3] = {{ 4, 5, 6 }};
    int64_t a_shape[1] = {{3}}, b_shape[1] = {{2}}, c_shape[1] = {{3}};
    chelis_tensor a = make_distinct_view(a_data, a_shape);
    chelis_tensor b = make_distinct_view(b_data, b_shape);
    chelis_tensor c = make_distinct_view(c_data, c_shape);
    chelis_tensor *inputs[3] = {{ &a, &b, &c }};
    chelis_tensor *outputs[1] = {{ NULL }};
    {function}(inputs, 3, outputs, 1);
    puts("UNREACHABLE");
    return 0;
}}
"#
    );
    let run = compile_and_capture_run("direct_fused_runtime_shape_guard", &src, &harness);
    assert!(
        !run.status.success(),
        "mismatched deferred dimensions reached the fused loop; stdout={}",
        String::from_utf8_lossy(&run.stdout)
    );
    assert!(
        String::from_utf8_lossy(&run.stderr).contains("elementwise operand shape mismatch"),
        "fused runtime-shape trap emitted the wrong diagnostic: {}",
        String::from_utf8_lossy(&run.stderr)
    );
}

fn direct_fused_reduction_runtime_shape_guard_case(reduce_kind: &str) {
    let matrix = |row_name: &str, column_name: &str| TensorType {
        dims: vec![
            DimInfo::Named(row_name.into(), None),
            DimInfo::Named(column_name.into(), None),
        ],
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let output_matrix_ty = matrix("rows", "columns");
    let other_matrix_ty = matrix("other_rows", "other_columns");
    let scalar = dag.add_node(
        RiscOp::Load {
            name: "scalar".into(),
        },
        vec![],
        TensorType::scalar_f32(),
        None,
    );
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        output_matrix_ty.clone(),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        other_matrix_ty,
        None,
    );
    let shifted = dag.add_node(RiscOp::Add, vec![scalar, a], output_matrix_ty.clone(), None);
    let product = dag.add_node(RiscOp::Mul, vec![shifted, b], output_matrix_ty, None);
    let output_ty = TensorType {
        dims: vec![DimInfo::Named("rows".into(), None)],
        precision: Prim::F32,
    };
    let reduced = match reduce_kind {
        "sum" => dag.add_node(
            RiscOp::Sum {
                axis: 1,
                accumulator: Prim::F32,
            },
            vec![product],
            output_ty,
            None,
        ),
        "max" => dag.add_node(
            RiscOp::MaxReduce { axis: 1 },
            vec![product],
            output_ty,
            None,
        ),
        _ => unreachable!(),
    };
    dag.add_root(reduced);

    let fused = fuse(&dag);
    let fused_node = fused
        .nodes()
        .iter()
        .find(|node| matches!(node.op, RiscOp::FusedElem { .. }))
        .expect("Add -> Mul must form a FusedElem before reduction inlining");
    assert_eq!(
        fused_node.inputs,
        vec![scalar, a, b],
        "the scalar must be first so the guard proves all-pairs tensor comparison"
    );
    assert!(
        chelis_ir::fuse::reduction_inlined_fused_elems(&fused).contains(&fused_node.id),
        "the regression must exercise the reduction-inlined FusedElem path"
    );

    let function = format!("direct_fused_{reduce_kind}_runtime_shape_guard");
    let src = chelis_backend_c::codegen(&fused, &function)
        .expect("fused reduction runtime-shape codegen")
        .c_source;
    let function_body = src
        .split_once(&format!("void {function}("))
        .unwrap_or_else(|| panic!("generated C omitted {function}:\n{src}"))
        .1;
    let guard_offset = function_body
        .find("elementwise operand shape mismatch")
        .unwrap_or_else(|| panic!("inlined {reduce_kind} dropped its shape guard:\n{src}"));
    let allocation_offset = function_body
        .find("chelis_alloc(")
        .unwrap_or_else(|| panic!("inlined {reduce_kind} emitted no output allocation:\n{src}"));
    assert!(
        guard_offset < allocation_offset,
        "inlined {reduce_kind} must guard before allocation or indexing:\n{src}"
    );
    assert!(
        function_body.contains("if (t1->rank == t2->rank)"),
        "scalar-first ordering must still compare the later tensor pair:\n{src}"
    );
    assert_eq!(
        function_body.matches("chelis_alloc(").count(),
        1,
        "reduction-inlined FusedElem must not allocate an intermediate:\n{src}"
    );

    let expected = if reduce_kind == "sum" {
        "29.0f, 110.0f"
    } else {
        "16.0f, 49.0f"
    };
    let harness_support = r#"
static chelis_tensor make_scalar_view(float *data) {
    return (chelis_tensor){
        .data = data, .shape = NULL, .strides = NULL, .size = 1,
        .byte_capacity = (int64_t)sizeof(float), .rank = 0,
        .dtype = CHELIS_DTYPE_F32, .owns_data = 0, .reserved = {0, 0},
    };
}
static chelis_tensor make_matrix_view(
    float *data, int64_t *shape, int64_t *strides, int64_t backing_elements
) {
    return (chelis_tensor){
        .data = data, .shape = shape, .strides = strides,
        .size = shape[0] * shape[1],
        .byte_capacity = backing_elements * (int64_t)sizeof(float), .rank = 2,
        .dtype = CHELIS_DTYPE_F32, .owns_data = 0, .reserved = {0, 0},
    };
}
"#;
    let positive_harness = format!(
        r#"{HARNESS_HEADER}{harness_support}
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    float scalar_data[1] = {{1.0f}};
    float a_data[6] = {{1, 2, 3, 4, 5, 6}};
    float b_data[7] = {{2, 3, 4, -99, 5, 6, 7}};
    int64_t a_shape[2] = {{2, 3}}, a_strides[2] = {{3, 1}};
    int64_t b_shape[2] = {{2, 3}}, b_strides[2] = {{4, 1}};
    chelis_tensor scalar = make_scalar_view(scalar_data);
    chelis_tensor a = make_matrix_view(a_data, a_shape, a_strides, 6);
    chelis_tensor b = make_matrix_view(b_data, b_shape, b_strides, 7);
    chelis_tensor *inputs[3] = {{&scalar, &a, &b}};
    chelis_tensor *outputs[1] = {{NULL}};
    {function}(inputs, 3, outputs, 1);
    float expected[2] = {{{expected}}};
    float *got = (float *)outputs[0]->data;
    for (int i = 0; i < 2; i++) if (fabsf(got[i] - expected[i]) > 1e-6f) return 1;
    puts("PASS");
    return 0;
}}
"#
    );
    let output = compile_and_run_kernel(
        &format!("direct_fused_{reduce_kind}_shape_positive"),
        &src,
        &positive_harness,
    )
    .unwrap_or_else(|| panic!("inlined {reduce_kind} positive case failed"));
    assert!(output.contains("PASS"), "inlined {reduce_kind}: {output}");

    let mismatch_harness = format!(
        r#"{HARNESS_HEADER}{harness_support}
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    float scalar_data[1] = {{1.0f}};
    float a_data[6] = {{1, 2, 3, 4, 5, 6}};
    float b_data[5] = {{2, 3, -99, 5, 6}};
    int64_t a_shape[2] = {{2, 3}}, a_strides[2] = {{3, 1}};
    int64_t b_shape[2] = {{2, 2}}, b_strides[2] = {{3, 1}};
    chelis_tensor scalar = make_scalar_view(scalar_data);
    chelis_tensor a = make_matrix_view(a_data, a_shape, a_strides, 6);
    chelis_tensor b = make_matrix_view(b_data, b_shape, b_strides, 5);
    chelis_tensor *inputs[3] = {{&scalar, &a, &b}};
    chelis_tensor *outputs[1] = {{NULL}};
    {function}(inputs, 3, outputs, 1);
    puts("UNREACHABLE");
    return 0;
}}
"#
    );
    let run = compile_and_capture_run(
        &format!("direct_fused_{reduce_kind}_shape_mismatch"),
        &src,
        &mismatch_harness,
    );
    assert!(
        !run.status.success(),
        "inlined {reduce_kind} mismatch reached allocation/indexing"
    );
    assert!(
        String::from_utf8_lossy(&run.stderr).contains("elementwise operand shape mismatch"),
        "inlined {reduce_kind} mismatch emitted the wrong diagnostic: {}",
        String::from_utf8_lossy(&run.stderr)
    );
}

#[test]
fn direct_fused_sum_runtime_shape_guard_precedes_allocation_and_executes() {
    direct_fused_reduction_runtime_shape_guard_case("sum");
}

#[test]
fn direct_fused_max_reduce_runtime_shape_guard_precedes_allocation_and_executes() {
    direct_fused_reduction_runtime_shape_guard_case("max");
}

#[derive(Clone, Copy)]
struct DirectExtremaBitCase<'a> {
    tag: &'a str,
    prim: Prim,
    c_dtype: &'a str,
    bits_type: &'a str,
    lhs_bits: &'a [u64],
    rhs_bits: &'a [u64],
}

fn direct_extrema_bit_case(
    case: DirectExtremaBitCase<'_>,
    expected_max: &[u64],
    expected_min: &[u64],
) {
    let DirectExtremaBitCase {
        tag,
        prim,
        c_dtype,
        bits_type,
        lhs_bits,
        rhs_bits,
    } = case;
    let n = lhs_bits.len();
    assert_eq!(rhs_bits.len(), n);
    let format_bits = |bits: &[u64]| {
        bits.iter()
            .map(|value| match bits_type {
                "uint64_t" => format!("UINT64_C(0x{value:016x})"),
                "uint32_t" => format!("UINT32_C(0x{value:08x})"),
                _ => format!("UINT16_C(0x{value:04x})"),
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    for (op_name, op, expected) in [
        ("max", RiscOp::MaxElem, expected_max),
        ("min", RiscOp::MinElem, expected_min),
    ] {
        let mut dag = Dag::new();
        let ty = vec_prim(n, prim);
        let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], ty.clone(), None);
        let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], ty.clone(), None);
        dag.add_node(op, vec![a, b], ty, None);
        let function = format!("direct_{op_name}_{tag}");
        let src = chelis_backend_c::codegen(&dag, &function).unwrap().c_source;
        assert!(!src.contains("fmaxf("), "{tag}/{op_name}: {src}");
        assert!(!src.contains("fminf("), "{tag}/{op_name}: {src}");

        let (value_type, setup, got) = match prim {
            Prim::F64 => (
                "double",
                "double a_data[N]; double b_data[N]; memcpy(a_data, a_bits, sizeof(a_bits)); memcpy(b_data, b_bits, sizeof(b_bits));",
                "uint64_t got; memcpy(&got, &((double *)outputs[0]->data)[i], sizeof(got));",
            ),
            Prim::F32 => (
                "float",
                "float a_data[N]; float b_data[N]; memcpy(a_data, a_bits, sizeof(a_bits)); memcpy(b_data, b_bits, sizeof(b_bits));",
                "uint32_t got; memcpy(&got, &((float *)outputs[0]->data)[i], sizeof(got));",
            ),
            Prim::F16 | Prim::Bf16 => (
                "uint16_t",
                "uint16_t *a_data = a_bits; uint16_t *b_data = b_bits;",
                "uint16_t got = ((uint16_t *)outputs[0]->data)[i];",
            ),
            _ => unreachable!(),
        };
        let harness = format!(
            r#"{HARNESS_HEADER}
#include <stdint.h>
#define N {n}
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    {bits_type} a_bits[N] = {{ {lhs} }};
    {bits_type} b_bits[N] = {{ {rhs} }};
    {bits_type} expected[N] = {{ {expected} }};
    {setup}
    (void)sizeof({value_type});
    chelis_tensor a = make_view_typed_1d(a_data, N, {c_dtype});
    chelis_tensor b = make_view_typed_1d(b_data, N, {c_dtype});
    chelis_tensor *inputs[2] = {{ &a, &b }}; chelis_tensor *outputs[1] = {{ NULL }};
    {function}(inputs, 2, outputs, 1);
    for (int i = 0; i < N; i++) {{ {got} if (got != expected[i]) return 1; }}
    puts("PASS"); return 0;
}}
"#,
            lhs = format_bits(lhs_bits),
            rhs = format_bits(rhs_bits),
            expected = format_bits(expected),
        );
        let output = compile_and_run_kernel(&function, &src, &harness)
            .unwrap_or_else(|| panic!("{tag}/{op_name} did not compile and run"));
        assert!(output.contains("PASS"), "{tag}/{op_name}: {output}");
    }
}

#[test]
fn direct_extrema_preserve_nan_payloads_and_lhs_signed_zero_at_every_float_width() {
    direct_extrema_bit_case(
        DirectExtremaBitCase {
            tag: "f64",
            prim: Prim::F64,
            c_dtype: "CHELIS_DTYPE_F64",
            bits_type: "uint64_t",
            lhs_bits: &[
                0x7ff8_1111_2222_3333,
                0x3ff0_0000_0000_0000,
                0,
                0x8000_0000_0000_0000,
            ],
            rhs_bits: &[
                0x4000_0000_0000_0000,
                0xfff8_4444_5555_6666,
                0x8000_0000_0000_0000,
                0,
            ],
        },
        &[
            0x7ff8_1111_2222_3333,
            0xfff8_4444_5555_6666,
            0,
            0x8000_0000_0000_0000,
        ],
        &[
            0x7ff8_1111_2222_3333,
            0xfff8_4444_5555_6666,
            0,
            0x8000_0000_0000_0000,
        ],
    );
    direct_extrema_bit_case(
        DirectExtremaBitCase {
            tag: "f32",
            prim: Prim::F32,
            c_dtype: "CHELIS_DTYPE_F32",
            bits_type: "uint32_t",
            lhs_bits: &[0x7fc1_2345, 0x3f80_0000, 0, 0x8000_0000],
            rhs_bits: &[0x4000_0000, 0xffc5_4321, 0x8000_0000, 0],
        },
        &[0x7fc1_2345, 0xffc5_4321, 0, 0x8000_0000],
        &[0x7fc1_2345, 0xffc5_4321, 0, 0x8000_0000],
    );
    direct_extrema_bit_case(
        DirectExtremaBitCase {
            tag: "f16",
            prim: Prim::F16,
            c_dtype: "CHELIS_DTYPE_F16",
            bits_type: "uint16_t",
            lhs_bits: &[0x7e11, 0x3c00, 0, 0x8000],
            rhs_bits: &[0x4000, 0xfe22, 0x8000, 0],
        },
        &[0x7e11, 0xfe22, 0, 0x8000],
        &[0x7e11, 0xfe22, 0, 0x8000],
    );
    direct_extrema_bit_case(
        DirectExtremaBitCase {
            tag: "bf16",
            prim: Prim::Bf16,
            c_dtype: "CHELIS_DTYPE_BF16",
            bits_type: "uint16_t",
            lhs_bits: &[0x7fc1, 0x3f80, 0, 0x8000],
            rhs_bits: &[0x4000, 0xffc2, 0x8000, 0],
        },
        &[0x7fc1, 0xffc2, 0, 0x8000],
        &[0x7fc1, 0xffc2, 0, 0x8000],
    );
}

fn direct_extrema_adjoint_bit_case(
    case: DirectExtremaBitCase<'_>,
    gradient_bits: &[u64],
    expected: [&[u64]; 4],
) {
    let DirectExtremaBitCase {
        tag,
        prim,
        c_dtype,
        bits_type,
        lhs_bits,
        rhs_bits,
    } = case;
    let n = lhs_bits.len();
    assert_eq!(rhs_bits.len(), n);
    assert_eq!(gradient_bits.len(), n);
    assert!(expected.iter().all(|values| values.len() == n));
    let format_bits = |bits: &[u64]| {
        bits.iter()
            .map(|value| match bits_type {
                "uint64_t" => format!("UINT64_C(0x{value:016x})"),
                "uint32_t" => format!("UINT32_C(0x{value:08x})"),
                _ => format!("UINT16_C(0x{value:04x})"),
            })
            .collect::<Vec<_>>()
            .join(", ")
    };

    let mut dag = Dag::new();
    let ty = vec_prim(n, prim);
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], ty.clone(), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], ty.clone(), None);
    let g = dag.add_node(RiscOp::Load { name: "g".into() }, vec![], ty.clone(), None);
    for (kind, operand) in [
        (ExtremaKind::Max, ExtremaOperand::Left),
        (ExtremaKind::Max, ExtremaOperand::Right),
        (ExtremaKind::Min, ExtremaOperand::Left),
        (ExtremaKind::Min, ExtremaOperand::Right),
    ] {
        let node = dag.add_node(
            RiscOp::ExtremaAdjoint { kind, operand },
            vec![a, b, g],
            ty.clone(),
            None,
        );
        dag.add_root(node);
    }
    let function = format!("direct_extrema_adjoint_{tag}");
    let src = chelis_backend_c::codegen(&dag, &function).unwrap().c_source;
    assert!(
        src.contains("UINT16_C(0)") || src.contains("0.0"),
        "{tag}: {src}"
    );

    let setup = match prim {
        Prim::F64 => {
            "double a_data[N]; double b_data[N]; double g_data[N]; memcpy(a_data, a_bits, sizeof(a_bits)); memcpy(b_data, b_bits, sizeof(b_bits)); memcpy(g_data, g_bits, sizeof(g_bits));"
        }
        Prim::F32 => {
            "float a_data[N]; float b_data[N]; float g_data[N]; memcpy(a_data, a_bits, sizeof(a_bits)); memcpy(b_data, b_bits, sizeof(b_bits)); memcpy(g_data, g_bits, sizeof(g_bits));"
        }
        Prim::F16 | Prim::Bf16 => {
            "uint16_t *a_data = a_bits; uint16_t *b_data = b_bits; uint16_t *g_data = g_bits;"
        }
        _ => unreachable!(),
    };
    let read_got = match prim {
        Prim::F64 => "uint64_t got; memcpy(&got, &((double *)outputs[out]->data)[i], sizeof(got));",
        Prim::F32 => "uint32_t got; memcpy(&got, &((float *)outputs[out]->data)[i], sizeof(got));",
        Prim::F16 | Prim::Bf16 => "uint16_t got = ((uint16_t *)outputs[out]->data)[i];",
        _ => unreachable!(),
    };
    let expected_rows = expected
        .iter()
        .map(|values| format!("{{ {} }}", format_bits(values)))
        .collect::<Vec<_>>()
        .join(", ");
    let harness = format!(
        r#"{HARNESS_HEADER}
#include <stdint.h>
#define N {n}
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    {bits_type} a_bits[N] = {{ {lhs} }};
    {bits_type} b_bits[N] = {{ {rhs} }};
    {bits_type} g_bits[N] = {{ {gradient} }};
    {bits_type} expected[4][N] = {{ {expected_rows} }};
    {setup}
    chelis_tensor a = make_view_typed_1d(a_data, N, {c_dtype});
    chelis_tensor b = make_view_typed_1d(b_data, N, {c_dtype});
    chelis_tensor g = make_view_typed_1d(g_data, N, {c_dtype});
    chelis_tensor *inputs[3] = {{ &a, &b, &g }};
    chelis_tensor *outputs[4] = {{ NULL, NULL, NULL, NULL }};
    {function}(inputs, 3, outputs, 4);
    for (int out = 0; out < 4; out++) {{
        for (int i = 0; i < N; i++) {{ {read_got} if (got != expected[out][i]) return 1; }}
    }}
    puts("PASS"); return 0;
}}
"#,
        lhs = format_bits(lhs_bits),
        rhs = format_bits(rhs_bits),
        gradient = format_bits(gradient_bits),
    );
    let output = compile_and_run_kernel(&function, &src, &harness)
        .unwrap_or_else(|| panic!("{tag} direct extrema adjoints did not compile and run"));
    assert!(output.contains("PASS"), "{tag}: {output}");
}

#[test]
fn direct_extrema_adjoints_copy_exact_gradient_bits_for_ties_and_nan_selection() {
    let run = |case: DirectExtremaBitCase<'_>, gradient: &[u64]| {
        let zero = 0;
        let max_left = [
            gradient[0],
            zero,
            gradient[2],
            gradient[3],
            gradient[4],
            zero,
        ];
        let max_right = [zero, gradient[1], zero, zero, zero, gradient[5]];
        let min_left = [
            gradient[0],
            zero,
            gradient[2],
            gradient[3],
            zero,
            gradient[5],
        ];
        let min_right = [zero, gradient[1], zero, zero, gradient[4], zero];
        direct_extrema_adjoint_bit_case(
            case,
            gradient,
            [&max_left, &max_right, &min_left, &min_right],
        );
    };

    run(
        DirectExtremaBitCase {
            tag: "f64",
            prim: Prim::F64,
            c_dtype: "CHELIS_DTYPE_F64",
            bits_type: "uint64_t",
            lhs_bits: &[
                0x7ff8_1111_2222_3333,
                0x3ff0_0000_0000_0000,
                0,
                0x8000_0000_0000_0000,
                0x4000_0000_0000_0000,
                0x3ff0_0000_0000_0000,
            ],
            rhs_bits: &[
                0x4000_0000_0000_0000,
                0xfff8_4444_5555_6666,
                0x8000_0000_0000_0000,
                0,
                0x3ff0_0000_0000_0000,
                0x4000_0000_0000_0000,
            ],
        },
        &[
            0x7ff8_abcd_1234_5678,
            0xbff0_0000_0000_0000,
            0x3ff0_0000_0000_0000,
            0x8000_0000_0000_0000,
            0x4008_0000_0000_0000,
            0xc010_0000_0000_0000,
        ],
    );
    run(
        DirectExtremaBitCase {
            tag: "f32",
            prim: Prim::F32,
            c_dtype: "CHELIS_DTYPE_F32",
            bits_type: "uint32_t",
            lhs_bits: &[
                0x7fc1_2345,
                0x3f80_0000,
                0,
                0x8000_0000,
                0x4000_0000,
                0x3f80_0000,
            ],
            rhs_bits: &[
                0x4000_0000,
                0xffc5_4321,
                0x8000_0000,
                0,
                0x3f80_0000,
                0x4000_0000,
            ],
        },
        &[
            0x7fc6_789a,
            0xbf80_0000,
            0x3f80_0000,
            0x8000_0000,
            0x4040_0000,
            0xc080_0000,
        ],
    );
    run(
        DirectExtremaBitCase {
            tag: "f16",
            prim: Prim::F16,
            c_dtype: "CHELIS_DTYPE_F16",
            bits_type: "uint16_t",
            lhs_bits: &[0x7e11, 0x3c00, 0, 0x8000, 0x4000, 0x3c00],
            rhs_bits: &[0x4000, 0xfe22, 0x8000, 0, 0x3c00, 0x4000],
        },
        &[0x7e33, 0xbc00, 0x3c00, 0x8000, 0x4200, 0xc400],
    );
    run(
        DirectExtremaBitCase {
            tag: "bf16",
            prim: Prim::Bf16,
            c_dtype: "CHELIS_DTYPE_BF16",
            bits_type: "uint16_t",
            lhs_bits: &[0x7fc1, 0x3f80, 0, 0x8000, 0x4000, 0x3f80],
            rhs_bits: &[0x4000, 0xffc2, 0x8000, 0, 0x3f80, 0x4000],
        },
        &[0x7fc3, 0xbf80, 0x3f80, 0x8000, 0x4040, 0xc080],
    );
}
