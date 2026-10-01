//! Direct, manually composed, and wrapped matmul retain the same arithmetic.
//!
//! [05-OP-30] fixes the adjacent-pair reduction tree. C must preserve that
//! tree through user-def boundaries; vendor GEMM is not an equivalent
//! replacement. The fixtures check emitted selection/storage and execute
//! independent positive and cancellation results for every calling form.
//! HIP dispatch is measured separately below and does not certify its
//! arithmetic conformance.

mod common;
use common::{authored_c_symbol, build_and_run, parse_tensor_data};

use std::fs;
use std::process::Command;

use assert_cmd::cargo::CommandCargoExt;
use tempfile::tempdir;

struct Counts {
    blas: usize,
    allocs: usize,
    reduction_scratch_allocs: usize,
    fused: usize,
    user_helper_defs: usize,
    /// Total bytes summed across every `chelis_alloc(N, (int64_t[]){...},
    /// CHELIS_DTYPE_F32)` call. This measures
    /// generated tensor-buffer submissions, excluding input storage,
    /// metadata, and reduction scratch; it is not peak process memory.
    total_alloc_bytes: usize,
}

fn build_and_count(source: &str, name: &str) -> Counts {
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}_out"));
    fs::write(&src_path, source).expect("write source");

    let status = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args([
            "build",
            "--emit-c",
            src_path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .status()
        .expect("chelis build should run");
    assert!(status.success(), "chelis build failed for {name}");

    let c_path = out_dir.join(format!("{name}.c"));
    let c = fs::read_to_string(&c_path).expect("read generated c");
    let blas = c.matches("cblas_sgemm").count() + c.matches("chelis_blas_matmul").count();
    let allocs = c.matches("chelis_alloc(").count();
    let reduction_scratch_allocs = c
        .lines()
        .filter(|line| {
            line.contains("chelis_tensor *__sum_scratch_")
                && line.contains("= chelis_alloc(1, &__sum_n_")
        })
        .count();
    let fused = c.matches("parallel for simd").count();
    let user_helper_defs = c
        .matches(&format!(
            "static void {}__tensor_",
            authored_c_symbol("my_mm")
        ))
        .count();

    // Sum bytes across every chelis_alloc(N, (int64_t[]){...}, CHELIS_DTYPE_F32) call.
    let mut total_alloc_bytes = 0usize;
    let mut idx = 0;
    while let Some(start) = c[idx..].find("chelis_alloc(") {
        let pos = idx + start;
        let after = &c[pos..];
        let window = &after[..(256.min(after.len()))];
        if let Some(brace_open) = window.find("(int64_t[]){") {
            let dims_start = brace_open + "(int64_t[]){".len();
            if let Some(brace_close) = window[dims_start..].find('}') {
                let dims_text = &window[dims_start..dims_start + brace_close];
                let mut const_factor: usize = 1;
                let mut ok = true;
                for tok in dims_text.split(',') {
                    let t = tok.trim();
                    if t.is_empty() {
                        continue;
                    }
                    if let Ok(n) = t.parse::<usize>() {
                        const_factor = const_factor.saturating_mul(n);
                    } else {
                        // Non-literal dim — skip this alloc.
                        ok = false;
                        break;
                    }
                }
                if ok {
                    total_alloc_bytes += const_factor * 4; // CHELIS_DTYPE_F32 only in this test
                }
            }
        } else if window.starts_with("chelis_alloc(0, NULL,") {
            total_alloc_bytes += 4;
        }
        idx = pos + "chelis_alloc(".len();
    }

    Counts {
        blas,
        allocs,
        reduction_scratch_allocs,
        fused,
        user_helper_defs,
        total_alloc_bytes,
    }
}

fn semantic_gap_sources() -> (&'static str, &'static str, &'static str, &'static str) {
    let direct = "def f(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                  -> tensor[8, 4, f32] = matmul(a, b)\n";
    let inline_manual = "def f(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                         -> tensor[8, 4, f32] = {\n  \
                           ae = insert(a, 2, 4i64)\n  \
                           be = insert(b, 0, 8i64)\n  \
                           sum(mul(ae, be), 1)\n\
                         }\n";
    let user_def_builtin = "def my_mm(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                            -> tensor[8, 4, f32] = matmul(a, b)\n\
                            def f(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                            -> tensor[8, 4, f32] = my_mm(a, b)\n";
    let user_def_manual = "def my_mm(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                           -> tensor[8, 4, f32] = {\n  \
                             ae = insert(a, 2, 4i64)\n  \
                             be = insert(b, 0, 8i64)\n  \
                             sum(mul(ae, be), 1)\n\
                           }\n\
                           def f(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                           -> tensor[8, 4, f32] = my_mm(a, b)\n";

    (direct, inline_manual, user_def_builtin, user_def_manual)
}

fn nested_matmul_source() -> &'static str {
    "def my_mm(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
     -> tensor[8, 4, f32] = matmul(a, b)\n\
     def wrap_mm(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
     -> tensor[8, 4, f32] = my_mm(a, b)\n\
     def f(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
     -> tensor[8, 4, f32] = wrap_mm(a, b)\n"
}

#[test]
fn inline_matmul_forms_preserve_primitive_arithmetic_and_storage() {
    let (direct, inline_manual, _, _) = semantic_gap_sources();
    let direct_c = build_and_count(direct, "sgap_direct");
    let inline_c = build_and_count(inline_manual, "sgap_inline_manual");
    for counts in [&direct_c, &inline_c] {
        assert_eq!(
            counts.blas, 0,
            "vendor GEMM changes the specified reduction tree"
        );
        // Two inserted operands and their product each have shape [8,16,4];
        // the reduced output is [8,4]. The fifth allocation site owns the
        // per-output 16-element scratch via its checked dynamic leaf count.
        // The literal-shape byte total below measures the four DAG submissions.
        assert_eq!(counts.allocs, 5);
        assert_eq!(counts.reduction_scratch_allocs, 1);
        assert_eq!(counts.total_alloc_bytes, (3 * 8 * 16 * 4 + 8 * 4) * 4);
    }
    assert_eq!(direct_c.total_alloc_bytes, inline_c.total_alloc_bytes);
    eprintln!(
        "direct/inline fused loops: {}/{}",
        direct_c.fused, inline_c.fused
    );
}

#[test]
fn user_def_matmul_helpers_preserve_primitive_arithmetic() {
    let (_, _, user_def_builtin, user_def_manual) = semantic_gap_sources();
    for (source, name) in [
        (user_def_builtin, "sgap_user_def_builtin"),
        (user_def_manual, "sgap_user_def_manual"),
        (nested_matmul_source(), "sgap_nested_user_def"),
    ] {
        let counts = build_and_count(source, name);
        assert_eq!(
            counts.blas, 0,
            "helper summaries must not restore vendor GEMM: {name}"
        );
        assert!(
            counts.user_helper_defs >= 1,
            "the original helper must still be emitted: {name}"
        );
    }
}

#[test]
fn every_matmul_calling_form_executes_the_canonical_tree() {
    let (direct, manual, wrapped_builtin, wrapped_manual) = semantic_gap_sources();
    for (form, definitions) in [
        direct,
        manual,
        wrapped_builtin,
        wrapped_manual,
        nested_matmul_source(),
    ]
    .into_iter()
    .enumerate()
    {
        // Ordinary positive values sum to 136. Adjacent pairs of the
        // cancellation case sum to zero, whereas sequential vendor GEMM
        // and the former stride-four tree can both produce a nonzero result.
        for (case, row, expected) in [
            (
                "positive",
                (1..=16).map(|n| format!("{n}.0f32")).collect::<Vec<_>>(),
                136.0,
            ),
            (
                "cancellation",
                [
                    "1e20", "1.0", "-1e20", "1.0", "-1e20", "1.0", "1e20", "1.0", "0.0", "0.0",
                    "0.0", "0.0", "0.0", "0.0", "0.0", "0.0",
                ]
                .map(|n| format!("{n}f32"))
                .to_vec(),
                0.0,
            ),
        ] {
            let a = vec![row.join(","); 8].join(",");
            let b = vec!["1.0f32"; 16 * 4].join(",");
            let source = format!(
                "{definitions}a: tensor[8,16,f32] = reshape(to_tensor([{a}]),[8i64,16i64])\nb: tensor[16,4,f32] = reshape(to_tensor([{b}]),[16i64,4i64])\nresult = f(a,b)\n"
            );
            let dir = tempdir().unwrap();
            let path = dir.path().join("calling_form.ch");
            fs::write(&path, &source).unwrap();
            let evaluated = Command::cargo_bin("chelis")
                .unwrap()
                .env("CHELIS_STYLE_GATE_DISABLE", "1")
                .args(["eval", "--file"])
                .arg(&path)
                .output()
                .unwrap();
            assert!(
                evaluated.status.success(),
                "{}",
                String::from_utf8_lossy(&evaluated.stderr)
            );
            for actual in [
                String::from_utf8(evaluated.stdout).unwrap(),
                build_and_run(&source, &format!("calling_form_{form}_{case}")),
            ] {
                assert_eq!(
                    parse_tensor_data(&actual, "result"),
                    vec![expected; 8 * 4],
                    "form {form}, {case}"
                );
            }
        }
    }
}

// ===========================================================================
// HIP target — Gap 5 M5(a): user-def matmul helpers must emit hipBLAS on HIP
// independently of the C backend's primitive arithmetic policy. The HIP target
// inlines pure-tensor helpers at DAG-lowering time via `program_defs`, so the
// emitted HIP `.cpp` should contain `chelis_hipblas_sgemm_row_major(` for
// each of the four matmul forms.
//
// These assertions use exact-substring match on the host-side hipBLAS
// dispatch (`chelis_hipblas_sgemm_row_major(`). They are NOT `contains()` on
// a loose pattern — they pin the row-major rank-2 dispatch the HIP backend
// emits at `crates/chelis-backend-hip/src/emit.rs:1754` for a fixed 8x16 @
// 16x4 matmul. Strided-batched / batched dispatch variants are locked
// elsewhere (e.g. `crates/chelis-backend-hip/tests/perf_f1_strided_batched_default.rs`)
// and intentionally not loosened here.
// ===========================================================================

struct HipCounts {
    /// Exact-substring occurrences of `chelis_hipblas_sgemm_row_major(` in
    /// the emitted HIP `.cpp`. The HIP emitter writes both a host-side entry
    /// (`<func>(...)`) and a device-side entry (`<func>_device(...)`) for
    /// the same DAG, so a single BLAS-eligible matmul produces TWO matches.
    hipblas_row_major_calls: usize,
    /// Occurrences of `cblas_sgemm(` — must be zero on the HIP target.
    cblas_calls: usize,
    /// Whether the link flags include `-lhipblas` (i.e. the HIP backend
    /// detected hipBLAS dispatch).
    has_lhipblas: bool,
}

fn build_hip_and_count(source: &str, name: &str) -> HipCounts {
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}_out"));
    fs::write(&src_path, source).expect("write source");

    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args([
            "build",
            "--emit-c",
            src_path.to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("chelis build should run");
    assert!(
        output.status.success(),
        "chelis build --target hip failed for {name}: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let cpp_path = out_dir.join(format!("{name}_hip.cpp"));
    let cpp = fs::read_to_string(&cpp_path).expect("read generated hip cpp");
    let hipblas_row_major_calls = cpp.matches("chelis_hipblas_sgemm_row_major(").count();
    let cblas_calls = cpp.matches("cblas_sgemm(").count();
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let has_lhipblas = stdout.contains("-lhipblas");

    HipCounts {
        hipblas_row_major_calls,
        cblas_calls,
        has_lhipblas,
    }
}

#[test]
fn hip_direct_matmul_hits_hipblas() {
    let (direct, _, _, _) = semantic_gap_sources();
    let h = build_hip_and_count(direct, "hip_sgap_direct");

    // Locked: every form must hit `chelis_hipblas_sgemm_row_major(` at
    // least twice — once in the host-side entry function and once in the
    // device-side entry function (HIP backend emits both for each DAG).
    assert_eq!(
        h.hipblas_row_major_calls, 2,
        "direct `matmul(a, b)` on HIP must emit exactly two \
         `chelis_hipblas_sgemm_row_major(` calls (host entry + device \
         entry). got={}",
        h.hipblas_row_major_calls
    );
    assert_eq!(
        h.cblas_calls, 0,
        "HIP target must NOT emit `cblas_sgemm(`; got {} on direct matmul",
        h.cblas_calls
    );
    assert!(
        h.has_lhipblas,
        "HIP link flags should include `-lhipblas` when hipBLAS is dispatched"
    );
}

#[test]
fn hip_inline_matmul_hits_hipblas() {
    let (_, inline_manual, _, _) = semantic_gap_sources();
    let h = build_hip_and_count(inline_manual, "hip_sgap_inline_manual");

    assert_eq!(
        h.hipblas_row_major_calls, 2,
        "inline insert+mul+sum on HIP must emit exactly two \
         `chelis_hipblas_sgemm_row_major(` calls (the Tier 2 specializer \
         recognizes the hand-written Einstein-form matmul before HIP \
         codegen). got={}",
        h.hipblas_row_major_calls
    );
    assert_eq!(h.cblas_calls, 0);
    assert!(h.has_lhipblas);
}

#[test]
fn hip_user_def_builtin_matmul_helper_hits_hipblas() {
    let (_, _, user_def_builtin, _) = semantic_gap_sources();
    let h = build_hip_and_count(user_def_builtin, "hip_sgap_user_def_builtin");

    // This is the M5(a) target behavior: `def my_mm(a, b) = matmul(a, b);
    // def f(a, b) = my_mm(a, b)` on HIP retains its dispatch. On HIP the
    // mechanism is DAG-level helper inlining at lowering time, threaded
    // through `lower_named_tensor_entry_dag` in `crates/chelis-ir/src/host.rs`.
    assert_eq!(
        h.hipblas_row_major_calls, 2,
        "user-def builtin matmul wrapper on HIP must emit exactly two \
         `chelis_hipblas_sgemm_row_major(` calls through helper inlining. got={}",
        h.hipblas_row_major_calls
    );
    assert_eq!(h.cblas_calls, 0);
    assert!(h.has_lhipblas);
}

#[test]
fn hip_user_def_manual_matmul_helper_hits_hipblas() {
    let (_, _, _, user_def_manual) = semantic_gap_sources();
    let h = build_hip_and_count(user_def_manual, "hip_sgap_user_def_manual");

    assert_eq!(
        h.hipblas_row_major_calls, 2,
        "user-def insert+mul+sum wrapper on HIP must emit exactly two \
         `chelis_hipblas_sgemm_row_major(` calls (the helper's body is \
         specialized into a BLAS matmul before HIP emit). got={}",
        h.hipblas_row_major_calls
    );
    assert_eq!(h.cblas_calls, 0);
    assert!(h.has_lhipblas);
}

#[test]
fn hip_nested_user_def_matmul_helper_hits_hipblas() {
    let nested = "def my_mm(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                  -> tensor[8, 4, f32] = matmul(a, b)\n\
                  def wrap_mm(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                  -> tensor[8, 4, f32] = my_mm(a, b)\n\
                  def f(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                  -> tensor[8, 4, f32] = wrap_mm(a, b)\n";
    let h = build_hip_and_count(nested, "hip_sgap_nested_user_def");

    assert_eq!(
        h.hipblas_row_major_calls, 2,
        "nested user-def matmul wrapper on HIP must emit exactly two \
         `chelis_hipblas_sgemm_row_major(` calls. got={}",
        h.hipblas_row_major_calls
    );
    assert_eq!(h.cblas_calls, 0);
    assert!(h.has_lhipblas);
}
