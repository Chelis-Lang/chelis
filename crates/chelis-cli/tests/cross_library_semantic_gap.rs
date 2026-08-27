//! Test 5 — Cross-Library AD Semantic Gap.
//!
//! Background: Chelis advertises that AD "flows through" Coral / Nautilus /
//! Octant because everything compiles to the same RISC primitive set. The
//! third-party concern: in practice, the Tier 2 pattern matcher then has to
//! recognize what high-level operation a graph of basic math ops *used to
//! be* in order to dispatch the right BLAS / cuDNN / scatter kernel. If the
//! recognizer is brittle, library code that semantically computes a matmul
//! gets degraded codegen.
//!
//! This test compiles four semantically-equivalent matmul programs and
//! tabulates which ones hit the C backend's BLAS specializer
//! (`cblas_sgemm`):
//!
//!   1. `f(a, b) = matmul(a, b)` — direct builtin call.
//!   2. `f(a, b) = { ae = expand(a, ...); be = expand(b, ...); sum(mul(ae, be), 1) }`
//!      — hand-written expand+mul+sum *inline in the same function body*.
//!   3. `def my_mm(a, b) = matmul(a, b); def f(a, b) = my_mm(a, b)`
//!      — builtin matmul *wrapped in a user def* and called from another fn.
//!   4. `def my_mm(a, b) = { ae = ...; ... }; def f(a, b) = my_mm(a, b)`
//!      — hand-written expand+mul+sum in a user def.
//!
//! Findings and target behavior locked here:
//!
//!   * (1) and (2) both fire `cblas_sgemm`. The IR optimizer treats inline
//!     expand+mul+sum identically to a Tier 2 desugared matmul. **Semantic
//!     gap is bridged at the IR level for inline code.**
//!   * (3) and (4) are the cross-function target: user-level helpers remain
//!     emitted as helper functions for debugging and non-specialized callers,
//!     while eligible call sites may bypass the helper and emit BLAS directly.
//!     The generated C should contain `cblas_sgemm`/`chelis_blas_matmul`
//!     without relying on clang/gcc LTO to rediscover the computation after
//!     emission.
//!
//! Implication for cross-library AD: a Coral `groupby+sum` or a Nautilus
//! Simpson's-rule integrator written as a user `def` won't get BLAS or
//! cuDNN specialization just by virtue of decomposing into RISC primitives.
//! The "semantic gap" the third-party identified is real for any
//! library-level abstraction that sits behind a function-call boundary.
//! The mitigation under test is compiler-derived specialization for helper
//! DAGs. There are no source annotations in these fixtures.

use std::fs;
use std::process::Command;

use assert_cmd::cargo::CommandCargoExt;
use tempfile::tempdir;

struct Counts {
    blas: usize,
    allocs: usize,
    fused: usize,
    user_helper_defs: usize,
    /// Total bytes summed across every `chelis_alloc(N, (int64_t[]){...},
    /// CHELIS_DTYPE_F32)` call. This approximates peak working set under the C
    /// backend's Phase-0 free-all-at-end strategy.
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
    let fused = c.matches("parallel for simd").count();
    let user_helper_defs = c.matches("static void my_mm__tensor_").count();

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
                           ae = expand(a, 2, 4i64)\n  \
                           be = expand(b, 0, 8i64)\n  \
                           sum(mul(ae, be), 1)\n\
                         }\n";
    let user_def_builtin = "def my_mm(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                            -> tensor[8, 4, f32] = matmul(a, b)\n\
                            def f(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                            -> tensor[8, 4, f32] = my_mm(a, b)\n";
    let user_def_manual = "def my_mm(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                           -> tensor[8, 4, f32] = {\n  \
                             ae = expand(a, 2, 4i64)\n  \
                             be = expand(b, 0, 8i64)\n  \
                             sum(mul(ae, be), 1)\n\
                           }\n\
                           def f(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                           -> tensor[8, 4, f32] = my_mm(a, b)\n";

    (direct, inline_manual, user_def_builtin, user_def_manual)
}

#[test]
fn inline_matmul_forms_hit_blas_and_allocate_only_result() {
    let (direct, inline_manual, _, _) = semantic_gap_sources();
    let direct_c = build_and_count(direct, "sgap_direct");
    let inline_c = build_and_count(inline_manual, "sgap_inline_manual");

    // Inline forms must hit BLAS — direct call OR hand-written Einstein math.
    // This is the "semantic gap is bridged at IR level for inline code"
    // assertion.
    assert!(
        direct_c.blas >= 1,
        "direct `matmul(a,b)` MUST emit at least one cblas_sgemm; got {}",
        direct_c.blas
    );
    assert!(
        inline_c.blas >= 1,
        "inline expand+mul+sum (the canonical Tier 2 pattern written by hand) \
         MUST be detected by the BLAS specializer; got {}. If this \
         regresses, the Tier 2 pattern matcher has lost the ability to \
         recognize hand-written Einstein-form matmul.",
        inline_c.blas
    );

    // ---- Cost profile ----
    //
    // Same logical 8×16 @ 16×4 matmul, four code paths, four working sets.
    // The result is 8×4×4 = 128 bytes. The minimum sane allocation is 1
    // result buffer. Anything beyond that is intermediate scaffolding.
    eprintln!("== Cost profile (8×16 @ 16×4 matmul, all f32) ==");
    eprintln!("  result tensor                  : 128 bytes (8×4×4)");
    eprintln!(
        "  direct `matmul(a, b)`          : blas={} allocs={} fused={} total_bytes={}",
        direct_c.blas, direct_c.allocs, direct_c.fused, direct_c.total_alloc_bytes
    );
    eprintln!(
        "  inline expand+mul+sum          : blas={} allocs={} fused={} total_bytes={}",
        inline_c.blas, inline_c.allocs, inline_c.fused, inline_c.total_alloc_bytes
    );
    // ---- Closed finding: dead-Mul intermediate ----
    //
    // M1 moved matmul specialization into an IR pass. Direct and inline
    // matmul now replace the Sum/Mul/Expand subgraph before DCE/fusion, so
    // the 3-D Mul intermediate is not allocated or computed on BLAS-hit paths.

    // Locked assertion: direct and inline_manual must produce the same
    // working set (both go through the same IR DAG → same codegen).
    assert_eq!(
        direct_c.total_alloc_bytes, inline_c.total_alloc_bytes,
        "direct `matmul(a, b)` and inline expand+mul+sum should produce \
         identical IR DAGs and therefore identical working sets. direct={} \
         bytes, inline={} bytes.",
        direct_c.total_alloc_bytes, inline_c.total_alloc_bytes
    );
    assert_eq!(
        direct_c.total_alloc_bytes,
        8 * 4 * 4,
        "direct BLAS-hit matmul should allocate only the 8x4 f32 result \
         buffer after the IR specialization pass removes the dead Mul"
    );
    assert_eq!(
        inline_c.total_alloc_bytes,
        8 * 4 * 4,
        "inline expand+mul+sum should also specialize to result-only memory"
    );

    let proj_direct_bytes = 1024_u64 * 1024 * 4;
    let proj_direct_mib = proj_direct_bytes as f64 / (1024.0 * 1024.0);
    eprintln!(
        "Linear projection to 1024×1024 @ 1024×1024 inputs:\n  \
         direct (BLAS hit, no dead Mul): {proj_direct_mib:.2} MiB result buffer\n  \
         inline (same IR, same codegen): {proj_direct_mib:.2} MiB result buffer"
    );
}

#[test]
fn user_def_matmul_helpers_hit_blas() {
    let (_, _, user_def_builtin, user_def_manual) = semantic_gap_sources();
    let user_b = build_and_count(user_def_builtin, "sgap_user_def_builtin");
    let user_m = build_and_count(user_def_manual, "sgap_user_def_manual");

    assert!(
        user_b.blas >= 1,
        "target behavior: user-def wrapper around builtin matmul should emit BLAS"
    );
    assert!(
        user_m.blas >= 1,
        "target behavior: user-def wrapper around expand+mul+sum should emit BLAS"
    );
    assert!(
        user_b.user_helper_defs >= 1,
        "target behavior: builtin matmul helper should still be emitted for debug/fallback"
    );
    assert!(
        user_m.user_helper_defs >= 1,
        "target behavior: manual matmul helper should still be emitted for debug/fallback"
    );

    eprintln!("== User-def helper target behavior ==");
    eprintln!(
        "  user_def of builtin matmul     : blas={} allocs={} fused={} total_bytes={}",
        user_b.blas, user_b.allocs, user_b.fused, user_b.total_alloc_bytes
    );
    eprintln!(
        "  user_def of expand+mul+sum     : blas={} allocs={} fused={} total_bytes={}",
        user_m.blas, user_m.allocs, user_m.fused, user_m.total_alloc_bytes
    );
}

#[test]
fn nested_user_def_matmul_helper_hits_blas() {
    let nested = "def my_mm(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                  -> tensor[8, 4, f32] = matmul(a, b)\n\
                  def wrap_mm(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                  -> tensor[8, 4, f32] = my_mm(a, b)\n\
                  def f(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                  -> tensor[8, 4, f32] = wrap_mm(a, b)\n";
    let nested_c = build_and_count(nested, "sgap_nested_user_def");

    assert!(
        nested_c.blas >= 1,
        "nested user-def wrapper around builtin matmul should emit BLAS"
    );
    assert!(
        nested_c.user_helper_defs >= 1,
        "nested fixture should still emit the original helper surface"
    );

    eprintln!("== Nested user-def helper target behavior ==");
    eprintln!(
        "  nested user_def matmul       : blas={} allocs={} fused={} total_bytes={}",
        nested_c.blas, nested_c.allocs, nested_c.fused, nested_c.total_alloc_bytes
    );
}

// ===========================================================================
// HIP target — Gap 5 M5(a): user-def matmul helpers must emit hipBLAS on HIP
// the same way they emit cblas_sgemm on the C backend (above). The HIP target
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
        "inline expand+mul+sum on HIP must emit exactly two \
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
    // def f(a, b) = my_mm(a, b)` on HIP must NOT lose hipBLAS the way it
    // would lose cblas_sgemm without summary consumption on C. On HIP the
    // mechanism is DAG-level helper inlining at lowering time, threaded
    // through `lower_named_tensor_entry_dag` in `crates/chelis-ir/src/host.rs`.
    assert_eq!(
        h.hipblas_row_major_calls, 2,
        "user-def builtin matmul wrapper on HIP must emit exactly two \
         `chelis_hipblas_sgemm_row_major(` calls (M5(a) closure: helper \
         calls hit hipBLAS the same way they hit cblas_sgemm on C). got={}",
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
        "user-def expand+mul+sum wrapper on HIP must emit exactly two \
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
