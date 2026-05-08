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
//! Findings (locked here):
//!
//!   * (1) and (2) both fire `cblas_sgemm`. The IR optimizer treats inline
//!     expand+mul+sum identically to a Tier 2 desugared matmul. **Semantic
//!     gap is bridged at the IR level for inline code.**
//!   * (3) and (4) both miss `cblas_sgemm`. When matmul math (or anything
//!     that decomposes into expand+mul+sum) lives inside a user-level `def`
//!     called from another `def`, the user def is emitted as a separate
//!     C function (`my_mm__tensor_0(inputs, n_in, outputs, n_out)`) and the
//!     BLAS detector keys off the *caller* DAG, not the helper. The Tier 2
//!     fused expansion and stride-view expand still happen (so it's not as
//!     bad as a scalar fallback), but the BLAS specialization is lost.
//!   * (4) additionally fails to lower `expand` in the host lane in some
//!     historical revisions; in the current snapshot the helper goes
//!     through the IR/tier2 path, but the lack of BLAS specialization
//!     remains the dominant cost difference.
//!
//! Implication for cross-library AD: a Coral `groupby+sum` or a Nautilus
//! Simpson's-rule integrator written as a user `def` won't get BLAS or
//! cuDNN specialization just by virtue of decomposing into RISC primitives.
//! The "semantic gap" the third-party identified is real for any
//! library-level abstraction that sits behind a function-call boundary.
//! The mitigations that would close it: aggressive cross-function inlining
//! before the optimize/fuse pass, or moving the BLAS detector to operate
//! on the call-graph rather than per-function DAGs.

use std::fs;
use std::process::Command;

use assert_cmd::cargo::CommandCargoExt;
use tempfile::tempdir;

struct Counts {
    blas: usize,
    allocs: usize,
    fused: usize,
    /// Total bytes summed across every `chelis_alloc(N, (int[]){...},
    /// CHELIS_F32)` call. This approximates peak working set under the C
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

    // Sum bytes across every chelis_alloc(N, (int[]){...}, CHELIS_F32) call.
    let mut total_alloc_bytes = 0usize;
    let mut idx = 0;
    while let Some(start) = c[idx..].find("chelis_alloc(") {
        let pos = idx + start;
        let after = &c[pos..];
        let window = &after[..(256.min(after.len()))];
        if let Some(brace_open) = window.find("(int[]){") {
            let dims_start = brace_open + "(int[]){".len();
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
                    total_alloc_bytes += const_factor * 4; // CHELIS_F32 only in this test
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
        total_alloc_bytes,
    }
}

#[test]
fn semantic_gap_inline_vs_user_def() {
    let direct = "def f(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                  -> tensor[8, 4, f32] = matmul(a, b)\n";
    let inline_manual = "def f(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                         -> tensor[8, 4, f32] = {\n  \
                           ae = expand(a, 2, 4)\n  \
                           be = expand(b, 0, 8)\n  \
                           sum(mul(ae, be), 1)\n\
                         }\n";
    let user_def_builtin = "def my_mm(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                            -> tensor[8, 4, f32] = matmul(a, b)\n\
                            def f(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                            -> tensor[8, 4, f32] = my_mm(a, b)\n";
    let user_def_manual = "def my_mm(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                           -> tensor[8, 4, f32] = {\n  \
                             ae = expand(a, 2, 4)\n  \
                             be = expand(b, 0, 8)\n  \
                             sum(mul(ae, be), 1)\n\
                           }\n\
                           def f(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) \
                           -> tensor[8, 4, f32] = my_mm(a, b)\n";

    let direct_c = build_and_count(direct, "sgap_direct");
    let inline_c = build_and_count(inline_manual, "sgap_inline_manual");
    let user_b = build_and_count(user_def_builtin, "sgap_user_def_builtin");
    let user_m = build_and_count(user_def_manual, "sgap_user_def_manual");

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

    // User-def forms must NOT hit BLAS — this is the semantic-gap finding.
    // If a future change starts inlining or cross-function pattern-matching,
    // this assertion will flip and the finding above must be updated.
    assert_eq!(
        user_b.blas, 0,
        "matmul wrapped in a user def is NOT specialized to BLAS in the \
         current snapshot. If this changes (cross-function inlining or \
         call-graph-aware pattern matching shipped), update the test docs \
         to reflect the new behavior. Got {} BLAS calls.",
        user_b.blas
    );
    assert_eq!(
        user_m.blas, 0,
        "hand-written Einstein-form matmul wrapped in a user def is NOT \
         specialized to BLAS. Same caveat as the previous assertion. Got \
         {} BLAS calls.",
        user_m.blas
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
    eprintln!(
        "  user_def of builtin matmul     : blas={} allocs={} fused={} total_bytes={}",
        user_b.blas, user_b.allocs, user_b.fused, user_b.total_alloc_bytes
    );
    eprintln!(
        "  user_def of expand+mul+sum     : blas={} allocs={} fused={} total_bytes={}",
        user_m.blas, user_m.allocs, user_m.fused, user_m.total_alloc_bytes
    );

    // ---- Surprising finding: dead-Mul intermediate ----
    //
    // At this 8×16 shape, direct/inline/user_def_builtin all allocate
    // 2176 bytes = 2048-byte 3-D Mul intermediate (`[8, 16, 4]`) + 128-byte
    // 2-D result. The Mul intermediate is allocated and computed even when
    // BLAS specialization fires, because BLAS replaces only the *Sum* step
    // (Sum-Mul-Expand-Expand → cblas_sgemm reading inputs directly), not
    // the Mul that was already emitted by Tier 2 lowering. The Mul output
    // is then freed without ever being read — pure dead compute and dead
    // memory.
    //
    // This is a NEW gap not in the original five: **DCE does not eliminate
    // the Tier-2 Mul intermediate when its consumer (Sum) is replaced by
    // a BLAS call.** Cost grows cubically with matmul size — for a
    // 1024×1024 matmul the dead intermediate is 1024³ × 4 B = 4 GiB,
    // wasted on every BLAS-hit matmul. Tracking is left as a follow-up.
    //
    // The original semantic-gap finding (BLAS misses on user-def helpers)
    // remains valid as a *compute-throughput* finding — the user_def
    // variants take the scalar reduction path on the Mul output, while
    // direct/inline take cblas_sgemm. The memory cost is comparable; the
    // compute cost differs by ~50-100×.

    // Locked assertion: direct and inline_manual must produce the same
    // working set (both go through the same IR DAG → same codegen).
    assert_eq!(
        direct_c.total_alloc_bytes, inline_c.total_alloc_bytes,
        "direct `matmul(a, b)` and inline expand+mul+sum should produce \
         identical IR DAGs and therefore identical working sets. direct={} \
         bytes, inline={} bytes.",
        direct_c.total_alloc_bytes, inline_c.total_alloc_bytes
    );

    // Linear projection: scale the inputs to 1024×1024 @ 1024×1024.
    // The 3-D Mul intermediate (allocated under both BLAS-hit and
    // BLAS-miss paths) becomes 1024×1024×1024×4 B = 4 GiB. This is the
    // dominant working-set cost for any non-trivial matmul shape, and
    // it's wasted compute under BLAS specialization. Closing the
    // dead-Mul gap (run DCE *after* BLAS detection, or fold BLAS detection
    // into the optimize pass instead of codegen) would reduce a 1024
    // matmul's working set from ~4 GiB to ~4 MiB — a 1000× reduction.
    let scale_factor: u64 = (1024_u64 * 1024 * 1024) / (8 * 16 * 4);
    let proj_direct_bytes = (direct_c.total_alloc_bytes as u64) * scale_factor;
    let proj_direct_gib = proj_direct_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "Linear projection to 1024×1024 @ 1024×1024 inputs:\n  \
         direct (BLAS hit, dead Mul): {proj_direct_gib:.2} GiB peak working set\n  \
         inline (same IR, same codegen): {proj_direct_gib:.2} GiB\n  \
         user_def variants: comparable peak (Mul is allocated either way), \
         but compute path is scalar reduction instead of sgemm — much slower."
    );
}
