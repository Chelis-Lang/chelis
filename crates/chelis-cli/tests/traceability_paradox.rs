//! Test 3 — Traceability vs Fusion
//!
//! The third-party review predicted that a heavily-fused transformer block
//! either (a) drops `// span:` comments down to one-per-fused-block (making
//! the audit chain useless) or (b) maintains line-by-line mapping at a perf
//! cost.
//!
//! This test compiles the existing 4-head MHA + FFN block in
//! `examples/transformer_block.ch` and inspects the emitted C source. The
//! recorded measurements are: number of span comments, share that are
//! synthesized markers vs Surf parser byte-range spans, number of fused
//! kernels, number of BLAS specializations actually fired, and the buffer
//! allocation footprint in bytes (peak working set under the C backend's
//! Phase-0 allocate-per-node strategy).
//!
//! Findings (locked here as assertions):
//!   * Emitted `// span:` lines include Surf parser byte-range IDs of the
//!     form `surf:<start>..<end>`. Synthesized markers may still appear for
//!     nodes that genuinely have no source range, but the traceability
//!     chain no longer bottoms out at `__synthesized_*__` for ordinary Surf
//!     bodies.
//!   * Despite the file containing 13 `matmul` calls, zero of them
//!     specialize to BLAS. The reason is the symbolic `seq` dimension on the
//!     input tensor: `extract_matmul_dims` in
//!     `crates/chelis-backend-c/src/blas.rs:90-95` requires concrete dim
//!     literals, and a purely symbolic `Named(_, None)` axis short-circuits
//!     the detector. This is the matmul analogue of the Test 2 brittleness —
//!     the pattern matcher misses on symbolic-dim inputs.
//!
//! ## Cost profile (computed from emitted C, parameterised by `seq`)
//!
//! Empirically the working set is a polynomial in `seq` whose coefficients
//! are read directly off `chelis_alloc(N, (int[]){...})` calls:
//!
//! | term | bytes | dominant source |
//! |---|---|---|
//! | `c2` (× seq²) | ~2 KiB | 4-head `[seq, 64, seq]` score Mul intermediates (1024 B), 4-head `[seq, seq, 64]` probs@v Mul intermediates (1024 B), `[seq, seq]` score+softmax buffers (~32 B) |
//! | `c1` (× seq) | **~3.18 MiB** | the 3-D `Mul` intermediates produced by every Tier-2-lowered matmul: 12× QKV `[seq, 256, 64]` (786 KiB/seq) + FFN1 `[seq, 256, 1024]` (1.0 MiB/seq) + FFN2 `[seq, 1024, 256]` (1.0 MiB/seq) + head-output projections (~64 KiB/seq) |
//! | `c0` | 0 | none — every allocation has at least one `seq` factor |
//!
//! Projected peak working set:
//!
//! | seq | peak | dominant term |
//! |---|---|---|
//! | 128 | 421 MiB | `c1 × seq` (Mul intermediates) |
//! | 512 | 2.1 GiB | `c1 × seq` |
//! | 2048 | **14.6 GiB** | `c1 × seq` |
//! | 4096 | 46.0 GiB | `c1 × seq` (`c2 × seq²` starts catching up) |
//!
//! **Why this is so high:** every matmul lowers via Tier 2 to
//! `expand → mul → sum`. The `Mul` step materializes a 3-D
//! `[i, j, k]` tensor whose volume is the *full triple product* of the
//! contraction. Normally the BLAS detector recognises the
//! `Sum → Mul → (Expand, Expand)` shape and replaces the materialised
//! `Mul` with a direct `cblas_sgemm` call (no 3-D intermediate). But the
//! detector requires concrete dim literals (`crates/chelis-backend-c/src/
//! blas.rs:90-95`) and `seq` is symbolic, so detection misses for every
//! matmul in this file. The result is roughly a 10× working-set
//! amplification compared to the post-BLAS path.
//!
//! Two passes would close this:
//!   1. Generalise the BLAS detector to symbolic-dim matmul (Gap 4 in
//!      `docs/identified_gaps.md`).
//!   2. Ship the C-backend memory planner so non-overlapping `Mul`
//!      intermediates can share buffers (Gap 1).
//!
//! Even with both, vanilla attention is `seq`-quadratic in the score
//! buffers — FlashAttention-style fusion (not on the roadmap today)
//! would eliminate the score materialisation entirely.

use std::fs;
use std::process::Command;

use assert_cmd::cargo::CommandCargoExt;
use tempfile::tempdir;

/// Parse `chelis_alloc` calls and compute a polynomial in `seq` describing
/// the working set. Returns (constant_bytes, seq1_bytes, seq2_bytes) such
/// that total ≈ constant + seq * seq1 + seq * seq * seq2.
///
/// Unhandled allocations (e.g. `chelis_alloc(0, NULL, ...)`) contribute
/// `dtype_size` bytes each to the constant term.
fn measure_seq_polynomial(c_source: &str) -> (usize, usize, usize, usize) {
    let mut alloc_count = 0;
    let mut const_bytes = 0usize;
    let mut seq1_bytes = 0usize;
    let mut seq2_bytes = 0usize;

    let mut idx = 0;
    while let Some(start) = c_source[idx..].find("chelis_alloc(") {
        let pos = idx + start;
        let after = &c_source[pos..];
        alloc_count += 1;

        // Look ahead a bounded window — long enough to cover the type tag.
        let window_end = (256).min(after.len());
        let window = &after[..window_end];
        let dtype_size: usize = if window.contains("CHELIS_F64") {
            8
        } else if window.contains("CHELIS_BOOL") {
            1
        } else if window.contains("CHELIS_I64") {
            8
        } else {
            // CHELIS_F32 / CHELIS_I32 / unknown all default to 4 bytes.
            4
        };

        // Find the shape list `(int[]){ ... }`.
        if let Some(brace_open) = window.find("(int[]){") {
            let dims_start = brace_open + "(int[]){".len();
            if let Some(brace_close) = window[dims_start..].find('}') {
                let dims_text = &window[dims_start..dims_start + brace_close];
                // Each dim is either a literal int (e.g. 256), a `seq`
                // symbol (treat as one factor of `seq`), or some other
                // identifier we don't model. We aggregate the literal-int
                // factor and count `seq` powers separately.
                let mut const_factor: usize = 1;
                let mut seq_power: usize = 0;
                let mut unrecognized = false;
                for dim_token in dims_text.split(',') {
                    let t = dim_token.trim();
                    if t.is_empty() {
                        continue;
                    }
                    if let Ok(n) = t.parse::<usize>() {
                        const_factor = const_factor.saturating_mul(n);
                    } else if t == "seq" {
                        seq_power += 1;
                    } else {
                        // Unknown symbolic dim — skip.
                        unrecognized = true;
                    }
                }
                if unrecognized {
                    // Fall through: don't attribute.
                    continue;
                }
                let bytes = const_factor * dtype_size;
                match seq_power {
                    0 => const_bytes += bytes,
                    1 => seq1_bytes += bytes,
                    2 => seq2_bytes += bytes,
                    // Anything higher: lump into seq² conservatively.
                    _ => seq2_bytes += bytes,
                }
            }
        } else if window.starts_with("chelis_alloc(0, NULL,") {
            const_bytes += dtype_size;
        }

        idx = pos + "chelis_alloc(".len();
    }

    (alloc_count, const_bytes, seq1_bytes, seq2_bytes)
}

#[test]
fn transformer_block_traceability_state_is_locked() {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("transformer_out");

    let status = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args([
            "build",
            "../../examples/transformer_block.ch",
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .status()
        .expect("chelis build should run");
    assert!(status.success(), "chelis build failed");

    let c_path = out_dir.join("transformer_block.c");
    let source = fs::read_to_string(&c_path).expect("read generated c");

    // Tabulate.
    let span_lines: Vec<&str> = source.lines().filter(|l| l.contains("// span:")).collect();
    let total_spans = span_lines.len();
    let synthesized_spans = span_lines
        .iter()
        .filter(|l| l.contains("__synthesized"))
        .count();
    let surf_spans = span_lines
        .iter()
        .filter(|l| l.contains("// span: surf:"))
        .count();
    let fused_kernels = source.matches("parallel for simd").count();
    let allocations = source.matches("chelis_alloc(").count();
    let blas_calls = source.matches("cblas_sgemm").count()
        + source.matches("chelis_blas_matmul").count()
        + source.matches("chelis_blas_sgemm").count();

    // Lock the findings.
    assert!(
        total_spans > 0,
        "expected the emitter to write some `// span:` lines; got zero"
    );
    assert!(
        surf_spans > 0,
        "expected Surf parser byte-range spans to reach emitted C; got zero \
         `// span: surf:<start>..<end>` lines out of {total_spans} total spans"
    );
    assert!(
        synthesized_spans < total_spans,
        "expected ordinary Surf source spans to replace the old all-synthesized \
         traceability state; got {synthesized_spans} synthesized spans out of \
         {total_spans} total spans"
    );
    assert!(
        fused_kernels >= 10,
        "expected at least 10 fused parallel-for-simd kernels for a 4-head \
         MHA+FFN block; got {fused_kernels}"
    );
    assert!(
        allocations > 50,
        "expected many intermediate buffer allocations for a 4-head MHA \
         (per-head Q/K/V/O + softmax + residual + FFN); got {allocations}"
    );
    assert_eq!(
        blas_calls, 0,
        "expected ZERO matmul→BLAS specializations because the symbolic `seq` \
         dim short-circuits the detector. If a future change generalizes \
         the BLAS specializer to symbolic dims, update this test."
    );

    // ---- Cost profile ----
    let (alloc_count, c0, c1, c2) = measure_seq_polynomial(&source);
    eprintln!("transformer_block: {alloc_count} allocs. Working-set polynomial in seq:");
    eprintln!("  const term ............ {c0:>10} bytes");
    eprintln!("  linear term * seq ..... {c1:>10} * seq bytes");
    eprintln!("  quadratic term * seq² . {c2:>10} * seq² bytes");

    let projections = [128usize, 512, 2048, 4096];
    for seq in projections {
        let total = c0 + c1 * seq + c2 * seq * seq;
        let mib = total as f64 / (1024.0 * 1024.0);
        eprintln!("  peak at seq = {seq:>5}: {total:>12} bytes ({mib:>8.2} MiB)");
    }

    assert!(
        c2 > 0,
        "expected a `seq`-quadratic component (attention scores [seq, seq] \
         and softmax probs [seq, seq] across 4 heads). Got c2 = {c2}. If this \
         drops to zero, either the score/softmax buffers are no longer \
         allocated or the codegen changed how it emits `[seq, seq]` shapes."
    );
    assert!(
        c1 > 0,
        "expected a `seq`-linear component (per-head Q/K/V activations and \
         residual / LN / FFN intermediates of shape `[seq, hidden]`). Got \
         c1 = {c1}."
    );
}
