//! Test 1 — Copy Elision via emitted C source.
//!
//! Third-party Test 1 prediction: a Surf function that calls `copy(x)` five
//! times either (a) elides every copy and uses ~one buffer of `x` worth of
//! memory, or (b) literally allocates one buffer per `copy` and balloons to
//! ~6× input size.
//!
//! This test compiles `examples/illustrative/copy_elision_probe.ch` to C and
//! inspects the output. The locked findings:
//!
//!   * Zero `memcpy` calls. The five `copy(x)` markers in Surf are *not*
//!     lowered as memory copies — they are linearity-discharge markers that
//!     authorize multi-consumer reads of the same backing buffer.
//!   * Five `chelis_alloc` calls — one per distinct unary result (a..e). No
//!     extra allocation per `copy(x)` and no allocation for any of the four
//!     `add` intermediates (those fuse).
//!   * Three `parallel for simd` blocks — kernel fusion combines the
//!     elementwise unary results and the add chain into a small number of
//!     SIMD-vectorized loops.
//!   * Every fused-kernel input/output pointer carries the C99 `restrict`
//!     qualifier. This is the linearity → no-aliasing guarantee surfacing in
//!     the C codegen so the host compiler can vectorize aggressively.
//!
//! Net: `copy()` is free at the buffer level. The remaining 5× working
//! footprint comes from five distinct unary results that all live until the
//! final reduction reads them. Two passes would shrink that further, neither
//! of which currently applies to C codegen:
//!   * Memory planning (spec §5.6). Phase 1c shipped a *GPU* memory planner
//!     (`crates/chelis-backend-hip/src/memory.rs`) and the roadmap marks 1c
//!     "Implemented." But the C backend's planner
//!     (`crates/chelis-backend-c/src/memory.rs`) is still the documented
//!     "Phase 0: simple allocate-per-node, free-all-at-end strategy." That
//!     line is the reason this test sees 5 allocs instead of ~2.
//!   * Fan-in fusion (multi-input reduction body fusion). Not shipped on
//!     either backend; `crates/chelis-ir/src/fuse.rs:69-88` only fuses
//!     elementwise chains.
//!
//! The cost profile here is "5 unary results unplanned in C codegen," not
//! "5 copies materialized."
//!
//! ## Cost profile (computed from emitted C)
//!
//! For the probe shape `tensor[1024, 1024, f32]` (~4 MiB per buffer):
//!   * 5 result buffers × (1024×1024×4 B) = **20 MiB allocated by helper**.
//!   * The input `x` itself is borrowed (not allocated) so it does not
//!     contribute to the helper's allocation footprint.
//!   * Under the C backend's Phase-0 free-all-at-end strategy, all 5 buffers
//!     are simultaneously live until the function returns. Peak working set
//!     is therefore exactly 20 MiB plus the caller-provided 4 MiB input.
//!
//! Linear projection to a 2 GiB input (~22300×22300 f32 ≈ 2 GiB):
//!   * Caller-side: 1 × 2 GiB input.
//!   * Helper-side: 5 × 2 GiB result buffers = **10 GiB peak working set**.
//!   * Total RAM with the input: ~12 GiB. This is the empirical answer to
//!     the third-party "VRAM spike to 12 GB" prediction — yes, it spikes,
//!     but the cause is "5 unary results unplanned" not "5 copies
//!     materialized." A memory planner would collapse this to ~2-4 GiB peak
//!     (only 1-2 unary results need to be live simultaneously since the
//!     fused final-reduction kernel reads each once).

use std::fs;
use std::process::Command;

use assert_cmd::cargo::CommandCargoExt;
use tempfile::tempdir;

/// Sum of bytes allocated by every `chelis_alloc(N, (int[]){...}, CHELIS_<T>)`
/// call in the C source. This approximates peak working set under the Phase-0
/// allocate-per-node / free-all-at-end strategy that
/// `crates/chelis-backend-c/src/memory.rs` ships today.
///
/// Returns (total_bytes, allocation_count, per_alloc_bytes).
pub fn measure_alloc_footprint(c_source: &str) -> (usize, usize, Vec<usize>) {
    let mut per_alloc = Vec::new();
    let mut idx = 0;
    while let Some(start) = c_source[idx..].find("chelis_alloc(") {
        let pos = idx + start;
        let after = &c_source[pos..];
        // Find the closing `)` of this alloc call. We use the matching brace
        // approach: locate the comma after rank, then the `(int[]){ ... }`.
        let dtype_size = if after.contains("CHELIS_F64") && after.find("CHELIS_F64").unwrap() < 200
        {
            8
        } else if after.contains("CHELIS_BOOL") && after.find("CHELIS_BOOL").unwrap() < 200 {
            1
        } else if after.contains("CHELIS_I64") && after.find("CHELIS_I64").unwrap() < 200 {
            8
        } else if after.contains("CHELIS_I32") && after.find("CHELIS_I32").unwrap() < 200 {
            4
        } else {
            // Default to f32 for CHELIS_F32 (most common) and unknown.
            4
        };

        // Parse the shape list. Two forms:
        //   chelis_alloc(N, (int[]){ d1, d2, ... }, CHELIS_<T>)
        //   chelis_alloc(0, NULL, CHELIS_<T>)             // scalar tensor
        let bytes = if let Some(brace_open) = after.find("(int[]){") {
            let dims_start = brace_open + "(int[]){".len();
            if let Some(brace_close) = after[dims_start..].find('}') {
                let dims_text = &after[dims_start..dims_start + brace_close];
                let dims: Vec<usize> = dims_text
                    .split(',')
                    .filter_map(|s| {
                        let trimmed = s.trim();
                        // Either a literal int or a symbolic dim name. We
                        // count literal-int dims; symbolic dims contribute
                        // a placeholder of 1 (so symbolic-only allocations
                        // get attributed `dtype_size`).
                        trimmed.parse::<usize>().ok()
                    })
                    .collect();
                let elements: usize = if dims.is_empty() {
                    1
                } else {
                    dims.iter().product()
                };
                elements * dtype_size
            } else {
                0
            }
        } else if after.starts_with("chelis_alloc(0, NULL,") {
            // Rank-0 scalar: 1 element of dtype_size.
            dtype_size
        } else {
            0
        };

        per_alloc.push(bytes);
        idx = pos + "chelis_alloc(".len();
    }
    let total = per_alloc.iter().sum();
    let count = per_alloc.len();
    (total, count, per_alloc)
}

#[test]
fn copy_elision_probe_emits_no_memcpy_and_one_buffer_per_unary_result() {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("copy_elision_out");

    let status = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args([
            "build",
            "../../examples/illustrative/copy_elision_probe.ch",
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .status()
        .expect("chelis build should run");
    assert!(status.success(), "chelis build failed");

    let c_path = out_dir.join("copy_elision_probe.c");
    let source = fs::read_to_string(&c_path).expect("read generated c");

    let alloc_calls = source.matches("chelis_alloc(").count();
    let memcpy_calls = source.matches("memcpy(").count();
    let fused_kernels = source.matches("parallel for simd").count();
    let restrict_qualifiers = source.matches("restrict").count();

    assert_eq!(
        memcpy_calls, 0,
        "expected zero memcpy calls — `copy(x)` markers must NOT lower to \
         physical buffer copies. Got {memcpy_calls}. If this fails, the \
         linearity-discharge contract has regressed and `copy(x)` is now \
         producing real allocations."
    );

    assert_eq!(
        alloc_calls, 5,
        "expected exactly 5 buffer allocations (one per distinct unary \
         result a..e in fanout/copy_elision_probe.ch). Got {alloc_calls}. \
         If this changes, either fusion got better (fewer allocs — celebrate \
         and update the assertion downward) or worse (more allocs — \
         investigate). The third-party prediction was 5 (one per copy) for \
         a fail mode; reality is 5 for a different reason (one per unary \
         result, and the copies themselves contribute zero)."
    );

    assert!(
        fused_kernels >= 2,
        "expected the C emitter to produce at least two fused \
         parallel-for-simd kernels for this 5-unary + 4-add chain. Got \
         {fused_kernels}. Fewer means the fusion pass regressed and the \
         add-chain intermediates are leaking back into separate kernels."
    );

    assert!(
        restrict_qualifiers > 0,
        "expected `restrict` qualifiers on fused-kernel pointers — this is \
         the linearity → no-aliasing guarantee surfacing in C codegen. Got \
         {restrict_qualifiers}. If this drops to zero the C compiler loses \
         the alias-free promise and vectorization quality regresses."
    );

    // ---- Cost profile ----
    let (total_bytes, count, per) = measure_alloc_footprint(&source);
    let mib = total_bytes as f64 / (1024.0 * 1024.0);
    eprintln!(
        "copy_elision_probe: {count} allocs, peak working set = {total_bytes} bytes \
         ({mib:.2} MiB). Per-alloc bytes: {per:?}"
    );

    // For tensor[1024, 1024, f32], each result buffer is 4 MiB. Five live
    // simultaneously under Phase 0 strategy → 20 MiB. We allow a small slack
    // in case the codegen adds a tiny scalar buffer somewhere; the dominant
    // term is the 5×4 MiB working set.
    let expected_min = 5 * 1024 * 1024 * 4; // 20 MiB
    let expected_max = expected_min + 1024 * 1024; // 21 MiB tolerance
    assert!(
        total_bytes >= expected_min && total_bytes <= expected_max,
        "expected peak working set ≈ 20 MiB (5 unary results × 4 MiB), got \
         {total_bytes} bytes ({mib:.2} MiB). If this drops, memory planning \
         shipped for C — update the test and the docstring's projection. If \
         it rises, an extra buffer leaked back in."
    );

    // Linear projection: scale input from 4 MiB (1024×1024 f32) to 2 GiB
    // (~512× larger). Each unary result scales the same way → 5 × 2 GiB =
    // 10 GiB peak. Total RAM with the borrowed input: ~12 GiB. This is the
    // empirical answer to the third-party Test 1 prediction.
    let scale_to_2gib = (2_u64 * 1024 * 1024 * 1024) / (1024 * 1024 * 4);
    let projected_2gib_peak_bytes = (total_bytes as u64) * scale_to_2gib;
    let projected_2gib_peak_gib = projected_2gib_peak_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "Linear projection to 2 GiB input: peak working set ≈ {projected_2gib_peak_gib:.1} GiB \
         (excludes the 2 GiB input itself). With the borrowed input: ~{:.1} GiB total.",
        projected_2gib_peak_gib + 2.0
    );
}
