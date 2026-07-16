//! Test 1 — Explicit Copy Materialization via emitted C source.
//!
//! The `copy-drop` contract makes source-level `copy(x)` explicit in IR as
//! `RiscOp::Copy`. This canary checks the emitted C follows that new wire
//! shape instead of silently erasing the copies.
//!
//! This test compiles `examples/illustrative/copy_elision_probe.ch` to C and
//! inspects the output. The locked findings:
//!
//!   * Zero `memcpy` calls. Explicit copies materialize through the same
//!     contiguous realization loop used by `realize`, not through raw byte
//!     copying.
//!   * Six large `chelis_slot*` backing allocations for the current
//!     conservative planner: explicit copy materialization plus fused fan-in
//!     intermediates.
//!   * Multiple `parallel for simd` blocks — kernel fusion combines the
//!     elementwise unary results and the add chain into SIMD-vectorized
//!     loops without source-level add intermediates.
//!   * Every fused-kernel input/output pointer carries the C99 `restrict`
//!     qualifier. This is the linearity → no-aliasing guarantee surfacing in
//!     the C codegen so the host compiler can vectorize aggressively.
//!
//! Net: explicit `copy()` is now visible to the IR/cost surface. The planner
//! may still reuse backing slots when liveness proves non-overlap, but this
//! canary no longer asserts that source copies are free.
//!
//! ## Cost profile (computed from emitted C)
//!
//! For the probe shape `tensor[1024, 1024, f32]` (~4 MiB per buffer):
//!   * 6 backing slots × (1024×1024×4 B) = **24 MiB allocated by helper**.
//!   * The input `x` itself is borrowed (not allocated) so it does not
//!     contribute to the helper's allocation footprint.
//!   * Metadata wrappers are still freed at function epilogue, but backing
//!     slots are reused as soon as planned liveness permits.
//!
//! Linear projection to a 2 GiB input (~22300×22300 f32 ≈ 2 GiB):
//!   * Caller-side: 1 × 2 GiB input.
//!   * Helper-side: 6 × 2 GiB backing slots = **12 GiB peak working set**.
//!   * Total RAM with the input: ~14 GiB. A later fan-in/in-place fusion pass
//!     could collapse this further, but that is not part of M2a.

use std::fs;
use std::process::Command;

use assert_cmd::cargo::CommandCargoExt;
use tempfile::tempdir;

fn build_copy_elision_c_source() -> String {
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
    fs::read_to_string(&c_path).expect("read generated c")
}

/// Sum of bytes allocated by every `chelis_alloc(N, (int[]){...}, CHELIS_<T>)`
/// call in the C source. After M2a this approximates the slot-planned helper
/// working set because slot backing allocations still use `chelis_alloc`.
///
/// Returns (`total_bytes`, `allocation_count`, `per_alloc_bytes`).
#[must_use]
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
fn copy_probe_materializes_explicit_copies_without_memcpy() {
    let source = build_copy_elision_c_source();

    let alloc_calls = source.matches("chelis_alloc(").count();
    let memcpy_calls = source.matches("memcpy(").count();
    let fused_kernels = source.matches("parallel for simd").count();
    let restrict_qualifiers = source.matches("restrict").count();

    assert_eq!(
        memcpy_calls, 0,
        "expected zero memcpy calls: `copy(x)` materializes through tensor \
         realization loops, not raw byte copies. Got {memcpy_calls}."
    );

    assert_eq!(
        alloc_calls, 6,
        "expected 6 C backing-slot allocations after explicit Copy reached \
         IR. Fewer means copy materialization was optimized away; more means \
         slot reuse regressed."
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
        "expected `restrict` qualifiers on fused-kernel pointers. This is \
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

    // For tensor[1024, 1024, f32], each backing slot is 4 MiB. Explicit
    // copy materialization currently brings this probe to six slots.
    let expected = 6 * 1024 * 1024 * 4; // 24 MiB
    assert_eq!(
        total_bytes, expected,
        "expected peak C backing-slot footprint of 24 MiB (6 slots × 4 MiB), \
         got {total_bytes} bytes ({mib:.2} MiB). Per-alloc bytes: {per:?}"
    );

    // Linear projection: scale input from 4 MiB (1024×1024 f32) to 2 GiB
    // (~512× larger). Six backing slots scale to 12 GiB helper-side peak.
    let scale_to_2gib = (2_u64 * 1024 * 1024 * 1024) / (1024 * 1024 * 4);
    let projected_2gib_peak_bytes = (total_bytes as u64) * scale_to_2gib;
    let projected_2gib_peak_gib = projected_2gib_peak_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "Linear projection to 2 GiB input: helper-side slot footprint ≈ {projected_2gib_peak_gib:.1} GiB \
         (excludes the 2 GiB input itself). With the borrowed input: ~{:.1} GiB total.",
        projected_2gib_peak_gib + 2.0
    );
}
