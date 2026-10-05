//! Test 1 — Explicit Copy Materialization via emitted C source.
//!
//! The `copy-drop` contract makes source-level `copy(x)` explicit in IR as
//! `RiscOp::Copy`. This canary checks the emitted C follows that new wire
//! shape instead of silently erasing the copies.
//!
//! This test compiles `examples/illustrative/copy_elision_probe.ch` to C and
//! inspects the output. The locked findings:
//!
//!   * Zero `memcpy` calls in the code lowered from the probe. Explicit copies
//!     materialize through the same contiguous realization loop used by
//!     `realize`, not through raw byte copying. The linked correctly rounded
//!     kernels (`exp`, `log`, `sin`) are `chelis-crmath`'s fixed text, whose
//!     float bit casts are `memcpy`; they are set aside before counting.
//!   * Ten independently owned tensor values: six fresh physical allocations
//!     plus four exact-capacity repurposes selected by Phase 3's proof-bearing
//!     planner. Borrowed `chelis_slot*` / `chelis_alloc_view` wrappers remain
//!     absent.
//!   * Multiple `parallel for simd` blocks — kernel fusion combines the
//!     elementwise unary results and the add chain into SIMD-vectorized
//!     loops without source-level add intermediates.
//!   * Every fused-kernel input/output pointer carries the C99 `restrict`
//!     qualifier. This is the linearity → no-aliasing guarantee surfacing in
//!     the C codegen so the host compiler can vectorize aggressively.
//!
//! Net: explicit `copy()` remains visible to the IR/cost surface while Phase 3
//! reuses physical storage only where its shared proof establishes safe
//! non-overlap and exact capacity.
//!
//! ## Cost profile (computed from emitted C)
//!
//! For the probe shape `tensor[1024, 1024, f32]` (~4 MiB per buffer):
//!   * 6 physical buffers × (1024×1024×4 B) = **24 MiB peak helper storage**.
//!   * The input `x` itself is borrowed (not allocated) so it does not
//!     contribute to the helper's allocation footprint.
//!   * Every physical buffer remains live until the function epilogue; four
//!     logical results repurpose exact-capacity storage without allocating.
//!
//! Linear projection to a 2 GiB input (~22300×22300 f32 ≈ 2 GiB):
//!   * Caller-side: 1 × 2 GiB input.
//!   * Helper-side: 6 × 2 GiB physical buffers = **12 GiB peak working set**.
//!   * Total RAM with the input: ~14 GiB.

use std::fs;
use std::process::Command;

use assert_cmd::cargo::CommandCargoExt;
use chelis_crmath::c_source::{Kernel, kernel_text};
use tempfile::tempdir;

fn build_copy_elision_c_source() -> String {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("copy_elision_out");

    let status = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args([
            "build",
            "--emit-c",
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

/// Sum of bytes allocated by every `chelis_alloc(N, (int64_t[]){...}, CHELIS_<T>)`
/// call in the C source. Every Phase 3 physical slot remains live through the
/// final allocation in this probe, so this is the helper peak.
///
/// Returns (total_bytes, allocation_count, per_alloc_bytes).
pub fn measure_alloc_footprint(c_source: &str) -> (usize, usize, Vec<usize>) {
    let mut per_alloc = Vec::new();
    let mut idx = 0;
    while let Some(start) = c_source[idx..].find("chelis_alloc(") {
        let pos = idx + start;
        let after = &c_source[pos..];
        // Find the closing `)` of this alloc call. We use the matching brace
        // approach: locate the comma after rank, then the `(int64_t[]){ ... }`.
        let dtype_size = if after.contains("CHELIS_DTYPE_F64")
            && after.find("CHELIS_DTYPE_F64").unwrap() < 200
        {
            8
        } else if after.contains("CHELIS_DTYPE_BOOL")
            && after.find("CHELIS_DTYPE_BOOL").unwrap() < 200
        {
            1
        } else if after.contains("CHELIS_DTYPE_I64")
            && after.find("CHELIS_DTYPE_I64").unwrap() < 200
        {
            8
        } else if after.contains("CHELIS_DTYPE_I32")
            && after.find("CHELIS_DTYPE_I32").unwrap() < 200
        {
            4
        } else {
            // Default to f32 for CHELIS_DTYPE_F32 (most common) and unknown.
            4
        };

        // Parse the shape list. Two forms:
        //   chelis_alloc(N, (int64_t[]){ d1, d2, ... }, CHELIS_<T>)
        //   chelis_alloc(0, NULL, CHELIS_<T>)             // scalar tensor
        let bytes = if let Some(brace_open) = after.find("(int64_t[]){") {
            let dims_start = brace_open + "(int64_t[]){".len();
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

/// The unit without the correctly rounded kernel text the C backend linked
/// into it: exactly the bytes `kernel_text` gives for the entries it calls.
fn without_linked_kernels(source: &str) -> String {
    let mut called: Vec<Kernel> = source
        .split(|c: char| !(c == '_' || c.is_ascii_alphanumeric()))
        .filter_map(Kernel::from_entry)
        .collect();
    called.sort();
    called.dedup();
    assert!(
        !called.is_empty(),
        "the probe's exp, log, and sin call correctly rounded kernels"
    );
    let text = kernel_text(&called);
    assert_eq!(
        source.matches(text.as_str()).count(),
        1,
        "the unit carries its kernels' text exactly once"
    );
    source.replacen(text.as_str(), "", 1)
}

#[test]
fn copy_probe_materializes_explicit_copies_without_memcpy() {
    let source = without_linked_kernels(&build_copy_elision_c_source());

    let alloc_calls = source.matches("chelis_alloc(").count();
    let repurpose_calls = source.matches("chelis_tensor_repurpose(").count();
    let memcpy_calls = source.matches("memcpy(").count();
    let fused_kernels = source.matches("omp for simd").count();
    let restrict_qualifiers = source.matches("restrict").count();
    let borrowed_slot_wrappers = source.matches("chelis_slot").count();
    let legacy_view_allocations = source.matches("chelis_alloc_view").count();

    assert_eq!(
        memcpy_calls, 0,
        "expected zero memcpy calls: `copy(x)` materializes through tensor \
         realization loops, not raw byte copies. Got {memcpy_calls}."
    );

    assert_eq!(
        (alloc_calls, repurpose_calls),
        (6, 4),
        "expected Phase 3 to materialize ten logical tensor values as six fresh \
         physical allocations and four exact-capacity repurposes. A different \
         split changes the temporary ownership/cost profile."
    );
    assert_eq!(
        alloc_calls + repurpose_calls,
        10,
        "explicit copy materialization and fused fan-in must still produce all \
         ten logical tensor values even when physical storage is reused"
    );

    assert_eq!(
        (borrowed_slot_wrappers, legacy_view_allocations),
        (0, 0),
        "Phase 3 must reuse proven storage directly, without borrowed slot \
         wrappers or the removed chelis_alloc_view path"
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

    // For tensor[1024, 1024, f32], each physical buffer is 4 MiB. Phase 3
    // allocates six slots and repurposes four of them for later logical values.
    let expected = 6 * 1024 * 1024 * 4; // 24 MiB
    assert_eq!(
        total_bytes, expected,
        "expected peak C physical-buffer footprint of 24 MiB (6 × 4 MiB), \
         got {total_bytes} bytes ({mib:.2} MiB). Per-alloc bytes: {per:?}"
    );

    let last_allocation = source.rfind("chelis_alloc(").expect("owned allocation");
    let first_terminal_release = source
        .find("    chelis_tensor_release(t")
        .expect("terminal tensor release");
    assert!(
        first_terminal_release > last_allocation,
        "the summed allocation footprint is a peak only while every temporary \
         survives through the final allocation"
    );

    // Linear projection: scale input from 4 MiB (1024×1024 f32) to 2 GiB
    // (~512× larger). Six physical buffers scale to a 12 GiB helper-side peak.
    let scale_to_2gib = (2_u64 * 1024 * 1024 * 1024) / (1024 * 1024 * 4);
    let projected_2gib_peak_bytes = (total_bytes as u64) * scale_to_2gib;
    let projected_2gib_peak_gib = projected_2gib_peak_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "Linear projection to 2 GiB input: helper-side owned-buffer footprint ≈ {projected_2gib_peak_gib:.1} GiB \
         (excludes the 2 GiB input itself). With the borrowed input: ~{:.1} GiB total.",
        projected_2gib_peak_gib + 2.0
    );
}
