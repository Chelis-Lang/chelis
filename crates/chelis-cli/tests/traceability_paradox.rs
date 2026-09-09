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
//! kernels, number of BLAS specializations actually fired, and the verified
//! Phase 3 buffer allocation/reuse footprint in bytes.
//!
//! The emitted spans retain Surf byte ranges through fusion. C contractions
//! retain their primitive arithmetic, so this fixture has no vendor GEMM
//! substitutions and keeps the dense product intermediates.
//!
//! The inspected ownership plan produces 74 tensors with 32 physical owned
//! allocations and 42 descriptor repurposes. Those physical tensor buffers
//! occupy `1028 + 3357720 * seq + 780 * seq^2` bytes. This measures the retained
//! tensor storage only: inputs, runtime metadata, and temporary reduction-tree
//! scratch are outside the polynomial. It is not a whole-process peak estimate.
//! Any future optimization must preserve the decided arithmetic and update
//! these measurements from its emitted ownership plan.

use std::fs;
use std::process::Command;

use assert_cmd::cargo::CommandCargoExt;
use tempfile::tempdir;

fn is_owned_tensor_allocation(line: &str) -> bool {
    let Some(after_prefix) = line.trim_start().strip_prefix("chelis_tensor *t") else {
        return false;
    };
    let Some((name_suffix, _arguments)) = after_prefix.split_once(" = chelis_alloc(") else {
        return false;
    };
    !name_suffix.is_empty() && name_suffix.bytes().all(|byte| byte.is_ascii_digit())
}

fn is_owned_tensor_release(line: &str) -> bool {
    let Some(name_suffix) = line
        .trim()
        .strip_prefix("chelis_tensor_release(t")
        .and_then(|suffix| suffix.strip_suffix(");"))
    else {
        return false;
    };
    !name_suffix.is_empty() && name_suffix.bytes().all(|byte| byte.is_ascii_digit())
}

/// Parse physical owned `tN = chelis_alloc(...)` calls and compute a polynomial
/// in `seq` describing retained physical tensor storage. Returns
/// (allocation_count, constant_bytes, seq1_bytes, seq2_bytes) such
/// that total ≈ constant + seq * seq1 + seq * seq * seq2.
///
/// Unhandled allocations (e.g. `chelis_alloc(0, NULL, ...)`) contribute
/// `dtype_size` bytes each to the constant term.
fn measure_seq_polynomial(c_source: &str) -> (usize, usize, usize, usize) {
    let mut alloc_count = 0;
    let mut const_bytes = 0usize;
    let mut seq1_bytes = 0usize;
    let mut seq2_bytes = 0usize;

    for line in c_source
        .lines()
        .filter(|line| is_owned_tensor_allocation(line))
    {
        let pos = line
            .find("chelis_alloc(")
            .expect("filtered allocation line");
        let after = &line[pos..];
        alloc_count += 1;

        // Look ahead a bounded window — long enough to cover the type tag.
        let window_end = (256).min(after.len());
        let window = &after[..window_end];
        let dtype_size: usize = if window.contains("CHELIS_DTYPE_F64") {
            8
        } else if window.contains("CHELIS_DTYPE_BOOL") {
            1
        } else if window.contains("CHELIS_DTYPE_I64") {
            8
        } else {
            // CHELIS_DTYPE_F32 / CHELIS_DTYPE_I32 / unknown all default to 4 bytes.
            4
        };

        // Find the shape list `(int64_t[]){ ... }`.
        if let Some(brace_open) = window.find("(int64_t[]){") {
            let dims_start = brace_open + "(int64_t[]){".len();
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
    let owned_allocations = source
        .lines()
        .filter(|line| is_owned_tensor_allocation(line))
        .count();
    let storage_repurposes = source.matches("chelis_tensor_repurpose(").count();
    let slot_allocations = source.matches("chelis_tensor *chelis_slot").count();
    let legacy_view_allocations = source.matches("chelis_alloc_view").count();
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
    assert_eq!(
        owned_allocations, 32,
        "expected the verified plan to map 74 produced tensors to \
         32 physical owned tN allocations; got \
         {owned_allocations}"
    );
    assert_eq!(
        storage_repurposes, 42,
        "expected the verified Phase 3 plan to repurpose exactly 42 physical \
         slots for the remaining produced tensors; got {storage_repurposes}"
    );
    assert_eq!(
        owned_allocations + storage_repurposes,
        74,
        "every transformer result must be accounted for by either a fresh \
         physical allocation or a proof-authorized descriptor repurpose"
    );
    assert_eq!(
        (slot_allocations, legacy_view_allocations),
        (0, 0),
        "Phase 3 reuse must use proof-authorized tN aliases and descriptor \
         repurposes, not borrowed chelis_slot wrappers or chelis_alloc_view"
    );
    assert_eq!(
        blas_calls, 0,
        "primitive contractions must keep their canonical arithmetic"
    );
    for dense_product_shape in [
        "(int64_t[]){ seq, 256, 64 }",
        "(int64_t[]){ seq, 64, seq }",
        "(int64_t[]){ seq, 256, 1024 }",
    ] {
        assert!(
            source.contains(dense_product_shape),
            "the retained primitive graph includes this product buffer: {dense_product_shape}"
        );
    }

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
        eprintln!("  tensor bytes at seq = {seq:>5}: {total:>12} bytes ({mib:>8.2} MiB)");
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
    assert_eq!(
        (alloc_count, c0, c1, c2),
        (32, 1028, 3_357_720, 780),
        "unexpected transformer_block Phase 3 physical working-set polynomial; \
         update the locked cost profile only after inspecting the emitted C and \
         its proof-authorized descriptor repurposes"
    );

    let final_owned_allocation = source
        .lines()
        .enumerate()
        .filter_map(|(line_number, line)| is_owned_tensor_allocation(line).then_some(line_number))
        .last()
        .expect("owned tensor allocation");
    let first_owned_release = source
        .lines()
        .position(is_owned_tensor_release)
        .expect("owned tensor release");
    assert!(
        first_owned_release > final_owned_allocation,
        "the summed owned-allocation polynomial is simultaneously live only while every \
         temporary survives through the final allocation"
    );
}
