//! Wave 5 red-team — Adjacent (M2b) span survival across multiple corpora.
//!
//! The existing `traceability_paradox.rs::transformer_block_traceability_state_is_locked`
//! verifies Surf byte-range spans survive every pass on `transformer_block.ch`.
//! The brief explicitly asked for span survival on **multiple corpora**, not
//! just one. This file probes two additional executable Phase-0 examples:
//!
//! * `linreg.ch` — rank-2 matmul + bias + elementwise; smaller MLP shape.
//! * `hello_tensor.ch` — to_tensor + add — minimal tensor program; tests
//!   the span-survival path through the smallest possible non-trivial DAG.
//!
//! For each: after `chelis build --target c`, the emitted C source must
//! contain at least one `// span: surf:<start>..<end>` line and the
//! synthesized-only line count must be strictly less than total span
//! lines (M2b's "ordinary Surf bodies have real spans" invariant).
//!
//! The brief: "Synthesized markers should appear only for genuinely
//! spanless inputs." For `chelis build` from a .ch source file, NOTHING
//! is genuinely spanless — every node should have a Surf byte range or
//! a synthesized tag inherited from a parent that does. This test
//! requires `surf_spans > 0` and asserts that even small programs see
//! the survival path.

use std::process::Command;

use assert_cmd::cargo::CommandCargoExt;
use tempfile::tempdir;

/// Build a Surf example through the CLI and return the emitted C source.
fn build_emit_c(example_rel_path: &str, out_subdir: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join(out_subdir);
    let status = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args([
            "build",
            example_rel_path,
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .status()
        .expect("chelis build should run");
    assert!(
        status.success(),
        "chelis build failed for {example_rel_path}"
    );

    // The C file name is derived from the source stem.
    let stem = example_rel_path
        .rsplit('/')
        .next()
        .unwrap()
        .trim_end_matches(".ch");
    let c_path = out_dir.join(format!("{stem}.c"));
    std::fs::read_to_string(&c_path).expect("read generated c")
}

fn count_span_lines(source: &str) -> (usize, usize, usize) {
    let span_lines: Vec<&str> = source.lines().filter(|l| l.contains("// span:")).collect();
    let total = span_lines.len();
    let synthesized = span_lines
        .iter()
        .filter(|l| l.contains("__synthesized"))
        .count();
    let surf = span_lines
        .iter()
        .filter(|l| l.contains("// span: surf:"))
        .count();
    (total, synthesized, surf)
}

/// linreg.ch — rank-2 matmul + add + elementwise + reduce. Real Surf
/// source, so Surf byte-range spans must reach the emitted C.
#[test]
fn linreg_corpus_surf_spans_survive_to_c_codegen() {
    let source = build_emit_c("../../examples/linreg.ch", "linreg_out");
    let (total, synthesized, surf) = count_span_lines(&source);
    assert!(
        total > 0,
        "linreg.ch: expected `// span:` lines in emitted C; got zero"
    );
    assert!(
        surf > 0,
        "linreg.ch: expected at least one `// span: surf:<start>..<end>` line \
         (M2b adjacent invariant); got {surf} surf spans out of {total} total. \
         A synthesized-only state regresses the traceability claim."
    );
    assert!(
        synthesized < total,
        "linreg.ch: expected ordinary Surf-source spans to replace the legacy \
         all-synthesized traceability state; got {synthesized} synthesized \
         out of {total} total."
    );
}

/// hello_tensor.ch — minimal viable Surf program. Even at this size, the
/// span-survival invariant must hold; if it doesn't, M2b is fragile to
/// shape rather than universally enforced.
#[test]
fn hello_tensor_corpus_surf_spans_survive_to_c_codegen() {
    let source = build_emit_c("../../examples/hello_tensor.ch", "hello_tensor_out");
    let (total, synthesized, surf) = count_span_lines(&source);
    assert!(
        total > 0,
        "hello_tensor.ch: expected `// span:` lines in emitted C; got zero"
    );
    assert!(
        surf > 0,
        "hello_tensor.ch: expected at least one Surf byte-range span; got \
         {surf} surf out of {total} total. A minimal-program span miss \
         indicates M2b's coverage is shape-dependent."
    );
    assert!(
        synthesized < total,
        "hello_tensor.ch: got {synthesized} synthesized out of {total} total"
    );
}
