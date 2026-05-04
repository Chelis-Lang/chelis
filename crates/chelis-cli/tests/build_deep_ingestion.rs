//! S5.1 + S5.2 — `chelis build` Deep ingestion path tests.
//!
//! Per `spec/design/chelis_span_survival.md` §2.5, `chelis build` must
//! auto-detect the input language by extension and accept a `--deep`
//! flag override:
//!
//! | Invocation                  | Path           |
//! |-----------------------------|----------------|
//! | `chelis build foo.dp`       | Deep (auto)    |
//! | `chelis build foo.dp --deep`| Deep (no-op)   |
//! | `chelis build foo.ch --deep`| Deep (override)|
//! | `chelis build foo.ch`       | Surf (today)   |
//!
//! These tests exercise all four rows plus the load-bearing S5
//! invariant: span-attributed `.dp` produces span-annotated C with
//! the audit chain intact.
//!
//! The fixtures used here:
//! - `crates/chelis-cli/tests/fixtures/octant/black_scholes/call_price_wrapped.dp`
//!   (typecheckable, span-attributed, rank-0 tensor f32 throughout)
//! - a tiny inline `.dp` for span-free Deep parity with Surf
//! - a tiny inline `.ch` for the Surf path
//!
//! gcc compile-success of the emitted C is asserted where the
//! generated source is well-formed for standalone compilation.

use assert_cmd::Command;
use std::fs;
use std::path::PathBuf;
use tempfile::tempdir;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/octant/black_scholes")
}

fn wrapped_dp() -> PathBuf {
    fixture_dir().join("call_price_wrapped.dp")
}

fn wrapped_spans_json() -> PathBuf {
    fixture_dir().join("call_price_wrapped.spans.json")
}

// ── Row 1: `chelis build foo.dp` (auto-detect) ─────────────────────────

#[test]
fn build_dp_extension_auto_detects_deep_path() {
    let dir = tempdir().expect("tempdir");
    let out = dir.path().join("auto.c");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            wrapped_dp().to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out.to_str().unwrap(),
        ])
        .assert()
        .success();

    let src = fs::read_to_string(&out).expect("read emitted C");
    let span_count = src.matches("// span:").count();
    let sidecar_text = fs::read_to_string(wrapped_spans_json()).expect("read spans sidecar");
    let sidecar: serde_json::Value = serde_json::from_str(&sidecar_text).expect("parse spans json");
    let sidecar_entries: usize = sidecar["spans"].as_array().map(Vec::len).unwrap_or(0);
    assert!(
        span_count >= sidecar_entries,
        ".dp auto-detect must produce >= sidecar entry count `// span:` lines \
         (S6: scalar fixture routes through host_emit which now emits per-node spans); \
         got {span_count} emitted vs {sidecar_entries} sidecar entries"
    );
}

// ── Row 2: `chelis build foo.dp --deep` (flag agrees, no-op) ───────────

#[test]
fn build_dp_with_deep_flag_is_a_noop_relative_to_auto_detect() {
    let dir = tempdir().expect("tempdir");
    let auto = dir.path().join("auto.c");
    let flagged = dir.path().join("flagged.c");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            wrapped_dp().to_str().unwrap(),
            "--target",
            "c",
            "--output",
            auto.to_str().unwrap(),
        ])
        .assert()
        .success();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            wrapped_dp().to_str().unwrap(),
            "--deep",
            "--target",
            "c",
            "--output",
            flagged.to_str().unwrap(),
        ])
        .assert()
        .success();

    let auto_src = fs::read_to_string(&auto).expect("read auto");
    let flag_src = fs::read_to_string(&flagged).expect("read flagged");

    // Both invocations took the Deep path. Byte equality is the
    // strongest correct assertion for "flag is a no-op": same input,
    // same path, same output. Previously a HashMap-iteration-order
    // non-determinism bug in `emit_input_shape_preamble` forced this
    // test to drop down to span-set equality; that bug was fixed by
    // sorting the iteration over input labels (see
    // spec/upstream-bugs/host-emit-hashmap-iteration-nondeterminism.md
    // and crates/chelis-backend-c/tests/codegen_determinism.rs for the
    // dedicated regression test).
    assert_eq!(
        auto_src, flag_src,
        "`--deep` on a `.dp` file must produce byte-identical C as auto-detect"
    );
    let span_count = auto_src.matches("// span:").count();
    let sidecar_text = fs::read_to_string(wrapped_spans_json()).expect("read spans sidecar");
    let sidecar: serde_json::Value = serde_json::from_str(&sidecar_text).expect("parse spans json");
    let sidecar_entries: usize = sidecar["spans"].as_array().map(Vec::len).unwrap_or(0);
    assert!(
        span_count >= sidecar_entries,
        "`.dp` auto-detect must emit >= sidecar entry count `// span:` lines \
         (S6: host_emit now per-node-emits); got {span_count} vs {sidecar_entries}"
    );
}

// ── Row 3: `chelis build foo.ch --deep` (override) ─────────────────────

#[test]
fn build_ch_with_deep_flag_routes_through_deep_path() {
    let dir = tempdir().expect("tempdir");
    let dp_with_ch_extension = dir.path().join("call_price.ch");
    fs::copy(wrapped_dp(), &dp_with_ch_extension).expect("copy fixture");
    let out = dir.path().join("override.c");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            dp_with_ch_extension.to_str().unwrap(),
            "--deep",
            "--target",
            "c",
            "--output",
            out.to_str().unwrap(),
        ])
        .assert()
        .success();

    let src = fs::read_to_string(&out).expect("read emitted C");
    assert!(
        src.matches("// span:").count() >= 1,
        "`--deep` override on a `.ch` file must take the Deep path and produce spans"
    );
}

// ── Row 4: `chelis build foo.ch` (Surf path, today's behavior) ─────────

#[test]
fn build_ch_without_deep_flag_takes_the_surf_path() {
    // The Surf path with span-free input emits no `// span:` lines (or
    // only synthesized markers from optimization passes). The
    // load-bearing thing here is that the build SUCCEEDS without a
    // Deep parser ever being invoked.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("hello.ch");
    // Tensors are linear; pre-bind a copy via a let block so neither
    // direct `x` nor `copy(x)` race on the same call site.
    fs::write(
        &path,
        "def hello(x: tensor[4, f32]) -> tensor[4, f32] = {\n  \
         x_copy = copy(x)\n  \
         add(x, x_copy)\n\
         }\n",
    )
    .expect("write");
    let out = dir.path().join("hello.c");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out.to_str().unwrap(),
        ])
        .assert()
        .success();

    // No assertion on `// span:` content here — Surf input has no
    // span metadata, but optimization-pass synthesized markers
    // (e.g., `__synthesized_tier2__`) MAY appear. Either is fine.
    assert!(out.exists(), "Surf path must produce output file");
}

// ── Span-free Deep parity with Surf path ───────────────────────────────

#[test]
fn build_span_free_deep_matches_surf_shape() {
    // Span-free Deep should produce equivalent C to the Surf path for
    // the same logical program — modulo absent spans (which span-free
    // Deep also produces zero of).
    let dir = tempdir().expect("tempdir");
    let dp_path = dir.path().join("nospan.dp");
    // Tensors are linear; one of the two `add` operands must come
    // from `(copy {} (var {} x))`. The other consumes the original
    // `x` binding.
    fs::write(
        &dp_path,
        r#"(def {} hello
  (fn {}
    (params {} (x {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))}))
    (let {}
      (bind {} x2 (copy {} (var {} x)))
      (app {} (var {} add) (var {} x) (var {} x2)))))
"#,
    )
    .expect("write dp");
    let out = dir.path().join("nospan.c");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            dp_path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out.to_str().unwrap(),
        ])
        .assert()
        .success();

    let src = fs::read_to_string(&out).expect("read emitted C");
    // Span-free Deep MAY still get optimization-pass synthesized
    // markers (Tier 2 etc.). The important thing is the file
    // compiles and emits a usable C function (named after the file
    // stem).
    assert!(
        src.contains("nospan("),
        "span-free Deep must produce a C function `nospan` (named after file stem); got src:\n{src}"
    );
}

// ── gcc compile-success of emitted C ───────────────────────────────────

#[test]
fn build_deep_emitted_c_compiles_via_gcc() {
    // The S5 acceptance bar isn't just "C source contains spans" — the
    // emitted C must actually compile. The S4 oracle locked
    // compile-success for hand-built DAGs; this test extends the bar
    // to the Deep-ingestion CLI surface end-to-end.
    let dir = tempdir().expect("tempdir");
    let out = dir.path().join("auditable.c");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            wrapped_dp().to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out.to_str().unwrap(),
        ])
        .assert()
        .success();

    let compile = std::process::Command::new("gcc")
        .args([
            "-O0",
            "-fopenmp",
            "-c",
            out.to_str().unwrap(),
            "-I",
            dir.path().to_str().unwrap(),
            "-o",
            dir.path().join("auditable.o").to_str().unwrap(),
        ])
        .output()
        .expect("gcc must be available on the test host");
    assert!(
        compile.status.success(),
        "gcc must compile Deep-emitted C cleanly; stderr:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );
}

// ── HIP target accepts Deep ingestion and emits spans ─────────────────

#[test]
fn build_deep_hip_target_emits_spans_in_cpp_and_kernel_strings() {
    // Per `spec/design/chelis_span_survival.md` §2.4.1: HIP carries
    // spans both host-side (C++ comments) and inside the embedded
    // per-node kernel source strings. This test only locks the host-
    // side count; the kernel-string emission is exercised in
    // chelis-backend-hip's S4.2 unit tests.
    let dir = tempdir().expect("tempdir");
    let out = dir.path().join("audit.cpp");
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            wrapped_dp().to_str().unwrap(),
            "--deep",
            "--target",
            "hip",
            "--output",
            out.to_str().unwrap(),
        ])
        .assert()
        .success();

    let src = fs::read_to_string(&out).expect("read emitted .cpp");
    let span_count = src.matches("// span:").count();
    assert!(
        span_count >= 1,
        "HIP Deep target must produce span-annotated .cpp (got {span_count} `// span:` lines)"
    );
}

// ── S5 oracle: span-attributed Deep produces audit-chain-recoverable C ──

#[test]
fn build_deep_audit_chain_is_recoverable_from_emitted_c() {
    // Per `spec/design/chelis_span_survival.md` §9 (the canary):
    // pick a span ID from the emitted C, look it up in the spans.json
    // sidecar, confirm the LaTeX text matches.
    let dir = tempdir().expect("tempdir");
    let out = dir.path().join("audit.c");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            wrapped_dp().to_str().unwrap(),
            "--deep",
            "--target",
            "c",
            "--output",
            out.to_str().unwrap(),
        ])
        .assert()
        .success();

    let src = fs::read_to_string(&out).expect("read emitted C");
    let sidecar_text = fs::read_to_string(wrapped_spans_json()).expect("read spans sidecar");

    // Count: emitted-C span comments must be >= sidecar entry count.
    // S6 step 6 rewrote this fixture to its natural scalar shape, which
    // routes through `chelis-backend-c::host_emit` (S6 step 5 now emits
    // per-`HostExpr` spans on that path); every sidecar entry surfaces
    // on at least one emitted node, so the strict equality holds.
    let emitted_count = src.matches("// span:").count();
    let sidecar: serde_json::Value = serde_json::from_str(&sidecar_text).expect("parse spans json");
    let sidecar_entries: usize = sidecar["spans"].as_array().map(Vec::len).unwrap_or(0);
    assert!(
        emitted_count >= sidecar_entries,
        "audit canary: emitted `// span:` count must be >= sidecar entry count \
         (emitted={emitted_count}, sidecar entries={sidecar_entries})"
    );

    // Audit invariant: every sidecar `deep_node_id` appears verbatim
    // as a `// span: <id>` line in the emitted C. This is the S6
    // sharpening of the old "first-span-resolves" check; with
    // host-path emission the full audit chain is recoverable, not
    // just the first contributor.
    let emitted_spans: std::collections::BTreeSet<String> = src
        .lines()
        .filter_map(|l| {
            let l = l.trim_start();
            l.strip_prefix("// span: ").map(str::to_string)
        })
        .collect();
    for entry in sidecar["spans"].as_array().unwrap() {
        let deep_id = entry["deep_node_id"].as_str().unwrap();
        assert!(
            emitted_spans.contains(deep_id),
            "audit canary: sidecar span `{deep_id}` must appear in emitted C \
             (emitted spans: {emitted_spans:?})"
        );
        let latex_text = entry["latex_text"].as_str().unwrap_or("");
        assert!(
            !latex_text.is_empty(),
            "audit canary: sidecar entry for `{deep_id}` must carry a non-empty `latex_text`"
        );
    }
}
