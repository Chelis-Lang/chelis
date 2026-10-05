//! S6 named oracle — end-to-end Black-Scholes audit chain on the natural
//! scalar form per `spec/design/chelis_span_survival.md` §9 canary recipe.
//!
//! With S6 step 5 landing per-`HostExpr` span emission in
//! `chelis-backend-c::host_emit` and S6 step 6 rewriting the
//! `call_price_wrapped.dp` fixture to its natural scalar shape (which
//! routes through that host-emit path), the audit chain
//!
//!   LaTeX byte range → Deep node → IR/HostExpr node → C source line
//!
//! is exercisable end-to-end on the canonical customer-shape program.
//! This test is the S6 named acceptance oracle.
//!
//! Recipe (mirrors §9 of the spec):
//!   1. `chelis build call_price_wrapped.dp --deep --target c -o /tmp/bs.c`
//!   2. `grep -c '// span:' /tmp/bs.c >= sidecar entry count`
//!   3. Pick a span ID from the emitted C; resolve it through
//!      `call_price_wrapped.spans.json` to its `latex_text`.
//!
//! The test asserts the audit invariant AT THE EMISSION LEVEL — every
//! input Deep span appears as a `// span:` comment in the emitted
//! host-side C source — not just on intermediate IR nodes.

use assert_cmd::Command;
use std::collections::BTreeSet;
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

#[test]
fn s6_oracle_audit_chain_resolves_span_to_latex_text() {
    // §9 canary: build the scalar-shape wrapped fixture, count spans in
    // the emitted host C, pick a representative span ID, and resolve it
    // through the sidecar to its original LaTeX byte range.
    let dir = tempdir().expect("tempdir");
    let out = dir.path().join("bs.c");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
    let sidecar_text = fs::read_to_string(wrapped_spans_json()).expect("read sidecar");
    let sidecar: serde_json::Value = serde_json::from_str(&sidecar_text).expect("parse spans json");
    let sidecar_arr = sidecar["spans"].as_array().expect("sidecar spans array");
    let sidecar_count = sidecar_arr.len();

    // §9 step 1: count.
    let emitted_count = src.matches("// span:").count();
    assert!(
        emitted_count >= sidecar_count,
        "§9 audit canary: emitted span count must be >= sidecar entry count \
         (emitted={emitted_count}, sidecar entries={sidecar_count}); \
         this is the load-bearing emission-level audit invariant"
    );

    // §9 step 2: every sidecar span ID must surface as a `// span:`
    // line in the emitted C. This is the AT-THE-EMISSION-LEVEL
    // invariant — host_emit (S6 step 5) is the audit-trace resolution
    // target per spec §2.4.2 (host source IS where audit traces
    // resolve), so the chain breaks immediately if any input span goes
    // missing in the emitted C.
    let emitted_spans: BTreeSet<String> = src
        .lines()
        .filter_map(|l| {
            let l = l.trim_start();
            l.strip_prefix("// span: ").map(str::to_string)
        })
        .collect();

    let mut missing: Vec<&str> = Vec::new();
    for entry in sidecar_arr {
        let deep_id = entry["deep_node_id"].as_str().unwrap();
        if !emitted_spans.contains(deep_id) {
            missing.push(deep_id);
        }
    }
    assert!(
        missing.is_empty(),
        "§9 audit canary: every sidecar span ID must appear in emitted C; \
         missing: {missing:?}\n\
         (emitted spans: {emitted_spans:?})"
    );

    // §9 step 3: pick a representative span ID from the emitted C and
    // resolve it through the sidecar to its `latex_text`. The recipe in
    // the spec uses the FIRST span; we pick `eq:d1_002` (Octant's
    // canonical d_1 span ID, the head of the d_1 LaTeX equation) to
    // avoid coupling to emission ordering — the oracle proves the
    // chain regardless of which span we pick first.
    let target_id = "eq:d1_002";
    assert!(
        emitted_spans.contains(target_id),
        "oracle: representative span `{target_id}` must appear in emitted C"
    );
    let entry = sidecar_arr
        .iter()
        .find(|e| e["deep_node_id"].as_str() == Some(target_id))
        .unwrap_or_else(|| panic!("oracle: sidecar must carry `{target_id}`"));
    let latex_text = entry["latex_text"]
        .as_str()
        .expect("sidecar entry has latex_text");
    // For eq:d1_002 the LaTeX is the full d_1 RHS (a fraction). The
    // exact byte string is locked by the sidecar; we assert
    // non-emptiness and that it contains the canonical d_1 marker
    // `\\frac` (LaTeX) rather than over-specifying the byte sequence.
    assert!(
        !latex_text.is_empty(),
        "oracle: sidecar `latex_text` for `{target_id}` must be non-empty"
    );
    assert!(
        latex_text.contains("\\frac"),
        "oracle: `latex_text` for `{target_id}` must contain `\\frac` (the d_1 RHS \
         is a LaTeX fraction); got `{latex_text}`"
    );
    let latex = entry["latex"]
        .as_object()
        .expect("sidecar entry has latex byte-range");
    let start_byte = latex["start_byte"].as_u64().unwrap_or(0);
    let end_byte = latex["end_byte"].as_u64().unwrap_or(0);
    assert!(
        end_byte > start_byte,
        "oracle: byte range for `{target_id}` must be non-empty \
         (start={start_byte}, end={end_byte})"
    );

    // S6 backward-compat lock: the emitted C must compile via the platform's
    // default C compiler. Drop `-fopenmp` for cross-platform portability —
    // Apple's clang (which `gcc` resolves to on macOS CI) doesn't support
    // OpenMP without libomp, and the audit-chain assertion is about span
    // comments surviving emission as valid C, not about OpenMP runtime
    // wiring. Linux gcc silently ignores `#pragma omp …` directives without
    // `-fopenmp`; the compile-success check still holds.
    let compile = std::process::Command::new("gcc")
        .args([
            "-O0",
            "-c",
            out.to_str().unwrap(),
            "-I",
            dir.path().to_str().unwrap(),
            "-o",
            dir.path().join("bs.o").to_str().unwrap(),
        ])
        .output()
        .expect("gcc must be available on the test host");
    assert!(
        compile.status.success(),
        "S6 oracle: emitted host C must compile via gcc; stderr:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );
}

#[test]
fn s6_backward_compat_span_free_dp_emits_no_span_comments_on_host_path() {
    // S6 backward-compat lock: span-free Deep input on the host path
    // (scalar f32 throughout, routing through `host_emit`) emits ZERO
    // `// span:` comments. The S6 step 5 host_emit emitter must not
    // fabricate spans — the no-op path is load-bearing for span-free
    // programs (today's hand-written Chelis).
    let dir = tempdir().expect("tempdir");
    let dp_path = dir.path().join("nospan_scalar.dp");
    // Pure-scalar f32 program with no span metadata anywhere. Reuses
    // the same host_emit code path the wrapped fixture exercises.
    fs::write(
        &dp_path,
        r#"(def {} double_it
  (fn {}
    (params {} (x {type: (t-prim {} f32)}))
    (app {} (var {} mul) (var {} x) (lit {type: (t-prim {} f32)} 2.0))))
"#,
    )
    .expect("write dp");
    let out = dir.path().join("nospan_scalar.c");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            dp_path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out.to_str().unwrap(),
        ])
        .assert()
        .success();

    let src = fs::read_to_string(&out).expect("read emitted C");
    let span_count = src.matches("// span:").count();
    // Span-free Deep MAY pick up optimization-pass synthesized markers
    // (Tier 2, etc.) in the DAG path, but the host_emit path itself
    // doesn't run those passes — span-free input means span-free
    // output. Strictest correct assertion is "zero".
    assert_eq!(
        span_count, 0,
        "span-free Deep on host path must emit zero `// span:` comments \
         (S6 backward-compat lock); got {span_count}\n\nSource:\n{src}"
    );
}
