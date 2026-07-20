//! S6 step 5 — host-path C backend emits `// span: <id>` comments
//! preceding each `HostExpr` node.
//!
//! Per `spec/design/chelis_span_survival.md` §2.4 (host-path emission rule)
//! and §2.3 host-side table (host emit row): backend host-side emission
//! prepends one `// span:` line per `span_id ∪ merged_spans` immediately
//! before the C statement(s) implementing the node. Order: canonical
//! first, then `merged_spans` lex-sorted (deterministic).
//!
//! These tests exercise `chelis_backend_c::host_emit::emit_host_program`
//! directly with hand-built `HostProgram` / `HostFunction` / `HostExpr`
//! values, so they cover the host-emit code path even when the upstream
//! lowering pipeline elides spans.
//!
//! Mirrors the DAG-side coverage in `span_comments.rs`.

use chelis_backend_c::host_emit::emit_host_program;
use chelis_ir::host::{HostExpr, HostExprKind, HostFunction, HostParam, HostProgram, HostType};

fn make_program(body: HostExpr) -> HostProgram {
    HostProgram {
        globals: Vec::new(),
        global_tensor_helpers: Vec::new(),
        functions: vec![HostFunction {
            name: "the_fn".to_string(),
            params: vec![HostParam {
                name: "x".to_string(),
                ty: HostType::Float64,
            }],
            ret_ty: HostType::Float64,
            body,
            tensor_helpers: Vec::new(),
            specialization: None,
            summary_rejections: Vec::new(),
        }],
        summary_rejections: Vec::new(),
    }
}

#[test]
fn s6_host_canonical_span_id_emitted_as_comment() {
    // A scalar Var expression carrying a span_id.
    let body = HostExpr::with_span(
        HostExprKind::Var("x".to_string(), HostType::Float64),
        Some("op.var".to_string()),
    );
    let program = make_program(body);
    let src = emit_host_program(&program, "host_canonical").unwrap();

    assert!(
        src.contains("// span: op.var"),
        "expected `// span: op.var` in host C source:\n{src}"
    );
}

#[test]
fn s6_host_merged_spans_emitted_lex_sorted_after_canonical() {
    // Hand-craft a HostExpr with span_id + merged_spans (the typical S3+S6
    // shape after host-side lowering N→1 collapses).
    let mut body = HostExpr::with_span(
        HostExprKind::Var("x".to_string(), HostType::Float64),
        Some("op.x".to_string()),
    );
    body.merged_spans = vec!["op.b".into(), "op.a".into(), "op.c".into()];

    let program = make_program(body);
    let src = emit_host_program(&program, "host_merged").unwrap();

    let span_lines: Vec<&str> = src
        .lines()
        .map(str::trim_start)
        .filter(|l| l.starts_with("// span:"))
        .collect();

    // Canonical first, then merged lex-sorted: op.x, op.a, op.b, op.c.
    assert_eq!(
        span_lines,
        vec![
            "// span: op.x",
            "// span: op.a",
            "// span: op.b",
            "// span: op.c"
        ],
        "expected canonical first then merged lex-sorted; got:\n{span_lines:?}\n\nFull source:\n{src}"
    );
}

#[test]
fn s6_host_merged_spans_dedup_against_canonical() {
    // Defensive: even if a merged span equals the canonical span_id, the
    // emitter must dedupe — same shape as the DAG-side dedup test.
    let mut body = HostExpr::with_span(
        HostExprKind::Var("x".to_string(), HostType::Float64),
        Some("op.dup".to_string()),
    );
    body.merged_spans = vec!["op.dup".into(), "op.other".into()];

    let program = make_program(body);
    let src = emit_host_program(&program, "host_dedup").unwrap();

    let canonical_count = src.matches("// span: op.dup").count();
    assert_eq!(
        canonical_count, 1,
        "merged_span equal to canonical span_id must be deduped; saw {canonical_count} occurrences"
    );
    assert!(
        src.contains("// span: op.other"),
        "non-duplicate merged span must still appear:\n{src}"
    );
}

#[test]
fn s6_host_no_spans_emits_no_comment_block() {
    // Backwards-compat: span-free HostExpr (the common case for hand-
    // written Chelis or for span-free Deep input) must produce zero
    // `// span:` lines on the host path.
    let body = HostExpr::new(HostExprKind::Var("x".to_string(), HostType::Float64));
    let program = make_program(body);
    let src = emit_host_program(&program, "host_nospan").unwrap();

    assert!(
        !src.contains("// span:"),
        "span-free HostExpr must not emit any `// span:` comments:\n{src}"
    );
}

#[test]
fn s6_host_forbidden_newline_in_span_is_escaped_at_emit() {
    // The injection probe — without sanitization, `// span: op\nint
    // INJECTED = 42;` would terminate the line comment and emit live C.
    // The shared `chelis_ir::span_sanitize::sanitize_for_comment` must
    // apply on the host path too.
    let body = HostExpr::with_span(
        HostExprKind::Var("x".to_string(), HostType::Float64),
        Some("op\nint INJECTED_HOST = 42;".to_string()),
    );
    let program = make_program(body);
    let src = emit_host_program(&program, "host_forbidden_newline").unwrap();

    assert!(
        src.contains("// span: op\\nint INJECTED_HOST = 42;"),
        "expected escaped `\\n` form in host `// span:` comment; source:\n{src}"
    );
    assert!(
        !src.contains("\nint INJECTED_HOST = 42;"),
        "raw injected line escaped the host-path sanitizer; source:\n{src}"
    );
}

#[test]
fn s6_host_forbidden_newline_in_merged_spans_is_escaped_at_emit() {
    // Same probe but via merged_spans on the host path.
    let mut body = HostExpr::with_span(
        HostExprKind::Var("x".to_string(), HostType::Float64),
        Some("op.clean".to_string()),
    );
    body.merged_spans = vec!["op\nint INJECTED_VIA_HOST_MERGED = 1;".into()];

    let program = make_program(body);
    let src = emit_host_program(&program, "host_forbidden_merged").unwrap();

    assert!(
        src.contains("// span: op\\nint INJECTED_VIA_HOST_MERGED = 1;"),
        "host merged-span path missing escape; source:\n{src}"
    );
    assert!(
        !src.contains("\nint INJECTED_VIA_HOST_MERGED = 1;"),
        "host merged-span injection escaped the sanitizer; source:\n{src}"
    );
}

#[test]
fn s6_host_clean_span_emitted_verbatim_audit_invariant() {
    // Audit invariant lock: a clean span ID (no forbidden chars) must
    // emit byte-identical to its input on the host path. Same invariant
    // as the DAG-side `s4_c_clean_span_emitted_verbatim_audit_invariant`.
    let body = HostExpr::with_span(
        HostExprKind::Var("x".to_string(), HostType::Float64),
        Some("eq1.σ_body".to_string()),
    );
    let program = make_program(body);
    let src = emit_host_program(&program, "host_clean_audit").unwrap();

    assert!(
        src.contains("// span: eq1.σ_body"),
        "Unicode host span ID not preserved verbatim; source:\n{src}"
    );
}
