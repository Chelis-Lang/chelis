//! S4.3 — Metal backend emits `// span: <id>` comments host-side and
//! inside embedded MSL kernel source strings.
//!
//! Per `spec/design/chelis_span_survival.md` §2.4: HIP and Metal emit the
//! same comment shape inside the embedded device-kernel source strings.
//!
//! Metal's emitter constructs one kernel per DAG node (`k_unary_<id>`,
//! `k_binary_<id>`, `k_reduce_<kind>_<id>`, `k_matmul_<id>`), so unlike
//! HIP we can prepend the originating node's spans inside *every* kernel
//! source string unambiguously.
//!
//! These structural tests run on every platform in default CI. The
//! load-bearing compile-success oracle for Metal lives in the
//! `#[ignore]`'d M6 manual gate (`tests/gpu_correctness.rs`), which
//! requires `xcrun -sdk macosx clang++` and runs on Mac only. See
//! `m6_span_attributed_program_compiles_and_matches_evaluator` for the
//! span-bearing case added in S4.3.

mod support;
use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};
use chelis_types::types::Prim;
use support::codegen_metal;

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

/// Strip the `static NSString *const <var>_src = @R"MSL(...)MSL";`
/// blocks from the generated `.mm` source, returning only the host-side
/// text. Used to verify that span comments appear on the host side, not
/// only inside the embedded kernel raw-string literals.
fn host_only(mm_source: &str) -> String {
    let mut out = String::new();
    let mut in_kernel_str = false;
    for line in mm_source.lines() {
        let trimmed = line.trim_start();
        if !in_kernel_str
            && trimmed.starts_with("static NSString *const ")
            && trimmed.contains("_src = @R\"MSL(")
        {
            in_kernel_str = true;
            // The opening line itself is part of the kernel string block.
            continue;
        }
        if in_kernel_str {
            // Block ends at the line containing `)MSL"`.
            if trimmed.contains(")MSL\"") {
                in_kernel_str = false;
            }
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Extract the body of a single named MSL kernel-source declaration from
/// the generated `.mm`. Returns the raw text between `@R"MSL(` and
/// `)MSL"`, exclusive.
fn extract_kernel_string(mm_source: &str, var_name: &str) -> Option<String> {
    let needle = format!("static NSString *const {var_name}_src = @R\"MSL(");
    let idx = mm_source.find(&needle)?;
    let after = &mm_source[idx + needle.len()..];
    let end = after.find(")MSL\"")?;
    Some(after[..end].to_string())
}

#[test]
fn s4_metal_canonical_span_id_emitted_host_side_and_in_kernel() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(4),
        Some("op.load".into()),
    );
    let n = dag.add_node(
        decl,
        RiscOp::Neg,
        vec![a],
        vec_f32(4),
        Some("op.neg".into()),
    );
    dag.add_root(n);

    let result = codegen_metal(&dag, "s4_metal_canonical");
    let src = &result.mm_source;

    // Both span comments appear somewhere in the source.
    assert!(
        src.contains("// span: op.load"),
        "expected `// span: op.load` in generated .mm:\n{src}"
    );
    assert!(
        src.contains("// span: op.neg"),
        "expected `// span: op.neg` in generated .mm:\n{src}"
    );

    // Host-side (outside MSL kernel raw-string literals) must carry the
    // span comments — these mark the launch sites.
    let host = host_only(src);
    assert!(
        host.contains("// span: op.load"),
        "host-side missing op.load comment\n\nHost:\n{host}\n\nFull:\n{src}"
    );
    assert!(
        host.contains("// span: op.neg"),
        "host-side missing op.neg comment\n\nHost:\n{host}\n\nFull:\n{src}"
    );

    // The unary Neg kernel (`pso_<id>_src`) must carry op.neg INSIDE its
    // MSL raw-string body, so the span survives into runtime-compiled
    // MSL.
    let neg_kernel = extract_kernel_string(src, &format!("pso_{}", n.0))
        .expect("pso_<neg-id>_src kernel block missing");
    assert!(
        neg_kernel.contains("// span: op.neg"),
        "kernel string for Neg node must embed `// span: op.neg`; body:\n{neg_kernel}"
    );
}

#[test]
fn s4_metal_merged_spans_emitted_lex_sorted_after_canonical() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let neg_id = dag.add_node(decl, RiscOp::Neg, vec![a], vec_f32(4), Some("op.x".into()));
    {
        let node = dag.node_mut(neg_id).unwrap();
        node.merged_spans = vec!["op.b".into(), "op.a".into(), "op.c".into()];
    }
    dag.add_root(neg_id);

    let result = codegen_metal(&dag, "s4_metal_merged");
    let src = &result.mm_source;

    // Look at host-side ordering. Find the canonical block — `op.x` is
    // unique and is the canonical that comes first.
    let host = host_only(src);
    let span_lines: Vec<&str> = host
        .lines()
        .map(str::trim_start)
        .filter(|l| l.starts_with("// span:"))
        .collect();
    let block_start = span_lines
        .iter()
        .position(|l| *l == "// span: op.x")
        .expect("missing canonical `// span: op.x` host-side");
    let block = &span_lines[block_start..(block_start + 4).min(span_lines.len())];
    assert_eq!(
        block,
        &[
            "// span: op.x",
            "// span: op.a",
            "// span: op.b",
            "// span: op.c"
        ],
        "expected canonical first then merged lex-sorted; got block:\n{block:?}\n\nHost:\n{host}"
    );

    // Inside the kernel string, same ordering applies for the Neg node
    // (its kernel embeds the same span block).
    let kernel = extract_kernel_string(src, &format!("pso_{}", neg_id.0))
        .expect("pso_<neg>_src kernel block missing");
    let kernel_span_lines: Vec<&str> = kernel
        .lines()
        .map(str::trim_start)
        .filter(|l| l.starts_with("// span:"))
        .collect();
    assert_eq!(
        kernel_span_lines,
        vec![
            "// span: op.x",
            "// span: op.a",
            "// span: op.b",
            "// span: op.c"
        ],
        "kernel string must embed the same canonical-first-then-lex-sorted span block; got:\n{kernel_span_lines:?}\n\nKernel:\n{kernel}"
    );
}

#[test]
fn s4_metal_merged_spans_dedup_against_canonical() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let neg_id = dag.add_node(
        decl,
        RiscOp::Neg,
        vec![a],
        vec_f32(4),
        Some("op.dup".into()),
    );
    {
        let node = dag.node_mut(neg_id).unwrap();
        node.merged_spans = vec!["op.dup".into(), "op.other".into()];
    }
    dag.add_root(neg_id);

    let result = codegen_metal(&dag, "s4_metal_dedup");
    let src = &result.mm_source;

    // Host-side count of `op.dup` for the Neg node should be 1 (deduped).
    let host = host_only(src);
    let host_dup = host.matches("// span: op.dup").count();
    assert_eq!(
        host_dup, 1,
        "merged_span equal to canonical span_id must be deduped host-side; saw {host_dup}"
    );
    assert!(
        host.contains("// span: op.other"),
        "non-duplicate merged span must still appear host-side"
    );

    // Inside the kernel string, same dedup rule applies.
    let kernel =
        extract_kernel_string(src, &format!("pso_{}", neg_id.0)).expect("kernel block missing");
    let kernel_dup = kernel.matches("// span: op.dup").count();
    assert_eq!(
        kernel_dup, 1,
        "kernel string must dedup canonical against merged; saw {kernel_dup}"
    );
}

#[test]
fn s4_metal_no_spans_emits_no_comment_block() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let n = dag.add_node(decl, RiscOp::Neg, vec![a], vec_f32(4), None);
    dag.add_root(n);

    let result = codegen_metal(&dag, "s4_metal_nospan");
    let src = &result.mm_source;

    assert!(
        !src.contains("// span:"),
        "span-free DAG must not emit any `// span:` comments:\n{src}"
    );
}

#[test]
fn s4_metal_oracle_richer_combinations_grep() {
    // S4 named oracle for Metal: 5 nodes with mixed combinations
    // (span_id only, merged_spans only, both, neither). Asserts grep
    // count >= sum.
    //
    // Compile-success for Metal lives in gpu_correctness.rs (#[ignore]
    // M6 manual gate) — this structural oracle runs in default CI on
    // every platform.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let n1 = dag.add_node(
        decl,
        RiscOp::Load { name: "in".into() },
        vec![],
        vec_f32(4),
        Some("n1.canonical".into()),
    );
    let n2 = dag.add_node(
        decl,
        RiscOp::Neg,
        vec![n1],
        vec_f32(4),
        Some("n2.canonical".into()),
    );
    {
        let node = dag.node_mut(n2).unwrap();
        node.merged_spans = vec!["n2.m1".into(), "n2.m2".into()];
    }
    let n3 = dag.add_node(decl, RiscOp::Sqrt, vec![n2], vec_f32(4), None);
    {
        let node = dag.node_mut(n3).unwrap();
        node.merged_spans = vec!["n3.m1".into(), "n3.m2".into()];
    }
    let n4 = dag.add_node(decl, RiscOp::Abs, vec![n3], vec_f32(4), None);
    let n5 = dag.add_node(
        decl,
        RiscOp::Floor,
        vec![n4],
        vec_f32(4),
        Some("n5.canonical".into()),
    );
    dag.add_root(n5);

    let result = codegen_metal(&dag, "s4_metal_oracle");
    let src = &result.mm_source;

    // Each node's spans appear at least twice — once in the kernel
    // string, once host-side. So grep count >= 2 * sum-of-spans for
    // nodes with kernels (Load goes through emit_load which is
    // host-side only — no kernel string for Loads). For this oracle,
    // count host-side AND kernel-side together: total expected is
    // (Load: 1) + (Neg: 3 host + 3 kernel) + (Sqrt: 2 host + 2 kernel)
    // + (Abs: 0) + (Floor: 1 host + 1 kernel) = 1 + 6 + 4 + 0 + 2 = 13.
    let expected_min: usize = 1 + 3 + 2 + 1; // host-side per node
    let actual = src.matches("// span: ").count();
    assert!(
        actual >= expected_min,
        "expected at least {expected_min} `// span:` lines, got {actual}\n\nSource:\n{src}"
    );

    for span in &[
        "n1.canonical",
        "n2.canonical",
        "n2.m1",
        "n2.m2",
        "n3.m1",
        "n3.m2",
        "n5.canonical",
    ] {
        assert!(
            src.contains(&format!("// span: {span}")),
            "missing `// span: {span}` in generated .mm:\n{src}"
        );
    }

    // Kernel-side embedding sanity: the Neg node's kernel must carry
    // both its canonical span and at least one of its merged spans.
    let neg_kernel =
        extract_kernel_string(src, &format!("pso_{}", n2.0)).expect("Neg kernel block missing");
    assert!(
        neg_kernel.contains("// span: n2.canonical"),
        "Neg kernel must embed canonical span:\n{neg_kernel}"
    );
    assert!(
        neg_kernel.contains("// span: n2.m1") && neg_kernel.contains("// span: n2.m2"),
        "Neg kernel must embed both merged spans:\n{neg_kernel}"
    );

    // n3 has merged_spans only (no canonical) — its kernel must still
    // embed the merged spans.
    let exp_kernel =
        extract_kernel_string(src, &format!("pso_{}", n3.0)).expect("Exp kernel block missing");
    assert!(
        exp_kernel.contains("// span: n3.m1") && exp_kernel.contains("// span: n3.m2"),
        "Exp kernel (merged_spans-only) must embed both merged spans:\n{exp_kernel}"
    );
    // And must NOT contain a canonical span for n3 (it has none).
    let exp_canonical_count = exp_kernel.matches("// span: n3.canonical").count();
    assert_eq!(
        exp_canonical_count, 0,
        "Exp kernel had no canonical span_id; must not embed one"
    );

    // Suppress unused-variable lint while still binding NodeId for the
    // assertion above. (n2.0 / n3.0 / n5.0 are used.)
    let _ = (n1, n4, n5);
    let _ = NodeId(0);
}

// ── Span-charset defense in depth (post-S4 red-team finding) ──────────
//
// Mirror of the C and HIP backend tests. Per `spec/03-deep-syntax.md`
// §1.1.1 the parser rejects forbidden span chars; programmatic IR
// construction bypasses that check, so the Metal backend's sanitizer
// (`chelis_ir::span_sanitize::sanitize_for_comment`) must escape forbidden
// bytes inside `// span:` comments — both host-side and inside embedded
// MSL kernel raw-string literals — before emitting `.mm` source.
//
// Compile-success for Metal lives in `gpu_correctness.rs` (#[ignore] M6
// manual gate) and runs only on macOS via `xcrun -sdk macosx`. This file
// stays structural so it runs on every platform in default CI.

#[test]
fn s4_metal_forbidden_newline_in_span_is_escaped_at_emit() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(4),
        Some("op\nint INJECTED_METAL_CODE = 42;".into()),
    );
    let n = dag.add_node(decl, RiscOp::Neg, vec![a], vec_f32(4), None);
    dag.add_root(n);

    let result = codegen_metal(&dag, "s4_metal_forbidden_newline");
    let src = &result.mm_source;

    assert!(
        src.contains("// span: op\\nint INJECTED_METAL_CODE = 42;"),
        "expected escaped `\\n` form in `// span:` comment; source:\n{src}"
    );
    assert!(
        !src.contains("\nint INJECTED_METAL_CODE = 42;"),
        "raw injected line escaped the sanitizer; source:\n{src}"
    );
}

#[test]
fn s4_metal_forbidden_newline_in_per_node_kernel_string_is_escaped() {
    // Metal emits one MSL kernel per DAG node, embedded inside a
    // raw-string literal `@R"MSL(...)MSL"`. A forbidden byte in the span
    // ID must be escaped before it enters the literal so the runtime
    // metallib compile sees a comment-safe identifier (and so the host
    // .mm source itself remains parseable).
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(4),
        Some("op.load".into()),
    );
    let n = dag.add_node(
        decl,
        RiscOp::Neg,
        vec![a],
        vec_f32(4),
        Some("op.neg\nINJECTED_METAL_KERNEL".into()),
    );
    dag.add_root(n);

    let result = codegen_metal(&dag, "s4_metal_forbidden_kernel");
    let src = &result.mm_source;

    assert!(
        src.contains("// span: op.neg\\nINJECTED_METAL_KERNEL"),
        "expected escaped `\\n` form host-side; source:\n{src}"
    );
    let kernel = extract_kernel_string(src, &format!("pso_{}", n.0))
        .expect("Neg kernel string block missing");
    assert!(
        kernel.contains("// span: op.neg\\nINJECTED_METAL_KERNEL"),
        "expected escaped span inside MSL kernel raw-string literal; body:\n{kernel}"
    );
    // Raw newline must NOT split the kernel-side comment line.
    assert!(
        !kernel.contains("// span: op.neg\nINJECTED_METAL_KERNEL"),
        "raw newline leaked into kernel raw-string literal; body:\n{kernel}"
    );
}

#[test]
fn s4_metal_clean_span_emitted_verbatim_audit_invariant() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(4),
        Some("eq1.σ_body".into()),
    );
    let n = dag.add_node(
        decl,
        RiscOp::Neg,
        vec![a],
        vec_f32(4),
        Some("__synthesized_grad__".into()),
    );
    dag.add_root(n);

    let result = codegen_metal(&dag, "s4_metal_clean");
    let src = &result.mm_source;

    assert!(
        src.contains("// span: eq1.σ_body"),
        "Unicode span ID not preserved verbatim; source:\n{src}"
    );
    assert!(
        src.contains("// span: __synthesized_grad__"),
        "synthesized marker not preserved verbatim; source:\n{src}"
    );
}
