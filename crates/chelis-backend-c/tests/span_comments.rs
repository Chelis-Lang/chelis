//! S4.1 — C backend emits `// span: <id>` comments preceding each node.
//!
//! Per `spec/design/chelis_span_survival.md` §2.4: each backend emits one
//! `// span: <id>` line per `span_id ∪ merged_spans` immediately preceding
//! the line(s) that implement the operation. Ordering: canonical first,
//! then `merged_spans` lex-sorted.
//!
//! The load-bearing assertion is **compile-success via gcc** — structural
//! grep alone is insufficient because malformed comment emission could pass
//! the grep but break the C source.

use chelis_backend_c::codegen;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

mod common;

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn runtime_include_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-runtime/include")
}

fn target_debug_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe failed");
    exe.parent()
        .and_then(Path::parent)
        .map(PathBuf::from)
        .expect("could not resolve target/debug dir from current_exe")
}

fn ensure_runtime_static_lib(canonical: &Path) -> std::io::Result<()> {
    if canonical.exists() {
        return Ok(());
    }
    let deps_dir = canonical
        .parent()
        .expect("canonical lib path has no parent")
        .join("deps");
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in fs::read_dir(&deps_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("libchelis_runtime-") && name.ends_with(".a") {
            let meta = entry.metadata()?;
            let mtime = meta.modified()?;
            match &newest {
                Some((cur, _)) if *cur >= mtime => {}
                _ => newest = Some((mtime, entry.path())),
            }
        }
    }
    let Some((_, hashed)) = newest else {
        return Err(std::io::Error::other(format!(
            "no libchelis_runtime-*.a found in {}",
            deps_dir.display()
        )));
    };
    // PID-suffixed tmp so concurrent test binaries (nextest runs sister
    // exec-style tests in parallel; they all materialize the same
    // canonical path) do not race on a shared tmp filename and trip
    // ENOENT on rename when a peer renames it away first.
    static NEXT_TEMP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tmp = canonical.with_extension(format!(
        "a.tmp.{}.{}",
        std::process::id(),
        NEXT_TEMP.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    fs::copy(&hashed, &tmp)?;
    match fs::rename(&tmp, canonical) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && canonical.exists() => Ok(()),
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            Err(e)
        }
    }
}

fn runtime_lib_path() -> PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let canonical = target_debug_dir().join("libchelis_runtime.a");
        if let Err(e) = ensure_runtime_static_lib(&canonical) {
            panic!(
                "failed to materialize libchelis_runtime.a at {}: {}",
                canonical.display(),
                e
            );
        }
        canonical
    })
    .clone()
}

/// Compile generated C source standalone (no harness). Returns Ok(()) on
/// successful compile, Err(stderr) otherwise.
fn compile_kernel_only(test_name: &str, c_source: &str) -> Result<(), String> {
    let probe = common::probe_dir(&format!("s4c_{test_name}"));
    let dir = probe.path().to_path_buf();
    fs::write(dir.join("kernel.c"), c_source).unwrap();

    let include_dir = runtime_include_dir();
    for hdr in &[
        "chelis_runtime.h",
        "chelis_runtime_dtype.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        let src = fs::read_to_string(include_dir.join(hdr)).unwrap();
        fs::write(dir.join(hdr), src).unwrap();
    }

    // Touch the runtime lib so the runtime headers are available.
    let _ = runtime_lib_path();

    let obj = dir.join("kernel.o");
    let compile = Command::new("gcc")
        .args([
            "-O0",
            "-std=c11",
            "-c",
            "-Werror",
            "-I",
            dir.to_str().unwrap(),
            dir.join("kernel.c").to_str().unwrap(),
            "-o",
            obj.to_str().unwrap(),
        ])
        .output()
        .expect("failed to invoke gcc");

    if compile.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&compile.stderr).into_owned())
    }
}

#[test]
fn s4_c_canonical_span_id_emitted_as_comment() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(4),
        Some("op.load".into()),
    );
    dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), Some("op.neg".into()));

    let result = codegen(&dag, "s4_canonical").unwrap();
    let src = &result.c_source;

    assert!(
        src.contains("// span: op.load"),
        "expected `// span: op.load` in generated C:\n{src}"
    );
    assert!(
        src.contains("// span: op.neg"),
        "expected `// span: op.neg` in generated C:\n{src}"
    );

    compile_kernel_only("canonical", src).expect("generated C must compile via gcc");
}

#[test]
fn s4_c_merged_spans_emitted_lex_sorted_after_canonical() {
    // Hand-craft a node carrying span_id + merged_spans (the typical S3
    // shape after Fusion / CSE / fold merges).
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(4), None);
    let neg_id = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), Some("op.x".into()));
    {
        let node = dag.node_mut(neg_id).unwrap();
        node.merged_spans = vec!["op.b".into(), "op.a".into(), "op.c".into()];
    }

    let result = codegen(&dag, "s4_merged").unwrap();
    let src = &result.c_source;

    // Find the order of `// span:` lines emitted for the Neg node.
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

    compile_kernel_only("merged", src).expect("generated C must compile via gcc");
}

#[test]
fn s4_c_merged_spans_dedup_against_canonical() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(4), None);
    let neg_id = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), Some("op.dup".into()));
    {
        // Manually inject a merged span equal to the canonical span_id.
        // span_merge helpers normally prevent this, but the emitter must
        // still dedupe defensively.
        let node = dag.node_mut(neg_id).unwrap();
        node.merged_spans = vec!["op.dup".into(), "op.other".into()];
    }

    let result = codegen(&dag, "s4_dedup").unwrap();
    let src = &result.c_source;

    let canonical_count = src.matches("// span: op.dup").count();
    assert_eq!(
        canonical_count, 1,
        "merged_span equal to canonical span_id must be deduped; saw {canonical_count} occurrences"
    );
    assert!(
        src.contains("// span: op.other"),
        "non-duplicate merged span must still appear:\n{src}"
    );

    compile_kernel_only("dedup", src).expect("generated C must compile via gcc");
}

#[test]
fn s4_c_no_spans_emits_no_comment_block() {
    // Backwards-compat: span-free DAG (the common case for hand-written
    // Chelis) must produce zero `// span:` lines.
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(4), None);
    dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), None);

    let result = codegen(&dag, "s4_nospan").unwrap();
    let src = &result.c_source;

    assert!(
        !src.contains("// span:"),
        "span-free DAG must not emit any `// span:` comments:\n{src}"
    );

    compile_kernel_only("nospan", src).expect("generated C must compile via gcc");
}

#[test]
fn s4_c_oracle_richer_combinations_compile_and_grep() {
    // Per the S4 named oracle: hand-craft a DAG with at least 5 nodes carrying
    // various combinations of (span_id only, merged_spans only, both, neither).
    // Assert structural grep count >= sum and compile-success.
    let mut dag = Dag::new();

    // Node 1: span_id only.
    let n1 = dag.add_node(
        RiscOp::Load { name: "in".into() },
        vec![],
        vec_f32(4),
        Some("n1.canonical".into()),
    );

    // Node 2: span_id + merged_spans.
    let n2 = dag.add_node(
        RiscOp::Neg,
        vec![n1],
        vec_f32(4),
        Some("n2.canonical".into()),
    );
    {
        let node = dag.node_mut(n2).unwrap();
        node.merged_spans = vec!["n2.m1".into(), "n2.m2".into()];
    }

    // Node 3: merged_spans only (no canonical). Defensive case per the spec —
    // shouldn't happen via normal pass output but emitter must handle it.
    let n3 = dag.add_node(RiscOp::Exp, vec![n2], vec_f32(4), None);
    {
        let node = dag.node_mut(n3).unwrap();
        node.merged_spans = vec!["n3.m1".into(), "n3.m2".into()];
    }

    // Node 4: neither (span-free).
    let n4 = dag.add_node(RiscOp::Log, vec![n3], vec_f32(4), None);

    // Node 5: span_id only.
    dag.add_node(
        RiscOp::Sin,
        vec![n4],
        vec_f32(4),
        Some("n5.canonical".into()),
    );

    let result = codegen(&dag, "s4_oracle_c").unwrap();
    let src = &result.c_source;

    // Expected counts:
    //   n1: 1 (canonical)
    //   n2: 1 (canonical) + 2 (merged) = 3
    //   n3: 0 (canonical) + 2 (merged) = 2
    //   n4: 0
    //   n5: 1 (canonical)
    // Total: 7
    let expected: usize = 1 + 3 + 2 + 1;
    let actual = src.matches("// span: ").count();
    assert!(
        actual >= expected,
        "expected at least {expected} `// span:` lines, got {actual}\n\nSource:\n{src}"
    );

    // Specific spans present
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
            "missing `// span: {span}` in generated C:\n{src}"
        );
    }

    // Load-bearing: compile success.
    compile_kernel_only("oracle_c", src).expect("S4 oracle: generated C must compile via gcc");
}

// ── Span-charset defense in depth (post-S4 red-team finding) ──────────
//
// Per `spec/03-deep-syntax.md` §1.1.1, span ID strings carrying ASCII
// control characters are spec-illegal and the Deep parser rejects them.
// Programmatic IR construction bypasses the parser, so the C backend
// applies a defense-in-depth sanitizer
// (`chelis_ir::span_sanitize::sanitize_for_comment`) before interpolating
// spans into `// span: <id>` line comments. These tests cover that fallback
// directly: build a `DagNode` with a forbidden span_id and assert the
// emitted C is comment-safe (no `\n` terminating the comment line) and
// compiles cleanly via gcc.

#[test]
fn s4_c_forbidden_newline_in_span_is_escaped_at_emit() {
    // The injection probe from the red-team gate. Without escaping,
    // `// span: op\nint INJECTED = 42;` would compile with INJECTED as a
    // real top-level declaration. With the sanitizer, the `\n` becomes a
    // literal `\n` two-char sequence inside the comment.
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(4),
        Some("op\nint INJECTED_C_CODE = 42;".into()),
    );
    dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), None);

    let result = codegen(&dag, "s4_c_forbidden_newline").unwrap();
    let src = &result.c_source;

    // Escaped form present.
    assert!(
        src.contains("// span: op\\nint INJECTED_C_CODE = 42;"),
        "expected escaped `\\n` form in `// span:` comment; source:\n{src}"
    );
    // No raw injected line.
    assert!(
        !src.contains("\nint INJECTED_C_CODE = 42;"),
        "raw injected line escaped the sanitizer; source:\n{src}"
    );

    compile_kernel_only("forbidden_newline", src)
        .expect("sanitized output must still compile via gcc");
}

#[test]
fn s4_c_forbidden_newline_in_merged_spans_is_escaped_at_emit() {
    // Same probe but via `merged_spans` rather than `span_id` — the
    // sanitizer must apply to both code paths in `emit_span_comments`.
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(4), None);
    let neg_id = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), Some("op.clean".into()));
    {
        let node = dag.node_mut(neg_id).unwrap();
        node.merged_spans = vec!["op\nint INJECTED_VIA_MERGED = 1;".into()];
    }

    let result = codegen(&dag, "s4_c_forbidden_merged").unwrap();
    let src = &result.c_source;

    assert!(
        src.contains("// span: op\\nint INJECTED_VIA_MERGED = 1;"),
        "merged-span path missing escape; source:\n{src}"
    );
    assert!(
        !src.contains("\nint INJECTED_VIA_MERGED = 1;"),
        "merged-span injection escaped the sanitizer; source:\n{src}"
    );
    compile_kernel_only("forbidden_merged", src)
        .expect("sanitized output (via merged_spans) must still compile via gcc");
}

#[test]
fn s4_c_clean_span_emitted_verbatim_audit_invariant() {
    // Audit invariant lock: a clean span ID (no forbidden chars) must
    // emit byte-identical to its input. The customer's `.spans.json`
    // sidecar entry must match the `// span: <id>` text in the .c file
    // verbatim for well-behaved producers.
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(4),
        Some("eq1.σ_body".into()),
    );
    dag.add_node(
        RiscOp::Neg,
        vec![a],
        vec_f32(4),
        Some("__synthesized_grad__".into()),
    );

    let result = codegen(&dag, "s4_c_clean").unwrap();
    let src = &result.c_source;

    // Both verbatim — no escapes inserted.
    assert!(
        src.contains("// span: eq1.σ_body"),
        "Unicode span ID not preserved verbatim; source:\n{src}"
    );
    assert!(
        src.contains("// span: __synthesized_grad__"),
        "synthesized marker not preserved verbatim; source:\n{src}"
    );
    compile_kernel_only("clean_audit", src).expect("clean span must compile via gcc");
}
