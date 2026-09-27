//! S4.2 — HIP backend emits `// span: <id>` comments host-side and inside
//! embedded device-kernel source strings.
//!
//! Per `spec/design/chelis_span_survival.md` §2.4: HIP and Metal emit the
//! same comment shape inside the embedded device-kernel source strings.
//!
//! Design choice (documented in commit message): generic shared kernels
//! (`kernel_neg`, `kernel_add`, …) are launched from multiple DAG nodes,
//! so embedding per-node spans inside their source string would be
//! ambiguous. Per-node kernels (`kernel_fused_*` for FusedElem and
//! `kernel_fused_sum_*`/`kernel_fused_maxred_*` for fused reductions) get
//! the originating node's spans prepended inside the source string.
//! Host-side launch sites get span comments for every node, making the
//! audit chain comprehensive.
//!
//! The load-bearing assertion is **compile-success via hipcc** — structural
//! grep alone is insufficient because malformed comment emission could pass
//! the grep but break the C/HIP source.

mod support;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::fuse::fuse;
use chelis_types::types::Prim;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use support::codegen_hip;

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn write_temp_file(dir: &Path, name: &str, contents: &str) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, contents).expect("write temp file");
    path
}

fn hipcc_available() -> bool {
    Command::new("hipcc")
        .arg("--version")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

fn hip_runtime_src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(chelis_backend_hip::runtime_dir())
}

fn cpu_runtime_include_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-runtime/include")
}

fn copy_runtime_headers(dst: &Path) {
    let include_dir = cpu_runtime_include_dir();
    for header in &[
        "chelis_runtime.h",
        "chelis_runtime_views.h",
        "chelis_runtime_dtype.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        write_temp_file(
            dst,
            header,
            &fs::read_to_string(include_dir.join(header))
                .unwrap_or_else(|_| panic!("read {header}")),
        );
    }
}

/// Compile a generated HIP `.cpp` source via `hipcc -c` (object only).
/// Returns Ok(()) on compile-success, Err(stderr) otherwise. Skips with
/// Ok(()) when `hipcc` is unavailable, matching the existing `s14_*` test
/// shape.
fn compile_hip_kernel_only(test_name: &str, c_source: &str) -> Result<(), String> {
    if !hipcc_available() {
        eprintln!("skipping compile assertion for `{test_name}`: hipcc unavailable");
        return Ok(());
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    let hip_rt = hip_runtime_src_dir();
    write_temp_file(
        tmp.path(),
        "chelis_hip_runtime.h",
        &fs::read_to_string(hip_rt.join("chelis_hip_runtime.h")).expect("hip runtime header"),
    );
    copy_runtime_headers(tmp.path());
    write_temp_file(tmp.path(), "model.cpp", c_source);

    let obj = tmp.path().join("model.o");
    let output = Command::new("hipcc")
        .arg("-c")
        .arg("-O0")
        .arg("-I")
        .arg(tmp.path())
        .arg(tmp.path().join("model.cpp"))
        .arg("-o")
        .arg(&obj)
        .output()
        .expect("run hipcc");
    if output.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).into_owned())
    }
}

/// Strip the embedded `*_src` const-char* declarations from the host
/// source, returning only the host-side text. Used to count span
/// comments that appear OUTSIDE the embedded kernel strings.
fn host_only(c_source: &str) -> String {
    let mut out = String::new();
    let mut in_string_decl = false;
    for line in c_source.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("const char *") && trimmed.contains("_src =") {
            in_string_decl = true;
            continue;
        }
        if in_string_decl {
            // The kernel-string declaration ends at the line that ends with `;`
            if trimmed.ends_with(';') {
                in_string_decl = false;
            }
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Extract the body of a single named const-char* kernel-source
/// declaration from the generated HIP source. Returns the literal string
/// content (with `\n` escapes still in place — i.e. the raw content of
/// the `const char *<name>_src = "..." "...";` block).
fn extract_kernel_string(c_source: &str, kernel_name: &str) -> Option<String> {
    let needle = format!("const char *{kernel_name}_src =");
    let idx = c_source.find(&needle)?;
    let after = &c_source[idx + needle.len()..];
    // Find the terminating `;` for the declaration.
    let end = after.find(';')?;
    Some(after[..end].to_string())
}

#[test]
fn s4_hip_canonical_span_id_emitted_host_side() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(4),
        Some("op.load".into()),
    );
    dag.add_node(
        decl,
        RiscOp::Neg,
        vec![a],
        vec_f32(4),
        Some("op.neg".into()),
    );
    dag.add_root(chelis_ir::dag::NodeId(1));

    let result = codegen_hip(&dag, "s4_hip_canonical").unwrap();
    let src = &result.c_source;

    // Host-side span comments must appear (we can find them anywhere in
    // the generated source, so this is a coarse check).
    assert!(
        src.contains("// span: op.load"),
        "expected `// span: op.load` host-side:\n{src}"
    );
    assert!(
        src.contains("// span: op.neg"),
        "expected `// span: op.neg` host-side:\n{src}"
    );

    // Confirm the span comments are emitted in the *host* portion, not
    // only inside the embedded kernel string. (kernel_neg is a shared
    // generic kernel — its source string should NOT carry per-node
    // spans, by the design choice documented at the top of this file.)
    let host = host_only(src);
    assert!(
        host.contains("// span: op.load"),
        "host portion must carry span: op.load comment\n\nHost portion:\n{host}"
    );
    assert!(
        host.contains("// span: op.neg"),
        "host portion must carry span: op.neg comment\n\nHost portion:\n{host}"
    );

    compile_hip_kernel_only("canonical", src).expect("hipcc must accept generated source");
}

#[test]
fn s4_hip_merged_spans_emitted_lex_sorted_after_canonical() {
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

    let result = codegen_hip(&dag, "s4_hip_merged").unwrap();
    let src = &result.c_source;

    // Look at host-side ordering only (kernel_neg shared kernel does not
    // carry per-node spans).
    let host = host_only(src);
    let span_lines: Vec<&str> = host
        .lines()
        .map(str::trim_start)
        .filter(|l| l.starts_with("// span:"))
        .collect();

    // Find the block of consecutive span lines for the Neg node — we
    // identify it by `op.x` since `op.x` is unique to this node and is
    // the canonical that comes first in the block.
    let neg_block_start = span_lines
        .iter()
        .position(|l| *l == "// span: op.x")
        .expect("missing canonical `// span: op.x`");

    let block = &span_lines[neg_block_start..(neg_block_start + 4).min(span_lines.len())];
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

    compile_hip_kernel_only("merged", src).expect("hipcc must accept generated source");
}

#[test]
fn s4_hip_no_spans_emits_no_comment_block() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let neg = dag.add_node(decl, RiscOp::Neg, vec![a], vec_f32(4), None);
    dag.add_root(neg);

    let result = codegen_hip(&dag, "s4_hip_nospan").unwrap();
    let src = &result.c_source;

    assert!(
        !src.contains("// span:"),
        "span-free DAG must not emit any `// span:` comments:\n{src}"
    );

    compile_hip_kernel_only("nospan", src).expect("hipcc must accept span-free source");
}

#[test]
fn s4_hip_per_node_kernel_string_carries_spans_inside() {
    // FusedElem (after `fuse()` over a 2-op chain) gets a per-node kernel
    // `kernel_fused_<id>`. Per the design choice, that kernel's source
    // string carries the originating node's spans embedded inside the
    // string literal so they survive into the runtime-compiled HIP.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(4),
        Some("op.load".into()),
    );
    let e = dag.add_node(
        decl,
        RiscOp::Exp,
        vec![a],
        vec_f32(4),
        Some("op.exp".into()),
    );
    let n = dag.add_node(
        decl,
        RiscOp::Neg,
        vec![e],
        vec_f32(4),
        Some("op.neg".into()),
    );
    dag.add_root(n);
    let fused = fuse(&dag);

    let result = codegen_hip(&fused, "s4_hip_fused").unwrap();
    let src = &result.c_source;

    // Confirm host-side launch comments appear for every node.
    assert!(
        src.contains("// span: op.load"),
        "host span: op.load missing"
    );

    // Find the FusedElem kernel string and verify it carries the
    // originating node's span comment INSIDE the string literal. The
    // FusedElem node is the survivor of the fuse pass; its span is
    // either op.exp or op.neg (with the other in merged_spans, per
    // S3.8). Check both possibilities.
    //
    // Locate the per-node kernel name by grepping the source for
    // `kernel_fused_<id>_src`.
    let mut fused_kernel_name: Option<String> = None;
    for line in src.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("const char *")
            && let Some(end) = rest.find("_src =")
        {
            let name = &rest[..end];
            if name.starts_with("kernel_fused_") {
                fused_kernel_name = Some(name.to_string());
                break;
            }
        }
    }
    let kname = fused_kernel_name.expect("fused kernel string declaration not found");
    let body = extract_kernel_string(src, &kname).expect("could not extract fused kernel body");

    // The body is a sequence of C string literals (one per source line of
    // the kernel) — the embedded `// span:` comment should appear as text
    // inside one of those literals.
    assert!(
        body.contains("// span: op.exp") || body.contains("// span: op.neg"),
        "fused kernel source string `{kname}_src` must embed at least one `// span:` comment from its originating node; body:\n{body}"
    );

    compile_hip_kernel_only("fused", src).expect("hipcc must accept generated fused source");
}

#[test]
fn s4_hip_oracle_richer_combinations_compile_and_grep() {
    // S4 named oracle for HIP: 5 nodes with mixed (span_id only,
    // merged_spans only, both, neither) combinations. Asserts host-side
    // grep count >= sum and compile-success.
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
    let n3 = dag.add_node(decl, RiscOp::Exp, vec![n2], vec_f32(4), None);
    {
        let node = dag.node_mut(n3).unwrap();
        node.merged_spans = vec!["n3.m1".into(), "n3.m2".into()];
    }
    let n4 = dag.add_node(decl, RiscOp::Log, vec![n3], vec_f32(4), None);
    let n5 = dag.add_node(
        decl,
        RiscOp::Sin,
        vec![n4],
        vec_f32(4),
        Some("n5.canonical".into()),
    );
    dag.add_root(n5);

    let result = codegen_hip(&dag, "s4_hip_oracle").unwrap();
    let src = &result.c_source;

    // Host-side counts (same as C oracle):
    //   n1: 1, n2: 3, n3: 2, n4: 0, n5: 1; total 7.
    // Plus possibly some embedded-string occurrences for per-node
    // kernels (none here — these are all generic shared kernels).
    let expected: usize = 1 + 3 + 2 + 1;
    let actual = src.matches("// span: ").count();
    assert!(
        actual >= expected,
        "expected at least {expected} `// span:` lines, got {actual}\n\nSource:\n{src}"
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
            "missing `// span: {span}` in generated HIP:\n{src}"
        );
    }

    compile_hip_kernel_only("oracle_hip", src).expect("S4 oracle: HIP source must compile");
}

// ── Span-charset defense in depth (post-S4 red-team finding) ──────────
//
// Mirror of the C backend tests in `chelis-backend-c/tests/span_comments.rs`.
// Per `spec/03-deep-syntax.md` §1.1.1 the parser rejects forbidden span
// chars; this test class exercises programmatic IR construction that
// bypasses the parser, so the HIP backend's sanitizer
// (`chelis_ir::span_sanitize::sanitize_for_comment`) must escape forbidden
// bytes inside `// span:` comments — both host-side and inside embedded
// kernel source strings — before emitting C/HIP source.

#[test]
fn s4_hip_forbidden_newline_in_span_is_escaped_at_emit() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(4),
        Some("op\nint INJECTED_HIP_CODE = 42;".into()),
    );
    let n = dag.add_node(decl, RiscOp::Neg, vec![a], vec_f32(4), None);
    dag.add_root(n);

    let result = codegen_hip(&dag, "s4_hip_forbidden_newline").unwrap();
    let src = &result.c_source;

    assert!(
        src.contains("// span: op\\nint INJECTED_HIP_CODE = 42;"),
        "expected escaped `\\n` form in `// span:` comment; source:\n{src}"
    );
    assert!(
        !src.contains("\nint INJECTED_HIP_CODE = 42;"),
        "raw injected line escaped the sanitizer; source:\n{src}"
    );
    compile_hip_kernel_only("forbidden_newline", src)
        .expect("sanitized HIP output must still compile via hipcc");
}

#[test]
fn s4_hip_forbidden_newline_in_per_node_kernel_string_is_escaped() {
    // FusedElem (after `fuse()`) gets a per-node kernel and embeds the
    // node's spans inside the source string literal. A forbidden byte in
    // the span ID must be escaped before it enters the string literal,
    // both for host-side correctness and so the runtime-compiled HIP
    // kernel sees a comment-safe identifier.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(4),
        Some("op.load".into()),
    );
    let e = dag.add_node(
        decl,
        RiscOp::Exp,
        vec![a],
        vec_f32(4),
        Some("op.exp\nINJECTED_HIP_KERNEL".into()),
    );
    let n = dag.add_node(
        decl,
        RiscOp::Neg,
        vec![e],
        vec_f32(4),
        Some("op.neg".into()),
    );
    dag.add_root(n);
    let fused = fuse(&dag);

    let result = codegen_hip(&fused, "s4_hip_forbidden_kernel").unwrap();
    let src = &result.c_source;

    assert!(
        src.contains("op.exp\\nINJECTED_HIP_KERNEL"),
        "expected escaped form for the embedded forbidden span; source:\n{src}"
    );
    // Find the embedded kernel-source declaration and confirm the raw
    // newline didn't leak into it.
    let mut fused_kernel_name: Option<String> = None;
    for line in src.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("const char *")
            && let Some(end) = rest.find("_src =")
        {
            let name = &rest[..end];
            if name.starts_with("kernel_fused_") {
                fused_kernel_name = Some(name.to_string());
                break;
            }
        }
    }
    if let Some(kname) = fused_kernel_name {
        let body = extract_kernel_string(src, &kname).expect("could not extract fused body");
        // The kernel string emitter escapes backslashes once when writing
        // the source into a `const char *_src = "..."` C string literal,
        // so the sanitizer's `\n` two-char sequence becomes `\\n` in the
        // literal text we extract here.
        assert!(
            body.contains("op.exp\\\\nINJECTED_HIP_KERNEL")
                || body.contains("op.exp\\nINJECTED_HIP_KERNEL"),
            "fused kernel string must carry the escaped span; body:\n{body}"
        );
        // The raw newline must NOT appear unescaped after the `// span:`
        // prefix inside the string literal.
        assert!(
            !body.contains("// span: op.exp\nINJECTED_HIP_KERNEL"),
            "raw newline leaked into kernel string literal; body:\n{body}"
        );
    }

    compile_hip_kernel_only("forbidden_kernel", src)
        .expect("sanitized HIP source (per-node kernel) must compile via hipcc");
}

#[test]
fn s4_hip_clean_span_emitted_verbatim_audit_invariant() {
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

    let result = codegen_hip(&dag, "s4_hip_clean").unwrap();
    let src = &result.c_source;

    assert!(
        src.contains("// span: eq1.σ_body"),
        "Unicode span ID not preserved verbatim; source:\n{src}"
    );
    assert!(
        src.contains("// span: __synthesized_grad__"),
        "synthesized marker not preserved verbatim; source:\n{src}"
    );
    compile_hip_kernel_only("clean_audit", src).expect("clean span must compile via hipcc");
}
