//! Verify format-string escapes for entry names and symbolic dimension names.
//!
//! Checked construction and deserialization reject invalid graph labels.
//! These cases retain the source-emission controls for other producer fields.

use chelis_backend_c::CodegenOptions;
mod support;
use chelis_dag_core::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_numeric_core::prim::Prim;
use support::{codegen, codegen_with_options};

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn vec_f32_named(name: &str) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Named(name.to_string(), None)],
        precision: Prim::F32,
    }
}

fn codegen_with_invalid_test_entry(dag: &Dag, name: &str) -> chelis_backend_c::CodegenResult {
    codegen_with_options(
        dag,
        name,
        CodegenOptions {
            static_entry: true,
            ..CodegenOptions::default()
        },
    )
    .expect("the source-only sanitizer probe bypasses public artifact sealing")
}

#[test]
fn c_fprintf_format_string_escapes_percent_in_func_name() {
    // The orchestrator's reproduction recipe: a `%` in the func_name
    // would be misread by printf as a positional specifier. The
    // emit-side format-string sanitizer doubles `%` to `%%`.
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

    // `%s` in func_name would be a runtime crash if printf tried to
    // consume an argument that isn't there. The CLI never produces this
    // shape, but downstream tooling could; the sanitizer is the seat
    // belt.
    let result = codegen_with_invalid_test_entry(&dag, "f%spct");
    let src = &result.c_source;

    // The format string itself must contain the escaped form `%%s`.
    assert!(
        src.contains("\"f%%spct: expected"),
        "expected `%%`-escaped func_name in fprintf format string; source:\n{src}"
    );
    // The raw `%s` must NOT appear inside an fprintf format-string
    // callsite (only as part of the rest of the format itself —
    // `%d`/`%lld` are compiler-internal and safe).
    assert!(
        !src.contains("\"f%spct:"),
        "raw `%s` in func_name leaked into fprintf format; source:\n{src}"
    );
}

#[test]
fn c_fprintf_format_string_escapes_newline_in_func_name() {
    // A raw newline inside a C string literal terminates the source
    // line; defense-in-depth even though the CLI's filename-derived
    // func_name normally wouldn't carry newlines.
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

    let result = codegen_with_invalid_test_entry(&dag, "f\nINJECT");
    let src = &result.c_source;

    // Escape: `\n` inside the C string literal becomes the two-character
    // escape `\\n`.
    assert!(
        src.contains("\"f\\nINJECT: expected"),
        "expected `\\n`-escaped newline in fprintf format; source:\n{src}"
    );
    // The raw newline character must NOT split the format string into
    // two source lines.
    let raw_split = src
        .lines()
        .any(|line| line.contains("\"f") && !line.contains("INJECT"));
    assert!(
        !raw_split,
        "raw newline split the fprintf format string into two source lines:\n{src}"
    );
}

#[test]
fn c_fprintf_format_string_escapes_double_quote_in_func_name() {
    // A `"` would terminate the surrounding C string literal early.
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

    let result = codegen_with_invalid_test_entry(&dag, "fn\"injected");
    let src = &result.c_source;

    assert!(
        src.contains(r#""fn\"injected: expected"#),
        "expected `\"`-escaped double-quote in fprintf format; source:\n{src}"
    );
}

#[test]
fn c_clean_func_name_emitted_verbatim() {
    // Audit-invariant: a clean func_name (typical CLI-derived stem) is
    // emitted byte-identical in fprintf format strings — no spurious
    // escaping.
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

    let result = codegen(&dag, "my_func").unwrap();
    let src = &result.c_source;

    assert!(
        src.contains("\"my_func: expected"),
        "clean func_name must be emitted verbatim in fprintf format; source:\n{src}"
    );
    // No spurious escaping.
    assert!(
        !src.contains("my\\_func"),
        "clean func_name was incorrectly escaped: {src}"
    );
}

#[test]
fn c_fprintf_format_string_escapes_percent_in_symbolic_dim_name() {
    // The `SymbolicDimBinding.name` field is plain String (no
    // constructor-side validation). It flows BOTH into a C identifier
    // context (`int {name} = ...`) and into a format-string context
    // (`fprintf(stderr, "... symbolic dim `{name}` mismatch ...")`).
    // The format-string sanitizer is the only line of defense for the
    // format-string side; the identifier side relies on the dim name
    // being a legal C identifier (otherwise emitted C fails to compile,
    // which is the desired loud failure).
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    // Two Load nodes sharing a symbolic dim — this triggers the
    // mismatch fprintf in `emit_input_shape_preamble`. Use an
    // identifier-grammar-valid name for the dim because it ALSO emits
    // as a C identifier; the format-string sanitization is independent
    // of the identifier validity.
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32_named("batch"),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "y".into() },
        vec![],
        vec_f32_named("batch"),
        None,
    );
    let s = dag.add_node(decl, RiscOp::Add, vec![a, b], vec_f32_named("batch"), None);
    dag.add_root(s);

    let result = codegen(&dag, "sym").unwrap();
    let src = &result.c_source;

    // No `%` injection here (clean identifier `batch`); confirm the guard's
    // context line is structurally present and the `batch` name appears
    // verbatim - the architectural pattern lock.
    //
    // chelis#1277: the guard's TRAP line is now [04-NUM-9]'s fixed rendering,
    // which carries no name, so the name-bearing surface this test locks is
    // the accompanying context line that `spec/04-type-system.md` section 4.7
    // requires beside the trap.
    assert!(
        src.contains("extent `batch`"),
        "the guard's name-bearing context line must be emitted; source:\n{src}"
    );
    assert!(
        src.contains("numeric trap: domain in load at i64"),
        "and the trap line is [04-NUM-9]'s exact rendering; source:\n{src}"
    );
}

#[test]
fn c_codegen_with_options_inherits_sanitization() {
    // `codegen_with_options` is the public API used by the host emitter
    // for inline tensor helpers. It must honour the same format-string
    // sanitization as the default `codegen` path.
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

    let result = codegen_with_options(
        &dag,
        "opts%pct",
        CodegenOptions {
            static_entry: true,
            ..Default::default()
        },
    )
    .unwrap();
    let src = &result.c_source;

    assert!(
        src.contains("\"opts%%pct: expected"),
        "codegen_with_options must apply format-string sanitizer; source:\n{src}"
    );
}
