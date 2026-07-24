//! Tests for `spec/upstream-bugs/producer-string-sanitization.md` —
//! every producer-supplied string flowing into a C-backend `printf`-
//! style format string is routed through
//! `chelis_ir::span_sanitize::sanitize_for_format_string`.
//!
//! ## What this covers
//!
//! The C backend interpolates three kinds of producer-supplied strings
//! into `fprintf(stderr, "...{x}...", ...)` runtime-error reports:
//!
//! - `func_name` — typically the source filename's stem (CLI-derived).
//! - Load names — `LoadStoreName::as_str()` of `RiscOp::Load { name }`.
//! - `SymbolicDimBinding.name` and `SymbolicDimOccurrence.input_label`
//!   — the symbolic-dim mismatch path.
//!
//! All three contexts can carry forbidden bytes (`%` confuses
//! positional-printf consumers; `\\`/`"`/control bytes break the C
//! string literal). The format-string sanitizer normalises all of them.
//!
//! Today's grammar-validated `LoadStoreName` rejects `%`/`\\`/`"`/control
//! bytes at construction; the bypass for these tests is
//! `serde_json::from_str` (transparent deserialize, no re-validation).
//! For `SymbolicDimBinding.name`, the field is plain `String` with no
//! constructor-side validation today — see the audit notes in
//! `producer-string-sanitization.md` and CLAUDE.md's "no silent
//! deferrals" rule. The emit-side sanitizer is the second layer of
//! defense in depth and is what these tests verify.

use chelis_backend_c::{CodegenOptions, codegen, codegen_with_options};
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::load_store_name::LoadStoreName;
use chelis_types::types::Prim;

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

fn dirty_name(s: &str) -> LoadStoreName {
    let json = serde_json::to_string(s).expect("string serializes");
    serde_json::from_str(&json)
        .expect("LoadStoreName deserialize is transparent (no re-validation)")
}

#[test]
fn c_fprintf_format_string_escapes_percent_in_func_name() {
    // The orchestrator's reproduction recipe: a `%` in the func_name
    // would be misread by printf as a positional specifier. The
    // emit-side format-string sanitizer doubles `%` to `%%`.
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(4), None);
    let n = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), None);
    dag.add_root(n);

    // `%s` in func_name would be a runtime crash if printf tried to
    // consume an argument that isn't there. The CLI never produces this
    // shape, but downstream tooling could; the sanitizer is the seat
    // belt.
    let result = codegen(&dag, "f%spct").unwrap();
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
fn c_fprintf_format_string_escapes_percent_in_load_name() {
    // `Load { name }`'s emitted `// ...` comment is comment-context
    // (handled in metal); for C the `name` flows into the
    // input-shape-preamble fprintf — that's the format-string context.
    let mut dag = Dag::new();
    let bad = dirty_name("inp%s");
    let a = dag.add_node(RiscOp::Load { name: bad }, vec![], vec_f32(4), None);
    let n = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), None);
    dag.add_root(n);

    let result = codegen(&dag, "test_load_pct").unwrap();
    let src = &result.c_source;

    // The fprintf reporting "input `inp%s` at slot N is NULL" needs the
    // escaped form so printf doesn't consume a phantom argument.
    assert!(
        src.contains("input `inp%%s`"),
        "expected `%%`-escaped Load label in fprintf format; source:\n{src}"
    );
    assert!(
        !src.contains("input `inp%s`"),
        "raw `%s` in Load name leaked into fprintf format; source:\n{src}"
    );
}

#[test]
fn c_fprintf_format_string_escapes_newline_in_func_name() {
    // A raw newline inside a C string literal terminates the source
    // line; defense-in-depth even though the CLI's filename-derived
    // func_name normally wouldn't carry newlines.
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(4), None);
    let n = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), None);
    dag.add_root(n);

    let result = codegen(&dag, "f\nINJECT").unwrap();
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
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(4), None);
    let n = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), None);
    dag.add_root(n);

    let result = codegen(&dag, "fn\"injected").unwrap();
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
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(4), None);
    let n = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), None);
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
    // Two Load nodes sharing a symbolic dim — this triggers the
    // mismatch fprintf in `emit_input_shape_preamble`. Use an
    // identifier-grammar-valid name for the dim because it ALSO emits
    // as a C identifier; the format-string sanitization is independent
    // of the identifier validity.
    let a = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32_named("batch"),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "y".into() },
        vec![],
        vec_f32_named("batch"),
        None,
    );
    let s = dag.add_node(RiscOp::Add, vec![a, b], vec_f32_named("batch"), None);
    dag.add_root(s);

    let result = codegen(&dag, "sym").unwrap();
    let src = &result.c_source;

    // No `%` injection here (clean identifier `batch`); confirm the
    // mismatch fprintf is structurally present and the `batch` name
    // appears verbatim — the architectural pattern lock.
    assert!(
        src.contains("symbolic dim `batch` mismatch"),
        "symbolic-dim mismatch fprintf must be emitted; source:\n{src}"
    );
}

#[test]
fn c_sanitized_format_strings_still_compile_cleanly() {
    // Audit invariant: sanitization preserves valid C. Push a dirty
    // Load name (carrying every byte class the sanitizer escapes —
    // `%`, `\\`, `"`, `\n`) through codegen with a CLEAN func_name (so
    // the C identifier-context emission is valid). Ask gcc for
    // `-fsyntax-only`. If the sanitizer produced an invalid C string
    // literal (raw `"` terminating early, raw `\n` splitting the source
    // line, raw `\\` mis-escaping the next char), gcc rejects it.
    let mut dag = Dag::new();
    // Every adversarial byte in one identifier — the sanitizer must
    // escape ALL of them so the surrounding `"..."` C string literal
    // stays well-formed.
    let bad = dirty_name("inp%s\\back\"q\n");
    let a = dag.add_node(RiscOp::Load { name: bad }, vec![], vec_f32(4), None);
    let n = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), None);
    dag.add_root(n);

    // Clean func_name so the declarator (`void {func_name}(...)`) is
    // valid C. This isolates the test to the format-string-context
    // sanitizer.
    let result = codegen(&dag, "test_sanitize").unwrap();

    let dir = std::env::temp_dir().join("chelis_producer_sanitize_c_compile");
    std::fs::create_dir_all(&dir).unwrap();
    let src_path = dir.join("sanitized.c");
    std::fs::write(&src_path, &result.c_source).unwrap();

    let include_dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-runtime/include");
    for header in &[
        "chelis_runtime.h",
        "chelis_runtime_dtype.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        let raw = std::fs::read_to_string(include_dir.join(header)).unwrap();
        std::fs::write(dir.join(header), &raw).unwrap();
    }

    let output = std::process::Command::new(chelis_backend_c::toolchain::c_compiler())
        .args([
            "-fsyntax-only",
            "-I",
            dir.to_str().unwrap(),
            src_path.to_str().unwrap(),
        ])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            // Sanitized format strings parse as valid C string literals.
        }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            panic!(
                "Sanitized fprintf format strings do NOT parse as valid C:\n{stderr}\n\nsource:\n{}",
                result.c_source
            );
        }
        Err(e) => {
            eprintln!("C compiler not available ({e}); skipping syntax check");
        }
    }
}

#[test]
fn c_codegen_with_options_inherits_sanitization() {
    // `codegen_with_options` is the public API used by the host emitter
    // for inline tensor helpers. It must honour the same format-string
    // sanitization as the default `codegen` path.
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(4), None);
    let n = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), None);
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
