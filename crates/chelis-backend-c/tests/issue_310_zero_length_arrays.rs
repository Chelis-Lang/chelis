//! Issue #310 — `chelis build --target c` must not emit ISO-C-illegal
//! zero-length `chelis_value[0]` arrays for nullary ADT variants or empty
//! tuples constructed in a host body.
//!
//! A zero-length array (`chelis_value foo[0];`) is a GCC/Clang extension, not
//! ISO C. Under the project's emit-then-`gcc`/`hipcc` model a nullary-variant
//! or empty-tuple host body therefore produced C that strict/pedantic
//! compilers reject. This is the same defect class #306/#300 fixed for the
//! constant-only tensor-helper path: when there are no elements, pass a `NULL`
//! pointer with count `0` instead of declaring a zero-length array. The
//! consuming runtime helpers (`chelis_adt_construct`, `chelis_tuple_from_values`)
//! already guard `ptr.is_null() || len <= 0` and never dereference.
//!
//! These tests build hand-crafted `HostProgram`s and exercise the host-emit
//! code path directly via `emit_host_program`, mirroring the harness style of
//! `host_span_comments.rs`. The non-empty cases provide negative parity: the
//! `[N]` array form must still be emitted when there *are* elements.

mod support;
use chelis_ir::ConcreteHostType as HostType;
use chelis_ir::host::{
    ConcreteHostExpr as HostExpr, ConcreteHostExprKind as HostExprKind,
    ConcreteHostFunction as HostFunction, ConcreteHostParam as HostParam,
    ConcreteHostProgram as HostProgram,
};
use std::path::PathBuf;
use std::process::Command;
use support::emit_host_program;

mod common;

/// Wrap a single-expression body into a minimal `HostProgram`. The function
/// takes one scalar param so the emitted signature is well-formed; the body
/// expression determines what construction code is emitted.
fn program_with_body(ret_ty: HostType, body: HostExpr) -> HostProgram {
    HostProgram {
        globals: Vec::new(),
        global_tensor_helpers: Vec::new(),
        functions: vec![HostFunction {
            helper_result_claim_axes: Vec::new(),
            name: "the_fn".to_string(),
            params: vec![HostParam {
                name: "x".to_string(),
                ty: HostType::Float64,
            }],
            ret_ty,
            body,
            tensor_helpers: Vec::new(),
            origin: chelis_ir::host::HostFunctionOrigin::Authored,
            specialization: None,
            summary_rejections: Vec::new(),
        }],
        summary_rejections: Vec::new(),
    }
}

/// True if any emitted line declares a zero-length array, e.g.
/// `chelis_value adt_fields0[0];` or `chelis_value tuple_values0[0];`.
fn has_zero_length_array(src: &str) -> bool {
    src.lines().any(|line| line.contains("[0];"))
}

// ---- Nullary ADT variant -------------------------------------------------

#[test]
fn issue_310_nullary_adt_variant_emits_no_zero_length_array() {
    // `True` / `Nothing` style constructor with no payload fields.
    let body = HostExpr::new(HostExprKind::AdtConstruct {
        ctor: "Nothing".to_string(),
        fields: Vec::new(),
        ty: HostType::Adt("Maybe".to_string(), Vec::new()),
    });
    let program = program_with_body(HostType::Adt("Maybe".to_string(), Vec::new()), body);
    let src = emit_host_program(&program, "nullary_adt").unwrap();

    assert!(
        !has_zero_length_array(&src),
        "nullary ADT variant must not emit a zero-length `[0];` array:\n{src}"
    );
    assert!(
        src.contains("chelis_adt_construct("),
        "expected a chelis_adt_construct call:\n{src}"
    );
    // The construct call must pass a NULL field pointer with count 0.
    assert!(
        src.contains("chelis_string_from_cstr(\"Nothing\")"),
        "{src}"
    );
    assert!(
        src.lines()
            .any(|line| line.contains("chelis_adt_construct(") && line.contains(", NULL, 0)")),
        "nullary ADT construct must pass NULL fields with count 0:\n{src}"
    );
    assert!(
        src.contains("chelis_string_release("),
        "constructor-name owner must be released after the cloning ADT constructor:\n{src}"
    );
}

#[test]
fn issue_310_adt_variant_with_fields_still_emits_array() {
    // Negative parity: a payload-carrying variant must still declare the
    // `chelis_value <name>[N];` array and pass count N.
    let body = HostExpr::new(HostExprKind::AdtConstruct {
        ctor: "Just".to_string(),
        fields: vec![HostExpr::new(HostExprKind::Int(7))],
        ty: HostType::Adt("Maybe".to_string(), vec![HostType::Int64]),
    });
    let program = program_with_body(
        HostType::Adt("Maybe".to_string(), vec![HostType::Int64]),
        body,
    );
    let src = emit_host_program(&program, "adt_with_field").unwrap();

    assert!(
        src.contains("[1];"),
        "single-field ADT variant must still declare a `[1];` array:\n{src}"
    );
    assert!(src.contains("chelis_string_from_cstr(\"Just\")"), "{src}");
    assert!(src.contains("chelis_adt_construct("), "{src}");
    // Must NOT degrade to NULL/0 when fields are present.
    assert!(
        !src.lines()
            .any(|line| line.contains("chelis_adt_construct(") && line.contains(", NULL, 0)")),
        "field-carrying ADT variant must not pass NULL/0:\n{src}"
    );
}

// ---- Empty tuple ---------------------------------------------------------

#[test]
fn issue_310_empty_tuple_emits_no_zero_length_array() {
    let body = HostExpr::new(HostExprKind::Tuple(Vec::new(), HostType::Tuple(Vec::new())));
    let program = program_with_body(HostType::Tuple(Vec::new()), body);
    let src = emit_host_program(&program, "empty_tuple").unwrap();

    assert!(
        !has_zero_length_array(&src),
        "empty tuple must not emit a zero-length `[0];` array:\n{src}"
    );
    assert!(
        src.contains("chelis_tuple_from_values("),
        "expected a chelis_tuple_from_values call:\n{src}"
    );
    assert!(
        src.contains("chelis_tuple_from_values(NULL, 0)"),
        "empty tuple must pass NULL items with count 0:\n{src}"
    );
}

#[test]
fn issue_310_nonempty_tuple_still_emits_array() {
    // Negative parity: a populated tuple must still declare the array.
    let body = HostExpr::new(HostExprKind::Tuple(
        vec![
            HostExpr::new(HostExprKind::Int(1)),
            HostExpr::new(HostExprKind::Int(2)),
        ],
        HostType::Tuple(vec![HostType::Int64, HostType::Int64]),
    ));
    let program = program_with_body(
        HostType::Tuple(vec![HostType::Int64, HostType::Int64]),
        body,
    );
    let src = emit_host_program(&program, "pair_tuple").unwrap();

    assert!(
        src.contains("[2];"),
        "two-element tuple must still declare a `[2];` array:\n{src}"
    );
    assert!(
        !src.contains("chelis_tuple_from_values(NULL, 0)"),
        "populated tuple must not pass NULL/0:\n{src}"
    );
}

// ---- Strict ISO-C compile gate ------------------------------------------
//
// The pattern assertions above are the authoritative oracle (they run without
// a C toolchain). This test additionally proves the emitted C is accepted
// under `-std=c11 -pedantic-errors`, which rejects zero-length arrays. It is
// skipped gracefully when no `gcc` is available so it never flakes CI hosts
// lacking a compiler.

fn runtime_include_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-runtime/include")
}

fn gcc_available() -> bool {
    Command::new("gcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Compile `c_source` (host C) under strict ISO-C with `-pedantic-errors -Werror`.
/// Returns the compiler stderr on failure, or `None` on success.
fn pedantic_compile_error(test_name: &str, c_source: &str) -> Option<String> {
    let probe = common::probe_dir(&format!("issue310_{test_name}"));
    let dir = probe.path().to_path_buf();
    std::fs::write(dir.join("host.c"), c_source).unwrap();

    let include_dir = runtime_include_dir();
    for hdr in &[
        "chelis_runtime.h",
        "chelis_runtime_views.h",
        "chelis_runtime_dtype.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        let src = std::fs::read_to_string(include_dir.join(hdr)).unwrap();
        std::fs::write(dir.join(hdr), src).unwrap();
    }

    // `-fsyntax-only` checks the translation unit without linking the runtime,
    // which is all we need to catch zero-length-array constraint violations.
    let compile = Command::new("gcc")
        .args([
            "-std=c11",
            "-pedantic-errors",
            "-Werror",
            "-fsyntax-only",
            "-I",
            dir.to_str().unwrap(),
            dir.join("host.c").to_str().unwrap(),
        ])
        .output()
        .expect("failed to invoke gcc");

    if compile.status.success() {
        None
    } else {
        Some(String::from_utf8_lossy(&compile.stderr).into_owned())
    }
}

#[test]
fn issue_310_nullary_adt_compiles_under_pedantic_iso_c() {
    if !gcc_available() {
        eprintln!("skipping pedantic compile gate: gcc not available");
        return;
    }
    let body = HostExpr::new(HostExprKind::AdtConstruct {
        ctor: "Nothing".to_string(),
        fields: Vec::new(),
        ty: HostType::Adt("Maybe".to_string(), Vec::new()),
    });
    let program = program_with_body(HostType::Adt("Maybe".to_string(), Vec::new()), body);
    let src = emit_host_program(&program, "nullary_adt_pedantic").unwrap();

    if let Some(stderr) = pedantic_compile_error("nullary_adt", &src) {
        panic!(
            "nullary ADT host C must compile under strict ISO C:\n{stderr}\n--- source ---\n{src}"
        );
    }
}

#[test]
fn issue_310_empty_tuple_compiles_under_pedantic_iso_c() {
    if !gcc_available() {
        eprintln!("skipping pedantic compile gate: gcc not available");
        return;
    }
    let body = HostExpr::new(HostExprKind::Tuple(Vec::new(), HostType::Tuple(Vec::new())));
    let program = program_with_body(HostType::Tuple(Vec::new()), body);
    let src = emit_host_program(&program, "empty_tuple_pedantic").unwrap();

    if let Some(stderr) = pedantic_compile_error("empty_tuple", &src) {
        panic!(
            "empty tuple host C must compile under strict ISO C:\n{stderr}\n--- source ---\n{src}"
        );
    }
}
