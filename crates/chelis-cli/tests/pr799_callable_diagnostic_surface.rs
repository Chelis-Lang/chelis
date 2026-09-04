//! chelis#730 Phase 2 - CLI-lane diagnostic parity for unresolved
//! callable values (`docs/investigations/pr799_returned_function_values_redteam.md`,
//! finding F1).
//!
//! The compiler API rejects a used returned/dynamically-selected function
//! value with the frozen `unsupported:` function-value diagnostic
//! (`pr799_host_resolution_result.rs`). `chelis build --target c` must
//! surface the SAME class of diagnostic for the same source: the
//! grad/vmap gate text is reserved for defs carrying the unspellable
//! transform marker (`HOST_UNRESOLVED_TRANSFORM_MARKER`, scanned by
//! `host_program_unresolved_transform_sites`), and a plain unresolved
//! callable carries the callable marker and falls through to ABI
//! projection's frozen rejection (chelis#841). Every rejection here must
//! also leave zero emitted artifacts.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

/// `chelis build --target c`; (build_ok, stderr, emitted (file name, contents)).
fn c_build(program: &str, name: &str) -> (bool, String, Vec<(String, String)>) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("chelis build should run");
    let mut emitted = Vec::new();
    if out_dir.is_dir() {
        for entry in std::fs::read_dir(&out_dir).expect("read out dir") {
            let p = entry.expect("entry").path();
            if p.is_file()
                && let Ok(text) = std::fs::read_to_string(&p)
            {
                emitted.push((p.file_name().unwrap().to_string_lossy().into_owned(), text));
            }
        }
    }
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        emitted,
    )
}

fn assert_frozen_callable_rejection(stderr: &str, emitted: &[(String, String)]) {
    assert!(
        stderr.contains("unsupported:") && stderr.contains("function value"),
        "the CLI must surface the frozen function-value diagnostic:\n{stderr}"
    );
    assert!(
        !stderr.contains("grad") && !stderr.contains("vmap"),
        "callable rejection must not be misclassified as an AD transform failure:\n{stderr}"
    );
    let names: Vec<&str> = emitted.iter().map(|(name, _)| name.as_str()).collect();
    assert!(
        emitted.is_empty(),
        "a rejected build must emit no artifacts, found: {names:?}"
    );
}

/// The compiler-API fixture from
/// `used_returned_named_function_rejects_before_an_unresolved_c_call_is_emitted`,
/// driven through the CLI surface.
#[test]
fn used_returned_named_function_gets_the_frozen_callable_diagnostic() {
    let (ok, stderr, emitted) = c_build(
        "def increment(x: int8) -> int8 = add(x, cast(1, int8))\n\
         def choose() -> int8 -> int8 = increment\n\
         chosen = choose()\n\
         out = print(chosen(cast(6, int8)))\n",
        "returned_named_used",
    );
    assert!(!ok, "a used returned function value must not build for C");
    assert_frozen_callable_rejection(&stderr, &emitted);
}

/// The compiler-API fixture from
/// `dynamically_selected_named_callback_has_no_c_host_value_abi`, driven
/// through the CLI surface.
#[test]
fn dynamically_selected_named_callback_gets_the_frozen_callable_diagnostic() {
    let (ok, stderr, emitted) = c_build(
        "def increment(x: int8) -> int8 = add(x, cast(1, int8))\n\
         def decrement(x: int8) -> int8 = sub(x, cast(1, int8))\n\
         selected = if true then increment else decrement\n\
         out = print(selected(cast(6, int8)))\n",
        "selected_callback",
    );
    assert!(!ok, "a dynamically selected callback must not build for C");
    assert_frozen_callable_rejection(&stderr, &emitted);
}

/// A def named `call` no longer collides with the lowerer's fallback
/// marker (unspellable since chelis#841); the supported program builds.
/// The full end-to-end lock lives in `issue_841_sentinel_collisions.rs`.
#[test]
fn a_def_named_call_is_an_ordinary_supported_program() {
    let (ok, stderr, _emitted) = c_build(
        "def call(x: int32) -> int32 = add(x, 1)\n\
         out = print(call(5))\n",
        "def_named_call",
    );
    assert!(
        ok,
        "a legal identifier must not read as a compiler marker:\n{stderr}"
    );
}

/// Control: a program that actually applies `grad` in a position the
/// host lane can't resolve (grad through host-lane `fold`) must keep the
/// AD-transform workaround text, not the generic callable diagnostic.
#[test]
fn unresolved_grad_positions_keep_the_ad_workaround_text() {
    let (ok, stderr, emitted) = c_build(
        "def sum_list(theta: f32) -> f32 =\n\
           fold(fn (acc: f32, x: f32) -> add(acc, mul(theta, x)), 0.0, [1.0, 2.0])\n\
         def gradient(theta: f32) -> f32 = grad(sum_list)(theta)\n\
         out = print(gradient(3.0))\n",
        "grad_through_fold",
    );
    assert!(!ok, "grad through host-lane fold still rejects");
    assert!(
        stderr.contains("applies/binds `grad`"),
        "an actual grad position must keep the workaround guidance:\n{stderr}"
    );
    assert!(
        emitted.is_empty(),
        "a rejected build must emit no artifacts"
    );
}

/// Positive neighbor: a declared callback parameter taking a named
/// callback keeps its exact typed C ABI and builds cleanly.
#[test]
fn declared_callback_parameter_with_named_callback_still_builds() {
    let (ok, stderr, emitted) = c_build(
        "def increment(x: int8) -> int8 = add(x, cast(1, int8))\n\
         def apply8(callback: int8 -> int8, value: int8) -> int8 = callback(value)\n\
         out = print(apply8(increment, cast(6, int8)))\n",
        "declared_callback_positive",
    );
    assert!(
        ok,
        "a contextual named callback must keep building:\n{stderr}"
    );
    // The program's own translation unit, not the copied runtime headers
    // (those legitimately contain `void *` in unrelated helpers).
    let (_, generated_c) = emitted
        .iter()
        .find(|(name, _)| name == "declared_callback_positive.c")
        .expect("the program translation unit must be emitted");
    assert!(
        generated_c.contains("int8_t (*callback)(int8_t)"),
        "the callback must keep its exact typed declarator:\n{generated_c}"
    );
    for erased_callback_declarator in [
        "void *(*callback)(",
        "void* (*callback)(",
        "(*callback)(void *)",
        "(*callback)(void*)",
        "void *apply8(",
        "void* apply8(",
        "apply8(void *",
        "apply8(void*",
    ] {
        assert!(
            !generated_c.contains(erased_callback_declarator),
            "callback ABI contains erased declarator \
             {erased_callback_declarator:?}:\n{generated_c}"
        );
    }
}
