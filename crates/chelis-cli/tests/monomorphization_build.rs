//! WS-A8 build acceptance suite: monomorphization closes the
//! BLOCKER-class gap that made every polymorphic-precision sig
//! invisible to `chelis build`. The pre-WS-A8 WS-C v3 acceptance
//! suite (`stdlib_precision_generalization_followups.rs`) only exercised `chelis check`;
//! these tests pin `chelis build` and the reachable C-codegen
//! pipeline against the production stdlib polymorphic sigs.
//!
//! Per the project's "`acceptance_surface`" memory rule: codegen tests
//! must compile generated output, not just check string patterns.
//! At least two tests in this file invoke gcc on the emitted C and
//! assert the link/compile step succeeds, so the suite catches
//! regressions where monomorphization produces well-formed Rust IR
//! but broken C symbol shapes.
//!
//! Spec authority: spec/04-type-system.md sections 5.4, 5.7.2, 5.8,
//! 5.8.1.

use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

fn run_build_in(dir: &Path, source: &Path) -> std::process::Output {
    let bin = assert_cmd::cargo::cargo_bin("chelis");
    StdCommand::new(bin)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(dir)
        .args(["build", source.to_str().unwrap()])
        .output()
        .expect("spawn chelis build")
}

#[cfg(not(target_os = "macos"))]
fn gcc_available() -> bool {
    StdCommand::new("gcc")
        .arg("--version")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

/// Compile the emitted C with gcc into an object file. Returns the
/// process output (stdout/stderr/status). Caller asserts on
/// `output.status.success()`.
#[cfg(not(target_os = "macos"))]
fn gcc_compile_object(work_dir: &Path, c_file: &Path, object: &Path) -> std::process::Output {
    StdCommand::new("gcc")
        .arg("-O2")
        .arg("-fopenmp")
        .arg("-c")
        .arg(c_file)
        .arg("-I")
        .arg(work_dir)
        .arg("-o")
        .arg(object)
        .current_dir(work_dir)
        .output()
        .expect("spawn gcc")
}

// ============================================================
// Section A. Production stdlib build acceptance.
// ============================================================

// ============================================================
// Section B. Polymorphic sig + concrete call site builds and
// produces compilable C. Per the acceptance_surface rule the test
// invokes gcc on the emitted C.
// ============================================================

// macOS's gcc is a clang alias; clang requires libomp for -fopenmp which
// isn't installed on the GitHub macOS runner. Linux gcc supports OpenMP
// natively.
#[cfg(not(target_os = "macos"))]
#[test]
fn build_polymorphic_linear_call_site_compiles_with_gcc() {
    if !gcc_available() {
        eprintln!("gcc not available, skipping WS-A8 gcc-compile test");
        return;
    }
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("poly_linear_call.ch");
    write_file(
        &src,
        r#"sig forward: &tensor[a, b, p] -> &tensor[b, c, p] -> tensor[a, c, p]
def forward(x, w) = matmul(x, w)
def call(x: &tensor[2, 3, f32], w: &tensor[3, 4, f32]) -> tensor[2, 4, f32] = forward(x, w)
"#,
    );
    let output = run_build_in(dir.path(), &src);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "WS-A8: build of polymorphic linear sig + concrete call site \
         must succeed. stderr={stderr}"
    );
    let c_file = dir.path().join("poly_linear_call.c");
    assert!(
        c_file.exists(),
        "WS-A8: build must emit `poly_linear_call.c`. stderr={stderr}"
    );
    let object = dir.path().join("poly_linear_call.o");
    let gcc_out = gcc_compile_object(dir.path(), &c_file, &object);
    let gcc_stderr = String::from_utf8_lossy(&gcc_out.stderr);
    assert!(
        gcc_out.status.success(),
        "WS-A8: gcc must compile the emitted C cleanly. \
         gcc stderr={gcc_stderr}"
    );
    assert!(
        object.exists(),
        "WS-A8: gcc must produce an object file at {}",
        object.display()
    );
}

// macOS's gcc is a clang alias; clang requires libomp for -fopenmp which
// isn't installed on the GitHub macOS runner. Linux gcc supports OpenMP
// natively.
#[cfg(not(target_os = "macos"))]
#[test]
fn build_polymorphic_id_t_call_site_compiles_with_gcc() {
    // Smaller probe: a polymorphic id_t threaded through a concrete
    // f32 call site. The simplest possible monomorphization shape.
    if !gcc_available() {
        eprintln!("gcc not available, skipping WS-A8 gcc-compile test");
        return;
    }
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("poly_id_call.ch");
    write_file(
        &src,
        r#"sig id_t: tensor[n, p] -> tensor[n, p]
def id_t(x) = x
def call(x: tensor[3, f32]) -> tensor[3, f32] = id_t(x)
"#,
    );
    let output = run_build_in(dir.path(), &src);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "WS-A8: simple polymorphic id_t build must succeed. stderr={stderr}"
    );
    let c_file = dir.path().join("poly_id_call.c");
    assert!(
        c_file.exists(),
        "WS-A8: build must emit `poly_id_call.c`. stderr={stderr}"
    );
    let object = dir.path().join("poly_id_call.o");
    let gcc_out = gcc_compile_object(dir.path(), &c_file, &object);
    let gcc_stderr = String::from_utf8_lossy(&gcc_out.stderr);
    assert!(
        gcc_out.status.success(),
        "WS-A8: gcc must compile the simple polymorphic id_t emit \
         cleanly. gcc stderr={gcc_stderr}"
    );
}

// ============================================================
// Section C. Cross-row enforcement at build time. The pre-WS-A8
// path silently accepted polymorphic-int matmul; WS-A8 rejects at
// type-check, so build also rejects (front-end gates build).
// ============================================================

fn build_must_reject(src: &str, name: &str, expected_in_stderr: &str) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    write_file(&path, src);
    let output = run_build_in(dir.path(), &path);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !output.status.success(),
        "WS-A8: build must reject `{name}`. stdout={stdout} stderr={stderr}"
    );
    let combined = format!("{stdout}\n{stderr}");
    assert!(
        combined.contains(expected_in_stderr),
        "WS-A8: build rejection of `{name}` must mention `{expected_in_stderr}`. \
         output={combined}"
    );
}

#[test]
fn build_rejects_polymorphic_matmul_at_int_call_site() {
    let src = r"sig my_linear: &tensor[a, b, p] -> &tensor[b, c, p] -> tensor[a, c, p]
def my_linear(x, w) = matmul(x, w)
def use_int(x: &tensor[2, 3, int32], w: &tensor[3, 4, int32]) -> tensor[2, 4, int32] = my_linear(x, w)
";
    build_must_reject(src, "poly_int_matmul", "5.7.2");
}

#[test]
fn build_rejects_polymorphic_softmax_at_int_call_site() {
    let src = r"sig wrap: &tensor[n, m, p] -> tensor[n, m, p]
def wrap(x) = softmax(x, -1)
def use_int(x: &tensor[3, 4, int32]) -> tensor[3, 4, int32] = wrap(x)
";
    build_must_reject(src, "poly_int_softmax", "5.4");
}
