//! Shared test fixtures for `chelis-cli` integration tests.
//!
//! ## Why this exists
//!
//! Pre-v0.2.8, every test file under `tests/` re-implemented `make_app` ≈
//! `tempdir + copy_dir_recursive(packages/chelis-std) + chelis reef publish`
//! per test. With ~12 integration test files and 80+ tests in aggregate,
//! that meant 80+ `chelis reef publish` invocations per CI run. The
//! v0.2.7 release CI run measured 33m 42s on the integration step, with
//! per-test publish overhead the dominant cost.
//!
//! `SharedReef` publishes chelis-std exactly once per test binary and
//! also pre-warms the lazy archive-extraction cache (a check-then-act
//! race surfaces at `--test-threads >= 8` without the warm pass — see
//! `feedback_shared_test_fixtures.md`). Tests then call `make_app` to
//! allocate a fresh per-test app shell pointing at the shared registry.
//!
//! ## Concurrency
//!
//! Per `crates/chelis-reef/src/lib.rs:353` (`load_package_graph_for_eval`)
//! eval is read-only against `CHELIS_REEF_HOME`. Build/check write only
//! the app's own `reef.lock`, which lives under the per-test app dir.
//! After the cache pre-warm, intra-binary thread parallelism is safe at
//! `--test-threads=8` and `--test-threads=16` (verified empirically).
//!
//! ## Usage
//!
//! ```ignore
//! #[path = "common/mod.rs"]
//! mod common;
//! use common::{make_app, COMPILER_VERSION};
//! ```
//!
//! Then:
//! ```ignore
//! let (_dir, reef_home, app_pkg) = make_app("my-test-name");
//! ```
//!
//! The returned `_dir: TempDir` keeps the per-test app alive for the
//! test scope; drop it to clean up.

// Each integration-test binary `#[path]`-includes this whole module but
// uses only the subset of fixtures it needs: the reef/std tests use
// `make_app` (and the publish chain it pulls in), while the codegen
// build/run tests use `build_and_run` (and friends). An item that is
// unused *in a given includer* is therefore expected, not rot. Allow it
// module-wide — the standard `tests/common/mod.rs` idiom — so the
// `-D warnings` clippy gate does not flag the unused subset per binary.
#![allow(dead_code)]

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use std::sync::LazyLock;
use tempfile::{TempDir, tempdir};

/// Pinned compiler version for fixture `reef.toml` files. Re-exported from
/// `chelis_compiler_api::COMPILER_VERSION`, which uses `env!("CARGO_PKG_VERSION")`
/// and therefore auto-syncs with `workspace.package.version` on bumps.
pub use chelis_compiler_api::COMPILER_VERSION;

pub fn package_std() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../packages/chelis-std")
        .canonicalize()
        .expect("chelis-std package must exist")
}

pub fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, contents).expect("write file");
}

pub fn copy_dir_recursive(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).expect("create dir");
    for entry in fs::read_dir(src).expect("read dir") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        let target = dst.join(entry.file_name());
        if path.is_dir() {
            copy_dir_recursive(&path, &target);
        } else {
            fs::copy(&path, &target).expect("copy file");
        }
    }
}

pub struct SharedReef {
    _dir: TempDir,
    pub reef_home: PathBuf,
}

pub static SHARED_REEF: LazyLock<SharedReef> = LazyLock::new(|| {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    copy_dir_recursive(&package_std(), &std_pkg);
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    // Pre-warm the lazy archive-extract cache. The first `chelis check`
    // against a chelis-std-importing app extracts `chelis-std-0.1.0.tar.zst`
    // into `reef_home/cache/<hash>/`. Without this serializing pass, threads
    // racing on the extract surfaced `failed to read .../reef.toml: No such
    // file or directory` at --test-threads=8.
    let warm_app = dir.path().join("__cache_warm");
    fs::create_dir_all(warm_app.join("src")).expect("mkdir warm app");
    fs::write(
        warm_app.join("reef.toml"),
        format!(
            r#"[package]
name = "cache-warm"
version = "0.1.0"
compiler = "={COMPILER_VERSION}"
module_prefix = "Demo"

[dependencies]
chelis-std = {{ version = "0.4.0" }}
"#
        ),
    )
    .expect("write warm reef.toml");
    fs::write(
        warm_app.join("src/main.ch"),
        "module Demo.Main\nimport Std.Test (assert_true)\nwarmed = true\n",
    )
    .expect("write warm main.ch");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&warm_app)
        .args(["check", warm_app.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success();

    SharedReef {
        _dir: dir,
        reef_home,
    }
});

/// Allocate a fresh per-test app shell that depends on the shared
/// `chelis-std` published in `SHARED_REEF`. Returns
/// `(TempDir, reef_home, app_pkg)`. Drop the `TempDir` to clean up.
pub fn make_app(dir_name: &str) -> (TempDir, PathBuf, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let app_pkg = dir.path().join(dir_name);
    fs::create_dir_all(app_pkg.join("src")).expect("mkdir app src");
    write_file(
        &app_pkg.join("reef.toml"),
        &format!(
            r#"[package]
name = "{dir_name}"
version = "0.1.0"
compiler = "={COMPILER_VERSION}"
module_prefix = "Demo"

[dependencies]
chelis-std = {{ version = "0.4.0" }}
"#
        ),
    );
    (dir, SHARED_REEF.reef_home.clone(), app_pkg)
}

// ---------------------------------------------------------------------------
// Codegen build/run harness
//
// Shared by the `chelis build --target c` integration tests that compile the
// generated C with the host toolchain and run it. The suites that share the
// identical build-link-run recipe — issue_527, issue_218, issue_289, and
// issue_345 — pull `build_and_run` / `link_generated` / `parse_tensor_data` /
// `write_file` from here, so a change to the host link recipe lands in one
// place instead of drifting across per-file copies.
//
// Suites whose recipe genuinely differs keep their own variant and are NOT
// routed through here: jit_par_runtime_gap parses f32 output, while
// numeric_dtype_adversarial and monomorphization_build only compile (the
// latter object-compiles) rather than build-link-run this shape.
// ---------------------------------------------------------------------------

/// Whether the chelis-generated C `source` (already written under `out_dir`)
/// pulls in the BLAS-backed runtime path, so the host link step must add the
/// BLAS link flags.
pub fn generated_source_needs_blas(out_dir: &Path, source: &str) -> bool {
    fs::read_to_string(out_dir.join(source))
        .is_ok_and(|text| text.contains("cblas_sgemm(") || text.contains("\"chelis_blas.h\""))
}

/// Link the chelis-generated C against the platform host toolchain, resolving
/// compile/link flags (OpenMP, and BLAS when the source needs it) from
/// `chelis_backend_c::toolchain`.
pub fn link_generated(out_dir: &Path, source: &str, binary: &str) -> std::process::ExitStatus {
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas: generated_source_needs_blas(out_dir, source),
        },
    );
    let mut cmd = StdCommand::new(&toolchain.compiler);
    cmd.current_dir(out_dir);
    cmd.arg("-O2");
    cmd.args(&toolchain.compile_flags);
    cmd.arg(source);
    cmd.args(["-L.", "-lchelis_runtime"]);
    cmd.args(&toolchain.link_flags);
    cmd.args(["-o", binary]);
    cmd.status().expect("host compiler should run")
}

/// Parse a printed Chelis tensor of the form
/// `name = tensor(shape=[..], data=[v0, v1, ...])` into its flat data.
pub fn parse_tensor_data(stdout: &str, name: &str) -> Vec<f64> {
    let prefix = format!("{name} = tensor(");
    let line = stdout
        .lines()
        .find(|l| l.starts_with(&prefix))
        .unwrap_or_else(|| panic!("output does not contain `{prefix}` line:\n{stdout}"));
    let data_marker = "data=[";
    let start = line
        .find(data_marker)
        .unwrap_or_else(|| panic!("no `data=[` in line: {line}"))
        + data_marker.len();
    let end = line[start..]
        .find(']')
        .unwrap_or_else(|| panic!("no closing `]` after data: {line}"));
    line[start..start + end]
        .split(',')
        .map(|s| s.trim().parse::<f64>().expect("numeric"))
        .collect()
}

/// Whether a host C compiler is available, so codegen build/run tests can
/// skip cleanly on machines without one.
pub fn gcc_available() -> bool {
    StdCommand::new(chelis_backend_c::toolchain::c_compiler())
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Build `source` to C, link it, run it, and return its stdout. Panics with a
/// diagnostic if any stage fails.
pub fn build_and_run(source: &str, name: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, source);

    Command::cargo_bin("chelis")
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
        .assert()
        .success();

    let source_file = format!("{name}.c");
    let status = link_generated(&out_dir, &source_file, name);
    assert!(status.success(), "link failed: {status}");

    let run_output = StdCommand::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "binary failed: {}\nstderr: {}",
        run_output.status,
        String::from_utf8_lossy(&run_output.stderr),
    );
    String::from_utf8(run_output.stdout).expect("utf-8 stdout")
}

// ---------------------------------------------------------------------------
// WS-C packaging-orchestration fixtures
//
// Shared by reef_setup.rs, reef_doctor_unified.rs, and
// unknown_subcommand_hint.rs, which all synthesize a pinned shell
// `reef.toml` (and, for the store-reading paths, a stub toolchain) inside
// an isolated `CHELIS_HOME`.
// ---------------------------------------------------------------------------

/// Write a shell `reef.toml` pinned to `pin`, appending `extra` verbatim
/// (e.g. a `[chelis-src]` or `[artifacts]` section). Creates the directory.
pub fn write_pinned_reef_toml(dir: &Path, pin: &str, extra: &str) {
    fs::create_dir_all(dir).unwrap();
    fs::write(
        dir.join("reef.toml"),
        format!(
            "[package]\nname = \"shelly\"\nversion = \"0.1.0\"\n\
             compiler = \"={pin}\"\nmodule_prefix = \"Shelly\"\n{extra}"
        ),
    )
    .unwrap();
}

/// Stub an installed toolchain in the chelisup store: a real `bin/chelis`
/// file under `<home>/toolchains/<ver>/`. `Store::is_installed` only checks
/// that this path is a file.
pub fn stub_toolchain(home: &Path, ver: &str) {
    let bin = home.join("toolchains").join(ver).join("bin");
    fs::create_dir_all(&bin).unwrap();
    fs::write(bin.join("chelis"), b"#!/bin/true\n").unwrap();
}
