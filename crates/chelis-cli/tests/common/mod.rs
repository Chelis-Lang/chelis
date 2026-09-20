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
//! `SharedReef` publishes chelis-std and pre-warms the lazy archive-extraction
//! cache (a check-then-act race surfaces at `--test-threads >= 8` without the
//! warm pass — see `feedback_shared_test_fixtures.md`). Local runs do this once
//! per test binary in an isolated tempdir. CI sets
//! `CHELIS_TEST_SHARED_REEF_HOME` to one absolute, job-scoped root; an atomic
//! cross-process lock and versioned sentinel then prepare that root once across
//! every integration-test binary. Tests call `make_app` to allocate a fresh
//! per-test app shell pointing at the shared registry.
//!
//! ## Concurrency
//!
//! Eval, check, and build can populate content-addressed compiled-context and
//! prepared-graph caches under the shared `CHELIS_REEF_HOME`. Those writers use
//! per-process temporary files and atomic renames, while each app's `reef.lock`
//! remains under its isolated per-test directory. The fixture pre-warm reduces
//! cache contention, and the atomic writers make remaining intra- and
//! cross-binary races safe. An empty, relative, partially initialized, or
//! incompatible configured root fails loudly rather than falling back to a
//! per-binary registry.
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
use std::env;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use std::sync::LazyLock;
use std::thread;
use std::time::{Duration, Instant};
use tempfile::{TempDir, tempdir};

/// Pinned compiler version for fixture `reef.toml` files. Re-exported from
/// `chelis_compiler_api::COMPILER_VERSION`, which uses `env!("CARGO_PKG_VERSION")`
/// and therefore auto-syncs with `workspace.package.version` on bumps.
pub use chelis_compiler_api::COMPILER_VERSION;

/// Return the injective C ABI symbol for an authored Chelis definition.
///
/// The emitter encodes the authored UTF-8 bytes as lowercase hexadecimal so
/// source names never borrow platform or C implementation namespaces.
pub fn authored_c_symbol(name: &str) -> String {
    let mut symbol = "chelis_fn_".to_string();
    for byte in name.bytes() {
        symbol.push_str(&format!("{byte:02x}"));
    }
    symbol
}

/// Locate the owned implementation body for an authored Chelis definition.
///
/// `host_body_definition` intentionally accepts an exact C symbol so its
/// synthetic parser controls remain independent of ABI policy. Generated-C
/// tests should use this wrapper instead of reconstructing the owned-body
/// symbol from the source spelling.
pub fn authored_host_body_definition<'a>(emitted: &'a str, name: &str) -> &'a str {
    host_body_definition(
        emitted,
        &format!("{}__chelis_owned_body", authored_c_symbol(name)),
    )
}

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
    _dir: Option<TempDir>,
    pub reef_home: PathBuf,
}

pub const SHARED_REEF_READY_FILE: &str = ".chelis-test-shared-reef-ready-v1";
const SHARED_REEF_LOCK_FILE: &str = ".chelis-test-shared-reef-prepare.lock";
const SHARED_REEF_WAIT_TIMEOUT: Duration = Duration::from_secs(120);
const SHARED_REEF_POLL_INTERVAL: Duration = Duration::from_millis(50);

fn shared_reef_ready_contract() -> String {
    format!("chelis-test-shared-reef-v1\ncompiler={COMPILER_VERSION}\nchelis-std=0.4.0\n")
}

pub fn resolve_configured_shared_reef_home(
    value: Option<OsString>,
) -> Result<Option<PathBuf>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_empty() {
        return Err("CHELIS_TEST_SHARED_REEF_HOME must not be empty".to_owned());
    }
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err(format!(
            "CHELIS_TEST_SHARED_REEF_HOME must be absolute, got {}",
            path.display()
        ));
    }
    Ok(Some(path))
}

fn ready_state(root: &Path) -> Result<bool, String> {
    let ready = root.join(SHARED_REEF_READY_FILE);
    if !ready.exists() {
        return Ok(false);
    }
    let actual = fs::read_to_string(&ready)
        .map_err(|error| format!("read shared Reef sentinel {}: {error}", ready.display()))?;
    let expected = shared_reef_ready_contract();
    if actual != expected {
        return Err(format!(
            "shared Reef sentinel {} has an incompatible contract",
            ready.display()
        ));
    }
    Ok(true)
}

struct PrepareLock {
    path: PathBuf,
}

impl Drop for PrepareLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

pub fn ensure_job_shared_reef_with<F>(root: &Path, prepare: F) -> Result<(), String>
where
    F: FnOnce(&Path) -> Result<(), String>,
{
    if !root.is_absolute() {
        return Err(format!(
            "shared Reef root must be absolute: {}",
            root.display()
        ));
    }
    fs::create_dir_all(root)
        .map_err(|error| format!("create shared Reef root {}: {error}", root.display()))?;
    let lock_path = root.join(SHARED_REEF_LOCK_FILE);
    let started = Instant::now();
    let mut prepare = Some(prepare);
    loop {
        if ready_state(root)? {
            return Ok(());
        }
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
        {
            Ok(mut lock_file) => {
                let _lock = PrepareLock {
                    path: lock_path.clone(),
                };
                writeln!(lock_file, "pid={}", std::process::id()).map_err(|error| {
                    format!("write shared Reef lock {}: {error}", lock_path.display())
                })?;
                if ready_state(root)? {
                    return Ok(());
                }
                let unexpected = fs::read_dir(root)
                    .map_err(|error| {
                        format!("inspect shared Reef root {}: {error}", root.display())
                    })?
                    .filter_map(Result::ok)
                    .map(|entry| entry.path())
                    .filter(|path| path != &lock_path)
                    .collect::<Vec<_>>();
                if !unexpected.is_empty() {
                    return Err(format!(
                        "shared Reef root {} is not empty and has no valid sentinel: {:?}",
                        root.display(),
                        unexpected
                    ));
                }
                prepare
                    .take()
                    .expect("initializer is consumed by only one lock owner")(root)?;
                let ready = root.join(SHARED_REEF_READY_FILE);
                let temporary = root.join(format!(
                    ".chelis-test-shared-reef-ready-{}.tmp",
                    std::process::id()
                ));
                fs::write(&temporary, shared_reef_ready_contract()).map_err(|error| {
                    format!(
                        "write shared Reef sentinel {}: {error}",
                        temporary.display()
                    )
                })?;
                fs::rename(&temporary, &ready).map_err(|error| {
                    format!("publish shared Reef sentinel {}: {error}", ready.display())
                })?;
                return Ok(());
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if started.elapsed() >= SHARED_REEF_WAIT_TIMEOUT {
                    return Err(format!(
                        "timed out waiting for shared Reef initializer lock {}",
                        lock_path.display()
                    ));
                }
                thread::sleep(SHARED_REEF_POLL_INTERVAL);
            }
            Err(error) => {
                return Err(format!(
                    "create shared Reef initializer lock {}: {error}",
                    lock_path.display()
                ));
            }
        }
    }
}

fn prepare_shared_reef(reef_home: &Path, scratch: &Path) {
    let std_pkg = scratch.join("chelis-std");
    copy_dir_recursive(&package_std(), &std_pkg);
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    // Pre-warm the lazy archive-extract cache. The first `chelis check`
    // against a chelis-std-importing app extracts `chelis-std-0.1.0.tar.zst`
    // into `reef_home/cache/<hash>/`. Without this serializing pass, threads
    // racing on the extract surfaced `failed to read .../reef.toml: No such
    // file or directory` at --test-threads=8.
    let warm_app = scratch.join("__cache_warm");
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
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(&warm_app)
        .args(["check", warm_app.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success();
}

pub static SHARED_REEF: LazyLock<SharedReef> = LazyLock::new(|| {
    let configured =
        resolve_configured_shared_reef_home(env::var_os("CHELIS_TEST_SHARED_REEF_HOME"))
            .unwrap_or_else(|error| panic!("invalid shared Reef configuration: {error}"));
    if let Some(reef_home) = configured {
        ensure_job_shared_reef_with(&reef_home, |root| {
            let scratch = tempdir().map_err(|error| error.to_string())?;
            prepare_shared_reef(root, scratch.path());
            Ok(())
        })
        .unwrap_or_else(|error| panic!("prepare job-scoped shared Reef: {error}"));
        SharedReef {
            _dir: None,
            reef_home,
        }
    } else {
        let dir = tempdir().expect("tempdir");
        let reef_home = dir.path().join("reef-home");
        prepare_shared_reef(&reef_home, dir.path());
        SharedReef {
            _dir: Some(dir),
            reef_home,
        }
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
        .map(|text| text.contains("cblas_sgemm(") || text.contains("\"chelis_blas.h\""))
        .unwrap_or(false)
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
        .map(|o| o.status.success())
        .unwrap_or(false)
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

/// Build a Reef-linked application to C, link the generated translation unit,
/// run it, and return stdout. This is the package-aware counterpart to
/// [`build_and_run`]: imports resolve through `reef_home`, and build output is
/// kept inside the per-test application directory.
pub fn build_and_run_app(reef_home: &Path, app_pkg: &Path, name: &str) -> String {
    let out_dir = app_pkg.join(format!("{name}-out"));
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app_pkg)
        .args([
            "build",
            app_pkg.join("src/main.ch").to_str().unwrap(),
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

// ---------------------------------------------------------------------------
// Domain-validity checker (chelis#729 Phase 0)
//
// Implements ONLY the "value set" column of the §C1 table in
// `spec/design/dtype_semantics.md`: pure membership of printed values in
// their declared dtype's value set. No finalize logic, no traps, no
// formatting rules (formatting faithfulness is chelis#732's contract, not
// this checker's; a value-preserving formatting lie like `750.0` for an
// i64 is IN domain here).
//
// FROZEN AT chelis#729 PHASE 0 EXIT: the API below
// (`assert_elements_in_domain(prim, printed, context)` plus the pure
// decision fn `element_domain_violation(prim, token)`) and the lane
// drivers' verbatim-string discipline (drivers return printed strings;
// nothing re-parses through f64 to compare values). Later phases treat
// "domain checker green" as acceptance evidence; changing the membership
// rules requires editing dtype_semantics.md and chelis#729 in the same
// change set (its §B1 protocol).
//
// Reading membership off PRINTED text has one documented consequence:
//
// TEXT AMBIGUITY IS RESOLVED TOWARD NO-FALSE-POSITIVES. A token that
//    is the shortest-round-trip rendering of an f32 at f32 width (for
//    example `0.1`) is accepted, even though the same text could have
//    been printed from an out-of-domain f64. Controls must never fail;
//    a missed violation surfaces later through the exact-string rows.
// ---------------------------------------------------------------------------

/// Retired `%.16g`-style print-truncation slack. chelis#732 Phase 2 made
/// compiled rendering shortest-round-trip, so Phase 0's promised ratchet is
/// now exact: near-but-distinct decimal tokens are domain violations.
pub const DOMAIN_PRINT_TRUNCATION_SLACK: f64 = 0.0;

/// Strip `List[...]` wrappers (the drivers pass return types like
/// `List[i64]` for `to_list` rows) down to the element prim name.
fn normalize_prim(prim: &str) -> &str {
    let mut p = prim.trim();
    while let Some(inner) = p.strip_prefix("List[").and_then(|s| s.strip_suffix(']')) {
        p = inner.trim();
    }
    p
}

/// Extract the printed value tokens from one lane's output line: the
/// `data=[...]` payload of a `tensor(...)` form, the elements of a
/// (possibly nested) `[...]` list, or the bare scalar payload. A leading
/// `name = ` binding echo is stripped.
pub fn printed_value_tokens(printed: &str) -> Vec<String> {
    let s = printed.trim();
    if let Some(dstart) = s.find("data=[") {
        let rest = &s[dstart + "data=[".len()..];
        if let Some(dend) = rest.find(']') {
            return split_value_tokens(&rest[..dend]);
        }
    }
    let s = match s.split_once(" = ") {
        Some((_, rhs)) if !s.starts_with('[') => rhs.trim(),
        _ => s,
    };
    if s.starts_with('[') && s.ends_with(']') {
        let inner: String = s.chars().filter(|c| *c != '[' && *c != ']').collect();
        return split_value_tokens(&inner);
    }
    vec![s.to_string()]
}

fn split_value_tokens(s: &str) -> Vec<String> {
    s.split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(String::from)
        .collect()
}

/// Pure membership decision for ONE printed element token against the
/// §C1 value-set column. `None` means member; `Some(reason)` names the
/// violation. Panics on an unknown prim name so a typo cannot silently
/// skip checking.
pub fn element_domain_violation(prim: &str, token: &str) -> Option<String> {
    let t = token.trim();
    match normalize_prim(prim) {
        "bool" => match t {
            "true" | "false" | "0" | "1" | "0.0" | "1.0" => None,
            _ => Some(format!("`{t}` is not a bool value (value set is {{0, 1}})")),
        },
        "f64" => match t.parse::<f64>() {
            Ok(_) => None,
            Err(_) => Some(format!("`{t}` is not parseable as a number")),
        },
        "f32" => narrow_float_violation(
            t,
            "f32",
            |d| (d as f32) as f64,
            |t| t.parse::<f32>().is_ok_and(|v| format!("{v:?}") == t),
        ),
        "f16" => narrow_float_violation(
            t,
            "f16",
            |d| chelis_types::f16_from_f64_rne(d).to_f64(),
            |t| {
                t.parse::<f64>().is_ok_and(|v| {
                    let w = chelis_types::f16_from_f64_rne(v);
                    chelis_types::observation::format_element(
                        chelis_types::types::Prim::F16,
                        chelis_types::observation::ElementRef::F16(w),
                    ) == t
                })
            },
        ),
        "bf16" => narrow_float_violation(
            t,
            "bf16",
            |d| chelis_types::bf16_from_f64_rne(d).to_f64(),
            |t| {
                t.parse::<f64>().is_ok_and(|v| {
                    let w = chelis_types::bf16_from_f64_rne(v);
                    chelis_types::observation::format_element(
                        chelis_types::types::Prim::Bf16,
                        chelis_types::observation::ElementRef::Bf16(w),
                    ) == t
                })
            },
        ),
        "i64" => int_violation(t, "i64", i64::MIN as i128, i64::MAX as i128),
        "i32" => int_violation(t, "i32", i32::MIN as i128, i32::MAX as i128),
        "i16" => int_violation(t, "i16", i16::MIN as i128, i16::MAX as i128),
        "i8" => int_violation(t, "i8", i8::MIN as i128, i8::MAX as i128),
        "f8e4m3" => Some(format!(
            "`{t}` claims dtype f8e4m3, which the checker rejects (spec/04 §1.1.1); \
             no runtime value may carry it"
        )),
        other => panic!("domain checker: unknown prim `{other}` (from `{prim}`)"),
    }
}

/// Shared narrow-float membership: exact widened rendering or own-width
/// shortest rendering. The zero slack below is retained as the executable
/// Phase 0 ratchet that rejects the retired `%.16g` accommodation.
/// This verifier alone reconstructs a value from already-rendered text: it
/// follows spec/05 §8.1 (`strtod`/f64, then one narrowing to the declared
/// half width). Runtime values never pass through this helper; storage and
/// transport remain at the value's declared dtype width.
fn narrow_float_violation(
    t: &str,
    prim: &str,
    round_trip: impl Fn(f64) -> f64,
    is_own_width_shortest: impl Fn(&str) -> bool,
) -> Option<String> {
    let d: f64 = match t.parse() {
        Ok(d) => d,
        Err(_) => return Some(format!("`{t}` is not parseable as a number")),
    };
    if d.is_nan() || d.is_infinite() {
        return None;
    }
    let h = round_trip(d);
    if h.is_infinite() {
        return Some(format!(
            "`{t}` is finite but exceeds the {prim} finite range (nearest {prim} is {h})"
        ));
    }
    if d == h {
        return None;
    }
    if is_own_width_shortest(t) {
        return None;
    }
    let rel = (d - h).abs() / d.abs().max(h.abs()).max(f64::MIN_POSITIVE);
    if rel <= DOMAIN_PRINT_TRUNCATION_SLACK {
        return None;
    }
    Some(format!(
        "`{t}` is not representable in {prim}: nearest {prim} value is {h:?}, \
         relative deviation {rel:e} exceeds the exact domain slack \
         {DOMAIN_PRINT_TRUNCATION_SLACK:e}"
    ))
}

/// Integer membership: integral value inside the width's range. Accepts
/// both integer-formatted and float-formatted renderings (the eval tensor
/// lane prints `100.0` for int tensors today; a value-preserving
/// formatting lie is chelis#732's problem, not a domain violation).
fn int_violation(t: &str, prim: &str, min: i128, max: i128) -> Option<String> {
    let looks_integral = {
        let body = t.strip_prefix('-').unwrap_or(t);
        !body.is_empty() && body.bytes().all(|b| b.is_ascii_digit())
    };
    if looks_integral {
        return match t.parse::<i128>() {
            Ok(v) if (min..=max).contains(&v) => None,
            Ok(v) => Some(format!("`{v}` is outside the {prim} range [{min}, {max}]")),
            Err(_) => Some(format!("`{t}` does not fit any integer width")),
        };
    }
    let d: f64 = match t.parse() {
        Ok(d) => d,
        Err(_) => return Some(format!("`{t}` is not parseable as a number")),
    };
    if !d.is_finite() {
        return Some(format!(
            "`{t}` is not a finite value; {prim} has no specials"
        ));
    }
    if d.fract() != 0.0 {
        return Some(format!("`{t}` is fractional; {prim} holds integers only"));
    }
    // Width bounds compared in f64. For widths below 64 bits both bounds
    // are exactly representable. For i64 the exclusive upper bound 2^63
    // is exact in f64 while i64::MAX is not; every integral f64 strictly
    // below 2^63 is <= i64::MAX (the f64 grid near 2^63 steps by 1024),
    // so `d < 2^63` is the correct membership test.
    let (lo, hi_exclusive) = if max == i64::MAX as i128 {
        (-(2f64.powi(63)), 2f64.powi(63))
    } else {
        (min as f64, max as f64 + 1.0)
    };
    if !(lo..hi_exclusive).contains(&d) {
        return Some(format!("`{t}` is outside the {prim} range [{min}, {max}]"));
    }
    None
}

/// THE chelis#729 Phase 0 detector entry point: assert every printed
/// element of `printed` is a member of `prim`'s value set (§C1's value
/// set column). Wire this wherever a lane driver's output and its declared
/// dtype meet. Panics with the offending token and reason.
pub fn assert_elements_in_domain(prim: &str, printed: &str, context: &str) {
    for token in printed_value_tokens(printed) {
        if let Some(reason) = element_domain_violation(prim, &token) {
            panic!(
                "domain violation [{context}]: {reason}. Full printed payload: \
                 `{printed}`. Value-set contract: spec/design/dtype_semantics.md \
                 §C1 (chelis#729 Phase 0 detector)."
            );
        }
    }
}

/// One emitted host body's DEFINITION, located by NAME rather than by its
/// exact parameter list, returned from the definition's first character to
/// the end of `emitted`. Callers slice their own end.
///
/// The emitted file carries a forward declaration and a definition for the
/// same symbol, so the name alone is ambiguous; the definition is the
/// occurrence whose parameter list is followed by `{` instead of `;`.
///
/// Matching the full signature instead is what chelis#1808 and chelis#1820
/// were: chelis#1799 gave every host body a `chelis_rng_state` parameter, and
/// a literal match on the old parameter list then failed before the row
/// counted anything, reporting a changed SIGNATURE as a missing definition. A
/// precondition that cannot tell a changed signature from a missing or
/// relocated body is worse than no precondition, so this one keys on the
/// structure it actually needs. chelis#1810 established the shape in
/// `runtime_extent_slice_b.rs` and chelis#1834 closed its two misreads there;
/// this carries the same two closures, shared. When that file next moves, it
/// can drop its private copy for this one.
///
/// The two misreads, both closed here. A match with no left word boundary
/// accepts `g_run__chelis_owned_body(` as `run__chelis_owned_body`, and the
/// wrong function's body then satisfies the caller's assertions for the wrong
/// reason. And taking the first `)` as the end of the parameter list mistakes
/// a nested parenthesis for the end of the signature, so the `{` test fails
/// and this reports a body that is present as having left the host lane --
/// chelis#1808's own misdiagnosis in a new spelling.
///
/// A third shape is unmodelled and stays that way deliberately. Neither
/// closure describes what sits BETWEEN the `)` and the `{`, and `trim_start`
/// consumes only whitespace, so a definition carrying an attribute or a
/// calling convention there -- `void f(int x) __attribute__((hot)) {` -- is
/// rejected and produces misread two's wrong diagnosis again. Nothing in
/// `crates/chelis-backend-c/src/` emits `__attribute__` or `__asm__`, so it is
/// latent rather than live; a case would pin a spelling the emitter does not
/// have. If one ever appears, this is where it lands.
pub fn host_body_definition<'a>(emitted: &'a str, name: &str) -> &'a str {
    /// The `)` that closes the first `(` in `text`, counting nesting.
    fn closing_paren(text: &str) -> Option<usize> {
        let open = text.find('(')?;
        let mut depth = 0usize;
        for (offset, byte) in text.bytes().enumerate().skip(open) {
            match byte {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(offset);
                    }
                }
                _ => {}
            }
        }
        None
    }

    let needle = format!("{name}(");
    let mut at = 0;
    while let Some(found) = emitted[at..].find(&needle) {
        let start = at + found;
        at = start + needle.len();
        let preceded_by_identifier = emitted[..start]
            .chars()
            .next_back()
            .is_some_and(|previous| previous.is_alphanumeric() || previous == '_');
        if preceded_by_identifier {
            continue;
        }
        let rest = &emitted[start..];
        if let Some(close) = closing_paren(rest)
            && rest[close + 1..].trim_start().starts_with('{')
        {
            return rest;
        }
    }
    panic!(
        "no definition of `{name}` in the emitted C: a forward declaration alone means the body \
         is not emitted here:\n{emitted}"
    );
}
