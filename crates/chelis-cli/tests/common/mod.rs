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

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
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
chelis-std = {{ version = "0.2.0" }}
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
chelis-std = {{ version = "0.2.0" }}
"#
        ),
    );
    (dir, SHARED_REEF.reef_home.clone(), app_pkg)
}
