//! Phase A bundled-chelis-std loader property oracle.
//!
//! This file owns the named test
//! `phaseA_bundled_chelis_std_loader_property_oracle`, which is the
//! single comprehensive oracle for the architectural correction that
//! makes chelis-std's bytes reachable from the chelis binary itself
//! (via `chelis-std-bundle` + `include_bytes!`) and synthesized into
//! every project's lockfile via `compiler =` pin.
//!
//! Sub-cases (asserted in one test so the oracle is one entrypoint):
//!
//! 1. **Implicit-runtime project builds.** A reef project whose
//!    `[dependencies]` does NOT list chelis-std builds successfully
//!    against an empty `$CHELIS_REEF_HOME`. The lockfile carries a
//!    `chelis-std` entry with `LockSource::Bundled`.
//! 2. **Explicit-runtime project builds.** Same shape, but the
//!    project lists `chelis-std = { version = "0.4.0" }` explicitly.
//!    The lockfile is structurally identical to the implicit case
//!    (synthesis is idempotent).
//! 3. **Wipe-and-rebuild.** After the first build, the test wipes
//!    `$CHELIS_REEF_HOME/packages/chelis-std/` (simulating "registry
//!    state has no chelis-std at all"). A second `chelis reef build`
//!    must succeed because the bundled bytes are reachable from the
//!    chelis binary, not from the registry.
//! 4. **No registry at all.** A pristine `$CHELIS_REEF_HOME` with no
//!    `index.json` and no `packages/` dir builds successfully. This
//!    is the "fresh dev clone with `chelis reef build` works" case
//!    the bundle is designed to support.
//!
//! Negative parity: `phaseA_item1_negative_parity` (separate test,
//! same file) asserts that a project explicitly listing chelis-std at
//! a non-bundled version fails with the soft-verify error and does
//! NOT silently get papered over by synthesis.
//!
//! No `#[ignore]` here. The test must be green by default. If it
//! cannot be, the bundle/loader/synthesis design is under-implemented.

#![allow(non_snake_case)]

use assert_cmd::Command;
use chelis_reef::{LockSource, ReefLock};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;
use walkdir::WalkDir;

fn runtime_package_files(root: &Path) -> BTreeMap<PathBuf, String> {
    let mut files = BTreeMap::new();
    let manifest = PathBuf::from("reef.toml");
    files.insert(
        manifest.clone(),
        fs::read_to_string(root.join(&manifest)).unwrap_or_else(|error| {
            panic!("read runtime package file {}: {error}", manifest.display())
        }),
    );
    for entry in WalkDir::new(root.join("src")) {
        let entry = entry.expect("walk runtime package src tree");
        if !entry.file_type().is_file() {
            continue;
        }
        let relative = entry
            .path()
            .strip_prefix(root)
            .expect("runtime package entry stays below root")
            .to_path_buf();
        files.insert(
            relative,
            fs::read_to_string(entry.path()).expect("read runtime package source file"),
        );
    }
    files
}

#[test]
fn bundled_chelis_std_sources_match_the_checked_in_runtime_package() {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = crate_root
        .parent()
        .and_then(Path::parent)
        .expect("chelis-cli is a workspace crate");
    let package_root = repo_root.join("packages/chelis-std");
    let extracted = tempdir().expect("bundle extraction tempdir");
    chelis_std_bundle::extract_into(extracted.path()).expect("extract embedded chelis-std");

    let embedded_files = runtime_package_files(extracted.path());
    let checked_in_files = runtime_package_files(&package_root);
    assert_eq!(
        embedded_files.keys().collect::<Vec<_>>(),
        checked_in_files.keys().collect::<Vec<_>>(),
        "embedded chelis-std file inventory differs from packages/chelis-std"
    );
    for (path, checked_in) in &checked_in_files {
        assert!(
            embedded_files.get(path) == Some(checked_in),
            "embedded chelis-std source drift at {}; rerun scripts/regenerate_chelis_std_bundle.py and commit every generated distribution artifact",
            path.display()
        );
    }

    let version = chelis_std_bundle::BUNDLED_CHELIS_STD_VERSION;
    let package_dist = package_root.join("dist");
    assert!(
        fs::read(package_dist.join(format!("chelis-std-{version}.tar.zst")))
            .expect("read package archive artifact")
            == chelis_std_bundle::CHELIS_STD_ARCHIVE,
        "packages/chelis-std and the compile-time bundle must carry one archive"
    );
    assert!(
        fs::read(package_dist.join(format!("chelis-std-{version}.chb")))
            .expect("read package shell artifact")
            == chelis_std_bundle::CHELIS_STD_SHELL,
        "packages/chelis-std and the compile-time bundle must carry one shell"
    );

    let lock = read_lockfile(&package_root);
    let bundled = lock
        .dependencies
        .iter()
        .find(|dependency| dependency.name == "chelis-std")
        .expect("the runtime package lock carries its bundled self-dependency");
    assert_eq!(
        bundled.archive_sha256,
        chelis_std_bundle::archive_sha256(),
        "the runtime package lock must pin the shipped archive bytes"
    );
    assert_eq!(
        bundled.shell_sha256,
        chelis_std_bundle::shell_sha256(),
        "the runtime package lock must pin the shipped shell bytes"
    );
}

fn write_implicit_runtime_project(root: &Path, module_prefix: &str) {
    fs::create_dir_all(root.join("src")).expect("mkdir src");
    let compiler = format!("={}", env!("CARGO_PKG_VERSION"));
    fs::write(
        root.join("reef.toml"),
        format!(
            r#"[package]
name = "downstream"
version = "0.4.0"
compiler = "{compiler}"
module_prefix = "{module_prefix}"

[dependencies]
"#
        ),
    )
    .expect("write reef.toml");
    fs::write(
        root.join("src/main.ch"),
        format!("module {module_prefix}.Main\n\ndef id(x: i32) -> i32 = x\n"),
    )
    .expect("write main.ch");
}

fn write_explicit_runtime_project(root: &Path, module_prefix: &str, std_version: &str) {
    fs::create_dir_all(root.join("src")).expect("mkdir src");
    let compiler = format!("={}", env!("CARGO_PKG_VERSION"));
    fs::write(
        root.join("reef.toml"),
        format!(
            r#"[package]
name = "downstream"
version = "0.4.0"
compiler = "{compiler}"
module_prefix = "{module_prefix}"

[dependencies]
chelis-std = {{ version = "{std_version}" }}
"#
        ),
    )
    .expect("write reef.toml");
    fs::write(
        root.join("src/main.ch"),
        format!("module {module_prefix}.Main\n\ndef id(x: i32) -> i32 = x\n"),
    )
    .expect("write main.ch");
}

fn read_lockfile(pkg_root: &Path) -> ReefLock {
    let text = fs::read_to_string(pkg_root.join("reef.lock")).expect("read reef.lock");
    toml::from_str(&text).expect("parse reef.lock")
}

fn assert_chelis_std_bundled_entry(lock: &ReefLock) {
    let entry = lock
        .dependencies
        .iter()
        .find(|d| d.name == "chelis-std")
        .unwrap_or_else(|| {
            panic!(
                "lockfile must include chelis-std; got entries: {:?}",
                lock.dependencies
                    .iter()
                    .map(|d| (&d.name, &d.version))
                    .collect::<Vec<_>>()
            )
        });
    assert_eq!(entry.version, "0.4.0", "bundled chelis-std version");
    match &entry.source {
        LockSource::Bundled { compiler_version } => {
            // The compiler_version field records the version of the
            // compiler that supplied the bundled bytes — by
            // construction, the running test binary's
            // CARGO_PKG_VERSION.
            assert_eq!(compiler_version, env!("CARGO_PKG_VERSION"));
        }
        other => panic!("chelis-std must use LockSource::Bundled, got {other:?}"),
    }
    assert!(
        !entry.archive_sha256.is_empty(),
        "Bundled chelis-std must carry archive_sha256 from embedded bytes"
    );
    assert!(
        !entry.shell_sha256.is_empty(),
        "Bundled chelis-std must carry shell_sha256 from embedded bytes"
    );
}

#[test]
fn phaseA_bundled_chelis_std_loader_property_oracle() {
    // Sub-case 1: implicit-runtime project + empty CHELIS_REEF_HOME.
    {
        let dir = tempdir().expect("tempdir");
        let pkg_root = dir.path().join("implicit-pkg");
        let reef_home = dir.path().join("reef-home");
        write_implicit_runtime_project(&pkg_root, "Implicit");

        Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef_home)
            .args(["reef", "build", pkg_root.to_str().unwrap()])
            .assert()
            .success();
        let lock = read_lockfile(&pkg_root);
        assert_chelis_std_bundled_entry(&lock);

        // Sub-case 4: a pristine reef_home (no index.json, no
        // packages/) suffices because the bundled bytes are reachable
        // from the chelis binary itself. We've just demonstrated that
        // — pin it explicitly so a regression that re-introduced a
        // registry-write side effect surfaces here.
        // Note: chelis reef build creates index.json on first run for
        // the registry it touches, but it should NOT create a
        // chelis-std entry under packages/.
        let chelis_std_pkg_dir = reef_home.join("packages").join("chelis-std");
        assert!(
            !chelis_std_pkg_dir.exists(),
            "build must not write chelis-std into the local registry; \
             expected absent: {chelis_std_pkg_dir:?}"
        );
    }

    // Sub-case 2: explicit-runtime project produces structurally
    // identical lockfile.
    {
        let dir = tempdir().expect("tempdir");
        let pkg_root = dir.path().join("explicit-pkg");
        let reef_home = dir.path().join("reef-home");
        write_explicit_runtime_project(&pkg_root, "Explicit", "0.4.0");

        Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef_home)
            .args(["reef", "build", pkg_root.to_str().unwrap()])
            .assert()
            .success();
        let lock = read_lockfile(&pkg_root);
        assert_chelis_std_bundled_entry(&lock);
        // Explicit and implicit must produce identical chelis-std
        // entries: synthesis is a no-op when the closure already
        // produced the entry.
        let entry = lock
            .dependencies
            .iter()
            .find(|d| d.name == "chelis-std")
            .expect("chelis-std entry");
        // archive/shell sha256s come from the embedded bundle in
        // both paths so they MUST agree.
        assert_eq!(
            entry.archive_sha256,
            chelis_std_bundle::archive_sha256(),
            "explicit-listing path must use bundled archive_sha256"
        );
        assert_eq!(
            entry.shell_sha256,
            chelis_std_bundle::shell_sha256(),
            "explicit-listing path must use bundled shell_sha256"
        );
    }

    // Sub-case 3: wipe-then-rebuild. Build once; wipe everything in
    // CHELIS_REEF_HOME (including index.json); rebuild. Must
    // succeed because the bundled bytes don't depend on registry
    // state.
    {
        let dir = tempdir().expect("tempdir");
        let pkg_root = dir.path().join("wipe-pkg");
        let reef_home = dir.path().join("reef-home");
        write_implicit_runtime_project(&pkg_root, "Wipe");

        Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef_home)
            .args(["reef", "build", pkg_root.to_str().unwrap()])
            .assert()
            .success();

        // Wipe the entire reef home (registry).
        if reef_home.exists() {
            fs::remove_dir_all(&reef_home).expect("wipe reef home");
        }

        Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef_home)
            .args(["reef", "build", pkg_root.to_str().unwrap()])
            .assert()
            .success();

        // Lockfile is regenerated by the second build; it must
        // again carry the Bundled chelis-std entry.
        let lock = read_lockfile(&pkg_root);
        assert_chelis_std_bundled_entry(&lock);
    }

    // Sub-case 3b: wipe ONLY chelis-std out of the registry, keeping
    // index.json. This exercises the "MissingFromIndex" branch of
    // load_registry_package — but the bundle's special-case at the
    // top of that function returns before consulting index.json, so
    // the build still succeeds.
    {
        let dir = tempdir().expect("tempdir");
        let pkg_root = dir.path().join("wipe-std-only");
        let reef_home = dir.path().join("reef-home");
        write_explicit_runtime_project(&pkg_root, "WipeStdOnly", "0.4.0");

        Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef_home)
            .args(["reef", "build", pkg_root.to_str().unwrap()])
            .assert()
            .success();

        // Wipe only the chelis-std pkg dir (if it was ever written).
        let std_pkg_dir = reef_home.join("packages").join("chelis-std");
        if std_pkg_dir.exists() {
            fs::remove_dir_all(&std_pkg_dir).expect("wipe chelis-std pkg");
        }

        Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef_home)
            .args(["reef", "build", pkg_root.to_str().unwrap()])
            .assert()
            .success();
        let lock = read_lockfile(&pkg_root);
        assert_chelis_std_bundled_entry(&lock);
    }
}

/// Negative parity for Item 1: a project explicitly listing
/// chelis-std at a non-bundled version must error with the
/// soft-verify message, and the lockfile must NOT be silently
/// rewritten to the bundled version.
#[test]
fn phaseA_item1_negative_parity_explicit_version_mismatch() {
    let dir = tempdir().expect("tempdir");
    let pkg_root = dir.path().join("mismatch-pkg");
    let reef_home = dir.path().join("reef-home");
    write_explicit_runtime_project(&pkg_root, "Mismatch", "9.9.9");

    let assert_out = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "build", pkg_root.to_str().unwrap()])
        .assert()
        .failure();

    let stderr = String::from_utf8_lossy(&assert_out.get_output().stderr).to_string();
    assert!(
        stderr.contains("chelis-std") && stderr.contains("9.9.9") && stderr.contains("0.4.0"),
        "soft-verify error must name both versions; got stderr:\n{stderr}"
    );

    // The lockfile must NOT have been written silently behind a
    // mismatch error.
    assert!(
        !pkg_root.join("reef.lock").exists(),
        "soft-verify mismatch must not write reef.lock; the synthesis \
         path must not paper over the explicit declaration"
    );
}
