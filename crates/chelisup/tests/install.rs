//! Offline `install` integration via the `CHELISUP_RELEASE_BASE` local
//! seam, plus the CLI status surface and the end-to-end shim hop.
#![cfg(unix)]

mod common;

use common::{bin, build_fixture_tarball, run_cli};
use std::path::Path;
use std::process::Command;

fn slug() -> &'static str {
    chelisup::install::detect_slug().expect("host platform must be supported in CI")
}

fn stdout(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}
fn stderr(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// `CHELISUP_RELEASE_BASE`-seamed install from a local fixture tarball.
fn install(home: &Path, release: &Path, version: &str) -> std::process::Output {
    run_cli(
        home,
        home,
        &["install", version],
        &[("CHELISUP_RELEASE_BASE", release.to_str().unwrap())],
    )
}

#[test]
fn install_from_local_fixture_populates_store_and_seeds_default() {
    let home = tempfile::tempdir().unwrap();
    let release = tempfile::tempdir().unwrap();
    build_fixture_tarball(release.path(), "0.1.0", slug());

    let out = install(home.path(), release.path(), "0.1.0");
    assert!(out.status.success(), "stderr: {}", stderr(&out));

    // Toolchain unpacked, shim + installer copies written, default seeded.
    assert!(home.path().join("toolchains/0.1.0/bin/chelis").is_file());
    assert!(home.path().join("bin/chelis").is_file());
    assert!(home.path().join("bin/chelisup").is_file());
    assert_eq!(
        std::fs::read_to_string(home.path().join("default"))
            .unwrap()
            .trim(),
        "0.1.0"
    );
}

#[test]
fn install_is_idempotent() {
    let home = tempfile::tempdir().unwrap();
    let release = tempfile::tempdir().unwrap();
    build_fixture_tarball(release.path(), "0.1.0", slug());

    assert!(
        install(home.path(), release.path(), "0.1.0")
            .status
            .success()
    );
    let again = install(home.path(), release.path(), "0.1.0");
    assert!(again.status.success(), "stderr: {}", stderr(&again));
    assert!(
        stdout(&again).contains("already installed"),
        "stdout: {}",
        stdout(&again)
    );
}

#[test]
fn install_rejects_malformed_version() {
    let home = tempfile::tempdir().unwrap();
    let release = tempfile::tempdir().unwrap();
    let out = install(home.path(), release.path(), "0.13");
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("expected X.Y.Z"),
        "stderr: {}",
        stderr(&out)
    );
}

#[test]
fn install_missing_asset_is_a_loud_error() {
    let home = tempfile::tempdir().unwrap();
    let release = tempfile::tempdir().unwrap();
    // No fixture built: the asset is absent under the release base.
    let out = install(home.path(), release.path(), "0.1.0");
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("not found under CHELISUP_RELEASE_BASE"),
        "stderr: {}",
        stderr(&out)
    );
}

#[test]
fn installed_shim_resolves_and_runs_the_toolchain() {
    if !common::python3_available() {
        eprintln!("skipping: python3 not available for the fake toolchain");
        return;
    }
    let home = tempfile::tempdir().unwrap();
    let release = tempfile::tempdir().unwrap();
    build_fixture_tarball(release.path(), "0.1.0", slug());
    assert!(
        install(home.path(), release.path(), "0.1.0")
            .status
            .success()
    );

    // Invoke the installed shim copy directly (argv[0] basename "chelis").
    let work = tempfile::tempdir().unwrap();
    let out = Command::new(home.path().join("bin").join("chelis"))
        .args(["build", "main.ch"])
        .current_dir(work.path())
        .env("CHELIS_HOME", home.path())
        .env_remove("CHELIS_TOOLCHAIN")
        .output()
        .unwrap();
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(stdout(&out), "FAKE-CHELIS 0.1.0 build main.ch");
}

#[test]
fn list_installed_and_which_reflect_the_store() {
    let home = tempfile::tempdir().unwrap();
    let release = tempfile::tempdir().unwrap();
    build_fixture_tarball(release.path(), "0.1.0", slug());
    assert!(
        install(home.path(), release.path(), "0.1.0")
            .status
            .success()
    );

    let list = run_cli(home.path(), home.path(), &["list-installed"], &[]);
    assert!(list.status.success());
    assert!(
        stdout(&list).contains("0.1.0 (default)"),
        "stdout: {}",
        stdout(&list)
    );

    let which = run_cli(home.path(), home.path(), &["which"], &[]);
    assert!(which.status.success(), "stderr: {}", stderr(&which));
    assert!(
        stdout(&which)
            .trim()
            .ends_with("toolchains/0.1.0/bin/chelis"),
        "stdout: {}",
        stdout(&which)
    );
}

#[test]
fn which_is_loud_when_pin_is_not_installed() {
    let home = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    std::fs::write(
        work.path().join("reef.toml"),
        "[package]\ncompiler = \"=0.42.0\"\n",
    )
    .unwrap();
    let out = run_cli(home.path(), work.path(), &["which"], &[]);
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("chelisup install 0.42.0"),
        "stderr: {}",
        stderr(&out)
    );
}

#[test]
fn uninstall_removes_toolchain_and_unsets_default() {
    let home = tempfile::tempdir().unwrap();
    let release = tempfile::tempdir().unwrap();
    build_fixture_tarball(release.path(), "0.1.0", slug());
    assert!(
        install(home.path(), release.path(), "0.1.0")
            .status
            .success()
    );

    let out = run_cli(home.path(), home.path(), &["uninstall", "0.1.0"], &[]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert!(!home.path().join("toolchains/0.1.0").exists());
    // It was the default, so the default is now unset.
    assert!(!home.path().join("default").exists());
}

#[test]
fn default_to_uninstalled_version_is_rejected() {
    let home = tempfile::tempdir().unwrap();
    let out = run_cli(home.path(), home.path(), &["default", "0.7.0"], &[]);
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("not installed"),
        "stderr: {}",
        stderr(&out)
    );
}

#[test]
fn self_uninstall_ignores_packaging_roots_and_keeps_toolchains() {
    let home = tempfile::tempdir().unwrap();
    let release = tempfile::tempdir().unwrap();
    build_fixture_tarball(release.path(), "0.1.0", slug());
    assert!(
        install(home.path(), release.path(), "0.1.0")
            .status
            .success()
    );
    let gc_root = home.path().join("nix-gcroots/chelisup");
    let staging_root = home.path().join("nix-gcroots/chelisup.next");
    let partial_root = home.path().join("nix-gcroots/chelisup.partial");
    std::fs::create_dir_all(gc_root.parent().unwrap()).unwrap();
    std::os::unix::fs::symlink(release.path(), &gc_root).unwrap();
    std::os::unix::fs::symlink(release.path(), &staging_root).unwrap();
    std::os::unix::fs::symlink(release.path(), &partial_root).unwrap();

    let out = run_cli(home.path(), home.path(), &["self", "uninstall"], &[]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert!(!home.path().join("bin/chelis").exists());
    assert!(!home.path().join("bin/chelisup").exists());
    assert!(gc_root.is_symlink());
    assert!(staging_root.is_symlink());
    assert!(partial_root.is_symlink());
    // Packaging roots and the toolchain bytes remain.
    assert!(home.path().join("toolchains/0.1.0/bin/chelis").is_file());
}

#[test]
fn update_is_a_documented_stub() {
    let home = tempfile::tempdir().unwrap();
    let out = run_cli(home.path(), home.path(), &["update"], &[]);
    // Stub exits 0 but says it did nothing and how to upgrade.
    assert!(out.status.success());
    assert!(
        stderr(&out).contains("not implemented"),
        "stderr: {}",
        stderr(&out)
    );

    let unused = bin();
    assert!(!unused.is_empty());
}
