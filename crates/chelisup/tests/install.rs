//! Offline `install` integration via the `CHELISUP_RELEASE_BASE` local
//! seam, plus the CLI status surface and the end-to-end shim hop.
#![cfg(unix)]

mod common;

use common::{ReleaseRuntime, bin, build_fixture_tarball, build_release_tarball, run_cli, sha256};
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

/// A release after 0.18.11, so `install` checks its runtime files.
const CHECKED: &str = "0.19.0";

/// The next patch release after this chelisup's own version.
fn newer_than_chelisup() -> String {
    let (major_minor, patch) = env!("CARGO_PKG_VERSION").rsplit_once('.').unwrap();
    format!("{major_minor}.{}", patch.parse::<u64>().unwrap() + 1)
}

/// Nothing of `version` reached the store: no toolchain, shim or default.
fn assert_store_untouched(home: &Path, version: &str) {
    assert!(!home.join("toolchains").join(version).exists());
    assert!(!home.join("bin/chelis").exists());
    assert!(!home.join("default").exists());
}

#[test]
fn install_checks_runtime_files_against_the_release_export() {
    let home = tempfile::tempdir().unwrap();
    let release = tempfile::tempdir().unwrap();
    let runtime = ReleaseRuntime::matching(CHECKED);
    build_release_tarball(release.path(), CHECKED, slug(), &runtime);

    // A caller's runtime directory must not reach the export, which refuses one.
    let out = run_cli(
        home.path(),
        home.path(),
        &["install", CHECKED],
        &[
            ("CHELISUP_RELEASE_BASE", release.path().to_str().unwrap()),
            ("CHELIS_RUNTIME_DIR", "/elsewhere"),
        ],
    );
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert!(
        stdout(&out).contains(&format!(
            "match `chelis runtime export` (libchelis_runtime.a sha256 {})",
            sha256(&runtime.shipped_archive)
        )),
        "stdout: {}",
        stdout(&out)
    );
    assert!(
        !stderr(&out).contains("warning"),
        "stderr: {}",
        stderr(&out)
    );
    let toolchain = home.path().join("toolchains").join(CHECKED);
    assert!(toolchain.join("lib/libchelis_runtime.a").is_file());
    assert!(toolchain.join("include/chelis_math.h").is_file());
}

/// A refused release: its name, how it departs from a matching release, and
/// the error text `install` must print.
type Refusal = (&'static str, fn(&mut ReleaseRuntime), &'static str);

#[test]
fn install_refuses_runtime_files_its_export_does_not_report() {
    let cases: [Refusal; 13] = [
        (
            "swapped archive",
            |runtime| runtime.shipped_archive = b"another runtime".to_vec(),
            "ships lib/libchelis_runtime.a with SHA-256",
        ),
        (
            "linked archive",
            |runtime| runtime.archive_is_symlink = true,
            "no usable lib/libchelis_runtime.a: lib/libchelis_runtime.a is not a regular file",
        ),
        (
            "linked lib directory",
            |runtime| runtime.lib_is_symlink = true,
            "no usable lib/libchelis_runtime.a: lib is not a directory",
        ),
        (
            "linked release root",
            |runtime| runtime.root_is_symlink = true,
            "expected exactly one chelis-v* directory (not a link) in the tarball, found []",
        ),
        (
            "swapped header",
            |runtime| runtime.shipped_headers[0].1 = b"/* other */\n".to_vec(),
            "ships include/chelis_runtime.h with SHA-256",
        ),
        (
            "missing header",
            |runtime| {
                runtime.shipped_headers.pop();
            },
            "no usable include/chelis_math.h",
        ),
        (
            "development build",
            |runtime| runtime.receipt["mode"] = "development".into(),
            "records mode \"development\", not \"sealed\"",
        ),
        (
            "another version",
            |runtime| runtime.receipt["chelis_version"] = "0.19.1".into(),
            "records chelis_version \"0.19.1\", not \"0.19.0\"",
        ),
        (
            "another receipt schema",
            |runtime| runtime.receipt["schema"] = "chelis-runtime-staging/2".into(),
            "records schema",
        ),
        (
            "no archive digest",
            |runtime| runtime.receipt["archive_sha256"] = "unknown".into(),
            "records no archive_sha256",
        ),
        (
            "no headers",
            |runtime| runtime.receipt["headers"] = serde_json::json!({}),
            "lists no headers",
        ),
        (
            "header outside include",
            |runtime| {
                runtime.receipt["headers"] =
                    serde_json::json!({ "../escape.h": sha256(b"/* runtime */\n") })
            },
            "which is not a file name",
        ),
        (
            "failed export",
            |runtime| runtime.export_status = 1,
            "`chelis runtime export` from the chelis 0.19.0 release failed",
        ),
    ];
    for (case, change, expected) in cases {
        let home = tempfile::tempdir().unwrap();
        let release = tempfile::tempdir().unwrap();
        let mut runtime = ReleaseRuntime::matching(CHECKED);
        change(&mut runtime);
        build_release_tarball(release.path(), CHECKED, slug(), &runtime);

        let out = install(home.path(), release.path(), CHECKED);
        assert!(
            !out.status.success(),
            "{case}: installed; stdout: {}",
            stdout(&out)
        );
        assert!(
            stderr(&out).contains(expected),
            "{case}: stderr: {}",
            stderr(&out)
        );
        assert_store_untouched(home.path(), CHECKED);
    }
}

#[test]
fn releases_before_the_export_install_unchecked_with_a_warning() {
    // A release up to 0.18.11 has no export to run, so a failing one is never
    // consulted; the first later release is checked and refused.
    let home = tempfile::tempdir().unwrap();
    let release = tempfile::tempdir().unwrap();
    for version in ["0.18.11", "0.18.12"] {
        let mut runtime = ReleaseRuntime::matching(version);
        runtime.export_status = 1;
        build_release_tarball(release.path(), version, slug(), &runtime);
    }

    let old = install(home.path(), release.path(), "0.18.11");
    assert!(old.status.success(), "stderr: {}", stderr(&old));
    assert!(
        stderr(&old).contains(
            "warning: chelis 0.18.11 predates `chelis runtime export`, so its lib/ and include/ \
             runtime files were installed unchecked"
        ),
        "stderr: {}",
        stderr(&old)
    );
    assert!(home.path().join("toolchains/0.18.11/bin/chelis").is_file());

    let first_checked = install(home.path(), release.path(), "0.18.12");
    assert!(!first_checked.status.success());
    assert!(
        stderr(&first_checked).contains("from the chelis 0.18.12 release failed"),
        "stderr: {}",
        stderr(&first_checked)
    );
    assert!(!home.path().join("toolchains/0.18.12").exists());
}

#[test]
fn a_refused_release_newer_than_chelisup_says_how_to_get_the_latest() {
    let version = newer_than_chelisup();
    let home = tempfile::tempdir().unwrap();
    let release = tempfile::tempdir().unwrap();
    let mut runtime = ReleaseRuntime::matching(&version);
    runtime.receipt["schema"] = "chelis-runtime-staging/2".into();
    build_release_tarball(release.path(), &version, slug(), &runtime);

    let out = install(home.path(), release.path(), &version);
    assert!(!out.status.success(), "stdout: {}", stdout(&out));
    let expected = format!(
        "records schema \"chelis-runtime-staging/2\", not \"chelis-runtime-staging/1\". \
         This chelisup ({}) is older than the {version} release, which may use a format it \
         does not know; to get the latest chelisup, re-run the bootstrap installer",
        env!("CARGO_PKG_VERSION")
    );
    assert!(stderr(&out).contains(&expected), "stderr: {}", stderr(&out));
    assert_store_untouched(home.path(), &version);
}

#[test]
fn install_says_why_this_machine_cannot_start_a_release() {
    // The release is newer than this chelisup, yet a newer chelisup would
    // unpack the same binary, so the error must not send the user to one.
    let version = newer_than_chelisup();
    let home = tempfile::tempdir().unwrap();
    let release = tempfile::tempdir().unwrap();
    let mut runtime = ReleaseRuntime::matching(&version);
    runtime.chelis_interpreter = "/nonexistent/ld-linux-x86-64.so.2";
    build_release_tarball(release.path(), &version, slug(), &runtime);

    let out = install(home.path(), release.path(), &version);
    assert!(!out.status.success(), "stdout: {}", stdout(&out));
    let stderr = stderr(&out);
    assert!(
        stderr.contains(&format!(
            "this machine cannot run the chelis {version} release's bin/chelis ("
        )) && stderr.contains(
            "): the file exists, so the loader or interpreter it names is missing, as with a \
             glibc build on a musl-based system such as Alpine"
        ),
        "stderr: {stderr}"
    );
    assert!(!stderr.contains("bootstrap"), "stderr: {stderr}");
    assert_store_untouched(home.path(), &version);
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
fn self_uninstall_removes_shims_but_keeps_toolchains() {
    let home = tempfile::tempdir().unwrap();
    let release = tempfile::tempdir().unwrap();
    build_fixture_tarball(release.path(), "0.1.0", slug());
    assert!(
        install(home.path(), release.path(), "0.1.0")
            .status
            .success()
    );

    let out = run_cli(home.path(), home.path(), &["self", "uninstall"], &[]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert!(!home.path().join("bin/chelis").exists());
    assert!(!home.path().join("bin/chelisup").exists());
    // The toolchain bytes remain.
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
