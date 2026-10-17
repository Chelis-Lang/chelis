//! `chelisup link <dir>`: register an external toolchain directory under
//! the version its `bin/chelis --version` reports, and remove only the
//! link with `chelisup uninstall <ver>`
//! (`spec/design/chelis_packaging_and_install.md` §5.1.1).
#![cfg(unix)]

mod common;

use common::{
    bin as chelisup_bin, build_fixture_tarball, run_cli, run_shim, write_default,
    write_fake_toolchain,
};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

fn stdout(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}
fn stderr(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// An external toolchain directory whose `bin/chelis` prints
/// `version_output` for `--version` and echoes other arguments.
fn external_toolchain(parent: &Path, name: &str, version_output: &str) -> PathBuf {
    let dir = parent.join(name);
    let bin_dir = dir.join("bin");
    fs::create_dir_all(&bin_dir).unwrap();
    let chelis = bin_dir.join("chelis");
    let script = format!(
        "#!/bin/sh\n\
         if [ \"$1\" = --version ]; then printf '%s' '{version_output}'; exit 0; fi\n\
         printf 'EXTERNAL {name} %s' \"$*\"\n"
    );
    fs::write(&chelis, script).unwrap();
    fs::set_permissions(&chelis, fs::Permissions::from_mode(0o755)).unwrap();
    dir
}

fn link(home: &Path, dir: &Path) -> std::process::Output {
    run_cli(home, home, &["link", dir.to_str().unwrap()], &[])
}

fn entry(home: &Path, version: &str) -> PathBuf {
    home.join("toolchains").join(version)
}

/// Nothing named `version` exists in the toolchain store, not even a link.
fn assert_no_entry(home: &Path, version: &str) {
    let path = entry(home, version);
    assert!(
        fs::symlink_metadata(&path).is_err(),
        "{} must not exist",
        path.display()
    );
}

#[test]
fn link_registers_the_reported_version_and_the_shim_runs_it() {
    let home = tempfile::tempdir().unwrap();
    let external = tempfile::tempdir().unwrap();
    let dir = external_toolchain(external.path(), "pkg", "chelis 0.5.0\n");

    let out = link(home.path(), &dir);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let canonical = dir.canonicalize().unwrap();
    assert!(
        stdout(&out).contains(&format!("linked chelis 0.5.0 to {}", canonical.display())),
        "stdout: {}",
        stdout(&out)
    );

    let link_path = entry(home.path(), "0.5.0");
    assert!(fs::symlink_metadata(&link_path).unwrap().is_symlink());
    assert_eq!(fs::read_link(&link_path).unwrap(), canonical);
    // The first registered toolchain becomes the default, as with install.
    assert_eq!(
        fs::read_to_string(home.path().join("default")).unwrap(),
        "0.5.0\n"
    );

    let list = run_cli(home.path(), home.path(), &["list-installed"], &[]);
    assert!(list.status.success());
    assert!(
        stdout(&list).contains(&format!(
            "0.5.0 (default, linked to {})",
            canonical.display()
        )),
        "stdout: {}",
        stdout(&list)
    );

    let work = tempfile::tempdir().unwrap();
    let ran = run_shim(home.path(), work.path(), &["build", "main.ch"], &[]);
    assert!(ran.status.success(), "stderr: {}", stderr(&ran));
    assert_eq!(stdout(&ran), "EXTERNAL pkg build main.ch");

    let pinned = run_shim(home.path(), work.path(), &["+0.5.0", "check", "a.ch"], &[]);
    assert!(pinned.status.success(), "stderr: {}", stderr(&pinned));
    assert_eq!(stdout(&pinned), "EXTERNAL pkg check a.ch");
}

#[test]
fn link_canonicalizes_a_relative_directory() {
    let home = tempfile::tempdir().unwrap();
    let external = tempfile::tempdir().unwrap();
    let dir = external_toolchain(external.path(), "pkg", "chelis 0.5.0\n");

    let out = run_cli(home.path(), external.path(), &["link", "pkg"], &[]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let target = fs::read_link(entry(home.path(), "0.5.0")).unwrap();
    assert!(target.is_absolute(), "{}", target.display());
    assert_eq!(target, dir.canonicalize().unwrap());
}

#[test]
fn link_says_how_to_create_a_missing_shim() {
    let home = tempfile::tempdir().unwrap();
    let external = tempfile::tempdir().unwrap();
    let dir = external_toolchain(external.path(), "pkg", "chelis 0.5.0\n");

    let out = link(home.path(), &dir);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    // link never copies the shim. It names the command that does.
    assert!(!home.path().join("bin").join("chelis").exists());
    assert!(
        stdout(&out).contains("chelisup install 0.5.0"),
        "stdout: {}",
        stdout(&out)
    );
}

#[test]
fn link_keeps_an_existing_default() {
    let home = tempfile::tempdir().unwrap();
    let external = tempfile::tempdir().unwrap();
    write_fake_toolchain(home.path(), "0.1.0");
    write_default(home.path(), "0.1.0");
    let dir = external_toolchain(external.path(), "pkg", "chelis 0.5.0\n");

    let out = link(home.path(), &dir);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(
        fs::read_to_string(home.path().join("default")).unwrap(),
        "0.1.0\n"
    );
}

#[test]
fn link_rejects_a_directory_without_bin_chelis() {
    let home = tempfile::tempdir().unwrap();
    let external = tempfile::tempdir().unwrap();
    let dir = external.path().join("empty");
    fs::create_dir_all(dir.join("bin")).unwrap();

    let out = link(home.path(), &dir);
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("bin/chelis"),
        "stderr: {}",
        stderr(&out)
    );
    assert!(!home.path().join("toolchains").exists());
    assert!(!home.path().join("default").exists());
}

#[test]
fn link_rejects_a_missing_directory_and_a_file() {
    let home = tempfile::tempdir().unwrap();
    let external = tempfile::tempdir().unwrap();
    let missing = external.path().join("missing");
    let out = link(home.path(), &missing);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("missing"), "stderr: {}", stderr(&out));

    let file = external.path().join("file");
    fs::write(&file, b"x").unwrap();
    let out = link(home.path(), &file);
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("not a directory"),
        "stderr: {}",
        stderr(&out)
    );
    assert!(!home.path().join("toolchains").exists());
}

#[test]
fn link_rejects_every_malformed_version_report() {
    let home = tempfile::tempdir().unwrap();
    let external = tempfile::tempdir().unwrap();
    for (index, report) in [
        "chelis 0.5\n",
        "chelis v0.5.0\n",
        "chelis 0.5.0-rc1\n",
        "chelis 0.5.0 extra\n",
        "chelis ../0.5.0\n",
        "rustc 0.5.0\n",
        "0.5.0\n",
        "chelis 0.5.0\nchelis 0.6.0\n",
        "",
    ]
    .iter()
    .enumerate()
    {
        let dir = external_toolchain(external.path(), &format!("bad{index}"), report);
        let out = link(home.path(), &dir);
        assert!(!out.status.success(), "accepted {report:?}");
        assert!(
            stderr(&out).contains("chelis X.Y.Z"),
            "{report:?} stderr: {}",
            stderr(&out)
        );
    }
    assert!(!home.path().join("toolchains").exists());
    assert!(!home.path().join("default").exists());
}

#[test]
fn link_rejects_a_failing_version_command() {
    let home = tempfile::tempdir().unwrap();
    let external = tempfile::tempdir().unwrap();
    let dir = external.path().join("broken");
    fs::create_dir_all(dir.join("bin")).unwrap();
    let chelis = dir.join("bin").join("chelis");
    fs::write(
        &chelis,
        "#!/bin/sh\necho 'chelis 0.5.0'\necho boom >&2\nexit 3\n",
    )
    .unwrap();
    fs::set_permissions(&chelis, fs::Permissions::from_mode(0o755)).unwrap();

    let out = link(home.path(), &dir);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("boom"), "stderr: {}", stderr(&out));
    assert!(!home.path().join("toolchains").exists());
}

#[test]
fn link_refuses_a_version_installed_as_a_directory() {
    let home = tempfile::tempdir().unwrap();
    let external = tempfile::tempdir().unwrap();
    write_fake_toolchain(home.path(), "0.5.0");
    let dir = external_toolchain(external.path(), "pkg", "chelis 0.5.0\n");

    let out = link(home.path(), &dir);
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("chelisup uninstall 0.5.0"),
        "stderr: {}",
        stderr(&out)
    );
    let installed = entry(home.path(), "0.5.0");
    assert!(!fs::symlink_metadata(&installed).unwrap().is_symlink());
    assert!(installed.join("bin").join("chelis").is_file());
}

#[test]
fn link_replaces_an_existing_link_of_the_same_version() {
    let home = tempfile::tempdir().unwrap();
    let external = tempfile::tempdir().unwrap();
    let first = external_toolchain(external.path(), "first", "chelis 0.5.0\n");
    let second = external_toolchain(external.path(), "second", "chelis 0.5.0\n");

    assert!(link(home.path(), &first).status.success());
    let out = link(home.path(), &second);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert!(
        stdout(&out).contains(&format!(
            "replaced the link to {}",
            first.canonicalize().unwrap().display()
        )),
        "stdout: {}",
        stdout(&out)
    );
    assert_eq!(
        fs::read_link(entry(home.path(), "0.5.0")).unwrap(),
        second.canonicalize().unwrap()
    );
    let work = tempfile::tempdir().unwrap();
    let ran = run_shim(home.path(), work.path(), &["run"], &[]);
    assert_eq!(stdout(&ran), "EXTERNAL second run");
    // The first target stays in place.
    assert!(first.join("bin").join("chelis").is_file());
}

#[test]
fn uninstall_removes_the_link_and_keeps_the_target() {
    let home = tempfile::tempdir().unwrap();
    let external = tempfile::tempdir().unwrap();
    let dir = external_toolchain(external.path(), "pkg", "chelis 0.5.0\n");
    assert!(link(home.path(), &dir).status.success());

    let out = run_cli(home.path(), home.path(), &["uninstall", "0.5.0"], &[]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert!(
        stdout(&out).contains("removed the link"),
        "stdout: {}",
        stdout(&out)
    );
    assert_no_entry(home.path(), "0.5.0");
    assert!(!home.path().join("default").exists());
    assert!(dir.join("bin").join("chelis").is_file());
}

#[test]
fn uninstall_rejects_a_malformed_version_and_keeps_the_store() {
    let home = tempfile::tempdir().unwrap();
    write_fake_toolchain(home.path(), "0.1.0");
    for bad in ["../toolchains/0.1.0", "0.1", ".", ""] {
        let out = run_cli(home.path(), home.path(), &["uninstall", bad], &[]);
        assert!(!out.status.success(), "accepted {bad:?}");
        assert!(
            stderr(&out).contains("malformed version"),
            "{bad:?} stderr: {}",
            stderr(&out)
        );
    }
    assert!(entry(home.path(), "0.1.0").join("bin/chelis").is_file());
}

#[test]
fn a_dangling_link_is_not_installed_and_names_both_remedies() {
    let home = tempfile::tempdir().unwrap();
    let external = tempfile::tempdir().unwrap();
    let dir = external_toolchain(external.path(), "pkg", "chelis 0.5.0\n");
    assert!(link(home.path(), &dir).status.success());
    let canonical = dir.canonicalize().unwrap();
    fs::remove_dir_all(&dir).unwrap();

    let list = run_cli(home.path(), home.path(), &["list-installed"], &[]);
    assert_eq!(stdout(&list).trim(), "(no toolchains installed)");

    let work = tempfile::tempdir().unwrap();
    let ran = run_shim(home.path(), work.path(), &["build"], &[]);
    assert!(!ran.status.success());
    let message = stderr(&ran);
    assert!(
        message.contains(&canonical.display().to_string()),
        "{message}"
    );
    assert!(message.contains("chelisup link <dir>"), "{message}");
    assert!(message.contains("chelisup uninstall 0.5.0"), "{message}");

    let which = run_cli(home.path(), home.path(), &["which"], &[]);
    assert!(!which.status.success());

    let out = run_cli(home.path(), home.path(), &["uninstall", "0.5.0"], &[]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_no_entry(home.path(), "0.5.0");
}

#[test]
fn link_replaces_a_dangling_link() {
    let home = tempfile::tempdir().unwrap();
    let external = tempfile::tempdir().unwrap();
    let gone = external_toolchain(external.path(), "gone", "chelis 0.5.0\n");
    assert!(link(home.path(), &gone).status.success());
    fs::remove_dir_all(&gone).unwrap();

    let fresh = external_toolchain(external.path(), "fresh", "chelis 0.5.0\n");
    let out = link(home.path(), &fresh);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(
        fs::read_link(entry(home.path(), "0.5.0")).unwrap(),
        fresh.canonicalize().unwrap()
    );
}

#[test]
fn install_replaces_a_dangling_link_with_a_downloaded_toolchain() {
    let home = tempfile::tempdir().unwrap();
    let external = tempfile::tempdir().unwrap();
    let gone = external_toolchain(external.path(), "gone", "chelis 0.1.0\n");
    assert!(link(home.path(), &gone).status.success());
    fs::remove_dir_all(&gone).unwrap();

    let release = tempfile::tempdir().unwrap();
    let slug = chelisup::install::detect_slug().unwrap();
    let build =
        chelisup::install::release_build("0.1.0", slug, chelisup::install::host_is_musl()).unwrap();
    build_fixture_tarball(release.path(), "0.1.0", &build);
    let out = run_cli(
        home.path(),
        home.path(),
        &["install", "0.1.0"],
        &[("CHELISUP_RELEASE_BASE", release.path().to_str().unwrap())],
    );
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let installed = entry(home.path(), "0.1.0");
    assert!(!fs::symlink_metadata(&installed).unwrap().is_symlink());
    assert!(installed.join("bin").join("chelis").is_file());
}

#[test]
fn install_of_a_linked_version_keeps_the_link() {
    let home = tempfile::tempdir().unwrap();
    let external = tempfile::tempdir().unwrap();
    let dir = external_toolchain(external.path(), "pkg", "chelis 0.5.0\n");
    assert!(link(home.path(), &dir).status.success());

    let out = run_cli(home.path(), home.path(), &["install", "0.5.0"], &[]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(
        fs::read_link(entry(home.path(), "0.5.0")).unwrap(),
        dir.canonicalize().unwrap()
    );
    assert!(home.path().join("bin").join("chelis").is_file());
}

#[test]
fn link_refuses_a_directory_inside_the_chelis_home() {
    let home = tempfile::tempdir().unwrap();
    write_fake_toolchain(home.path(), "0.1.0");
    // The home itself holds bin/chelis once the shim exists.
    fs::create_dir_all(home.path().join("bin")).unwrap();
    fs::copy(
        entry(home.path(), "0.1.0").join("bin/chelis"),
        home.path().join("bin/chelis"),
    )
    .unwrap();
    for inside in [home.path().to_path_buf(), entry(home.path(), "0.1.0")] {
        let out = link(home.path(), &inside);
        assert!(!out.status.success(), "linked {}", inside.display());
        assert!(
            stderr(&out).contains("inside the chelis home"),
            "stderr: {}",
            stderr(&out)
        );
    }
    assert!(
        !fs::symlink_metadata(entry(home.path(), "0.1.0"))
            .unwrap()
            .is_symlink()
    );
}

#[test]
fn link_refuses_a_directory_whose_bin_chelis_is_the_shim() {
    let home = tempfile::tempdir().unwrap();
    let external = tempfile::tempdir().unwrap();
    let dir = external_toolchain(external.path(), "pkg", "chelis 0.5.0\n");
    assert!(link(home.path(), &dir).status.success());
    fs::create_dir_all(home.path().join("bin")).unwrap();
    fs::copy(chelisup_bin(), home.path().join("bin/chelis")).unwrap();

    let alias = external.path().join("alias");
    fs::create_dir_all(alias.join("bin")).unwrap();
    std::os::unix::fs::symlink(home.path().join("bin/chelis"), alias.join("bin/chelis")).unwrap();
    let out = link(home.path(), &alias);
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("is the chelis shim"),
        "stderr: {}",
        stderr(&out)
    );
    assert_eq!(
        fs::read_link(entry(home.path(), "0.5.0")).unwrap(),
        dir.canonicalize().unwrap()
    );
}

#[test]
fn uninstall_removes_a_half_removed_directory_that_link_refuses() {
    let home = tempfile::tempdir().unwrap();
    let external = tempfile::tempdir().unwrap();
    fs::create_dir_all(entry(home.path(), "0.5.0")).unwrap();
    let dir = external_toolchain(external.path(), "pkg", "chelis 0.5.0\n");

    let refused = link(home.path(), &dir);
    assert!(!refused.status.success());
    assert!(stderr(&refused).contains("chelisup uninstall 0.5.0"));

    let out = run_cli(home.path(), home.path(), &["uninstall", "0.5.0"], &[]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_no_entry(home.path(), "0.5.0");
    assert!(link(home.path(), &dir).status.success());
}

#[test]
fn link_leaves_no_scratch_entry_in_the_toolchain_store() {
    let home = tempfile::tempdir().unwrap();
    let external = tempfile::tempdir().unwrap();
    let dir = external_toolchain(external.path(), "pkg", "chelis 0.5.0\n");
    assert!(link(home.path(), &dir).status.success());
    let names: Vec<String> = fs::read_dir(home.path().join("toolchains"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, vec!["0.5.0".to_string()]);
}
