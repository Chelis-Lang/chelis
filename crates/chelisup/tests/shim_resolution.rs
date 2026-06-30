//! End-to-end shim resolution: spawn the binary as `chelis` and assert
//! which installed toolchain it `exec`s for each precedence level, plus
//! the two loud-error paths.
#![cfg(unix)]

mod common;

use common::{run_shim, write_default, write_fake_toolchain};
use std::fs;

/// Standard fixture: a store with 0.1.0 and 0.2.0 installed, default
/// 0.2.0, and an empty working directory.
fn fixture() -> (tempfile::TempDir, tempfile::TempDir) {
    let home = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    write_fake_toolchain(home.path(), "0.1.0");
    write_fake_toolchain(home.path(), "0.2.0");
    write_default(home.path(), "0.2.0");
    (home, work)
}

fn stdout(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}
fn stderr(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn level5_default_resolves_outside_any_package() {
    if !common::python3_available() {
        eprintln!("skipping: python3 not available for the fake toolchain");
        return;
    }
    let (home, work) = fixture();
    let out = run_shim(home.path(), work.path(), &["build", "x"], &[]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(stdout(&out), "FAKE-CHELIS 0.2.0 build x");
}

#[test]
fn level1_plus_arg_wins_and_is_stripped() {
    if !common::python3_available() {
        return;
    }
    let (home, work) = fixture();
    let out = run_shim(home.path(), work.path(), &["+0.1.0", "build", "x"], &[]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    // The +0.1.0 selected 0.1.0 AND was removed from the forwarded args.
    assert_eq!(stdout(&out), "FAKE-CHELIS 0.1.0 build x");
}

#[test]
fn level2_env_overrides_default() {
    if !common::python3_available() {
        return;
    }
    let (home, work) = fixture();
    let out = run_shim(
        home.path(),
        work.path(),
        &["run"],
        &[("CHELIS_TOOLCHAIN", "0.1.0")],
    );
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(stdout(&out), "FAKE-CHELIS 0.1.0 run");
}

#[test]
fn level3_chelis_toolchain_file_overrides_default() {
    if !common::python3_available() {
        return;
    }
    let (home, work) = fixture();
    fs::write(work.path().join("chelis-toolchain"), "0.1.0\n").unwrap();
    let out = run_shim(home.path(), work.path(), &["check"], &[]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(stdout(&out), "FAKE-CHELIS 0.1.0 check");
}

#[test]
fn level4_reef_pin_overrides_default() {
    if !common::python3_available() {
        return;
    }
    let (home, work) = fixture();
    fs::write(
        work.path().join("reef.toml"),
        "[package]\nname = \"x\"\ncompiler = \"=0.1.0\"\n",
    )
    .unwrap();
    let out = run_shim(home.path(), work.path(), &["build"], &[]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(stdout(&out), "FAKE-CHELIS 0.1.0 build");
}

#[test]
fn level3_beats_level4_toolchain_file_over_reef_pin() {
    if !common::python3_available() {
        return;
    }
    let (home, work) = fixture();
    fs::write(work.path().join("chelis-toolchain"), "0.1.0\n").unwrap();
    fs::write(
        work.path().join("reef.toml"),
        "[package]\ncompiler = \"=0.2.0\"\n",
    )
    .unwrap();
    let out = run_shim(home.path(), work.path(), &["build"], &[]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(stdout(&out), "FAKE-CHELIS 0.1.0 build");
}

#[test]
fn reef_pin_walks_up_from_subdirectory() {
    if !common::python3_available() {
        return;
    }
    let (home, work) = fixture();
    fs::write(
        work.path().join("reef.toml"),
        "[package]\ncompiler = \"=0.1.0\"\n",
    )
    .unwrap();
    let nested = work.path().join("a").join("b");
    fs::create_dir_all(&nested).unwrap();
    let out = run_shim(home.path(), &nested, &["build"], &[]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(stdout(&out), "FAKE-CHELIS 0.1.0 build");
}

#[test]
fn resolved_but_not_installed_is_a_loud_error() {
    // No python3 needed: this never reaches an exec.
    let (home, work) = fixture();
    fs::write(
        work.path().join("reef.toml"),
        "[package]\ncompiler = \"=0.99.0\"\n",
    )
    .unwrap();
    let out = run_shim(home.path(), work.path(), &["build"], &[]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("chelisup install 0.99.0"), "stderr: {err}");
    assert!(err.contains("not installed"), "stderr: {err}");
    // It must NOT silently fall through to the installed default.
    assert!(
        stdout(&out).is_empty(),
        "stdout should be empty: {}",
        stdout(&out)
    );
}

#[test]
fn nothing_resolved_is_a_loud_error() {
    let home = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    // No toolchains, no default, no files, no env.
    let out = run_shim(home.path(), work.path(), &["build"], &[]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(
        err.contains("no chelis toolchain resolved"),
        "stderr: {err}"
    );
    assert!(err.contains("chelisup default"), "stderr: {err}");
}

#[test]
fn plus_arg_not_installed_does_not_fall_back() {
    let (home, work) = fixture();
    let out = run_shim(home.path(), work.path(), &["+0.99.0", "build"], &[]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("chelisup install 0.99.0"), "stderr: {err}");
}
