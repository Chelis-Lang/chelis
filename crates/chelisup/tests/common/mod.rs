//! Shared helpers for the chelisup integration tests. Unix-only (the
//! shim `exec`s and tests set a custom `argv[0]`).
#![cfg(unix)]
#![allow(dead_code)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The built chelisup binary under test.
pub fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_chelisup")
}

/// A fake `chelis` toolchain binary: a python script that self-reports
/// the version from its own path (`.../toolchains/<ver>/bin/chelis`) and
/// echoes the forwarded args. Lets a test assert both which version ran
/// and that `+ver` was stripped from the args.
pub const FAKE_CHELIS: &str = "#!/usr/bin/env python3\n\
import sys, pathlib\n\
p = pathlib.Path(sys.argv[0]).resolve()\n\
ver = p.parent.parent.name\n\
sys.stdout.write('FAKE-CHELIS ' + ver + ' ' + ' '.join(sys.argv[1:]))\n";

/// True iff `python3` runs here (the fake toolchain needs it).
pub fn python3_available() -> bool {
    Command::new("python3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn set_exec(path: &Path) {
    let mut perms = fs::metadata(path).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(path, perms).unwrap();
}

/// Write a fake installed toolchain directly into the store (no install
/// flow). Used by the resolution tests.
pub fn write_fake_toolchain(home: &Path, version: &str) {
    let bin_dir = home.join("toolchains").join(version).join("bin");
    fs::create_dir_all(&bin_dir).unwrap();
    let chelis = bin_dir.join("chelis");
    fs::write(&chelis, FAKE_CHELIS).unwrap();
    set_exec(&chelis);
}

/// Record the default version in the store.
pub fn write_default(home: &Path, version: &str) {
    fs::create_dir_all(home).unwrap();
    fs::write(home.join("default"), format!("{version}\n")).unwrap();
}

/// Build a release-tarball fixture `chelis-v<ver>-<slug>.tar.gz` under
/// `release_dir`, whose top-level `chelis-v<ver>-<slug>/bin/chelis` is
/// the fake toolchain. This is what `CHELISUP_RELEASE_BASE` serves.
pub fn build_fixture_tarball(release_dir: &Path, version: &str, slug: &str) -> PathBuf {
    use flate2::Compression;
    use flate2::write::GzEncoder;

    fs::create_dir_all(release_dir).unwrap();
    let top = format!("chelis-v{version}-{slug}");

    // Stage the unpacked layout on disk so the tar entries carry real
    // file modes (the fake `chelis` is executable).
    let stage = tempfile::tempdir().unwrap();
    let root = stage.path().join(&top);
    let bin_dir = root.join("bin");
    fs::create_dir_all(&bin_dir).unwrap();
    let chelis = bin_dir.join("chelis");
    fs::write(&chelis, FAKE_CHELIS).unwrap();
    set_exec(&chelis);

    let asset = release_dir.join(format!("{top}.tar.gz"));
    let file = fs::File::create(&asset).unwrap();
    let enc = GzEncoder::new(file, Compression::fast());
    let mut builder = tar::Builder::new(enc);
    builder.append_dir_all(&top, &root).unwrap();
    let enc = builder.into_inner().unwrap();
    enc.finish().unwrap();
    asset
}

/// Spawn the binary as the `chelis` shim (argv[0] = "chelis"), with an
/// isolated `CHELIS_HOME`, the given cwd, and `CHELIS_TOOLCHAIN`
/// explicitly removed unless a test re-adds it via `extra_env`.
pub fn run_shim(
    home: &Path,
    cwd: &Path,
    args: &[&str],
    extra_env: &[(&str, &str)],
) -> std::process::Output {
    use std::os::unix::process::CommandExt;
    let mut cmd = Command::new(bin());
    cmd.arg0("chelis")
        .args(args)
        .current_dir(cwd)
        .env("CHELIS_HOME", home)
        .env_remove("CHELIS_TOOLCHAIN");
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    cmd.output().unwrap()
}

/// Spawn the binary as `chelisup` (normal argv[0]) with an isolated
/// `CHELIS_HOME` and the given cwd.
pub fn run_cli(
    home: &Path,
    cwd: &Path,
    args: &[&str],
    extra_env: &[(&str, &str)],
) -> std::process::Output {
    let mut cmd = Command::new(bin());
    cmd.args(args)
        .current_dir(cwd)
        .env("CHELIS_HOME", home)
        .env_remove("CHELIS_TOOLCHAIN");
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    cmd.output().unwrap()
}
