//! chelis#3812: `chelis build`, `check`, `cost`, `eval --file`, and `prove`
//! accept a package root with no `reef.lock` that the process cannot write.
//! A source in a read-only store is such a root. The commands resolve the
//! graph in memory and write nothing into that root. A writable root keeps
//! the lock repair, and a resolution error stays an error.
#![cfg(unix)]

use assert_cmd::Command;
use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

const PROGRAM: &str = "module Plock.Main
import Std.Text (join)
export (main)
def main() -> i64 = cast(1, i64)
";

fn chelis(package: &Path, reef_home: &Path) -> Command {
    let mut command = Command::cargo_bin("chelis").expect("binary");
    command
        .current_dir(package)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home);
    command
}

/// A package root that holds only `reef.toml` and `src/main.ch`, with the
/// manifest that `chelis reef init` writes. `dependencies` extends the
/// `[dependencies]` table.
fn lockless_package(dependencies: &str) -> (TempDir, PathBuf, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let scaffold = dir.path().join("scaffold");
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "reef",
            "init",
            "plock",
            "--module-prefix",
            "Plock",
            "--output",
        ])
        .arg(&scaffold)
        .assert()
        .success();
    let manifest = fs::read_to_string(scaffold.join("reef.toml")).expect("read the manifest");
    assert!(
        manifest.contains("\n[dependencies]\n"),
        "the scaffold manifest has a dependency table: {manifest}"
    );
    let manifest = manifest.replace(
        "\n[dependencies]\n",
        &format!("\n[dependencies]\n{dependencies}"),
    );
    let package = dir.path().join("plock");
    fs::create_dir_all(package.join("src")).expect("create the source directory");
    fs::write(package.join("reef.toml"), manifest).expect("write the manifest");
    fs::write(package.join("src/main.ch"), PROGRAM).expect("write the program");
    let reef_home = dir.path().join("reef-home");
    (dir, package, reef_home)
}

/// Every path under `root`, relative to it, with its file bytes.
fn package_tree(root: &Path) -> BTreeMap<PathBuf, Option<Vec<u8>>> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<PathBuf, Option<Vec<u8>>>) {
        for entry in fs::read_dir(dir).expect("read the package directory") {
            let path = entry.expect("read a directory entry").path();
            let relative = path
                .strip_prefix(root)
                .expect("relative path")
                .to_path_buf();
            if path.is_dir() {
                out.insert(relative, None);
                walk(root, &path, out);
            } else {
                out.insert(
                    relative,
                    Some(fs::read(&path).expect("read a package file")),
                );
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

/// Removes write permission from the package root and from its source
/// directory. The drop gives the permission back, so that the temporary
/// directory can go.
struct ReadOnlyPackage<'a>(&'a Path);

impl<'a> ReadOnlyPackage<'a> {
    fn new(package: &'a Path) -> Self {
        for path in [package.join("src"), package.to_path_buf()] {
            fs::set_permissions(&path, fs::Permissions::from_mode(0o555))
                .expect("remove write permission");
        }
        assert!(
            fs::write(package.join(".write-probe"), b"").is_err(),
            "the test process must not bypass directory permissions"
        );
        Self(package)
    }
}

impl Drop for ReadOnlyPackage<'_> {
    fn drop(&mut self) {
        for path in [self.0.to_path_buf(), self.0.join("src")] {
            let _ = fs::set_permissions(&path, fs::Permissions::from_mode(0o755));
        }
    }
}

#[test]
fn read_only_root_without_lock_runs_every_package_command() {
    let (dir, package, reef_home) = lockless_package("");
    let out = dir.path().join("out");
    let before = package_tree(&package);
    let _read_only = ReadOnlyPackage::new(&package);

    chelis(&package, &reef_home)
        .args(["check", "src/main.ch"])
        .assert()
        .success()
        .stdout(predicates::str::contains("\"errors\": []"));
    chelis(&package, &reef_home)
        .args(["build", "--emit-c", "--target", "c", "src/main.ch", "-o"])
        .arg(&out)
        .assert()
        .success();
    assert!(
        out.join("main.c").is_file(),
        "build emits C outside the root"
    );
    chelis(&package, &reef_home)
        .args(["cost", "src/main.ch"])
        .assert()
        .success()
        .stdout(predicates::str::contains("total_copy_count="));
    chelis(&package, &reef_home)
        .args(["eval", "--file", "src/main.ch"])
        .assert()
        .success()
        .stdout("main = 1\n");
    chelis(&package, &reef_home)
        .args(["prove", "src/main.ch"])
        .assert()
        .success()
        .stdout(predicates::str::contains("0 errors"));

    assert_eq!(
        package_tree(&package),
        before,
        "a package command wrote into a read-only package root"
    );
}

/// A writable root with no lock keeps the lock repair of `chelis check`.
#[test]
fn writable_root_without_lock_repairs_lock() {
    let (_dir, package, reef_home) = lockless_package("");
    chelis(&package, &reef_home)
        .args(["check", "src/main.ch"])
        .assert()
        .success();
    let lock = fs::read_to_string(package.join("reef.lock")).expect("check writes reef.lock");
    assert!(lock.contains("name = \"chelis-std\""), "{lock}");
    let ignore = fs::read_to_string(package.join(".gitignore")).expect("check writes .gitignore");
    assert!(
        ignore.lines().any(|line| line == ".reef-write.lock"),
        "{ignore}"
    );
}

/// Negative parity: a read-only root does not hide a resolution error. The
/// missing path dependency fails each lock-repairing command with its own
/// message, not a write error, and the root stays unchanged.
#[test]
fn read_only_root_without_lock_keeps_resolution_errors() {
    let (dir, package, reef_home) = lockless_package("absent = { path = \"./absent\" }\n");
    let out = dir.path().join("out");
    let before = package_tree(&package);
    let _read_only = ReadOnlyPackage::new(&package);

    let mut build = chelis(&package, &reef_home);
    build
        .args(["build", "--emit-c", "--target", "c", "src/main.ch", "-o"])
        .arg(&out);
    let mut check = chelis(&package, &reef_home);
    check.args(["check", "src/main.ch"]);
    let mut cost = chelis(&package, &reef_home);
    cost.args(["cost", "src/main.ch"]);
    let mut prove = chelis(&package, &reef_home);
    prove.args(["prove", "src/main.ch"]);
    for (name, command) in [
        ("build", &mut build),
        ("check", &mut check),
        ("cost", &mut cost),
        ("prove", &mut prove),
    ] {
        let output = command.output().expect("run the command");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!output.status.success(), "{name} must fail: {text}");
        assert!(
            text.contains("resolve path dependency `./absent`"),
            "{name}: {text}"
        );
        assert!(!text.contains("Permission denied"), "{name}: {text}");
    }

    assert_eq!(package_tree(&package), before);
}
