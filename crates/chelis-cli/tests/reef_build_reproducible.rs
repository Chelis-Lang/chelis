//! Regression coverage for chelis#970: `chelis reef build` artifacts are
//! reproducible for unchanged package inputs.

use assert_cmd::Command;
use predicates::prelude::*;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;
use tar::Archive;
use tempfile::tempdir;

const EXPLICIT_EPOCH: u64 = 1_234_567_890;
const LONG_SOURCE_FILE: &str = concat!(
    "properties/",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.ch"
);

fn stage_package(parent: &Path) -> PathBuf {
    let root = parent.join("reproducible");
    fs::create_dir_all(root.join("src/nested")).expect("create src tree");
    fs::create_dir_all(root.join("properties")).expect("create properties tree");
    fs::write(
        root.join("reef.toml"),
        format!(
            r#"[package]
name = "reproducible"
version = "0.1.0"
compiler = "={}"
module_prefix = "Repro"
additional_sources = ["properties"]
"#,
            env!("CARGO_PKG_VERSION")
        ),
    )
    .expect("write reef.toml");
    // Create files in deliberately non-lexical order. Archive order is a
    // packaging contract, not an accident of creation order or WalkDir.
    fs::write(
        root.join("src/nested/zeta.ch"),
        "module Repro.Nested.Zeta\n\ndef zeta() -> int32 = 2\n",
    )
    .expect("write zeta");
    fs::write(
        root.join("src/main.ch"),
        "module Repro.Main\n\
         export (main, identity, choose_left, tensor_identity, matrix_identity)\n\n\
         def main() -> int32 = 1\n\
         def identity[a](x: a) -> a = x\n\
         def choose_left[a, b](x: a, y: b) -> a = x\n\
         def tensor_identity[p](x: &tensor[..r, p]) -> &tensor[..r, p] = x\n\
         def matrix_identity[n, m, p](x: &tensor[n, m, p]) -> &tensor[n, m, p] = x\n",
    )
    .expect("write main");
    fs::write(
        root.join("properties/alpha.ch"),
        "module Repro.Properties.Alpha\n\ndef alpha() -> bool = true\n",
    )
    .expect("write alpha");
    let long_stem = Path::new(LONG_SOURCE_FILE)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .expect("UTF-8 long source stem");
    let long_module_component = format!("A{}", &long_stem[1..]);
    fs::write(
        root.join(LONG_SOURCE_FILE),
        format!(
            "module Repro.Properties.{long_module_component}\n\n\
             def long_name() -> int32 = 3\n"
        ),
    )
    .expect("write long-name source");
    root
}

fn run_build(root: &Path, reef_home: &Path, source_date_epoch: Option<&str>) {
    let mut command = Command::cargo_bin("chelis").expect("chelis binary");
    command
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .env_remove("SOURCE_DATE_EPOCH")
        .current_dir(root)
        .args(["reef", "build"]);
    if let Some(epoch) = source_date_epoch {
        command.env("SOURCE_DATE_EPOCH", epoch);
    }
    command.assert().success();
}

fn artifact_paths(root: &Path) -> (PathBuf, PathBuf) {
    (
        root.join("dist/reproducible-0.1.0.tar.zst"),
        root.join("dist/reproducible-0.1.0.chb"),
    )
}

fn sha256(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn assert_canonical_archive(archive_bytes: &[u8], expected_mtime: u64) {
    let decoded =
        zstd::stream::decode_all(Cursor::new(archive_bytes)).expect("decode source archive");
    let mut archive = Archive::new(Cursor::new(decoded));
    let mut paths = Vec::new();
    for entry in archive.entries().expect("read tar entries") {
        let entry = entry.expect("read tar entry");
        let header = entry.header();
        paths.push(
            entry
                .path()
                .expect("read tar path")
                .to_string_lossy()
                .into_owned(),
        );
        assert_eq!(header.mtime().expect("mtime"), expected_mtime);
        assert_eq!(header.uid().expect("uid"), 0);
        assert_eq!(header.gid().expect("gid"), 0);
        assert_eq!(header.mode().expect("mode"), 0o644);
    }
    assert_eq!(
        paths,
        vec![
            LONG_SOURCE_FILE.to_string(),
            "properties/alpha.ch".to_string(),
            "reef.lock".to_string(),
            "reef.toml".to_string(),
            "src/main.ch".to_string(),
            "src/nested/zeta.ch".to_string(),
        ],
        "members must use bytewise lexical package-relative order"
    );
}

#[test]
fn repeated_reef_builds_are_byte_identical_and_chb_pins_canonical_archive() {
    let outer = tempdir().expect("tempdir");
    let root = stage_package(outer.path());
    let reef_home = outer.path().join("reef-home");
    let epoch = EXPLICIT_EPOCH.to_string();
    let (archive_path, shell_path) = artifact_paths(&root);

    run_build(&root, &reef_home, Some(&epoch));
    let first_archive = fs::read(&archive_path).expect("read first archive");
    let first_shell = fs::read(&shell_path).expect("read first shell");

    // `reef build` rewrites reef.lock. Ensure its filesystem mtime changes
    // so this test exercises the original chelis#970 failure mode.
    thread::sleep(Duration::from_millis(1_100));
    run_build(&root, &reef_home, Some(&epoch));
    let second_archive = fs::read(&archive_path).expect("read second archive");
    let second_shell = fs::read(&shell_path).expect("read second shell");

    assert_eq!(
        first_archive, second_archive,
        "source archive bytes drifted"
    );
    assert_eq!(first_shell, second_shell, "CHB bytes drifted");
    assert_canonical_archive(&second_archive, EXPLICIT_EPOCH);

    let shell = chelis_shell::read_shell(&shell_path).expect("decode CHB");
    assert_eq!(
        shell.archive_sha256,
        sha256(&second_archive),
        "CHB must embed the canonical source archive SHA-256"
    );
}

#[test]
fn absent_source_date_epoch_uses_the_documented_zero_epoch() {
    let outer = tempdir().expect("tempdir");
    let root = stage_package(outer.path());
    let reef_home = outer.path().join("reef-home");
    let (archive_path, _) = artifact_paths(&root);

    run_build(&root, &reef_home, None);
    let archive = fs::read(archive_path).expect("read archive");
    assert_canonical_archive(&archive, 0);
}

#[test]
fn malformed_source_date_epoch_is_rejected_instead_of_silently_ignored() {
    let outer = tempdir().expect("tempdir");
    let root = stage_package(outer.path());
    let reef_home = outer.path().join("reef-home");

    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .env("SOURCE_DATE_EPOCH", "not-a-timestamp")
        .current_dir(root)
        .args(["reef", "build"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "SOURCE_DATE_EPOCH must be a non-negative integer",
        ));
}
