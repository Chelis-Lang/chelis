//! Records the declared source inputs of this runtime compilation
//! (`spec/08-backends.md` §2.1).
//!
//! A development build of a Chelis compiler checks this record against its
//! checkout before it stages the runtime it carries (`chelis-runtime-bundle`),
//! so it never stages a runtime older than its sources. The declared inputs
//! are, for this crate and for each workspace crate it depends on
//! (`chelis-abi`, `chelis-vocab`, and `chelis-unord` with `ownership-ledger`):
//! `Cargo.toml`, `build.rs` and `include/` when present, and every file under
//! `src/`; plus the workspace `Cargo.lock`. Names beginning with `.` are not
//! inputs. Paths are relative to the workspace root, so the record carries no
//! build path and is the same in every checkout of the same sources.
//!
//! The record is `$OUT_DIR/build_record.txt`, exposed as
//! `chelis_runtime::build_record::SOURCES`, whose documentation gives the
//! format. Cargo reruns this script when a declared root changes. A `build.rs`
//! or `include/` added later to one of those crates enters the record the next
//! time the script runs.
//!
//! A compilation outside the Chelis workspace, such as a per-crate Nix build,
//! records the required roots it cannot find. A development compiler refuses
//! to stage that runtime; a sealed compiler never reads the record.

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

/// This crate's directory, relative to the workspace root.
const RUNTIME_CRATE: &str = "crates/chelis-runtime";

fn main() {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    let mut crates = vec![RUNTIME_CRATE, "crates/chelis-abi", "crates/chelis-vocab"];
    if env::var_os("CARGO_FEATURE_OWNERSHIP_LEDGER").is_some() {
        crates.push("crates/chelis-unord");
    }

    let record = match workspace_root(&manifest_dir) {
        Some(root) => {
            let missing = missing_required_roots(root, &crates);
            if missing.is_empty() {
                available(root, &crates)
            } else {
                unavailable(&missing)
            }
        }
        None => unavailable(&[RUNTIME_CRATE.to_owned()]),
    };
    let destination = out_dir.join("build_record.txt");
    fs::write(&destination, record)
        .unwrap_or_else(|error| panic!("cannot write {}: {error}", destination.display()));
}

/// The workspace root, when this crate sits at [`RUNTIME_CRATE`] below it.
fn workspace_root(manifest_dir: &Path) -> Option<&Path> {
    let root = manifest_dir.ancestors().nth(2)?;
    let expected = fs::canonicalize(root.join(RUNTIME_CRATE)).ok()?;
    (fs::canonicalize(manifest_dir).ok()? == expected).then_some(root)
}

/// The roots every declared crate has: its manifest and `src/`, and the
/// workspace lockfile.
fn missing_required_roots(root: &Path, crates: &[&str]) -> Vec<String> {
    let mut required = vec!["Cargo.lock".to_owned()];
    for krate in crates {
        required.push(format!("{krate}/Cargo.toml"));
        required.push(format!("{krate}/src"));
    }
    required
        .into_iter()
        .filter(|path| !root.join(path).exists())
        .collect()
}

fn unavailable(missing: &[String]) -> String {
    // Watching an absent path would rerun this script on every build.
    println!("cargo:rerun-if-changed=build.rs");
    let mut record = String::new();
    for path in missing {
        writeln!(record, "unavailable {path}").expect("writing to a String succeeds");
    }
    record
}

fn available(root: &Path, crates: &[&str]) -> String {
    let mut directories = Vec::new();
    let mut files = BTreeMap::new();
    record_file(root, "Cargo.lock", &mut files);
    for krate in crates {
        record_file(root, &format!("{krate}/Cargo.toml"), &mut files);
        let build_script = format!("{krate}/build.rs");
        if root.join(&build_script).is_file() {
            record_file(root, &build_script, &mut files);
        }
        for name in ["src", "include"] {
            let directory = format!("{krate}/{name}");
            if root.join(&directory).is_dir() {
                record_directory(root, &directory, &mut files);
                directories.push(directory);
            }
        }
    }
    directories.sort();

    let mut record = String::new();
    for directory in &directories {
        println!("cargo:rerun-if-changed={}", root.join(directory).display());
        writeln!(record, "dir {directory}").expect("writing to a String succeeds");
    }
    for (path, digest) in &files {
        if !directories
            .iter()
            .any(|directory| path.starts_with(&format!("{directory}/")))
        {
            println!("cargo:rerun-if-changed={}", root.join(path).display());
        }
        writeln!(record, "sha256 {digest} {path}").expect("writing to a String succeeds");
    }
    record
}

/// Record every file below `directory`, skipping names that begin with `.`.
fn record_directory(root: &Path, directory: &str, files: &mut BTreeMap<String, String>) {
    let path = root.join(directory);
    let entries = fs::read_dir(&path)
        .unwrap_or_else(|error| panic!("cannot read runtime input {}: {error}", path.display()));
    for entry in entries {
        let entry = entry.unwrap_or_else(|error| {
            panic!("cannot read runtime input {}: {error}", path.display())
        });
        let name = entry.file_name();
        let name = name.to_str().unwrap_or_else(|| {
            panic!(
                "runtime input {} has a name that is not UTF-8",
                entry.path().display()
            )
        });
        if name.starts_with('.') {
            continue;
        }
        let relative = format!("{directory}/{name}");
        let metadata = fs::metadata(entry.path()).unwrap_or_else(|error| {
            panic!(
                "cannot read runtime input {}: {error}",
                entry.path().display()
            )
        });
        if metadata.is_dir() {
            record_directory(root, &relative, files);
        } else {
            record_file(root, &relative, files);
        }
    }
}

fn record_file(root: &Path, relative: &str, files: &mut BTreeMap<String, String>) {
    let path = root.join(relative);
    let bytes = fs::read(&path)
        .unwrap_or_else(|error| panic!("cannot read runtime input {}: {error}", path.display()));
    let mut digest = String::new();
    for byte in Sha256::digest(&bytes) {
        write!(digest, "{byte:02x}").expect("writing to a String succeeds");
    }
    files.insert(relative.to_owned(), digest);
}
