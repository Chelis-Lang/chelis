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
//! format. Cargo reruns this script when anything in a declared crate's
//! directory or the lockfile changes. Watching whole crate directories, not
//! only the roots they hold now, means a `build.rs` or `include/` added later
//! enters the record on the next build. An edit elsewhere in those directories,
//! such as to a test, also reruns the script, and Cargo then recompiles the
//! runtime even though the record is unchanged.
//!
//! Each watched path is declared to Cargo relative to this package's directory,
//! as `../../<path>`. A build-script execution cache, such as Kache from 0.26
//! on, replays a recorded run when the watched paths as spelled hold the same
//! bytes. It relocates only paths under the package directory to the checkout
//! being built, so an absolute spelling of a path outside it would let the cache
//! serve another checkout's record.
//!
//! A compilation outside the Chelis workspace, such as a per-crate Nix build,
//! records the required roots it cannot find, and declares them all so a cache
//! keys that record on their absence. A development compiler refuses to stage
//! that runtime; a sealed compiler never reads the record.

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

/// This crate's directory, relative to the workspace root.
const RUNTIME_CRATE: &str = "crates/chelis-runtime";
/// The workspace root, relative to this crate's directory.
const TO_WORKSPACE_ROOT: &str = "../..";

fn main() {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    let mut crates = vec![RUNTIME_CRATE, "crates/chelis-abi", "crates/chelis-vocab"];
    if env::var_os("CARGO_FEATURE_OWNERSHIP_LEDGER").is_some() {
        crates.push("crates/chelis-unord");
    }

    let required = required_roots(&crates);
    let record = match workspace_root(&manifest_dir) {
        Some(root) => {
            let missing = required
                .iter()
                .filter(|path| !root.join(path).exists())
                .cloned()
                .collect::<Vec<_>>();
            if missing.is_empty() {
                available(root, &crates)
            } else {
                unavailable(&required, &missing)
            }
        }
        None => unavailable(&required, &[RUNTIME_CRATE.to_owned()]),
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
fn required_roots(crates: &[&str]) -> Vec<String> {
    let mut required = vec!["Cargo.lock".to_owned()];
    for krate in crates {
        required.push(format!("{krate}/Cargo.toml"));
        required.push(format!("{krate}/src"));
    }
    required
}

/// Tell Cargo to rerun this script when the workspace path `relative` changes.
fn declare(relative: &str) {
    println!("cargo:rerun-if-changed={TO_WORKSPACE_ROOT}/{relative}");
}

fn unavailable(required: &[String], missing: &[String]) -> String {
    // Cargo reruns a script whose declared input is absent on every build; that
    // cost falls only on a compilation outside the workspace.
    for path in required {
        declare(path);
    }
    let mut record = String::new();
    for path in missing {
        writeln!(record, "unavailable {path}").expect("writing to a String succeeds");
    }
    record
}

fn available(root: &Path, crates: &[&str]) -> String {
    declare("Cargo.lock");
    for krate in crates {
        declare(krate);
    }

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
        writeln!(record, "dir {directory}").expect("writing to a String succeeds");
    }
    for (path, digest) in &files {
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
