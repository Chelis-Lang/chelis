//! A development consumer stages its runtime only while the runtime's sources in
//! the checkout it was built from are unchanged; a sealed consumer reads no
//! checkout (`spec/08-backends.md` §2.1).
//!
//! The test copies the runtime, its workspace dependencies and the runtime
//! bundle into a temporary workspace, builds a consumer that stages the carried
//! runtime, and then edits, rebuilds and finally deletes that checkout.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// The crates the consumer builds from source.
const CRATES: &[&str] = &[
    "chelis-abi",
    "chelis-runtime",
    "chelis-runtime-bundle",
    "chelis-runtime-bundle-macro",
    "chelis-unord",
    "chelis-vocab",
];

const CONSUMER_MANIFEST: &str = r#"[package]
name = "consumer"
version = "0.0.0"
edition = "2021"
publish = false

[features]
sealed = ["chelis-runtime-bundle/sealed"]

[dependencies]
chelis-runtime-bundle.workspace = true
"#;

const CONSUMER_MAIN: &str = r#"fn main() {
    let dir = std::env::args_os().nth(1).expect("a staging directory");
    match chelis_runtime_bundle::stage(std::path::Path::new(&dir)) {
        Ok(staged) => println!("{}", staged.archive_sha256),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
"#;

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the bundle lies two levels below the workspace root")
        .to_path_buf()
}

fn copy_tree(from: &Path, to: &Path) {
    fs::create_dir_all(to).expect("create directory");
    for entry in fs::read_dir(from).expect("read directory") {
        let entry = entry.expect("directory entry");
        let destination = to.join(entry.file_name());
        if entry.file_type().expect("file type").is_dir() {
            copy_tree(&entry.path(), &destination);
        } else {
            fs::copy(entry.path(), destination).expect("copy file");
        }
    }
}

/// `manifest` without its `[dev-dependencies]` table, which may name workspace
/// crates the copy leaves out.
fn without_dev_dependencies(manifest: &str) -> String {
    let mut kept = String::new();
    let mut dropping = false;
    for line in manifest.lines() {
        if line.starts_with('[') {
            dropping = line.trim() == "[dev-dependencies]";
        }
        if !dropping {
            kept.push_str(line);
            kept.push('\n');
        }
    }
    kept
}

/// A workspace holding the build inputs of [`CRATES`], the lockfile, and a
/// consumer crate. It is its own checkout: the runtime it builds records these
/// copies.
fn copy_checkout(checkout: &Path) {
    let source = workspace();
    for krate in CRATES {
        let from = source.join("crates").join(krate);
        let to = checkout.join("crates").join(krate);
        fs::create_dir_all(&to).expect("crate directory");
        let manifest = fs::read_to_string(from.join("Cargo.toml")).expect("crate manifest");
        fs::write(to.join("Cargo.toml"), without_dev_dependencies(&manifest))
            .expect("crate manifest copy");
        if from.join("build.rs").is_file() {
            fs::copy(from.join("build.rs"), to.join("build.rs")).expect("build script");
        }
        for directory in ["src", "include"] {
            if from.join(directory).is_dir() {
                copy_tree(&from.join(directory), &to.join(directory));
            }
        }
    }
    fs::copy(source.join("Cargo.lock"), checkout.join("Cargo.lock")).expect("lockfile");

    let manifest = fs::read_to_string(source.join("Cargo.toml")).expect("workspace manifest");
    let start = manifest.find("members = [").expect("workspace members");
    let end = start
        + manifest[start..]
            .find("\n]")
            .expect("end of workspace members")
        + 2;
    let members = CRATES
        .iter()
        .map(|krate| format!("    \"crates/{krate}\",\n"))
        .collect::<String>();
    fs::write(
        checkout.join("Cargo.toml"),
        format!(
            "{}members = [\n{members}    \"consumer\",\n]{}",
            &manifest[..start],
            &manifest[end..]
        ),
    )
    .expect("workspace manifest copy");

    fs::create_dir_all(checkout.join("consumer/src")).expect("consumer directory");
    fs::write(checkout.join("consumer/Cargo.toml"), CONSUMER_MANIFEST).expect("consumer manifest");
    fs::write(checkout.join("consumer/src/main.rs"), CONSUMER_MAIN).expect("consumer source");
}

/// Build the consumer from `checkout` and copy it to `binary`.
fn build_consumer(checkout: &Path, target: &Path, features: &[&str], binary: &Path) {
    // Run from this workspace so its toolchain file selects the compiler.
    let output = Command::new(env!("CARGO"))
        .current_dir(workspace())
        .env("CARGO_TARGET_DIR", target)
        .arg("build")
        .arg("--offline")
        .arg("--manifest-path")
        .arg(checkout.join("Cargo.toml"))
        .args(["-p", "consumer"])
        .args(features)
        .output()
        .expect("run cargo");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    fs::copy(target.join("debug/consumer"), binary).expect("copy the consumer");
}

fn stage(binary: &Path, dir: &Path) -> Output {
    fs::create_dir_all(dir).expect("staging directory");
    Command::new(binary)
        .arg(dir)
        .env_remove("CHELIS_RUNTIME_DIR")
        .output()
        .expect("run the consumer")
}

fn receipt_mode(dir: &Path) -> String {
    let receipt: serde_json::Value = serde_json::from_slice(
        &fs::read(dir.join(chelis_runtime_bundle::RECEIPT_FILE_NAME)).expect("receipt"),
    )
    .expect("receipt JSON");
    receipt["mode"].as_str().expect("receipt mode").to_owned()
}

fn is_empty(dir: &Path) -> bool {
    fs::read_dir(dir)
        .expect("staging directory")
        .next()
        .is_none()
}

#[test]
fn development_consumers_stage_only_from_their_unchanged_checkout() {
    let scratch = tempfile::tempdir().expect("tempdir");
    let checkout = scratch.path().join("checkout");
    let target = scratch.path().join("target");
    let development = scratch.path().join("development-consumer");
    let sealed = scratch.path().join("sealed-consumer");
    copy_checkout(&checkout);
    build_consumer(&checkout, &target, &[], &development);
    build_consumer(&checkout, &target, &["--features", "sealed"], &sealed);

    let fresh = scratch.path().join("fresh");
    let output = stage(&development, &fresh);
    assert!(
        output.status.success(),
        "an unchanged checkout stages: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(receipt_mode(&fresh), "development");

    let edited = "crates/chelis-vocab/src/lib.rs";
    let mut source = fs::read(checkout.join(edited)).expect("runtime dependency source");
    source.extend_from_slice(b"\n// edited after the consumer was built\n");
    fs::write(checkout.join(edited), source).expect("edit");
    let stale = scratch.path().join("stale");
    let output = stage(&development, &stale);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "a stale runtime stages: {stderr}");
    assert!(stderr.contains(&format!("changed: {edited}")), "{stderr}");
    assert!(is_empty(&stale), "a refused staging writes nothing");

    build_consumer(&checkout, &target, &[], &development);
    let rebuilt = scratch.path().join("rebuilt");
    let output = stage(&development, &rebuilt);
    assert!(
        output.status.success(),
        "a rebuild records the edited source: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    fs::remove_dir_all(&checkout).expect("remove the checkout");
    let orphaned = scratch.path().join("orphaned");
    let output = stage(&development, &orphaned);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "a development consumer stages without its checkout: {stderr}"
    );
    assert!(
        stderr.contains(&checkout.display().to_string()) && stderr.contains("sealed-runtime"),
        "{stderr}"
    );
    assert!(is_empty(&orphaned), "a refused staging writes nothing");

    let relocated = scratch.path().join("relocated");
    let output = stage(&sealed, &relocated);
    assert!(
        output.status.success(),
        "a sealed consumer stages without a checkout: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(receipt_mode(&relocated), "sealed");
}
