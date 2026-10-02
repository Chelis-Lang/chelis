//! Pack the chelis-std runtime into `OUT_DIR`.
//!
//! The runtime's archive and shell are produced while this crate builds, by
//! the same packing step `chelis reef build` runs, so the bytes a binary
//! embeds always match the `packages/chelis-std` sources it was built from
//! and nothing generated is committed. The inputs are staged first (see
//! `build/stage.rs`), and every archive member's mtime is 0 whatever
//! `SOURCE_DATE_EPOCH` says, so the bytes depend only on those inputs and
//! the packing code.
//!
//! Outputs, read by `src/lib.rs`:
//! - `chelis-std.tar.zst` and `chelis-std.chb`, the packed pair;
//! - `chelis-std.version`, the version `packages/chelis-std/reef.toml` names;
//! - `chelis-std.inputs`, the SHA-256 of this script's executable and the
//!   staged inputs. A rerun whose fingerprint matches skips the pack, so a
//!   touched file or a stray non-source file costs no repack.

use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

#[path = "build/stage.rs"]
mod stage;

fn main() {
    let manifest_dir = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"),
    );
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    let std_root = manifest_dir.join("../../packages/chelis-std");
    println!("cargo:rerun-if-changed=build/stage.rs");
    let watched = stage::watched_paths(&std_root)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", std_root.display()));
    for relative in watched {
        println!(
            "cargo:rerun-if-changed={}",
            std_root.join(relative).display()
        );
    }

    let staged_root = out_dir.join("stage").join("chelis-std");
    let staged = stage::stage(&std_root, &staged_root)
        .unwrap_or_else(|error| panic!("failed to stage {}: {error}", std_root.display()));

    let fingerprint = inputs_fingerprint(&staged);
    let archive_path = out_dir.join("chelis-std.tar.zst");
    let shell_path = out_dir.join("chelis-std.chb");
    let version_path = out_dir.join("chelis-std.version");
    let fingerprint_path = out_dir.join("chelis-std.inputs");
    let up_to_date = fs::read_to_string(&fingerprint_path).is_ok_and(|seen| seen == fingerprint)
        && [&archive_path, &shell_path, &version_path]
            .iter()
            .all(|path| path.is_file());
    if up_to_date {
        return;
    }

    let packed = stage::pack(&staged_root)
        .unwrap_or_else(|error| panic!("failed to pack {}: {error}", std_root.display()));
    write_if_changed(&archive_path, &packed.archive);
    write_if_changed(&shell_path, &packed.shell);
    write_if_changed(&version_path, packed.package.version.as_bytes());
    write_if_changed(&fingerprint_path, fingerprint.as_bytes());
}

/// SHA-256 over this build script's own executable, which carries the
/// packing code, and every staged input with its package-relative path.
fn inputs_fingerprint(staged: &[(PathBuf, Vec<u8>)]) -> String {
    let executable = std::env::current_exe().expect("locate the build script executable");
    let mut hasher = Sha256::new();
    hasher.update(
        fs::read(&executable)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", executable.display())),
    );
    for (relative, bytes) in staged {
        let relative = relative.to_string_lossy();
        hasher.update((relative.len() as u64).to_le_bytes());
        hasher.update(relative.as_bytes());
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    format!("{:x}", hasher.finalize())
}

fn write_if_changed(path: &Path, bytes: &[u8]) {
    if fs::read(path).is_ok_and(|existing| existing == bytes) {
        return;
    }
    fs::write(path, bytes)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
}
