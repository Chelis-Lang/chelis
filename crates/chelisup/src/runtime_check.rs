//! Check a release's runtime files against the runtime its compiler carries.
//!
//! A release tarball ships `lib/libchelis_runtime.a` and the public runtime
//! headers under `include/` for C programs that link the runtime themselves.
//! `chelis` never reads them: it stages the runtime it carries
//! (spec/08-backends.md §2.1). Before a release enters the store, `install`
//! runs the unpacked `chelis runtime export` into a scratch directory and
//! requires the shipped files to be what that export reports. Its staging
//! receipt must describe a sealed build of the version being installed, and
//! the shipped archive and every header the receipt lists must have the
//! SHA-256 it records (chelis#1354).
//!
//! Releases up to [`LAST_RELEASE_WITHOUT_EXPORT`] predate
//! `chelis runtime export`, so nothing can be checked; they install as
//! before, and the CLI warns that their runtime files are unchecked.

use std::fs;
use std::io;
use std::path::Path;
use std::process::Command;

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::version::is_safe_path_component;

/// The last release whose `chelis` has no `chelis runtime export`.
pub const LAST_RELEASE_WITHOUT_EXPORT: &str = "0.18.11";

const RECEIPT: &str = "chelis_runtime.receipt.json";
const RECEIPT_SCHEMA: &str = "chelis-runtime-staging/1";
const ARCHIVE: &str = "libchelis_runtime.a";

/// What the runtime check established about a newly unpacked release.
#[derive(Debug, PartialEq, Eq)]
pub enum RuntimeCheck {
    /// The shipped runtime files are the ones the release's `chelis` exports.
    Verified { archive_sha256: String },
    /// The release predates `chelis runtime export`; its runtime files were
    /// not checked.
    Unchecked,
}

/// Check the runtime files of release `version` unpacked at `unpacked`,
/// exporting into a new directory under `scratch`. `version` is a validated
/// `X.Y.Z`.
pub(crate) fn check(
    version: &str,
    unpacked: &Path,
    scratch: &Path,
) -> Result<RuntimeCheck, String> {
    if triple(version)? <= triple(LAST_RELEASE_WITHOUT_EXPORT)? {
        return Ok(RuntimeCheck::Unchecked);
    }
    let exported = tempfile::Builder::new()
        .prefix("runtime-export-")
        .tempdir_in(scratch)
        .map_err(|e| format!("could not create a runtime export directory: {e}"))?;
    let chelis = unpacked.join("bin").join("chelis");
    let output = Command::new(&chelis)
        .args(["runtime", "export"])
        .arg(exported.path())
        // The export refuses a runtime directory; this check concerns the
        // release, not the caller's environment.
        .env_remove("CHELIS_RUNTIME_DIR")
        .output()
        .map_err(|e| format!("could not run {} runtime export: {e}", chelis.display()))?;
    if !output.status.success() {
        return Err(format!(
            "`chelis runtime export` from the chelis {version} release failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let receipt_path = exported.path().join(RECEIPT);
    let receipt: Value = fs::read(&receipt_path)
        .map_err(|e| format!("could not read {}: {e}", receipt_path.display()))
        .and_then(|bytes| {
            serde_json::from_slice(&bytes)
                .map_err(|e| format!("{} is not JSON: {e}", receipt_path.display()))
        })?;
    for (field, expected) in [
        ("schema", RECEIPT_SCHEMA),
        ("archive", ARCHIVE),
        ("mode", "sealed"),
        ("chelis_version", version),
    ] {
        if receipt.get(field).and_then(Value::as_str) != Some(expected) {
            return Err(format!(
                "the chelis {version} release's runtime export records {field} {}, not {expected:?}",
                receipt
                    .get(field)
                    .map_or_else(|| "nothing".to_owned(), Value::to_string)
            ));
        }
    }

    let archive_sha256 = sha256_field(&receipt, "archive_sha256", version)?;
    require_shipped(
        unpacked,
        &Path::new("lib").join(ARCHIVE),
        archive_sha256,
        version,
    )?;
    let headers = receipt
        .get("headers")
        .and_then(Value::as_object)
        .filter(|headers| !headers.is_empty())
        .ok_or_else(|| format!("the chelis {version} release's runtime export lists no headers"))?;
    for (name, digest) in headers {
        if !is_safe_path_component(name) {
            return Err(format!(
                "the chelis {version} release's runtime export lists the header {name:?}, \
                 which is not a file name"
            ));
        }
        let digest = digest
            .as_str()
            .filter(|digest| is_sha256_hex(digest))
            .ok_or_else(|| {
                format!(
                    "the chelis {version} release's runtime export records no SHA-256 for {name}"
                )
            })?;
        require_shipped(unpacked, &Path::new("include").join(name), digest, version)?;
    }
    Ok(RuntimeCheck::Verified {
        archive_sha256: archive_sha256.to_owned(),
    })
}

/// Parse a validated `X.Y.Z` for ordering.
fn triple(version: &str) -> Result<(u64, u64, u64), String> {
    let mut parts = version.split('.').map(str::parse::<u64>);
    match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some(Ok(major)), Some(Ok(minor)), Some(Ok(patch)), None) => Ok((major, minor, patch)),
        _ => Err(format!("malformed version {version:?}; expected X.Y.Z")),
    }
}

fn sha256_field<'a>(receipt: &'a Value, field: &str, version: &str) -> Result<&'a str, String> {
    receipt
        .get(field)
        .and_then(Value::as_str)
        .filter(|digest| is_sha256_hex(digest))
        .ok_or_else(|| format!("the chelis {version} release's runtime export records no {field}"))
}

fn is_sha256_hex(digest: &str) -> bool {
    digest.len() == 64
        && digest
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

/// Require the regular file `relative` under `unpacked` to have SHA-256 `expected`.
fn require_shipped(
    unpacked: &Path,
    relative: &Path,
    expected: &str,
    version: &str,
) -> Result<(), String> {
    let path = unpacked.join(relative);
    let observed = fs::symlink_metadata(&path)
        .and_then(|metadata| {
            if metadata.is_file() {
                Ok(())
            } else {
                Err(io::Error::other("not a regular file"))
            }
        })
        .and_then(|()| sha256_file(&path))
        .map_err(|e| {
            format!(
                "the chelis {version} release has no usable {}: {e}",
                relative.display()
            )
        })?;
    if observed != expected {
        return Err(format!(
            "the chelis {version} release ships {} with SHA-256 {observed}, but its chelis \
             exports {expected}",
            relative.display()
        ));
    }
    Ok(())
}

fn sha256_file(path: &Path) -> io::Result<String> {
    let mut hasher = Sha256::new();
    io::copy(&mut fs::File::open(path)?, &mut hasher)?;
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}
