//! Check a release's runtime files against the runtime its compiler carries.
//!
//! A release tarball ships `lib/libchelis_runtime.a` and the public runtime
//! headers under `include/` for C programs that link the runtime themselves.
//! `chelis` never reads them: it stages the runtime it carries
//! (spec/08-backends.md §2.1). Before a release enters the store, `install`
//! runs the unpacked `chelis runtime export` into a scratch directory and
//! requires the shipped files to be what that export reports. Its staging
//! receipt must describe a sealed build of the version being installed, and
//! the shipped archive and exactly the six public headers must have the
//! SHA-256 digests that export records (chelis#1354).
//!
//! Releases up to [`LAST_RELEASE_WITHOUT_EXPORT`] predate
//! `chelis runtime export`, so nothing can be checked; they install as
//! before, and the CLI warns that their runtime files are unchecked.
//!
//! A refused release newer than the running chelisup may use a format this
//! chelisup does not know, and chelisup does not update itself, so that
//! refusal says how to get the latest chelisup. When the operating system
//! refuses to execute a release's `chelis`, the release is not refused for
//! its runtime files, and a newer chelisup would unpack the same binary, so
//! that error says why it cannot run instead. A dynamic loader that rejects
//! the binary after execution starts shows up as a failed export, with the
//! loader's message.

use std::fs;
use std::io;
use std::path::Path;
use std::process::{Command, Output};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::UPGRADE_ADVICE;
use crate::version::{is_safe_path_component, release_triple};

/// The last release whose `chelis` has no `chelis runtime export`.
pub const LAST_RELEASE_WITHOUT_EXPORT: &str = "0.18.11";

/// The running chelisup's version.
const CHELISUP_VERSION: &str = env!("CARGO_PKG_VERSION");

const RECEIPT: &str = "chelis_runtime.receipt.json";
const RECEIPT_SCHEMA: &str = "chelis-runtime-staging/1";
const ARCHIVE: &str = "libchelis_runtime.a";
const PUBLIC_HEADERS: [&str; 6] = [
    "chelis_runtime.h",
    "chelis_runtime_views.h",
    "chelis_runtime_dtype.h",
    "chelis_blas.h",
    "chelis_simd.h",
    "chelis_math.h",
];

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
    if release_triple(version)? <= release_triple(LAST_RELEASE_WITHOUT_EXPORT)? {
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
        .map_err(|e| cannot_start(version, &e))?;
    verify(version, &output, exported.path(), unpacked)
        .map(|archive_sha256| RuntimeCheck::Verified { archive_sha256 })
        .map_err(|refusal| advise_upgrade(refusal, version, CHELISUP_VERSION))
}

/// The error for a release whose `bin/chelis` the operating system refuses
/// to execute.
fn cannot_start(version: &str, error: &io::Error) -> String {
    // Extraction found `bin/chelis`, so what is missing is a file it names.
    let cause = if error.kind() == io::ErrorKind::NotFound {
        ": the file exists, so the loader or interpreter it names is missing, as with a \
         glibc build on a musl-based system such as Alpine"
    } else {
        ""
    };
    format!("this machine cannot run the chelis {version} release's bin/chelis ({error}){cause}")
}

/// Append how to get the latest chelisup to `refusal` when release `version`
/// is newer than `chelisup`, the running chelisup's version.
fn advise_upgrade(refusal: String, version: &str, chelisup: &str) -> String {
    match (release_triple(version), release_triple(chelisup)) {
        (Ok(release), Ok(running)) if release > running => format!(
            "{refusal}. This chelisup ({chelisup}) is older than the {version} release, which \
             may use a format it does not know; to get the latest chelisup, {UPGRADE_ADVICE}, \
             then retry"
        ),
        _ => refusal,
    }
}

/// Require `output` to report a sealed build of `version` whose exported
/// archive and headers match the receipt and whose shipped counterparts match
/// those same digests. Returns the archive SHA-256.
fn verify(
    version: &str,
    output: &Output,
    exported: &Path,
    unpacked: &Path,
) -> Result<String, String> {
    if !output.status.success() {
        return Err(format!(
            "`chelis runtime export` from the chelis {version} release failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let receipt_path = exported.join(RECEIPT);
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
    require_exported(exported, Path::new(ARCHIVE), archive_sha256, version)?;
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
    for name in headers.keys() {
        if !is_safe_path_component(name) {
            return Err(format!(
                "the chelis {version} release's runtime export lists the header {name:?}, \
                 which is not a file name"
            ));
        }
    }
    if headers.len() != PUBLIC_HEADERS.len()
        || PUBLIC_HEADERS
            .iter()
            .any(|name| !headers.contains_key(*name))
    {
        return Err(format!(
            "the chelis {version} release's runtime export does not list exactly the six public headers"
        ));
    }
    for (name, digest) in headers {
        let digest = digest
            .as_str()
            .filter(|digest| is_sha256_hex(digest))
            .ok_or_else(|| {
                format!(
                    "the chelis {version} release's runtime export records no SHA-256 for {name}"
                )
            })?;
        require_exported(exported, Path::new(name), digest, version)?;
        require_shipped(unpacked, &Path::new("include").join(name), digest, version)?;
    }
    Ok(archive_sha256.to_owned())
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

/// Require the regular exported file `relative` to have the digest in the receipt.
fn require_exported(
    exported: &Path,
    relative: &Path,
    expected: &str,
    version: &str,
) -> Result<(), String> {
    let observed = within_release(exported, relative)
        .and_then(|()| sha256_file(&exported.join(relative)))
        .map_err(|e| {
            format!(
                "the chelis {version} runtime export has no usable {}: {e}",
                relative.display()
            )
        })?;
    if observed != expected {
        return Err(format!(
            "the chelis {version} runtime export reports {} with SHA-256 {observed}, but its \
             receipt records {expected}",
            relative.display()
        ));
    }
    Ok(())
}

/// Require the regular file `relative` under `unpacked` to have SHA-256 `expected`.
fn require_shipped(
    unpacked: &Path,
    relative: &Path,
    expected: &str,
    version: &str,
) -> Result<(), String> {
    let observed = within_release(unpacked, relative)
        .and_then(|()| sha256_file(&unpacked.join(relative)))
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

/// Fail unless every component of `relative` below `root` is a real directory
/// and the last is a regular file, so the check reads nothing a link points to.
fn within_release(root: &Path, relative: &Path) -> io::Result<()> {
    let mut path = root.to_path_buf();
    let mut components = relative.components().peekable();
    while let Some(component) = components.next() {
        path.push(component);
        let file_type = fs::symlink_metadata(&path)?.file_type();
        let (expected, found) = if components.peek().is_some() {
            ("directory", file_type.is_dir())
        } else {
            ("regular file", file_type.is_file())
        };
        if !found {
            return Err(io::Error::other(format!(
                "{} is not a {expected}",
                path.strip_prefix(root).unwrap_or(&path).display()
            )));
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_release_newer_than_chelisup_gets_upgrade_advice() {
        let advised = advise_upgrade("refused".to_owned(), "0.19.10", "0.19.9");
        assert!(
            advised
                .starts_with("refused. This chelisup (0.19.9) is older than the 0.19.10 release"),
            "{advised}"
        );
        assert!(advised.contains(UPGRADE_ADVICE), "{advised}");
        for release in ["0.19.9", "0.9.10"] {
            assert_eq!(
                advise_upgrade("refused".to_owned(), release, "0.19.9"),
                "refused"
            );
        }
    }

    #[test]
    fn the_running_chelisup_version_is_ordered() {
        // An unordered version would silently drop every upgrade advice.
        assert!(release_triple(CHELISUP_VERSION).is_ok());
    }
}
