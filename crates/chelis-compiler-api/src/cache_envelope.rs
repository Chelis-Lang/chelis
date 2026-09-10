//! Shared on-disk cache envelope: atomic write, version + integrity
//! envelope, torn-write rejection.
//!
//! Both the Phase K whole-package `CompiledContext` cache (`context.rs`)
//! and the cross-process chelis-std typecheck sub-context cache
//! (`stdlib_cache.rs`) persist a bincode-serialized payload through this
//! one code path. The two caches differ only in *what they key on* — a
//! whole-graph source hash vs. a content-addressed chelis-std bundle hash
//! — not in *how the bytes hit disk*.
//!
//! ## On-disk layout
//!
//! ```text
//! [CACHE_MAGIC bytes][bincode-encoded Envelope]
//! ```
//!
//! The raw magic prefix sits BEFORE the bincode region so an "is this
//! even a cache file" check is robust against bincode's leading
//! length-prefix on `Vec<u8>` fields. The envelope carries the format
//! version, a 32-byte cache key, and a SHA-256 of the inner payload bytes
//! (torn-write detection).
//!
//! ## Atomicity
//!
//! The bytes are written to a unique `.tmp.<pid>.<tid>.<nanos>` file in
//! the same directory, `sync_all`'d, then `fs::rename`'d into place.
//! POSIX `rename` is atomic within a single filesystem, so a crashed or
//! racing writer can leave a `.tmp.*` orphan but never a half-written
//! final file. A reader sees either the absent old state or the fully
//! written file.
//!
//! ## Corruption handling
//!
//! `load` returns `Ok(None)` only for a clean miss (file absent, or the
//! stored key does not match the expected key). Every "the bytes are
//! present but unusable" condition — missing magic, bad envelope decode,
//! version skew, payload SHA-256 mismatch, inner decode failure — returns
//! `Err(CacheError::*)`. Callers treat `Err` as "recompute and
//! overwrite," never as a hard abort, but the distinct error means a
//! torn write is never silently mistaken for a valid miss.

use chelis_ir::lower::LoweredLibrary;
use chelis_unord::UnordMap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Magic header bytes. The trailing newline guards against accidental
/// concatenation with another file.
const CACHE_MAGIC: &[u8] = b"CHELIS_CACHE_ENV_V1\n";

/// Errors from the shared cache envelope layer.
#[derive(Debug)]
pub enum CacheError {
    /// I/O failure while reading or writing a cache file.
    Io {
        op: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    /// `bincode` failure on the encode side. Should be impossible for a
    /// well-formed payload, but surfaced for completeness.
    Encode(String),
    /// `bincode` failure on the decode side after the envelope and
    /// version checks passed — the inner payload shape changed without a
    /// version bump. Treat as a bug + miss.
    Decode(String),
    /// File contents are not a valid envelope: missing magic, bincode
    /// error during envelope decode, or payload SHA-256 mismatch (torn
    /// write). Callers MUST NOT silently accept the bytes.
    Corrupt(String),
    /// Envelope decoded but the on-disk format version is not the one the
    /// running binary supports.
    UnsupportedVersion { stored: u32, expected: u32 },
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheError::Io { op, path, source } => {
                write!(
                    f,
                    "cache I/O error during {op} on {}: {source}",
                    path.display()
                )
            }
            CacheError::Encode(msg) => write!(f, "cache encode error: {msg}"),
            CacheError::Decode(msg) => write!(f, "cache decode error: {msg}"),
            CacheError::Corrupt(msg) => write!(f, "cache file is corrupt: {msg}"),
            CacheError::UnsupportedVersion { stored, expected } => write!(
                f,
                "cache file format version {stored} not supported by this binary \
                 (expects {expected})"
            ),
        }
    }
}

impl std::error::Error for CacheError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CacheError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Envelope {
    /// On-disk format version. A schema change to `Envelope` bumps this
    /// so `load` rejects older files with `UnsupportedVersion` rather
    /// than risking a "decoded but wrong" payload.
    version: u32,
    /// 32-byte cache key. `load` compares this against the caller's
    /// expected key; a mismatch is a clean miss (`Ok(None)`).
    key: [u8; 32],
    /// SHA-256 of the inner payload bytes. Catches torn writes whose
    /// truncated payload still bincode-decodes successfully.
    payload_sha256: [u8; 32],
    /// Bincode-encoded payload body.
    payload: Vec<u8>,
}

const ENVELOPE_FORMAT_VERSION: u32 = 1;

fn sorted_map_bytes<T: Serialize>(map: &UnordMap<String, T>) -> Result<Vec<u8>, String> {
    let entries = map
        .to_sorted()
        .into_iter()
        .map(|(key, value)| (key.as_str(), value))
        .collect::<Vec<_>>();
    bincode::serialize(&entries).map_err(|error| error.to_string())
}

fn sorted_btree_map_bytes<T: Serialize>(
    map: &std::collections::BTreeMap<String, T>,
) -> Result<Vec<u8>, String> {
    let entries = map
        .iter()
        .map(|(key, value)| (key.as_str(), value))
        .collect::<Vec<_>>();
    bincode::serialize(&entries).map_err(|error| error.to_string())
}

/// Compare raw lower results without dependence on `UnordMap` iteration order.
///
/// Bincode preserves float bits. This comparison therefore accepts equal NaN
/// payloads but rejects every changed field in the raw cache carrier.
pub(crate) fn lowered_library_payload_matches(
    cached: &LoweredLibrary,
    expected: &LoweredLibrary,
) -> Result<bool, String> {
    if cached.linearity() != expected.linearity()
        || cached.rootless_defs() != expected.rootless_defs()
        || cached.library_proof_id() != expected.library_proof_id()
    {
        return Ok(false);
    }

    Ok(
        bincode::serialize(cached.dag()).map_err(|error| error.to_string())?
            == bincode::serialize(expected.dag()).map_err(|error| error.to_string())?
            && sorted_map_bytes(cached.symbol_table())?
                == sorted_map_bytes(expected.symbol_table())?
            && sorted_btree_map_bytes(cached.program_defs())?
                == sorted_btree_map_bytes(expected.program_defs())?
            && sorted_btree_map_bytes(cached.program_signatures())?
                == sorted_btree_map_bytes(expected.program_signatures())?
            && sorted_btree_map_bytes(cached.program_types())?
                == sorted_btree_map_bytes(expected.program_types())?
            && sorted_btree_map_bytes(cached.lowered_names())?
                == sorted_btree_map_bytes(expected.lowered_names())?,
    )
}

/// Atomically persist `payload` to `path` under the 32-byte `key`.
///
/// Parent directories are created lazily (private `0o700` on Unix when
/// this call has to create them; an existing directory's mode is left
/// untouched). The write is temp-file + `fs::rename`.
pub fn save<T: Serialize>(path: &Path, key: [u8; 32], payload: &T) -> Result<(), CacheError> {
    let payload_bytes =
        bincode::serialize(payload).map_err(|e| CacheError::Encode(format!("payload: {e}")))?;
    let payload_sha256: [u8; 32] = Sha256::digest(&payload_bytes).into();
    let envelope = Envelope {
        version: ENVELOPE_FORMAT_VERSION,
        key,
        payload_sha256,
        payload: payload_bytes,
    };
    let envelope_bytes =
        bincode::serialize(&envelope).map_err(|e| CacheError::Encode(format!("envelope: {e}")))?;

    let mut bytes = Vec::with_capacity(CACHE_MAGIC.len() + envelope_bytes.len());
    bytes.extend_from_slice(CACHE_MAGIC);
    bytes.extend_from_slice(&envelope_bytes);

    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
        && !parent.exists()
    {
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            fs::DirBuilder::new()
                .recursive(true)
                .mode(0o700)
                .create(parent)
                .map_err(|e| CacheError::Io {
                    op: "create_dir_all",
                    path: parent.to_path_buf(),
                    source: e,
                })?;
        }
        #[cfg(not(unix))]
        {
            fs::create_dir_all(parent).map_err(|e| CacheError::Io {
                op: "create_dir_all",
                path: parent.to_path_buf(),
                source: e,
            })?;
        }
    }

    // Same-directory temp file -> atomic rename. The temp name carries
    // pid + thread id + a fresh nanosecond timestamp so concurrent
    // writers (worker pools, ~200 racing nextest processes) never collide
    // on the temp path.
    use std::time::{SystemTime, UNIX_EPOCH};
    let tid = format!("{:?}", std::thread::current().id())
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect::<String>();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let tmp_name = format!(
        ".{}.tmp.{}.{}.{}",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("cache"),
        std::process::id(),
        tid,
        nanos,
    );
    let tmp_path = path
        .parent()
        .map(|p| p.join(&tmp_name))
        .unwrap_or_else(|| PathBuf::from(&tmp_name));

    {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)
            .map_err(|e| CacheError::Io {
                op: "open_tmp",
                path: tmp_path.clone(),
                source: e,
            })?;
        f.write_all(&bytes).map_err(|e| CacheError::Io {
            op: "write_tmp",
            path: tmp_path.clone(),
            source: e,
        })?;
        f.sync_all().map_err(|e| CacheError::Io {
            op: "sync_tmp",
            path: tmp_path.clone(),
            source: e,
        })?;
    }

    fs::rename(&tmp_path, path).map_err(|e| {
        let _ = fs::remove_file(&tmp_path);
        CacheError::Io {
            op: "rename",
            path: path.to_path_buf(),
            source: e,
        }
    })?;
    Ok(())
}

/// Load the payload at `path` if it is a valid envelope whose stored key
/// equals `expected_key`.
///
/// - `Ok(Some(payload))` — valid, integrity-verified, key matches.
/// - `Ok(None)` — clean miss: file absent, or stored key != expected key.
/// - `Err(_)` — the bytes are present but unusable (corrupt, version
///   skew, decode failure). Never a silent fall-through.
pub fn load<T: for<'de> Deserialize<'de>>(
    path: &Path,
    expected_key: [u8; 32],
) -> Result<Option<T>, CacheError> {
    // Read the whole file before any decode work — no streaming-decode
    // window where a half-written tail looks like a full envelope.
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(CacheError::Io {
                op: "read",
                path: path.to_path_buf(),
                source: e,
            });
        }
    };

    // An empty or truncated-before-magic file is Corrupt, never None: a
    // torn write must NOT silently fall through as a "valid miss."
    if bytes.is_empty() {
        return Err(CacheError::Corrupt("empty cache file".to_string()));
    }
    if bytes.len() < CACHE_MAGIC.len() || &bytes[..CACHE_MAGIC.len()] != CACHE_MAGIC {
        return Err(CacheError::Corrupt(
            "missing or wrong magic header".to_string(),
        ));
    }

    let envelope_bytes = &bytes[CACHE_MAGIC.len()..];
    let envelope: Envelope = match bincode::deserialize(envelope_bytes) {
        Ok(env) => env,
        Err(e) => return Err(CacheError::Corrupt(format!("envelope decode: {e}"))),
    };

    if envelope.version != ENVELOPE_FORMAT_VERSION {
        return Err(CacheError::UnsupportedVersion {
            stored: envelope.version,
            expected: ENVELOPE_FORMAT_VERSION,
        });
    }

    // Key mismatch is a clean miss — the file is a valid cache entry, just
    // for a different key (e.g. a different chelis-std bundle).
    if envelope.key != expected_key {
        return Ok(None);
    }

    // Verify the payload SHA-256 before paying the bincode-decode cost. A
    // torn write whose envelope happens to decode but whose payload was
    // truncated is caught here.
    let actual_payload_sha: [u8; 32] = Sha256::digest(&envelope.payload).into();
    if actual_payload_sha != envelope.payload_sha256 {
        return Err(CacheError::Corrupt(
            "payload sha256 does not match envelope".to_string(),
        ));
    }

    match bincode::deserialize(&envelope.payload) {
        Ok(payload) => Ok(Some(payload)),
        Err(e) => Err(CacheError::Decode(format!(
            "payload decode (envelope/version/key match but inner shape changed): {e}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Sample {
        a: u32,
        b: String,
    }

    fn key(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    #[test]
    fn round_trips_on_matching_key() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("entry.cache");
        let payload = Sample {
            a: 7,
            b: "hello".to_string(),
        };
        save(&path, key(1), &payload).unwrap();
        let loaded: Option<Sample> = load(&path, key(1)).unwrap();
        assert_eq!(loaded, Some(payload));
    }

    #[test]
    fn absent_file_is_clean_miss() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nope.cache");
        let loaded: Option<Sample> = load(&path, key(1)).unwrap();
        assert_eq!(loaded, None);
    }

    #[test]
    fn key_mismatch_is_clean_miss_not_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("entry.cache");
        save(
            &path,
            key(1),
            &Sample {
                a: 1,
                b: "x".to_string(),
            },
        )
        .unwrap();
        // Same file, different expected key: a valid entry for another
        // key, so a clean miss — not a corruption error.
        let loaded: Option<Sample> = load(&path, key(2)).unwrap();
        assert_eq!(loaded, None);
    }

    #[test]
    fn empty_file_is_corrupt_not_miss() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("entry.cache");
        fs::write(&path, b"").unwrap();
        let err = load::<Sample>(&path, key(1)).unwrap_err();
        assert!(matches!(err, CacheError::Corrupt(_)), "got {err:?}");
    }

    #[test]
    fn missing_magic_is_corrupt_not_miss() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("entry.cache");
        fs::write(&path, b"not a chelis cache file at all").unwrap();
        let err = load::<Sample>(&path, key(1)).unwrap_err();
        assert!(matches!(err, CacheError::Corrupt(_)), "got {err:?}");
    }

    #[test]
    fn truncated_payload_is_corrupt_not_miss() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("entry.cache");
        save(
            &path,
            key(1),
            &Sample {
                a: 42,
                b: "payload bytes here".to_string(),
            },
        )
        .unwrap();
        // Truncate to a third of the file: past the magic, into the
        // envelope/payload region — the classic torn-write shape.
        let full = fs::read(&path).unwrap();
        fs::write(&path, &full[..full.len() / 3]).unwrap();
        let err = load::<Sample>(&path, key(1)).unwrap_err();
        assert!(
            matches!(err, CacheError::Corrupt(_)),
            "truncated payload must be Corrupt, got {err:?}"
        );
    }

    #[test]
    fn save_is_atomic_under_concurrent_writers() {
        // N threads racing to write the same path must each either
        // complete a full write or fail cleanly; the final file must be
        // a valid, fully-decodable envelope, never a torn one.
        let dir = tempdir().unwrap();
        let path = dir.path().join("raced.cache");
        let handles: Vec<_> = (0..16)
            .map(|i| {
                let path = path.clone();
                std::thread::spawn(move || {
                    save(
                        &path,
                        key(9),
                        &Sample {
                            a: i,
                            b: format!("writer-{i}"),
                        },
                    )
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap().expect("each racing save must succeed");
        }
        // The surviving file must be a valid envelope.
        let loaded: Option<Sample> = load(&path, key(9)).expect("final file must not be torn");
        assert!(loaded.is_some(), "a valid payload must survive the race");
    }
}
