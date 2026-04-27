//! Phase G: composed Compiled Artifact Cache.
//!
//! `CompiledContext` is the cacheable artifact a reef package compiles
//! into once. New code (test files, eval inputs) is checked and lowered
//! against the context's pre-checked, pre-lowered library defs; the
//! context's contents are referenced, not recompiled.
//!
//! The pipeline split is:
//!
//! - `compile_reef_context` (this file): runs the full library pipeline
//!   ONCE. Parses + desugars + macro-expands the linked library decls,
//!   builds a Phase 0e [`TypeEnv`], then runs the monolithic Phase 0e
//!   checker, the effects checker, and the linearity checker on the
//!   library, and lowers it to a [`LoweredLibrary`].
//! - `eval_in_context` / `check_in_context` / `eval_many_in_context`
//!   (in `compiler.rs`): take a `CompiledContext` plus new source. Only
//!   the new source is re-compiled; the C/D/E/F `_with_context` variants
//!   stack the new code on top of the cached library state.
//!
//! See `/home/jeff/.claude/plans/now-plan-out-the-shimmying-wand.md`
//! for the full plan.

use chelis_ir::lower::{LoweredLibrary, lower_program_to_library};
use chelis_reef::{PreparedReefGraph, SourceDigest, prepare_reef_graph};
use chelis_types::{CheckedProgram, TypeEnv, build_compiled_library_context, check_linearity};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::compiler::{CompilerError, check_error_diagnostic, stage_error};
use crate::schema::Diagnostic;

/// 32-byte content hash of every source file that contributed to a
/// `CompiledContext`. Phase I disk cache keys on this for invalidation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextHash(pub [u8; 32]);

impl ContextHash {
    /// Combine a sorted slice of `SourceDigest` rows into a single
    /// fixed-width hash. Strings are length-prefixed (u64 little-endian)
    /// to disambiguate concatenation collisions; per-file `sha256` is
    /// fixed-width so it's appended directly.
    pub fn from_digests(digests: &[SourceDigest]) -> Self {
        let mut hasher = Sha256::new();
        for d in digests {
            hasher.update((d.package_name.len() as u64).to_le_bytes());
            hasher.update(d.package_name.as_bytes());
            hasher.update((d.package_version.len() as u64).to_le_bytes());
            hasher.update(d.package_version.as_bytes());
            hasher.update((d.module_name.len() as u64).to_le_bytes());
            hasher.update(d.module_name.as_bytes());
            hasher.update(d.sha256);
        }
        let bytes: [u8; 32] = hasher.finalize().into();
        ContextHash(bytes)
    }
}

/// A reef package compiled once into a reusable artifact.
///
/// Phase G composes the result of every pipeline stage:
/// - `source_hash`: content hash for disk-cache invalidation (Phase I).
/// - `reef_state`: linked library decls + reef metadata (Phase B).
/// - `type_env`: Phase 0e type-checker snapshot (Phase C). Used by
///   `check_phase0e_with_context` for new-code type checking.
/// - `library_checked`: monolithic Phase 0e + effects + linearity result
///   over the library decls (Phases C/D/E). Used by
///   `check_effects_with_context` and `check_linearity_with_context`.
/// - `library_dag`: lowered library DAG carrier (Phase F). Used by
///   `lower_program_with_context`.
///
/// All five fields are populated once by `compile_reef_context` and
/// thereafter treated as immutable. Cheap to clone (the heavy state is
/// `Arc`-shared inside `TypeEnv`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledContext {
    /// Content hash of every backed source file. Stable across calls
    /// on unchanged sources; changes when ANY source byte changes.
    pub source_hash: ContextHash,
    /// The reef state (lockfile-backed package graph + linked library
    /// decls + internal-name maps + dep shells).
    pub(crate) reef_state: PreparedReefGraph,
    /// Phase 0e type-checker snapshot — the outer scope for new-code
    /// type checking via `check_phase0e_with_context`.
    pub(crate) type_env: TypeEnv,
    /// Library Phase 0e + effects + linearity result. Feeds the
    /// `_with_context` variants of effects and linearity.
    pub(crate) library_checked: CheckedProgram,
    /// Lowered library carrier. Feeds `lower_program_with_context`.
    pub(crate) library_dag: LoweredLibrary,
}

impl CompiledContext {
    /// Bincode round-trip for the Phase H worker handoff and the
    /// Phase I disk cache. Phases C/F will add new fields; the
    /// encoder must continue to round-trip then.
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(self).map_err(|e| format!("encode CompiledContext: {e}"))
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        bincode::deserialize(bytes).map_err(|e| format!("decode CompiledContext: {e}"))
    }

    /// Phase I — atomically persist this context to `path`.
    ///
    /// The on-disk format is a bincode-serialized [`CacheEnvelope`]
    /// (see below) which embeds:
    /// - a magic byte string for format identification,
    /// - the format version (`CACHE_FORMAT_VERSION`),
    /// - a copy of `source_hash` outside the inner payload (cheap freshness
    ///   check before paying the bincode-decode cost),
    /// - a SHA-256 of the inner payload bytes (catches torn writes that
    ///   happen to bincode-decode anyway), and
    /// - the inner payload (bincode-encoded [`CompiledContext`]).
    ///
    /// **Atomic write:** the bytes are written to `<path>.tmp.<pid>` in
    /// the same directory, then `fs::rename`'d into place. POSIX `rename`
    /// is atomic within a single filesystem, so a crashed writer can leave
    /// a `.tmp.<pid>` orphan but never a half-written final file. Parent
    /// directories are created lazily.
    pub fn save(&self, path: &Path) -> Result<(), CacheError> {
        let payload =
            bincode::serialize(self).map_err(|e| CacheError::Encode(format!("payload: {e}")))?;
        let payload_sha256: [u8; 32] = Sha256::digest(&payload).into();
        let envelope = CacheEnvelope {
            version: CACHE_FORMAT_VERSION,
            source_hash: self.source_hash,
            payload_sha256,
            payload,
        };
        let envelope_bytes = bincode::serialize(&envelope)
            .map_err(|e| CacheError::Encode(format!("envelope: {e}")))?;
        // On-disk layout: raw magic prefix (so torn writes that don't even
        // get past the first sector are visibly non-cache files), then the
        // bincode-encoded envelope.
        let mut bytes = Vec::with_capacity(CACHE_MAGIC.len() + envelope_bytes.len());
        bytes.extend_from_slice(CACHE_MAGIC);
        bytes.extend_from_slice(&envelope_bytes);

        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            // RT-I4 fix: only create the directory if it's missing, and
            // do NOT silently rewrite permissions on every save call.
            // The previous unconditional `set_permissions(parent, 0o700)`
            // overwrote any user-chosen mode (e.g., a shared 0o755 cache
            // group dir) on every cache write, which is operationally
            // surprising. If the directory exists, we honor whatever
            // mode the user chose. If we have to create it, we create
            // it with private 0o700 mode via DirBuilder.
            if !parent.exists() {
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
        }

        // Same-directory temp file → atomic rename.
        // RT-I1 fix: include thread id and a fresh nanosecond timestamp
        // in the tempfile name. The previous `.<file>.tmp.<pid>` pattern
        // collided when multiple threads in the same process called
        // `save()` for the same canonical path — one thread's `rename`
        // consumed the tempfile while another's was still writing,
        // surfacing as ENOENT or torn writes. Worker pools, async
        // runtimes, and Phase H's concurrent test workers would all
        // hit this. Adding tid + nanos makes the temp name unique
        // per-call.
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
            path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("ctx_cache"),
            std::process::id(),
            tid,
            nanos,
        );
        let tmp_path = path
            .parent()
            .map(|p| p.join(&tmp_name))
            .unwrap_or_else(|| PathBuf::from(&tmp_name));

        // Open + write + sync + close; a crash before sync is fine because
        // we never touch the canonical path until rename.
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
            // Best-effort cleanup of the orphan; ignore errors here.
            let _ = fs::remove_file(&tmp_path);
            CacheError::Io {
                op: "rename",
                path: path.to_path_buf(),
                source: e,
            }
        })?;
        Ok(())
    }

    /// Phase I — load `path` if its stored `source_hash` matches the hash
    /// freshly recomputed from `package_dir`. Returns:
    ///
    /// - `Ok(Some(ctx))` on a valid, fresh cache hit;
    /// - `Ok(None)` on a clean miss: the file does not exist, OR the
    ///   stored `source_hash` no longer matches the recomputed one;
    /// - `Err(CacheError::Corrupt | HashMismatch | UnsupportedVersion |
    ///   Decode | Io | Reef)` on any condition where silently using the
    ///   bytes would be wrong.
    ///
    /// **Important:** if the stored hash matches the recomputed hash, the
    /// envelope's `payload_sha256` is verified against the inner payload
    /// bytes BEFORE bincode-decoding the payload. A torn write that
    /// happens to deserialize as a valid envelope but whose payload was
    /// truncated (so the inner sha256 disagrees) is rejected as
    /// [`CacheError::Corrupt`], NOT silently accepted.
    ///
    /// `_reef_home` is currently unused; reserved for future invalidation
    /// signals (e.g., compiler-version pinning) that depend on reef-home
    /// state rather than just `package_dir`.
    pub fn load_if_fresh(
        path: &Path,
        _reef_home: &Path,
        package_dir: &Path,
    ) -> Result<Option<Self>, CacheError> {
        // Read the entire file into memory before any decode work — no
        // streaming-decode windows where a half-written tail looks like
        // a full envelope.
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

        // Reject empty / truncated-before-magic files as Corrupt — never None,
        // because Ok(None) means "valid cache miss" and a torn write must NOT
        // silently fall through to the recompile path without flagging.
        if bytes.is_empty() {
            return Err(CacheError::Corrupt("empty cache file".to_string()));
        }
        if bytes.len() < CACHE_MAGIC.len() || &bytes[..CACHE_MAGIC.len()] != CACHE_MAGIC {
            return Err(CacheError::Corrupt(
                "missing or wrong magic header".to_string(),
            ));
        }

        // Strip the raw magic prefix; the rest is the bincode envelope.
        let envelope_bytes = &bytes[CACHE_MAGIC.len()..];
        let envelope: CacheEnvelope = match bincode::deserialize(envelope_bytes) {
            Ok(env) => env,
            Err(e) => return Err(CacheError::Corrupt(format!("envelope decode: {e}"))),
        };

        if envelope.version != CACHE_FORMAT_VERSION {
            return Err(CacheError::UnsupportedVersion {
                stored: envelope.version,
                expected: CACHE_FORMAT_VERSION,
            });
        }

        // Recompute the source hash from the live package_dir. If the file
        // was named with a hash prefix that collides with a different
        // package, the recomputed hash will not match → cache miss.
        let live_graph = prepare_reef_graph(package_dir).map_err(CacheError::Reef)?;
        let live_digests = live_graph.source_digests().map_err(CacheError::Reef)?;
        let live_hash = ContextHash::from_digests(&live_digests);
        if envelope.source_hash != live_hash {
            return Ok(None);
        }

        // Verify the payload SHA-256 matches before paying bincode-decode
        // cost on the inner CompiledContext. A torn write whose envelope
        // happens to bincode-decode but whose payload was truncated is
        // caught here.
        let actual_payload_sha: [u8; 32] = Sha256::digest(&envelope.payload).into();
        if actual_payload_sha != envelope.payload_sha256 {
            return Err(CacheError::Corrupt(
                "payload sha256 does not match envelope".to_string(),
            ));
        }

        // Decode the inner CompiledContext.
        let ctx: CompiledContext = match bincode::deserialize(&envelope.payload) {
            Ok(c) => c,
            Err(e) => {
                return Err(CacheError::Decode(format!(
                    "CompiledContext decode (envelope/version match but inner shape changed): {e}"
                )));
            }
        };

        // Belt-and-braces: the inner CompiledContext must agree with the
        // outer envelope on `source_hash`. If it doesn't, something
        // mutated the bytes between encode/decode → treat as corrupt.
        if ctx.source_hash != envelope.source_hash {
            return Err(CacheError::HashMismatch {
                envelope: envelope.source_hash,
                inner: ctx.source_hash,
            });
        }

        Ok(Some(ctx))
    }

    /// Convenience for callers that only have a `reef_home` + `package_dir`
    /// and want the canonical cache location. Phase H wires `cmd_eval` and
    /// `cmd_check` through this helper; the disk-cache key is
    /// `<reef_home>/.cache/compiled/<pkg_name>-<pkg_version>-<hash16>.ctx`.
    ///
    /// The hash prefix is 16 hex chars (8 bytes of the full hash). Full-hash
    /// verification still happens inside `load_if_fresh`, so a prefix
    /// collision on the path is recoverable (returns `Ok(None)`, not silent
    /// hit).
    pub fn cache_path_for(
        reef_home: &Path,
        package_id: (&str, &str),
        source_hash: ContextHash,
    ) -> PathBuf {
        let (name, version) = package_id;
        let prefix_hex = hex_prefix(&source_hash.0, 8);
        // Sanitize to keep the filename POSIX-friendly across odd package
        // names (reef enforces a stricter rule, but we don't trust it here).
        let safe_name = sanitize_path_component(name);
        let safe_version = sanitize_path_component(version);
        reef_home
            .join(".cache")
            .join("compiled")
            .join(format!("{safe_name}-{safe_version}-{prefix_hex}.ctx"))
    }

    /// Read-only borrow of the underlying [`PreparedReefGraph`]. Phase H
    /// `chelis test` workers use this to drive the legacy
    /// `compile_with_reef_graph` + `prepare_eval` evaluation path
    /// without paying for a per-worker `prepare_reef_graph` walk: the
    /// parent has already linked the library decls + lockfile state
    /// once and serialized them into the context, and decoded workers
    /// share the same in-memory snapshot.
    ///
    /// Until the in-context Phase G/G' evaluator (host-runtime + linearity)
    /// reaches feature parity with the monolithic evaluator on the
    /// chelis-std test corpus, this accessor is the safe correctness
    /// path for `chelis test`. The performance win still comes from
    /// skipping the per-worker reef walk.
    pub fn reef_state(&self) -> &PreparedReefGraph {
        &self.reef_state
    }
}

/// Phase K disk-cache wire-up: probe the on-disk cache for a fresh
/// [`CompiledContext`] for `package_dir`, falling back to a full
/// `compile_reef_context` build (with side-effect: save to disk) on miss.
///
/// The cache key is derived from `(reef_home, root_package_id, source_hash)`
/// where `source_hash` is the hash of every source file backing the
/// resolved package graph. The probe walks `prepare_reef_graph +
/// source_digests` first to compute the hash, which is the same work
/// `compile_reef_context` does — but the rest of the library compile
/// (desugar, macro expand, type/effects/linearity check, lower) is skipped
/// on a hit. On a Coral-shape package this drops 67s cold to ~5s warm
/// (just the hash probe + bincode decode).
///
/// `verbose_corruption_to_stderr` controls one piece of operator-facing
/// behavior: when a cache file exists but is corrupt / version-skewed /
/// hash-mismatched, the helper logs to stderr and treats it as a miss
/// (recompile + overwrite) rather than aborting. Set to `false` for
/// tests that want silence.
///
/// The fallback path is `Ok` even if the post-compile `save` fails (the
/// compile itself succeeded; surface a stderr warning and continue with
/// the in-memory context). The next invocation will retry the save.
pub fn load_or_compile_for_package(
    reef_home: &Path,
    package_dir: &Path,
    verbose_corruption_to_stderr: bool,
) -> Result<CompiledContext, CompilerError> {
    // Phase K guardrail: if `reef_home` is empty (caller unset
    // `CHELIS_REEF_HOME`), the disk cache would land at a relative
    // `.cache/compiled/...` path in CWD — leaking artifacts into the
    // user's working tree. Bypass the cache entirely in that case and
    // delegate to a pure `compile_reef_context` build. The user opts
    // in to the disk cache by setting `CHELIS_REEF_HOME`.
    if reef_home.as_os_str().is_empty() {
        return compile_reef_context(reef_home, package_dir);
    }
    // Step 1: walk the reef graph + hash every source file. This is the
    // mandatory pre-work for both the cache probe AND a full compile, so
    // we always pay it. On Coral-shape packages this is ~5s; the savings
    // come from skipping the rest of `compile_reef_context` on a hit.
    let live_graph = match prepare_reef_graph(package_dir) {
        Ok(g) => g,
        Err(e) => return Err(reef_error(&e)),
    };
    let live_digests = match live_graph.source_digests() {
        Ok(d) => d,
        Err(e) => {
            // LocalRegistry packages don't have source_digests support
            // yet. Surface the same `hash_error` shape `compile_reef_context`
            // would surface so the CLI's existing LocalRegistry-detection
            // fallback continues to work unchanged.
            return Err(hash_error(&e));
        }
    };
    let source_hash = ContextHash::from_digests(&live_digests);
    let (root_name, root_version) = live_graph.root_package_id();
    let cache_path =
        CompiledContext::cache_path_for(reef_home, (root_name, root_version), source_hash);

    // Step 2: probe the disk cache. A clean miss (Ok(None)) is fine.
    // Corrupt / version-skewed / hash-mismatched files fall through to a
    // full compile + overwrite, with a stderr breadcrumb for the operator.
    match CompiledContext::load_if_fresh(&cache_path, reef_home, package_dir) {
        Ok(Some(ctx)) => return Ok(ctx),
        Ok(None) => {}
        Err(e) => {
            if verbose_corruption_to_stderr {
                eprintln!(
                    "chelis: disk cache at {} unusable ({e}); recompiling and overwriting",
                    cache_path.display()
                );
            }
        }
    }

    // Step 3: cache miss. Run the full compile and save the result.
    // The save is best-effort — if it fails, the compile result is still
    // usable for this invocation; only the next invocation pays the cold
    // cost again.
    let ctx = compile_reef_context(reef_home, package_dir)?;
    if let Err(e) = ctx.save(&cache_path)
        && verbose_corruption_to_stderr
    {
        eprintln!(
            "chelis: warning: failed to save compiled context cache to {}: {e}",
            cache_path.display()
        );
    }
    Ok(ctx)
}

/// Magic header bytes for the Phase I disk-cache file format.
/// Trailing newline guards against accidental concatenation with another
/// file (e.g., a misuse that piped two cache files together).
const CACHE_MAGIC: &[u8] = b"CHELIS_CTX_V1\n";

/// On-disk format version for the cache envelope. Bumping this tells
/// `load_if_fresh` to reject older cache files with
/// [`CacheError::UnsupportedVersion`] rather than risk a "successful but
/// wrong" decode.
const CACHE_FORMAT_VERSION: u32 = 1;

/// On-disk envelope for the Phase I cache. The full file layout is:
///
/// ```text
/// [CACHE_MAGIC bytes][bincode-encoded CacheEnvelope]
/// ```
///
/// The raw magic prefix (sitting BEFORE the bincode region) makes the
/// "is this even a cache file" check robust against bincode's leading
/// length prefix on fields like `Vec<u8>`. The envelope itself carries
/// the format version, an outer copy of `source_hash` (cheap stale-
/// check), and a SHA-256 of the inner payload bytes (torn-write
/// detection).
#[derive(Serialize, Deserialize)]
struct CacheEnvelope {
    /// On-disk format version. Future schema changes bump this to force
    /// `load_if_fresh` to reject the file with `UnsupportedVersion`
    /// instead of risking a "decoded but wrong" payload.
    version: u32,
    /// A copy of the inner `CompiledContext::source_hash`. Held outside
    /// the inner payload so a fresh-check can be done without paying the
    /// bincode-decode cost on the full `CompiledContext`.
    source_hash: ContextHash,
    /// SHA-256 of the inner payload bytes. Catches torn writes whose
    /// truncated payload still bincode-decodes successfully.
    payload_sha256: [u8; 32],
    /// Bincode-encoded `CompiledContext` body.
    payload: Vec<u8>,
}

/// Errors from the Phase I disk cache. Distinct from [`CompilerError`]
/// because the cache layer's failure modes (corrupt files, version
/// skew, IO errors) are categorically different from compile failures
/// and benefit from separate match arms in callers.
#[derive(Debug)]
pub enum CacheError {
    /// I/O failure while reading or writing a cache file.
    Io {
        op: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    /// `bincode` failure on the *encode* side. Should be impossible in
    /// practice for well-formed `CompiledContext` values, but surfaced
    /// for completeness.
    Encode(String),
    /// `bincode` failure on the *decode* side after the envelope and
    /// version checks passed. Indicates the inner `CompiledContext`
    /// shape changed without a version bump — treat as a bug + miss.
    Decode(String),
    /// File contents are not a valid cache envelope: missing magic,
    /// bincode error during envelope decode, or payload SHA-256
    /// mismatch (torn write). Caller MUST NOT silently accept the bytes.
    Corrupt(String),
    /// Envelope decoded but the on-disk format version is not the one
    /// the running binary supports.
    UnsupportedVersion { stored: u32, expected: u32 },
    /// Outer envelope's `source_hash` and the inner `CompiledContext`'s
    /// `source_hash` disagree — bytes were tampered with between encode
    /// and decode.
    HashMismatch {
        envelope: ContextHash,
        inner: ContextHash,
    },
    /// `prepare_reef_graph` or `source_digests` failed while
    /// recomputing the live source hash for invalidation. The string
    /// is whatever `chelis_reef` returned.
    Reef(String),
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
                "cache file format version {stored} not supported by this binary (expects {expected})"
            ),
            CacheError::HashMismatch { envelope, inner } => write!(
                f,
                "cache hash mismatch: envelope={} inner={}",
                hex_prefix(&envelope.0, 32),
                hex_prefix(&inner.0, 32)
            ),
            CacheError::Reef(msg) => write!(f, "cache invalidation reef error: {msg}"),
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

/// Lower-case hex of the first `n` bytes of `data`. Local impl to avoid
/// pulling in the `hex` crate for one call site.
fn hex_prefix(data: &[u8], n: usize) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let n = n.min(data.len());
    let mut out = String::with_capacity(n * 2);
    for &b in &data[..n] {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

/// Replace anything outside `[A-Za-z0-9._-]` with `_` so a hostile or
/// surprising package name cannot escape the cache directory or hit
/// reserved characters on Windows-style filesystems.
fn sanitize_path_component(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push('_');
    }
    out
}

/// Build a `CompiledContext` from a reef package directory.
///
/// Pipeline:
/// 1. `prepare_reef_graph` — resolve the package graph, link library
///    decls, hash the source files.
/// 2. Surf-desugar + macro-expand the linked library decls into Deep.
/// 3. Build a Phase 0e [`TypeEnv`] over the library (Phase C).
/// 4. Run the monolithic Phase 0e checker, the effects checker, and the
///    linearity checker over the library to produce a
///    [`CheckedProgram`] (Phases C/D/E feed into this composite library
///    snapshot — the `_with_context` callers receive this as their
///    "library context" argument).
/// 5. Lower the library to a [`LoweredLibrary`] (Phase F).
/// 6. Combine everything in a [`CompiledContext`].
///
/// `_reef_home` is currently unused; reserved for the Phase I disk-cache
/// key (the cache lives under `$CHELIS_REEF_HOME/.cache/compiled/...`).
pub fn compile_reef_context(
    _reef_home: &Path,
    package_dir: &Path,
) -> Result<CompiledContext, CompilerError> {
    // Phase K profile instrumentation: when `CHELIS_PROFILE_COMPILE_CONTEXT=1`
    // is set, emit per-phase wall-clock to stderr so the operator can see
    // which stage dominates. Off by default — zero cost on the hot path.
    let profile = std::env::var_os("CHELIS_PROFILE_COMPILE_CONTEXT")
        .map(|v| v == "1")
        .unwrap_or(false);
    let mut t = std::time::Instant::now();
    let log_phase = |name: &str, t: &mut std::time::Instant| {
        if profile {
            let elapsed = t.elapsed();
            eprintln!(
                "compile_reef_context: {:>32} {:>8.3}s",
                name,
                elapsed.as_secs_f64()
            );
            *t = std::time::Instant::now();
        }
    };

    let reef_state = prepare_reef_graph(package_dir).map_err(|e| reef_error(&e))?;
    log_phase("prepare_reef_graph", &mut t);
    let digests = reef_state.source_digests().map_err(|e| hash_error(&e))?;
    log_phase("source_digests", &mut t);
    let source_hash = ContextHash::from_digests(&digests);
    log_phase("hash_digests", &mut t);

    // Surf → Deep desugar + macro expand of the library decls.
    // `linked_library_decls` is already linked + internal-name-rewritten
    // by `prepare_reef_graph`.
    let desugared = chelis_surf::desugar::desugar_program(&reef_state.linked_library_decls);
    log_phase("surf_desugar", &mut t);
    let deep_library_decls =
        chelis_macros::expand_program(&desugared, &chelis_macros::ExpansionOptions::default())
            .map_err(|err| stage_error("desugar", err.to_string(), "macro_error"))?
            .into_exprs();
    log_phase("macro_expand", &mut t);

    if profile {
        let (modules, decls) = library_structural_summary(&deep_library_decls);
        eprintln!(
            "compile_reef_context: structural_summary modules={} top_level_decls={}",
            modules, decls
        );
    }

    // Phase C+0e: build the type-env snapshot AND the library
    // `CheckedProgram` in a single pass. The unified helper
    // `build_compiled_library_context` runs the per-decl HM
    // inference + per-decl annotation ONCE and returns both the
    // [`TypeEnv`] (for downstream `_with_context` calls) and the
    // [`CheckedProgram`] (for effects + linearity + lowering).
    //
    // The previous code path called `build_type_env_from_library`
    // and `check_phase0e_with_context(&TypeEnv::empty(), library)`
    // sequentially; both ran a full inference + annotation pass,
    // duplicating ~16s of work on Coral.
    let (type_env, checked) =
        build_compiled_library_context(&deep_library_decls).map_err(|report| CompilerError {
            stage: "check".to_string(),
            errors: report.errors.iter().map(check_error_diagnostic).collect(),
        })?;
    log_phase("build_compiled_library_context", &mut t);
    let checked = chelis_effects::check_program(&checked).map_err(|errors| CompilerError {
        stage: "effects".to_string(),
        errors: errors
            .iter()
            .map(|error| Diagnostic {
                kind: "effect_error".to_string(),
                message: error.message.clone(),
                severity: 0.8,
                expected: None,
                got: None,
                suggestions: vec![],
                span: None,
            })
            .collect(),
    })?;
    log_phase("check_effects", &mut t);
    let library_checked = check_linearity(&checked).map_err(|errors| CompilerError {
        stage: "linearity".to_string(),
        errors: errors.iter().map(check_error_diagnostic).collect(),
    })?;
    log_phase("check_linearity", &mut t);

    // Phase F: lower the library to a `LoweredLibrary` carrier.
    let library_dag = lower_program_to_library(&library_checked);
    log_phase("lower_program_to_library", &mut t);

    Ok(CompiledContext {
        source_hash,
        reef_state,
        type_env,
        library_checked,
        library_dag,
    })
}

/// Profile-only: count modules and top-level decls in a library expr
/// list. Used for the `CHELIS_PROFILE_COMPILE_CONTEXT=1` structural
/// summary. Cheap O(N) walk; not on the hot path.
fn library_structural_summary(exprs: &[chelis_deep::ast::Expr]) -> (usize, usize) {
    let mut modules = 0usize;
    let mut decls = 0usize;
    for expr in exprs {
        let chelis_deep::ast::Expr::List(list, _) = expr else {
            continue;
        };
        // Match the `top_level_decl_items` walk: descend through
        // `(module {} name children...)`.
        let tag = list
            .elements
            .first()
            .and_then(|e| match e {
                chelis_deep::ast::Expr::Atom(chelis_deep::ast::Atom::Symbol(s), _) => {
                    Some(s.as_str())
                }
                _ => None,
            })
            .unwrap_or("");
        if tag == "module" {
            modules += 1;
            for child in list.elements.iter().skip(3) {
                if let chelis_deep::ast::Expr::List(child_list, _) = child {
                    let child_tag = child_list
                        .elements
                        .first()
                        .and_then(|e| match e {
                            chelis_deep::ast::Expr::Atom(chelis_deep::ast::Atom::Symbol(s), _) => {
                                Some(s.as_str())
                            }
                            _ => None,
                        })
                        .unwrap_or("");
                    if matches!(child_tag, "def" | "defsig" | "deftype" | "typealias") {
                        decls += 1;
                    }
                }
            }
        } else if matches!(tag, "def" | "defsig" | "deftype" | "typealias") {
            decls += 1;
        }
    }
    (modules, decls)
}

fn reef_error(msg: &str) -> CompilerError {
    let kind = if msg.contains("reef.toml") {
        "package_not_found"
    } else if msg.contains("lockfile") {
        "lockfile_error"
    } else {
        "reef_error"
    };
    CompilerError {
        stage: "compile_reef_context".to_string(),
        errors: vec![Diagnostic {
            kind: kind.to_string(),
            message: msg.to_string(),
            severity: 0.8,
            expected: None,
            got: None,
            suggestions: vec![],
            span: None,
        }],
    }
}

fn hash_error(msg: &str) -> CompilerError {
    CompilerError {
        stage: "compile_reef_context".to_string(),
        errors: vec![Diagnostic {
            kind: "hash_error".to_string(),
            message: msg.to_string(),
            severity: 0.8,
            expected: None,
            got: None,
            suggestions: vec![],
            span: None,
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    /// Mirrors the chelis-reef `shared_graph_fixture` shape: a root
    /// package with one `Path` dep called `mylib`. The path-dep is
    /// load-bearing for tests #2 and #3 — they exercise the cross-
    /// package walk inside `source_digests`.
    fn path_dep_fixture() -> (TempDir, PathBuf) {
        let dir = TempDir::new().expect("tempdir");
        let root = dir.path().join("myapp");
        fs::create_dir_all(root.join("src")).expect("mkdir src");
        fs::create_dir_all(root.join("mylib/src")).expect("mkdir mylib/src");
        fs::write(
            root.join("reef.toml"),
            "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\ncompiler = \"=0.2.7\"\nmodule_prefix = \"App\"\n\n[dependencies]\nmylib = { path = \"./mylib\" }\n",
        )
        .expect("write app reef.toml");
        fs::write(
            root.join("src/main.ch"),
            "module App.Main\n\ndef main_value -> int32 = cast(7, int32)\n",
        )
        .expect("write main.ch");
        fs::write(
            root.join("mylib/reef.toml"),
            "[package]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"=0.2.7\"\nmodule_prefix = \"Mylib\"\n",
        )
        .expect("write mylib reef.toml");
        fs::write(
            root.join("mylib/src/math.ch"),
            "module Mylib.Math\nexport (add)\n\ndef add(x: int32, y: int32) -> int32 = cast(0, int32)\n",
        )
        .expect("write math.ch");
        // Hand-write a minimal reef.lock so prepare_reef_graph hits
        // the lockfile fast path and doesn't try to resolve over the
        // network.
        fs::write(
            root.join("reef.lock"),
            "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\n\n[[dependencies]]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"=0.2.7\"\narchive_sha256 = \"\"\nshell_sha256 = \"\"\n\n[dependencies.source]\nkind = \"path\"\npath = \"./mylib\"\n",
        )
        .expect("write reef.lock");
        (dir, root)
    }

    #[test]
    fn compile_reef_context_succeeds_on_path_dep_fixture() {
        let (_dir, root) = path_dep_fixture();
        let ctx = compile_reef_context(Path::new("/tmp/reef_home_unused"), &root)
            .expect("compile_reef_context succeeds");
        // Hash must not be all-zeros — that would mean either no sources
        // were hashed or every source was empty.
        assert_ne!(ctx.source_hash.0, [0u8; 32]);
    }

    #[test]
    fn compile_reef_context_hash_is_stable_across_repeat_calls() {
        let (_dir, root) = path_dep_fixture();
        let ctx1 = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx1");
        let ctx2 = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx2");
        assert_eq!(
            ctx1.source_hash, ctx2.source_hash,
            "repeat calls on unchanged sources must produce identical hashes"
        );
    }

    #[test]
    fn compile_reef_context_hash_changes_when_path_dep_source_changes() {
        let (_dir, root) = path_dep_fixture();
        let ctx_before = compile_reef_context(Path::new("/tmp/x"), &root).expect("before");
        // Edit the path-dep file, not the root, to exercise the
        // cross-package walk inside source_digests.
        let math_path = root.join("mylib/src/math.ch");
        let mut math_src = fs::read_to_string(&math_path).expect("read math.ch");
        math_src.push_str("-- a comment that changes file content\n");
        fs::write(&math_path, math_src).expect("rewrite math.ch");
        let ctx_after = compile_reef_context(Path::new("/tmp/x"), &root).expect("after");
        assert_ne!(
            ctx_before.source_hash, ctx_after.source_hash,
            "editing a path-dep source file must invalidate the hash"
        );
    }

    #[test]
    fn context_round_trips_through_bincode() {
        let (_dir, root) = path_dep_fixture();
        let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");
        let bytes = ctx.encode().expect("encode");
        let restored = CompiledContext::decode(&bytes).expect("decode");
        assert_eq!(ctx.source_hash, restored.source_hash);
        assert_eq!(
            ctx.reef_state.package_root,
            restored.reef_state.package_root
        );
    }
}
