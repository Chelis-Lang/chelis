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
//!   builds an IR [`TypeEnv`], then runs the monolithic IR
//!   checker, the effects checker, and the linearity checker on the
//!   library, and lowers it to a [`LoweredLibrary`].
//! - `eval_in_context` / `check_in_context` / `eval_many_in_context`
//!   (in `compiler.rs`): take a `CompiledContext` plus new source. Only
//!   the new source is re-compiled; the C/D/E/F `_with_context` variants
//!   stack the new code on top of the cached library state.
//!
//! ## Checker-state deserialization boundary
//!
//! `CompiledContext` contains only checker-success artifacts: construction
//! fails before a `TypeEnv`/`CheckedProgram` is returned when diagnostics are
//! non-empty, and the checker totality invariant forbids `Type::Error` in a
//! successful result. Decode verifies the cache format/build identity and
//! payload integrity, but it does not rerun semantic checking. The ADT
//! registry's provisional `TypeResolutionEnv` is serde-skipped; a later
//! stacked check reconstructs it from validated ADT/alias definitions plus
//! that check unit's declarations. Rejected declaration headers therefore
//! cannot persist in either the reef context or the stdlib sub-context.
//!
//! See `/home/jeff/.claude/plans/now-plan-out-the-shimmying-wand.md`
//! for the full plan.

use chelis_deep::DeepTag;
use chelis_ir::lower::LoweredLibrary;
use chelis_reef::{PreparedReefGraph, SourceDigest, prepare_reef_graph_cached};
use chelis_types::{CheckedProgram, TypeEnv, build_compiled_library_context};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::compiler::{CompilerError, bail_if_cancelled, cancelled_or, check_error_diagnostic};
use crate::schema::{Diagnostic, GeneralKind};

/// 32-byte content hash of every source file that contributed to a
/// `CompiledContext`. Phase I disk cache keys on this for invalidation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextHash(pub [u8; 32]);

/// Package + build identity for a `CompiledContext` cache entry.
///
/// `source_hash` is a pure *content* check: two on-disk packages that
/// share `(package_name, package_version)` and byte-identical source
/// files hash identically. That is not enough to key the disk cache.
///
/// - Two distinct package checkouts (different `package_root`) with the
///   same name+version+source bytes would otherwise collide on one
///   cache file. The second package would silently load the first's
///   `CompiledContext` — including its `package_root` — which is a
///   wrong-result bug for `chelis eval --file` and a hard error for the
///   `chelis test` worker's `package_root` guard.
/// - A `chelis` binary built from different compiler source but the same
///   resolved package would otherwise read an older binary's cached
///   context, applying stale compiler semantics to fresh input.
///
/// `CacheIdentity` folds both into the cache file name AND into the
/// `load_if_fresh` freshness check, so a mismatched-identity entry is a
/// clean miss (recompile), never a stale hit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheIdentity {
    /// Canonicalized on-disk location of the root package. Two distinct
    /// checkouts have distinct roots even with identical source bytes.
    /// Stored as a string (lossy) so the identity round-trips through
    /// bincode on every platform.
    pub package_root: String,
    /// The compiler crate version (`COMPILER_VERSION`). A binary built
    /// from different compiler source must not read an older binary's
    /// cached context.
    pub compiler_version: String,
}

impl CacheIdentity {
    /// Build a `CacheIdentity` from a resolved package root. The path is
    /// canonicalized so equivalent paths (symlinks, `.`/`..` segments,
    /// trailing slashes) resolve to one identity; if canonicalization
    /// fails (path removed mid-build, permission error), the raw path is
    /// used so the identity is still distinct rather than empty.
    pub fn for_package_root(package_root: &Path) -> Self {
        let canonical = fs::canonicalize(package_root)
            .unwrap_or_else(|_| package_root.to_path_buf())
            .to_string_lossy()
            .into_owned();
        CacheIdentity {
            package_root: canonical,
            compiler_version: crate::COMPILER_VERSION.to_string(),
        }
    }

    /// 16-hex-char fingerprint of this identity. Folded into the cache
    /// file name so distinct identities never share a cache path.
    fn fingerprint_hex(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update((self.package_root.len() as u64).to_le_bytes());
        hasher.update(self.package_root.as_bytes());
        hasher.update((self.compiler_version.len() as u64).to_le_bytes());
        hasher.update(self.compiler_version.as_bytes());
        let bytes: [u8; 32] = hasher.finalize().into();
        hex_prefix(&bytes, 8)
    }
}

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
/// - `type_env`: IR type-checker snapshot. Used by
///   `check_ir_with_context` for new-code type checking.
/// - `library_checked`: monolithic IR + effects + linearity result
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
    /// Package + build identity (canonical `package_root` + compiler
    /// version). `source_hash` is a content check only; `identity`
    /// disambiguates two distinct checkouts that share name+version+
    /// source bytes, and pins the compiler build. `load_if_fresh`
    /// rejects an entry whose stored `identity` does not match the
    /// recomputed one as a clean miss.
    pub identity: CacheIdentity,
    /// The reef state (lockfile-backed package graph + linked library
    /// decls + internal-name maps + dep shells).
    pub(crate) reef_state: PreparedReefGraph,
    /// IR type-checker snapshot — the outer scope for new-code
    /// type checking via `check_ir_with_context`.
    pub(crate) type_env: TypeEnv,
    /// Library IR + effects + linearity result. Feeds the
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
            identity: self.identity.clone(),
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
    /// signals that depend on reef-home state rather than just
    /// `package_dir`.
    ///
    /// **Identity check:** beyond the `source_hash` content check, the
    /// stored `CacheIdentity` (canonical `package_root` + compiler
    /// version) is recomputed from the live `package_dir` and the
    /// running binary. A mismatch — a different on-disk checkout that
    /// shares name+version+source bytes, or a cache file written by a
    /// differently-built `chelis` binary — is treated as a clean miss
    /// (`Ok(None)`), never a stale hit.
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
        let live_graph = prepare_reef_graph_cached(package_dir).map_err(CacheError::Reef)?;
        let live_digests = live_graph.source_digests().map_err(CacheError::Reef)?;
        let live_hash = ContextHash::from_digests(&live_digests);
        if envelope.source_hash != live_hash {
            return Ok(None);
        }

        // Recompute the package + build identity from the live graph and
        // the running binary. `source_hash` is a content check only: two
        // distinct checkouts with identical name+version+source bytes
        // hash the same, and a differently-built binary produces the same
        // content hash for the same sources. A stored identity that does
        // not match the live one is a clean miss, not a stale hit.
        let live_identity = CacheIdentity::for_package_root(&live_graph.package_root);
        if envelope.identity != live_identity {
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

        // Same belt-and-braces check for the identity: the inner
        // CompiledContext's `identity` must agree with the outer
        // envelope's copy. A disagreement means the bytes were mutated
        // between encode and decode → treat as corrupt.
        if ctx.identity != envelope.identity {
            return Err(CacheError::IdentityMismatch {
                envelope: envelope.identity.clone(),
                inner: ctx.identity.clone(),
            });
        }

        Ok(Some(ctx))
    }

    /// Convenience for callers that only have a `reef_home` + `package_dir`
    /// and want the canonical cache location. Phase H wires `cmd_eval` and
    /// `cmd_check` through this helper; the disk-cache key is
    /// `<reef_home>/.cache/compiled/<pkg_name>-<pkg_version>-<src16>-<id16>.ctx`.
    ///
    /// `src16` is 16 hex chars of the source-content hash; `id16` is 16
    /// hex chars of the package + build identity (canonical
    /// `package_root` plus compiler version). Folding identity into the
    /// file name keeps two distinct checkouts that share
    /// name+version+source bytes on SEPARATE cache files. Full
    /// verification of both still happens inside `load_if_fresh`, so a
    /// prefix collision on the path is recoverable (returns `Ok(None)`,
    /// not a silent hit).
    pub fn cache_path_for(
        reef_home: &Path,
        package_id: (&str, &str),
        source_hash: ContextHash,
        identity: &CacheIdentity,
    ) -> PathBuf {
        reef_home
            .join(".cache")
            .join("compiled")
            .join(Self::cache_file_name(package_id, source_hash, identity))
    }

    /// The bare `<name>-<version>-<src16>-<id16>.ctx` cache file name for
    /// a package. Split out from [`cache_path_for`] so callers that
    /// resolve the cache directory via the XDG-fallback helper
    /// (`stdlib_cache::cache_dir_for`) can join the same file name onto
    /// it.
    pub fn cache_file_name(
        package_id: (&str, &str),
        source_hash: ContextHash,
        identity: &CacheIdentity,
    ) -> String {
        let (name, version) = package_id;
        let src_hex = hex_prefix(&source_hash.0, 8);
        let id_hex = identity.fingerprint_hex();
        // Sanitize to keep the filename POSIX-friendly across odd package
        // names (reef enforces a stricter rule, but we don't trust it here).
        let safe_name = sanitize_path_component(name);
        let safe_version = sanitize_path_component(version);
        format!("{safe_name}-{safe_version}-{src_hex}-{id_hex}.ctx")
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
    // Resolve the compiled-context cache directory. When `CHELIS_REEF_HOME`
    // is set, `reef_home` is non-empty and the cache lives at
    // `<reef_home>/.cache/compiled/`. When it is unset, `reef_home` is
    // empty and a bare `<reef_home>/.cache/compiled/...` join would land
    // at a relative path in CWD, leaking artifacts into the working tree
    // — so resolve the XDG fallback (`$XDG_CACHE_HOME/chelis/compiled/`
    // -> `~/.cache/chelis/compiled/`) instead. This is the same fallback
    // the chelis-std typecheck cache uses; it un-gates the disk cache for
    // the no-`CHELIS_REEF_HOME` test-worker workload. Only when none of
    // the three roots resolves do we fall through to an uncached compile.
    let cache_dir = if reef_home.as_os_str().is_empty() {
        match crate::stdlib_cache::cache_dir_for("compiled") {
            Some(dir) => dir,
            None => return compile_reef_context(reef_home, package_dir),
        }
    } else {
        reef_home.join(".cache").join("compiled")
    };
    // Step 1: walk the reef graph + hash every source file. This is the
    // mandatory pre-work for both the cache probe AND a full compile, so
    // we always pay it. On Coral-shape packages this is ~5s; the savings
    // come from skipping the rest of `compile_reef_context` on a hit.
    let live_graph = match prepare_reef_graph_cached(package_dir) {
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
    // Package + build identity disambiguates two distinct on-disk
    // checkouts that share name+version+source bytes (which `source_hash`
    // alone cannot) and pins the compiler build. Folded into the cache
    // file name AND re-verified inside `load_if_fresh`.
    let identity = CacheIdentity::for_package_root(&live_graph.package_root);
    let (root_name, root_version) = live_graph.root_package_id();
    let cache_path = cache_dir.join(CompiledContext::cache_file_name(
        (root_name, root_version),
        source_hash,
        &identity,
    ));

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

/// Which route [`load_or_compile_with_local_registry_fallback`] took to
/// produce its context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextLoadPath {
    /// The disk-cache-aware [`load_or_compile_for_package`] succeeded
    /// (cache hit or a cache-miss compile+save).
    Cached,
    /// [`load_or_compile_for_package`] failed with the LocalRegistry
    /// source-hash gap and the uncached [`compile_reef_context`] fallback
    /// succeeded instead.
    LocalRegistryFallback,
}

/// [`load_or_compile_for_package`] with the LocalRegistry hash-gap fallback
/// folded in — the single home for the `hash_error`/`"LocalRegistry"`
/// detection string-match (chelis#822 review, Fix D).
///
/// A dependency resolved from the `LocalRegistry` (e.g. `chelis-std`) cannot
/// be source-hashed yet, so the disk-cache probe errors with a `hash_error`
/// diagnostic naming `LocalRegistry`. That is not a real failure: fall back
/// to an uncached in-memory [`compile_reef_context`], which handles the
/// LocalRegistry case (digests = `None`). Any other error is a genuine
/// compile failure and propagates. The returned [`ContextLoadPath`] reports
/// which route produced the context.
///
/// Call sites: chelis-python's `load_reef_context` uses this today. The CLI
/// `chelis test` worker (`cmd_internal_test_file` /
/// `is_local_registry_hash_unsupported` in crates/chelis-cli/src/main.rs)
/// still carries its own copy of the detection + fallback and should migrate
/// here when the #830 `build --in-context` work lands, rather than growing a
/// third copy. (The CLI *eval* site deliberately differs: on the hash gap it
/// drops to the legacy `prepare_eval` path, not to `compile_reef_context`.)
pub fn load_or_compile_with_local_registry_fallback(
    reef_home: &Path,
    package_dir: &Path,
    verbose_corruption_to_stderr: bool,
) -> Result<(CompiledContext, ContextLoadPath), CompilerError> {
    match load_or_compile_for_package(reef_home, package_dir, verbose_corruption_to_stderr) {
        Ok(context) => Ok((context, ContextLoadPath::Cached)),
        Err(err) if is_local_registry_hash_gap(&err) => {
            compile_reef_context(reef_home, package_dir)
                .map(|context| (context, ContextLoadPath::LocalRegistryFallback))
        }
        Err(err) => Err(err),
    }
}

/// The LocalRegistry hash-gap detection predicate: a `hash_error` diagnostic
/// whose message names `LocalRegistry`. Named (rather than inline) so the
/// string-match has unit tests locking it against wording drift in the
/// upstream diagnostic: a `hash_error` NOT naming LocalRegistry, or any other
/// diagnostic kind, must propagate as a genuine failure rather than trigger
/// the uncached-recompile fallback (#822 review round 3, finding 4).
fn is_local_registry_hash_gap(err: &CompilerError) -> bool {
    err.errors.iter().any(|d| {
        d.kind() == chelis_vocab::DiagnosticKind::HashError && d.message.contains("LocalRegistry")
    })
}

/// Magic header bytes for the Phase I disk-cache file format.
/// Trailing newline guards against accidental concatenation with another
/// file (e.g., a misuse that piped two cache files together).
/// V7: chelis#878 changed `RiscOp::Pad::fill` to the sealed dtype-tagged
/// scalar, and chelis#942 made deferred positional-expand constraints part of
/// the serialized checker context. Both alter cached bincode shapes; V6 files
/// are stale.
/// V8: chelis#1182 emits the root package's modules LAST in the linker
/// assembly, so the `reef_state` (`PreparedReefGraph`) decl order serialized
/// in this cache changed. Without the bump a V7 file written by a pre-#1182
/// binary would replay the old decl order to a post-#1182 binary at the same
/// compiler version -- the same stale-order hazard the
/// `PREPARED_GRAPH_CACHE_VERSION` bump closes for the graph cache.
const CACHE_MAGIC: &[u8] = b"CHELIS_CTX_V8\n";

/// On-disk format version for the cache envelope. Bumping this tells
/// `load_if_fresh` to reject older cache files with
/// [`CacheError::UnsupportedVersion`] rather than risk a "successful but
/// wrong" decode.
const CACHE_FORMAT_VERSION: u32 = 8;

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
    /// A copy of the inner `CompiledContext::identity`. Held outside the
    /// inner payload so the identity check (canonical `package_root` +
    /// compiler version) can run before the bincode-decode cost. A
    /// stored identity that does not match the live one is a clean miss.
    identity: CacheIdentity,
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
    /// Outer envelope's `identity` and the inner `CompiledContext`'s
    /// `identity` disagree — bytes were tampered with between encode and
    /// decode.
    IdentityMismatch {
        envelope: CacheIdentity,
        inner: CacheIdentity,
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
            CacheError::IdentityMismatch { envelope, inner } => write!(
                f,
                "cache identity mismatch: envelope={} inner={}",
                envelope.fingerprint_hex(),
                inner.fingerprint_hex()
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
/// 3. Build an IR [`TypeEnv`] over the library.
/// 4. Run the monolithic IR checker, the effects checker, and the
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
    // RFC v5 (RT-1 F2 bypass): the entire reef library is linker output
    // (internal-name-mangled), so the reserved linker-name rejection is
    // off for this whole context build.
    let _linked = chelis_types::install_linked_program_guard();
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

    // chelis#930: the library-context build is the dominant per-call front-end
    // cost for a reef package (chelis#828 measures ~4.5 minutes with a real
    // library context), so it polls the cancellation token at every phase
    // boundary. The per-declaration polling that makes those bounds useful
    // lives in `chelis-types`; `prepare_reef_graph_cached` itself is
    // filesystem work and is not covered.
    bail_if_cancelled("reef")?;
    let reef_state = prepare_reef_graph_cached(package_dir).map_err(|e| reef_error(&e))?;
    log_phase("prepare_reef_graph", &mut t);
    bail_if_cancelled("check")?;
    let digests = reef_state
        .source_digests()
        .map_err(|error| hash_error(&error))?;
    log_phase("source_digests", &mut t);
    let source_hash = ContextHash::from_digests(&digests);
    log_phase("hash_digests", &mut t);

    // Phase C+0e / chelis#451: build the `(TypeEnv, library CheckedProgram,
    // LoweredLibrary)` triple. The layered path reuses the cross-process
    // chelis-std typecheck cache so the chelis-std half of the library is
    // never re-walked here; only the package's own (non-chelis-std) decls
    // are desugared + inferred + checked. On any miss (cache disabled, no
    // chelis-std in the graph, or the non-chelis-std decls don't compose
    // cleanly) it falls back to the monolithic whole-library build below,
    // which is byte-identical. See `build_library_triple_layered`.
    //
    // The full-library Surf → Deep desugar + macro expand is deferred to
    // the monolithic fallback so the layered path does NOT re-desugar
    // chelis-std (the layered helper desugars only the package's own decls).
    let (type_env, library_checked) = match build_library_triple_layered(&reef_state, &mut t) {
        Some(Ok(triple)) => triple,
        Some(Err(err)) => return Err(err),
        None => {
            // chelis#930: the layered path folds ANY failure into `None` so
            // the monolithic path can produce the byte-identical diagnostic.
            // An abandoned compile must not take that route: it would rerun
            // the entire library check it just abandoned, so a cancelled
            // library-context build would cost MORE than an uncancelled one.
            bail_if_cancelled("check")?;
            // Surf → Deep desugar + macro expand of the WHOLE library.
            // `linked_library_decls` is already linked + internal-name-
            // rewritten by `prepare_reef_graph`.
            let prepared =
                crate::pipeline::prepare_surf_decls(&reef_state.linked_library_decls, None)
                    .map_err(|error| {
                        crate::compiler::pipeline_rejection_to_compiler_error(
                            crate::pipeline::PipelineRejection::Preparation(error),
                        )
                    })?;
            log_phase("surf_desugar_macro_expand", &mut t);

            if profile {
                let (modules, decls) = library_structural_summary(prepared.expanded_deep());
                eprintln!(
                    "compile_reef_context: structural_summary modules={} top_level_decls={}",
                    modules, decls
                );
            }

            // Monolithic fallback: build the type-env snapshot AND the
            // library `CheckedProgram` in a single inference session.
            let (type_env, checked) = build_compiled_library_context(prepared.expanded_deep())
                .map_err(|report| CompilerError {
                    stage: "check".to_string(),
                    errors: report.errors.iter().map(check_error_diagnostic).collect(),
                })
                .map_err(|error| cancelled_or("check", error))?;
            log_phase("build_compiled_library_context", &mut t);
            let analysis = crate::pipeline::prepared_analysis_from_checked(prepared, checked);
            bail_if_cancelled("effects")?;
            let checked = crate::pipeline::complete_checks(
                analysis,
                crate::pipeline::SemanticContext::Isolated,
            )
            .map_err(|rejection| {
                crate::compiler::pipeline_rejection_to_compiler_error(rejection.into())
            })
            .map_err(|error| cancelled_or("effects", error))?;
            bail_if_cancelled("linearity")?;
            log_phase("semantic_checks", &mut t);
            let (_, _, library_checked, _) = checked.into_parts();
            (type_env, library_checked)
        }
    };

    // Lower the (possibly layer-composed) whole-library `CheckedProgram` to
    // a `LoweredLibrary` carrier. Lowering is cheap relative to the
    // typecheck, and lowering the composed program is byte-identical to
    // lowering the monolithic one (the composed program carries the same
    // chelis-std ++ package annotated bodies), so the layered path does not
    // need to reuse the cached chelis-std `library_dag` here.
    bail_if_cancelled("lower")?;
    let library_dag = crate::pipeline::lower_library(&library_checked)
        .map_err(crate::compiler::pipeline_rejection_to_compiler_error)
        .map_err(|error| cancelled_or("lower", error))?;
    log_phase("lower_program_to_library", &mut t);

    // Package + build identity: canonical `package_root` from the
    // resolved reef graph plus the compiler version. Distinguishes two
    // distinct checkouts that share name+version+source bytes and pins
    // the compiler build into the cache key.
    let identity = CacheIdentity::for_package_root(&reef_state.package_root);

    Ok(CompiledContext {
        source_hash,
        identity,
        reef_state,
        type_env,
        library_checked,
        library_dag,
    })
}

/// chelis#451 — build the `(whole-library TypeEnv, whole-library
/// CheckedProgram)` half of a `CompiledContext` while reusing the
/// cross-process chelis-std typecheck cache, so the chelis-std library is
/// not re-walked here.
///
/// This is the same layered seam `check_layered_for_build` uses, lifted
/// into the `CompiledContext` shape. The chelis-std half of the library
/// is the dominant per-process cost (`build_compiled_library_context` is
/// ~0.7s for a small package and grows with stdlib size); a sharded test
/// matrix or any package-source edit re-pays it on every fresh process
/// today because `compile_reef_context` ran the monolithic whole-library
/// build. The chelis-std sub-context is content-addressed on the linked
/// chelis-std decls + compiler version, so a warm hit is a disk read and
/// the result is byte-identical to a cold build.
///
/// ## Returns
/// - `None` — take the monolithic fallback. Fired when the cache is
///   disabled (`CHELIS_STDLIB_CACHE_DISABLE=1`), the graph has no
///   chelis-std package (nothing to amortize), or the non-chelis-std
///   decls do not compose cleanly against the cached sub-context (a type,
///   effect, linearity, or macro error). The monolithic path then
///   produces the byte-identical error report — exactly the error-path
///   handoff `check_layered` documents.
/// - `Some(Ok((type_env, library_checked)))` — the layered triple. The
///   `type_env` is the union of chelis-std + package declared types (so
///   `compile_new_source_in_context` resolves both layers); the
///   `library_checked` is `compose(stdlib_library_checked, package_checked)`
///   carrying chelis-std ++ package annotated bodies (so the
///   `_with_context` effect / linearity / lowering passes see the whole
///   library) — exactly the shape the monolithic build produces.
/// - `Some(Err(_))` — building the chelis-std sub-context itself failed (a
///   chelis-std regression). Surfaced, never swallowed into a fallback.
///
/// ## warm == cold
/// On a hit, the chelis-std sub-context loaded from disk is the same
/// artifact `build_stdlib_context` produces cold (content-addressed key);
/// the package half is checked the same way in both branches; and
/// `compose` is deterministic. So the composed `CheckedProgram` and
/// unioned `TypeEnv` equal what the monolithic
/// `build_compiled_library_context(whole_library)` path yields for every
/// `CompiledContext` consumer. The lowering step (run by the caller on
/// this composed program) is therefore byte-identical too.
fn build_library_triple_layered(
    reef_state: &PreparedReefGraph,
    t: &mut std::time::Instant,
) -> Option<Result<(TypeEnv, CheckedProgram), CompilerError>> {
    // Escape hatch: the disable seam routes the whole library build through
    // the monolithic path so the acceptance oracle's monolithic-vs-layered
    // comparison has a real monolithic leg, and so an operator can always
    // sidestep the cache.
    if crate::stdlib_cache::cache_disabled() {
        return None;
    }
    // No chelis-std in the graph (e.g. a package that depends on nothing):
    // there is nothing to amortize, so the monolithic path is both simpler
    // and avoids composing against an empty sub-context.
    if reef_state.linked_stdlib_decls.is_empty() {
        return None;
    }

    // Layer 1: the cached chelis-std sub-context. A miss builds + writes it
    // once; every later process reads it back. A build failure here is a
    // genuine chelis-std regression, surfaced rather than hidden behind the
    // monolithic fallback.
    let stdlib_ctx =
        match crate::stdlib_cache::load_or_build_stdlib_context(&reef_state.linked_stdlib_decls) {
            Ok(ctx) => ctx,
            Err(err) => return Some(Err(err)),
        };

    // Layer 2: desugar + macro-expand only the non-chelis-std library decls
    // (the package's own modules + non-stdlib path-deps). A macro-expansion
    // failure is a real front-end error the monolithic path also surfaces,
    // so hand back `None` for the byte-identical diagnostic.
    let prepared = match crate::pipeline::prepare_surf_decls(
        &reef_state.linked_non_stdlib_library_decls,
        None,
    ) {
        Ok(prepared) => prepared,
        Err(_) => return None,
    };

    // Type-check + annotate the non-chelis-std decls stacked on the cached
    // chelis-std sub-context. Returns the union `TypeEnv` and one type product.
    let (type_env, package_checked) = match chelis_types::build_compiled_library_context_with_base(
        &stdlib_ctx.type_env,
        prepared.expanded_deep(),
    ) {
        Ok(pair) => pair,
        Err(_) => return None,
    };
    let analysis = crate::pipeline::prepared_analysis_from_checked(prepared, package_checked);
    let package_checked = match crate::pipeline::complete_checks(
        analysis,
        crate::pipeline::SemanticContext::Library(&stdlib_ctx.library_checked),
    ) {
        Ok(checked) => checked,
        Err(crate::pipeline::SemanticRejection::Effects { .. })
        | Err(crate::pipeline::SemanticRejection::Linearity { .. }) => return None,
    };

    // Compose the cached chelis-std half with the freshly-checked package
    // half into the one whole-library `CheckedProgram` the rest of
    // `compile_reef_context` (and every `_with_context` consumer) expects.
    let library_checked =
        CheckedProgram::compose(&stdlib_ctx.library_checked, package_checked.program());

    if std::env::var_os("CHELIS_PROFILE_COMPILE_CONTEXT").map(|v| v == "1") == Some(true) {
        eprintln!(
            "compile_reef_context: {:>32} {:>8.3}s",
            "layered_stdlib_cached_check",
            t.elapsed().as_secs_f64()
        );
    }
    *t = std::time::Instant::now();
    Some(Ok((type_env, library_checked)))
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
        let tag = list.tag();
        if tag == Some(DeepTag::Module) {
            modules += 1;
            for child in list.elements.iter().skip(3) {
                if let chelis_deep::ast::Expr::List(child_list, _) = child
                    && matches!(
                        child_list.tag(),
                        Some(
                            DeepTag::Def | DeepTag::Defsig | DeepTag::Deftype | DeepTag::Typealias
                        )
                    )
                {
                    decls += 1;
                }
            }
        } else if matches!(
            tag,
            Some(DeepTag::Def | DeepTag::Defsig | DeepTag::Deftype | DeepTag::Typealias)
        ) {
            decls += 1;
        }
    }
    (modules, decls)
}

fn reef_error(msg: &str) -> CompilerError {
    CompilerError {
        stage: "compile_reef_context".to_string(),
        errors: vec![Diagnostic::general(GeneralKind::ReefError, msg, 0.8)],
    }
}

fn hash_error(msg: &str) -> CompilerError {
    CompilerError {
        stage: "compile_reef_context".to_string(),
        errors: vec![Diagnostic::general(GeneralKind::HashError, msg, 0.8)],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::stage_error;
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
            format!(
                "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\ncompiler = \"={ver}\"\nmodule_prefix = \"App\"\n\n[dependencies]\nmylib = {{ path = \"./mylib\" }}\n",
                ver = crate::COMPILER_VERSION,
            ),
        )
        .expect("write app reef.toml");
        fs::write(
            root.join("src/main.ch"),
            "module App.Main\n\ndef main_value() -> int32 = cast(7, int32)\n",
        )
        .expect("write main.ch");
        fs::write(
            root.join("mylib/reef.toml"),
            format!(
                "[package]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={ver}\"\nmodule_prefix = \"Mylib\"\n",
                ver = crate::COMPILER_VERSION,
            ),
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
            format!(
                "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\n\n[[dependencies]]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={ver}\"\narchive_sha256 = \"\"\nshell_sha256 = \"\"\n\n[dependencies.source]\nkind = \"path\"\npath = \"./mylib\"\n",
                ver = crate::COMPILER_VERSION,
            ),
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

    // #822 review round 3, finding 4: the LocalRegistry hash-gap detection is
    // a string match over the upstream diagnostic; these lock it in both
    // directions so wording drift cannot silently reroute genuine failures
    // into the uncached-recompile fallback (or vice versa).
    #[test]
    fn local_registry_hash_gap_predicate_matches_the_gap_shape() {
        let gap = stage_error(
            "context",
            "cannot source-hash dependency `chelis-std` resolved from the LocalRegistry",
            GeneralKind::HashError,
        );
        assert!(is_local_registry_hash_gap(&gap));
    }

    #[test]
    fn local_registry_hash_gap_predicate_rejects_other_failures() {
        // A hash_error about something else is a genuine failure.
        let other_hash = stage_error(
            "context",
            "content hash mismatch for src/lib.ch",
            GeneralKind::HashError,
        );
        assert!(!is_local_registry_hash_gap(&other_hash));
        // A non-hash diagnostic naming LocalRegistry is a genuine failure.
        let other_kind = stage_error(
            "context",
            "LocalRegistry package `chelis-std` failed to compile",
            GeneralKind::CompileError,
        );
        assert!(!is_local_registry_hash_gap(&other_kind));
    }

    #[test]
    fn reef_diagnostic_kind_does_not_depend_on_message_substrings() {
        for message in [
            "missing reef.toml",
            "malformed lockfile",
            "ordinary graph preparation failure",
        ] {
            let error = reef_error(message);
            assert_eq!(error.errors.len(), 1);
            assert_eq!(
                error.errors[0].kind(),
                chelis_vocab::DiagnosticKind::ReefError
            );
        }
    }
}
