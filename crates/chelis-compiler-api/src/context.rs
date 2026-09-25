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
//! `CompiledContext` contains only checker-success artifacts. Construction
//! rejects diagnostics before it creates a library proof. Decode verifies the
//! cache identity and payload integrity. It also checks the type-environment
//! relationship. It reruns the remaining semantic checks and the lower phase.
//! Then, it restores that proof. The provisional `TypeResolutionEnv` is serde-skipped.
//! A later stacked check reconstructs it from validated ADT and alias definitions.
//! Rejected declaration headers cannot persist in either compiler cache.
//!
//! See `/home/jeff/.claude/plans/now-plan-out-the-shimmying-wand.md`
//! for the full plan.

use chelis_deep::DeepTag;
use chelis_ir::lower::LoweredLibrary as IrLoweredLibrary;
use chelis_reef::{PreparedReefGraph, SourceDigest, prepare_reef_graph_cached};
use chelis_types::{CheckedProgram, TypeEnv};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::compiler::{CompilerError, bail_if_cancelled, cancelled_or};
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
    /// The compiler BUILD fingerprint (`build_fingerprint()`), not the
    /// bare crate version. A binary built from different compiler source
    /// must not read an older binary's cached context — and
    /// `COMPILER_VERSION` alone does not enforce that, because two builds
    /// from different commits share one `workspace.package.version` until
    /// the next release bump. Two such binaries can disagree about type
    /// semantics, so sharing a cache entry lets one check a program under
    /// the other's rules (chelis#1156).
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
            compiler_version: crate::build_fingerprint().to_string(),
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
/// - `library`: bound type environment and accepted program. Contextual
///   checks retain this proof through type, effect, and linearity stages.
/// - `library_dag`: lowered library DAG carrier (Phase F). Used by
///   `lower_program_with_context`.
///
/// Source artifacts are populated once by `compile_reef_context` and
/// thereafter treated as immutable. The evaluator memo is derived lazily
/// from the accepted source, never from serialized graph metadata.
/// Cheap to clone (the heavy state is
/// `Arc`-shared inside `TypeEnv`).
#[derive(Debug, Clone)]
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
    /// Bound type environment and semantically accepted library program.
    pub(crate) library: crate::pipeline::CheckedLibrary,
    /// Lowered library carrier. Feeds `lower_program_with_context`.
    pub(crate) library_dag: crate::pipeline::LoweredLibrary,
}

#[derive(Serialize, Deserialize)]
struct CompiledContextWire {
    source_hash: ContextHash,
    identity: CacheIdentity,
    reef_state: PreparedReefGraph,
    type_env: TypeEnv,
    library_checked: CheckedProgram,
    library_dag: IrLoweredLibrary,
}

impl Serialize for CompiledContext {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        CompiledContextWire {
            source_hash: self.source_hash,
            identity: self.identity.clone(),
            reef_state: self.reef_state.clone(),
            type_env: self.library.type_env().clone(),
            library_checked: self.library.program().clone(),
            library_dag: self.library_dag.raw().clone(),
        }
        .serialize(serializer)
    }
}

impl CompiledContextWire {
    /// Rebuild the producer's context from a payload the caller authenticated.
    ///
    /// The precondition is [`CompiledContext::decode_authenticated`]'s: the
    /// bytes were compared against a digest that did not travel with them. Under
    /// it, this reconstructs the producer's exact value. The transmitted
    /// `CheckedProgram` is already the output of the effect and linearity
    /// checkers, and rerunning them over it reproduces it byte for byte, so the
    /// reruns `Deserialize` performs would establish nothing here. The lowered
    /// library is re-derived rather than adopted, which keeps
    /// `chelis_pipeline_core::LoweredLibrary`'s rule that only `lower_library`
    /// can construct one, and leaves this route trusting a single transmitted
    /// artifact instead of two.
    fn into_authenticated_context(self) -> Result<CompiledContext, String> {
        let _linked = chelis_types::install_linked_program_guard();
        let library =
            chelis_pipeline_core::adopt_authenticated_library(self.type_env, self.library_checked)
                .map_err(|rejection| rejection.to_string())?;
        if self.library_dag.library_proof_id() != library.program().library_proof_id() {
            return Err("the lowered library does not match the checked library".to_string());
        }
        let library_dag = crate::pipeline::lower_library(&library).map_err(|e| e.to_string())?;
        Ok(CompiledContext {
            source_hash: self.source_hash,
            identity: self.identity,
            reef_state: self.reef_state,
            library,
            library_dag,
        })
    }
}

impl<'de> Deserialize<'de> for CompiledContext {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = CompiledContextWire::deserialize(deserializer)?;
        let _linked = chelis_types::install_linked_program_guard();
        let library =
            chelis_pipeline_core::validate_cached_library(wire.type_env, wire.library_checked)
                .map_err(serde::de::Error::custom)?;
        if wire.library_dag.library_proof_id() != library.program().library_proof_id() {
            return Err(serde::de::Error::custom(
                "the lowered library does not match the checked library",
            ));
        }
        let library_dag =
            crate::pipeline::lower_library(&library).map_err(serde::de::Error::custom)?;
        if !crate::cache_envelope::lowered_library_payload_matches(
            &wire.library_dag,
            library_dag.raw(),
        )
        .map_err(serde::de::Error::custom)?
        {
            return Err(serde::de::Error::custom(
                "the lowered library payload does not match the checked library",
            ));
        }
        Ok(Self {
            source_hash: wire.source_hash,
            identity: wire.identity,
            reef_state: wire.reef_state,
            library,
            library_dag,
        })
    }
}

impl CompiledContext {
    pub(crate) fn checked_library(&self) -> &crate::pipeline::CheckedLibrary {
        &self.library
    }

    pub(crate) fn library_checked(&self) -> &CheckedProgram {
        self.library.program()
    }

    /// Encode the same compatibility envelope used by the disk cache.
    /// Workers must check its format and build identity before decoding the
    /// positional checked-context payload.
    ///
    /// The bytes carry their own payload digest, which detects a torn write but
    /// not a deliberate rewrite, because whoever rewrote the payload rewrote the
    /// digest with it. A producer that can reach its reader over a second
    /// channel should use [`Self::encode_for_handoff`] instead.
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        self.envelope_bytes()
            .map(|(bytes, _)| bytes)
            .map_err(|e| format!("encode CompiledContext: {e}"))
    }

    /// Encode for a reader that authenticates the payload out of band.
    ///
    /// Returns the same envelope bytes as [`Self::encode`] together with the
    /// payload digest the envelope embeds. Deliver that digest to the reader
    /// over a channel this process controls and the bytes do not travel on --
    /// `chelis test` puts it in the worker's environment while the bytes go to a
    /// tempfile -- and the reader can then use [`Self::decode_authenticated`].
    pub fn encode_for_handoff(&self) -> Result<(Vec<u8>, HandoffDigest), String> {
        self.envelope_bytes()
            .map_err(|e| format!("encode CompiledContext: {e}"))
    }

    /// Reconstruct from bytes of unknown provenance.
    ///
    /// Every integrity claim available here comes out of the same bytes, so this
    /// route re-derives the library from the decoded `CheckedProgram` and
    /// compares the result against the transmitted lowered payload. That detects
    /// a payload whose parts no longer agree with each other. It cannot detect a
    /// payload that is internally consistent but was never produced from the
    /// sources it claims; nothing carried inside the bytes can.
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        CacheEnvelope::from_bytes(bytes)
            .and_then(CacheEnvelope::into_context)
            .map_err(|e| format!("decode CompiledContext: {e}"))
    }

    /// Reconstruct from bytes authenticated against a digest delivered out of
    /// band by [`Self::encode_for_handoff`].
    ///
    /// `expected` did not travel with `bytes`, so this rejects every payload
    /// but the producer's. [`Self::decode`] rejects one whose parts stopped
    /// agreeing with each other, which covers a rewrite of a few bytes and is
    /// what `cache_reconstruction_rejects_changed_numeric_bits_after_checksum_recomputed`
    /// demonstrates; it accepts a whole substituted payload that some other
    /// compilation by the same build produced, because that one is internally
    /// consistent (chelis#2257). This route rejects both.
    ///
    /// Given that, the effect and linearity reruns and the lowered-payload
    /// comparison have nothing left to establish here: they recompute a value
    /// equal to the transmitted one, as
    /// `both_decode_routes_reconstruct_identical_contexts` requires.
    pub fn decode_authenticated(bytes: &[u8], expected: &HandoffDigest) -> Result<Self, String> {
        CacheEnvelope::from_bytes(bytes)
            .and_then(|envelope| envelope.into_authenticated_context(expected))
            .map_err(|e| format!("decode CompiledContext: {e}"))
    }

    fn envelope_bytes(&self) -> Result<(Vec<u8>, HandoffDigest), CacheError> {
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
        let mut bytes = Vec::with_capacity(CACHE_MAGIC.len() + envelope_bytes.len());
        bytes.extend_from_slice(CACHE_MAGIC);
        bytes.extend_from_slice(&envelope_bytes);
        Ok((bytes, HandoffDigest(payload_sha256)))
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
        let (bytes, _) = self.envelope_bytes()?;

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

        let envelope = CacheEnvelope::from_bytes(&bytes)?;

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

        envelope.into_context().map(Some)
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
/// V10: `TypeEnv` now serializes transactional generalization levels,
/// transition watermarks, lowering overrides, and persisted-context resume
/// floors. Bincode is positional, so every V9 payload has the old checker
/// state shape and must be rejected before decode.
/// V11 adds quantified type-variable restrictions and their live
/// substitution ledger, so constrained function values retain their domain
/// through a compiled-context round trip.
/// V14 combines two independent V13 formats: chelis#1341 canonicalizes every
/// unordered collection that can reach encoded compiler-context bytes, while
/// chelis#1247 adds checker-owned nominal parameter kinds and kinded nominal
/// arguments. Either V13 payload has a branch-specific positional shape and
/// must clean-miss.
///
/// V12 records canonical source positions on deferred positional-expand and
/// reshape obligations. Their serialized checker state is therefore
/// structurally different from V11 even when a program has no cache-visible
/// type changes.
/// V9 unified two independent V8 formats. The pipeline-core
/// extraction sealed the lowered-library proof identity into the cached
/// context (branch V8). On main (main V8), chelis#878 (`RiscOp::Pad::fill`
/// sealed dtype-tagged scalar), chelis#942 (deferred positional-expand
/// constraints in the serialized checker context), and chelis#1182 (root
/// package modules emitted LAST, changing the serialized `reef_state`
/// (`PreparedReefGraph`) decl order) all landed. Both predecessors used V8
/// for their own shape at the same compiler version, and released 0.18.4
/// carries the main V8. The merged struct carries every field from both, so
/// bincode is positional and a V8 file of either lineage would decode to a
/// wrong shape; the magic check rejects it before any decode. A V6, V7, or
/// either V8 file is stale.
const CACHE_MAGIC: &[u8] = b"CHELIS_CTX_V23\n";

/// On-disk format version for the cache envelope. Bumping this tells
/// `load_if_fresh` to reject older cache files with
/// [`CacheError::UnsupportedVersion`] rather than risk a "successful but
/// wrong" decode.
///
/// V19 retains authored program signatures and checked extent carriers.
/// Older caches cannot reconstruct these call obligations.
///
/// V18 encodes sealed numeric scalar/storage payloads using exact dtype-tagged
/// bit codecs. Prior positional payloads must be regenerated.
///
/// V17 gives producer annotations an explicit opaque extension-data wire value.
/// Older AST payloads must be regenerated.
///
/// V16 removes both deferred-shape ledgers from the serialized `Subst` inside
/// `TypeEnv`. `spec/04-type-system.md` section 4.7.2 gives `expand` and
/// `insert` one result shape each, so nothing is deferred and the two fields
/// are gone. Bincode is positional, so a V15 entry carries two fields where
/// the following ones are now expected.
///
/// V15: that ledger carried a `DeferredShapeObligation` enum rather than a
/// bare expand constraint, so a comparison result could mirror its operand's
/// open choice.
///
/// V20 (chelis#1374/#1376): section 4.7.2's named half is now checked at
/// execution, so a V19 entry can only describe a program compiled before the
/// guard existed.
/// V21 (#1875): TypeEnv explicitly versions its direct encoding and retains
/// callable dimension labels independently from authored binder identities.
/// V22 is allocated to typed callable restrictions (#2071).
/// V23 carries exact result-claim witness roles in the lowered library.
/// V24 retains literal-result declaration tokens in the lowered library.
/// V25 retains checker-owned local tensor-ascription obligations.
/// V26 retains TypeEnv callable provenance for contextual grad selectors.
/// V27 (chelis#1125): Deep `Expr` and `Atom` lost the legacy list and tag
/// variants, so bincode variant indices shifted, and the lowered library's
/// program definitions and signatures are node-spelled on every ingress.
/// V29 (chelis#2413): the lowered library's random draws are key-operand
/// nodes fed by a counter-stream bridge operation and the baked random
/// variants are gone, so bincode variant indices shift. V28 was an intermediate state of the same change
/// and never shipped.
/// V30 (chelis#2413): the explicit key operations join `RiscOp` and `key`
/// becomes a storage dtype, so bincode variant indices shift again.
/// V31 (chelis#2413): the counter-stream bridge `RiscOp` variant, the `Random`
/// effect and the `random` handler kind are deleted with the counter stream,
/// so bincode variant indices shift again.
const CACHE_FORMAT_VERSION: u32 = 31;

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

impl CacheEnvelope {
    fn from_bytes(bytes: &[u8]) -> Result<Self, CacheError> {
        if bytes.is_empty() {
            return Err(CacheError::Corrupt("empty cache file".to_string()));
        }
        let envelope_bytes = bytes
            .strip_prefix(CACHE_MAGIC)
            .ok_or_else(|| CacheError::Corrupt("missing or wrong magic header".to_string()))?;
        // The fixed-int bincode envelope starts with a little-endian u32.
        // Reject its version before parsing the rest of an incompatible shape.
        let version_bytes = envelope_bytes
            .get(..4)
            .ok_or_else(|| CacheError::Corrupt("truncated cache format version".to_string()))?;
        let version = u32::from_le_bytes([
            version_bytes[0],
            version_bytes[1],
            version_bytes[2],
            version_bytes[3],
        ]);
        if version != CACHE_FORMAT_VERSION {
            return Err(CacheError::UnsupportedVersion {
                stored: version,
                expected: CACHE_FORMAT_VERSION,
            });
        }
        bincode::deserialize(envelope_bytes)
            .map_err(|e| CacheError::Corrupt(format!("envelope decode: {e}")))
    }

    /// Checks every route runs before it looks at the payload's contents:
    /// the compiler build the payload was produced by, and the payload's own
    /// embedded digest.
    ///
    /// The embedded digest catches a torn or truncated write. It cannot catch a
    /// deliberate rewrite, because it lives in the same bytes the rewriter
    /// controls; `into_authenticated_context` adds the check that does.
    fn check_build_and_embedded_digest(&self) -> Result<[u8; 32], CacheError> {
        if self.identity.compiler_version != crate::build_fingerprint() {
            return Err(CacheError::Corrupt(
                "compiled context belongs to an incompatible compiler build".to_string(),
            ));
        }
        let actual_payload_sha: [u8; 32] = Sha256::digest(&self.payload).into();
        if actual_payload_sha != self.payload_sha256 {
            return Err(CacheError::Corrupt(
                "payload sha256 does not match envelope".to_string(),
            ));
        }
        Ok(actual_payload_sha)
    }

    /// Checks every route runs after reconstructing the context: the envelope's
    /// outer copies of the source hash and identity must agree with the inner
    /// ones, so a spliced envelope header cannot relabel a payload.
    fn check_envelope_agreement(&self, context: &CompiledContext) -> Result<(), CacheError> {
        if context.source_hash != self.source_hash {
            return Err(CacheError::HashMismatch {
                envelope: self.source_hash,
                inner: context.source_hash,
            });
        }
        if context.identity != self.identity {
            return Err(CacheError::IdentityMismatch {
                envelope: self.identity.clone(),
                inner: context.identity.clone(),
            });
        }
        Ok(())
    }

    fn into_context(self) -> Result<CompiledContext, CacheError> {
        self.check_build_and_embedded_digest()?;
        let context: CompiledContext = bincode::deserialize(&self.payload).map_err(|e| {
            CacheError::Decode(format!(
                "CompiledContext decode (envelope/version match but inner shape changed): {e}"
            ))
        })?;
        self.check_envelope_agreement(&context)?;
        Ok(context)
    }

    /// Reconstruct a payload authenticated against a digest that did not travel
    /// with the bytes.
    ///
    /// `expected` is the authentication. The embedded digest checked above only
    /// says the bytes are self-consistent; this one says they are the bytes the
    /// producer wrote, because a rewriter who recomputed the embedded digest
    /// cannot also reach into the channel `expected` arrived on. Given that, the
    /// wire is deserialized straight into its context: the effect and linearity
    /// reruns `into_context` performs would recompute a value equal to the one
    /// transmitted, which is what
    /// `both_decode_routes_reconstruct_identical_contexts` locks.
    fn into_authenticated_context(
        self,
        expected: &HandoffDigest,
    ) -> Result<CompiledContext, CacheError> {
        let actual_payload_sha = self.check_build_and_embedded_digest()?;
        if actual_payload_sha != expected.0 {
            return Err(CacheError::HandoffDigestMismatch {
                expected: hex_prefix(&expected.0, 32),
                actual: hex_prefix(&actual_payload_sha, 32),
            });
        }
        let wire: CompiledContextWire = bincode::deserialize(&self.payload).map_err(|e| {
            CacheError::Decode(format!(
                "CompiledContext decode (envelope/version match but inner shape changed): {e}"
            ))
        })?;
        let context = wire
            .into_authenticated_context()
            .map_err(CacheError::Decode)?;
        self.check_envelope_agreement(&context)?;
        Ok(context)
    }
}

/// SHA-256 of a [`CompiledContext`] handoff payload, carried to the reader
/// separately from the bytes it authenticates.
///
/// A digest is only evidence when it reaches the reader by a route the bytes did
/// not take. `chelis test` mints one with
/// [`CompiledContext::encode_for_handoff`], writes the bytes to a tempfile, and
/// puts the digest in the worker process's environment, so an agent that can
/// replace the tempfile between the parent's write and the worker's read cannot
/// also replace the digest in an already-spawned child.
///
/// What that does not claim: the environment of a process is readable by other
/// processes of the same user, so this is not same-user isolation and does not
/// try to be. Anyone who can set the worker's environment or replace the
/// `chelis` binary already runs chosen code as this user, and the handoff is not
/// the weak point in that situation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HandoffDigest([u8; 32]);

impl HandoffDigest {
    /// Lower-case hex, the form that crosses a process boundary.
    pub fn to_hex(&self) -> String {
        hex_prefix(&self.0, 32)
    }

    /// Parse the [`Self::to_hex`] form.
    ///
    /// Rejects any other spelling rather than accepting a prefix: a short or
    /// mixed-case digest is a producer that did not follow the protocol, and
    /// silently repairing it would let a truncated value authenticate bytes it
    /// does not cover.
    pub fn from_hex(text: &str) -> Result<Self, String> {
        if text.len() != 64 {
            return Err(format!(
                "handoff digest must be 64 lower-case hex characters, got {} characters",
                text.len()
            ));
        }
        // The length check above makes the remainder empty, so every input
        // byte reaches `hex_nibble` and no prefix can be silently accepted.
        let (pairs, remainder) = text.as_bytes().as_chunks::<2>();
        debug_assert!(remainder.is_empty(), "64 is even");
        let mut bytes = [0u8; 32];
        for (index, [high, low]) in pairs.iter().enumerate() {
            bytes[index] = (hex_nibble(*high)? << 4) | hex_nibble(*low)?;
        }
        Ok(Self(bytes))
    }
}

fn hex_nibble(byte: u8) -> Result<u8, String> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        _ => Err(format!(
            "handoff digest must be 64 lower-case hex characters, found byte {byte:#04x}"
        )),
    }
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
    /// The payload's digest does not match the one the producer delivered out
    /// of band. The bytes are internally consistent — the embedded digest and
    /// the build identity both passed — so this is a payload that was replaced
    /// or rewritten after the producer wrote it, not a torn write.
    HandoffDigestMismatch { expected: String, actual: String },
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
            CacheError::HandoffDigestMismatch { expected, actual } => write!(
                f,
                "compiled context payload does not match the digest its producer delivered: \
                 expected={expected} actual={actual}"
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

    // Phase C+0e / chelis#451: build the checked library and its lowered DAG.
    // The layered path reuses the cross-process
    // chelis-std typecheck cache so the chelis-std half of the library is
    // never re-walked here; only the package's own (non-chelis-std) decls
    // are desugared + inferred + checked. On any miss (cache disabled, no
    // chelis-std in the graph, or the non-chelis-std decls don't compose
    // cleanly) it falls back to the monolithic whole-library build below,
    // which is byte-identical. See `build_checked_library_layered`.
    //
    // The full-library Surf → Deep desugar + macro expand is deferred to
    // the monolithic fallback so the layered path does NOT re-desugar
    // chelis-std (the layered helper desugars only the package's own decls).
    let library = match build_checked_library_layered(&reef_state, &mut t) {
        Some(Ok(library)) => library,
        Some(Err(err)) => return Err(err),
        None => {
            // chelis#930: the layered path folds a semantic rejection (type,
            // effect, or linearity) or an upstream build failure into `None`
            // so the monolithic path produces the byte-identical diagnostic.
            // A `LibraryRejection::ContextMismatch` is the one exception: it
            // is an internal proof-bind invariant failure, so
            // `build_checked_library_layered` returns it as `Some(Err(..))`
            // and it never reaches this fallback (see the carve-out there).
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

            // Build both type products in one session. Keep facade cancellation
            // checks between type analysis and the semantic suffix.
            let analysis = crate::pipeline::analyze_prepared_library(prepared)
                .map_err(library_rejection_to_compiler_error)
                .map_err(|error| cancelled_or("check", error))?;
            log_phase("build_compiled_library_context", &mut t);
            bail_if_cancelled("effects")?;
            let library = crate::pipeline::complete_library_checks(analysis)
                .map_err(library_rejection_to_compiler_error)
                .map_err(|error| cancelled_or("effects", error))?;
            bail_if_cancelled("linearity")?;
            log_phase("semantic_checks", &mut t);
            library
        }
    };

    // Lower the (possibly layer-composed) whole-library `CheckedProgram` to
    // a `LoweredLibrary` carrier. Lowering is cheap relative to the
    // typecheck, and lowering the composed program is byte-identical to
    // lowering the monolithic one (the composed program carries the same
    // chelis-std ++ package annotated bodies), so the layered path does not
    // need to reuse the cached chelis-std `library_dag` here.
    bail_if_cancelled("lower")?;
    let library_dag = crate::pipeline::lower_library(&library)
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
        library,
        library_dag,
    })
}

pub(crate) fn library_rejection_to_compiler_error(
    rejection: crate::pipeline::LibraryRejection,
) -> CompilerError {
    match rejection {
        crate::pipeline::LibraryRejection::Type { report } => {
            crate::compiler::check_errors_to_compiler_error("check", &report.errors)
        }
        crate::pipeline::LibraryRejection::ContextMismatch => CompilerError {
            transcript: Vec::new(),
            stage: "check".to_string(),
            errors: vec![Diagnostic::general(
                GeneralKind::Other,
                "the library type environment does not match its checked program".to_string(),
                crate::schema::numbers::UnitInterval::new(1.0).expect("constant severity"),
            )],
        },
        crate::pipeline::LibraryRejection::Effects { errors } => {
            crate::compiler::pipeline_rejection_to_compiler_error(
                crate::pipeline::PipelineRejection::Effects { errors },
            )
        }
        crate::pipeline::LibraryRejection::Linearity { errors } => {
            crate::compiler::pipeline_rejection_to_compiler_error(
                crate::pipeline::PipelineRejection::Linearity { errors },
            )
        }
    }
}

/// chelis#451 — build the `CheckedLibrary` half of a `CompiledContext`
/// while reusing the
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
fn build_checked_library_layered(
    reef_state: &PreparedReefGraph,
    t: &mut std::time::Instant,
) -> Option<Result<crate::pipeline::CheckedLibrary, CompilerError>> {
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
    let stdlib_ctx = match crate::stdlib_cache::load_or_build_stdlib_context(
        &reef_state.linked_stdlib_decls,
        reef_state.stdlib_source_digest(),
    ) {
        Ok(ctx) => ctx,
        Err(err) => return Some(Err(err)),
    };

    // Layer 2: desugar + macro-expand only the non-chelis-std library decls
    // (the package's own modules + non-stdlib path-deps). A macro-expansion
    // failure is a real front-end error the monolithic path also surfaces,
    // so hand back `None` for the byte-identical diagnostic.
    let prepared = match crate::pipeline::prepare_surf_decls_with_context(
        &reef_state.linked_non_stdlib_library_decls,
        stdlib_ctx.checked_library().program().exprs(),
        None,
    ) {
        Ok(prepared) => prepared,
        Err(_) => return None,
    };

    // Type-check + annotate the non-chelis-std declarations against the exact
    // checked chelis-std proof. The analysis retains that proof for later checks.
    let analysis = match crate::pipeline::analyze_prepared_library_with_base(
        prepared,
        stdlib_ctx.checked_library(),
    ) {
        Ok(analysis) => analysis,
        Err(_) => return None,
    };
    let library = match crate::pipeline::complete_context_library_checks(analysis) {
        Ok(library) => library,
        // chelis#930: a real semantic rejection folds into `None` so the
        // monolithic path reproduces the byte-identical diagnostic.
        Err(crate::pipeline::LibraryRejection::Effects { .. })
        | Err(crate::pipeline::LibraryRejection::Linearity { .. }) => return None,
        // A proof-bind mismatch (`ContextMismatch`) is NOT a user-program
        // rejection -- it means the freshly composed library and its type
        // environment disagree on the proof identity, reachable only through
        // an internal proof-threading bug. Folding it into the monolithic
        // fallback would yield a correct user result while permanently hiding
        // the invariant failure, which is the silent-fallback class the core
        // extraction exists to eliminate. Surface it loudly instead.
        Err(rejection) => return Some(Err(library_rejection_to_compiler_error(rejection))),
    };

    if std::env::var_os("CHELIS_PROFILE_COMPILE_CONTEXT").map(|v| v == "1") == Some(true) {
        eprintln!(
            "compile_reef_context: {:>32} {:>8.3}s",
            "layered_stdlib_cached_check",
            t.elapsed().as_secs_f64()
        );
    }
    *t = std::time::Instant::now();
    Some(Ok(library))
}

/// Profile-only: count modules and top-level decls in a library expr
/// list. Used for the `CHELIS_PROFILE_COMPILE_CONTEXT=1` structural
/// summary. Cheap O(N) walk; not on the hot path.
fn library_structural_summary(exprs: &[chelis_deep::ast::Expr]) -> (usize, usize) {
    let mut modules = 0usize;
    let mut decls = 0usize;
    for expr in exprs {
        let chelis_deep::ast::Expr::Node(node, _) = expr else {
            continue;
        };
        // Match the `top_level_decl_items` walk: descend through
        // `(module {} name children...)`.
        let tag = Some(node.tag());
        if tag == Some(DeepTag::Module) {
            modules += 1;
            for child in node.children_slice().iter().skip(1) {
                if matches!(
                    child.tag(),
                    Some(DeepTag::Def | DeepTag::Defsig | DeepTag::Deftype | DeepTag::Typealias)
                ) {
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
        transcript: Vec::new(),
        stage: "compile_reef_context".to_string(),
        errors: vec![Diagnostic::general(
            GeneralKind::ReefError,
            msg,
            crate::schema::numbers::UnitInterval::new(0.8).expect("constant severity"),
        )],
    }
}

fn hash_error(msg: &str) -> CompilerError {
    CompilerError {
        transcript: Vec::new(),
        stage: "compile_reef_context".to_string(),
        errors: vec![Diagnostic::general(
            GeneralKind::HashError,
            msg,
            crate::schema::numbers::UnitInterval::new(0.8).expect("constant severity"),
        )],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::stage_error;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    #[test]
    fn cache_format_version_tracks_the_key_operand_random_nodes() {
        assert_eq!(CACHE_MAGIC, b"CHELIS_CTX_V23\n");
        assert_eq!(CACHE_FORMAT_VERSION, 31);
    }

    /// chelis#1156: the cache identity must distinguish two BUILDS, not
    /// just two releases. Before the fix this field held
    /// `COMPILER_VERSION`, so a released `X.Y.Z` binary and a `main`
    /// binary still reporting `X.Y.Z` shared one identity and read each
    /// other's cached contexts — checking programs under the other
    /// build's type semantics. Observed both directions on 0.18.2 vs a
    /// post-`#1130` `main`: a spurious `precision mismatch: expected
    /// i32, got i64` on valid code, and (unsound) silent acceptance
    /// of code the running binary would reject on a cold cache.
    #[test]
    fn cache_identity_uses_the_build_fingerprint_not_the_bare_version() {
        let dir = TempDir::new().expect("tempdir");
        let identity = CacheIdentity::for_package_root(dir.path());
        assert_eq!(
            identity.compiler_version,
            crate::build_fingerprint(),
            "identity must carry the build fingerprint"
        );
        // The fingerprint is strictly finer than the release string on
        // every path, degraded included: both arms of `fingerprint_string`
        // extend `COMPILER_VERSION` with a discriminator, so this can
        // never be a conditional check.
        assert_ne!(
            identity.compiler_version,
            crate::COMPILER_VERSION,
            "a build-identity cache key must not collapse to the release version"
        );
    }

    /// A differing build fingerprint must change the on-disk cache file
    /// name, so two builds cannot even reach each other's entries.
    #[test]
    fn cache_file_name_separates_distinct_build_fingerprints() {
        let dir = TempDir::new().expect("tempdir");
        let mine = CacheIdentity::for_package_root(dir.path());
        let other = CacheIdentity {
            package_root: mine.package_root.clone(),
            compiler_version: format!("{}+other-build", mine.compiler_version),
        };
        let hash = ContextHash([7u8; 32]);
        assert_ne!(
            CompiledContext::cache_file_name(("pkg", "0.1.0"), hash, &mine),
            CompiledContext::cache_file_name(("pkg", "0.1.0"), hash, &other),
            "distinct build fingerprints must not share a cache file"
        );
    }

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
            "module App.Main\n\ndef main_value() -> i32 = cast(7, i32)\n",
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
            "module Mylib.Math\nexport (add)\n\ndef add(x: i32, y: i32) -> i32 = cast(0, i32)\n",
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

    #[test]
    fn current_surf_tensor_precision_round_trips_through_the_context_binary_codec() {
        let decls =
            chelis_surf::parser::parse_str("def identity(x: tensor[4, f32]) -> tensor[4, f32] = x")
                .expect("parse tensor declaration");
        let bytes = bincode::serialize(&decls).expect("encode current Surf AST with bincode");
        let restored: Vec<chelis_surf::ast::Decl> =
            bincode::deserialize(&bytes).expect("decode current Surf AST with bincode");
        assert_eq!(restored, decls);
    }

    #[test]
    fn worker_handoff_uses_the_exact_disk_compatibility_envelope() {
        let (_dir, root) = path_dep_fixture();
        let context = compile_reef_context(Path::new("/tmp/x"), &root).expect("context");
        let bytes = context.encode().expect("encode handoff");
        assert!(bytes.starts_with(CACHE_MAGIC));
        let path = root.join("handoff.ctx");
        context.save(&path).expect("save context");
        assert_eq!(bytes, fs::read(path).expect("read disk envelope"));
        let restored = CompiledContext::decode(&bytes).expect("decode handoff");
        assert_eq!(context.identity, restored.identity);
        assert_eq!(
            bincode::serialize(&context).expect("original payload"),
            bincode::serialize(&restored).expect("restored payload")
        );

        let unversioned = bincode::serialize(&context).expect("raw positional payload");
        let error = CompiledContext::decode(&unversioned).expect_err("no raw fallback");
        assert!(error.contains("magic"), "{error}");
        for version in [21_u32, 22, 23, 24, 25, 26, CACHE_FORMAT_VERSION + 1] {
            let mut truncated = CACHE_MAGIC.to_vec();
            truncated.extend_from_slice(&version.to_le_bytes());
            let error = CompiledContext::decode(&truncated)
                .expect_err("version rejection precedes even envelope payload parsing");
            assert!(error.contains("version"), "{error}");
        }
    }

    #[test]
    fn worker_handoff_checks_build_and_integrity_before_inner_decode() {
        let (_dir, root) = path_dep_fixture();
        let context = compile_reef_context(Path::new("/tmp/x"), &root).expect("context");
        let bytes = context.encode().expect("encode handoff");
        let mut envelope: CacheEnvelope =
            bincode::deserialize(&bytes[CACHE_MAGIC.len()..]).expect("envelope");
        envelope.identity.compiler_version = "incompatible-compiler-build".to_string();
        envelope.payload.clear();
        envelope.payload_sha256 = Sha256::digest(&envelope.payload).into();
        let encode_envelope = |envelope: &CacheEnvelope| {
            let mut bytes = CACHE_MAGIC.to_vec();
            bytes.extend(bincode::serialize(envelope).expect("encode envelope"));
            bytes
        };
        let error = CompiledContext::decode(&encode_envelope(&envelope))
            .expect_err("build rejection must precede invalid inner payload decode");
        assert!(error.contains("compiler build"), "{error}");

        envelope.identity = context.identity.clone();
        envelope.payload_sha256 = [0; 32];
        let error = CompiledContext::decode(&encode_envelope(&envelope))
            .expect_err("integrity rejection must precede invalid inner payload decode");
        assert!(error.contains("sha256"), "{error}");
    }

    // #822 review round 3, finding 4: the LocalRegistry hash-gap detection is
    // a string match over the upstream diagnostic; these lock it in both
    // directions so wording drift cannot silently reroute genuine failures
    // into the uncached-recompile fallback (or vice versa).
    fn encode_unchecked_context_wire(wire: &CompiledContextWire) -> Vec<u8> {
        let payload = bincode::serialize(wire).expect("invalid wire encodes");
        let envelope = CacheEnvelope {
            version: CACHE_FORMAT_VERSION,
            source_hash: wire.source_hash,
            identity: wire.identity.clone(),
            payload_sha256: Sha256::digest(&payload).into(),
            payload,
        };
        let mut bytes = CACHE_MAGIC.to_vec();
        bytes.extend(bincode::serialize(&envelope).expect("test envelope encodes"));
        bytes
    }

    #[test]
    fn context_decode_rejects_missing_or_forged_authored_signatures() {
        let (_dir, root) = path_dep_fixture();
        let context = compile_reef_context(Path::new("/tmp/x"), &root).expect("context");
        let lowered = serde_json::to_value(context.library_dag.raw()).unwrap();
        assert!(
            !lowered["program_signatures"]
                .as_object()
                .unwrap()
                .is_empty()
        );
        let mut missing = lowered.clone();
        missing
            .as_object_mut()
            .unwrap()
            .remove("program_signatures");
        assert!(serde_json::from_value::<IrLoweredLibrary>(missing).is_err());

        let mut forged = lowered;
        forged["program_signatures"] = serde_json::json!({});
        let wire = CompiledContextWire {
            source_hash: context.source_hash,
            identity: context.identity.clone(),
            reef_state: context.reef_state.clone(),
            type_env: context.checked_library().type_env().clone(),
            library_checked: context.library_checked().clone(),
            library_dag: serde_json::from_value(forged).unwrap(),
        };
        assert_eq!(
            wire.library_dag.library_proof_id(),
            context.library_checked().library_proof_id()
        );
        // A valid checksum and the original proof id cannot authorize an
        // erased signature ledger. Admission must compare fresh lowering.
        let error = CompiledContext::decode(&encode_unchecked_context_wire(&wire)).unwrap_err();
        assert!(error.contains("lowered library"), "{error}");
    }

    #[test]
    fn context_decode_rejects_a_foreign_type_environment() {
        let (_dir, root) = path_dep_fixture();
        let context = compile_reef_context(Path::new("/tmp/x"), &root).expect("context");
        let wire = CompiledContextWire {
            source_hash: context.source_hash,
            identity: context.identity.clone(),
            reef_state: context.reef_state.clone(),
            type_env: TypeEnv::empty(),
            library_checked: context.library_checked().clone(),
            library_dag: context.library_dag.raw().clone(),
        };
        let bytes = encode_unchecked_context_wire(&wire);
        let error = CompiledContext::decode(&bytes)
            .expect_err("the cache parser must reject mismatched library fields");

        assert!(error.contains("type environment"));
    }

    #[test]
    fn context_decode_rejects_a_foreign_lowered_library() {
        let (_first_dir, first_root) = path_dep_fixture();
        let first = compile_reef_context(Path::new("/tmp/x"), &first_root).expect("first context");
        let (_second_dir, second_root) = path_dep_fixture();
        fs::write(
            second_root.join("src/main.ch"),
            "module App.Main\n\ndef main_value() -> i32 = cast(8, i32)\n",
        )
        .expect("rewrite second main.ch");
        let second =
            compile_reef_context(Path::new("/tmp/x"), &second_root).expect("second context");
        let wire = CompiledContextWire {
            source_hash: first.source_hash,
            identity: first.identity.clone(),
            reef_state: first.reef_state.clone(),
            type_env: first.checked_library().type_env().clone(),
            library_checked: first.library_checked().clone(),
            library_dag: second.library_dag.raw().clone(),
        };
        let bytes = encode_unchecked_context_wire(&wire);
        let error = CompiledContext::decode(&bytes)
            .expect_err("the cache parser must reject a foreign lowered library");

        assert!(error.contains("lowered library"));
    }

    #[test]
    fn context_decode_rejects_a_changed_lowered_payload_with_the_same_identity() {
        let (_dir, root) = path_dep_fixture();
        let context = compile_reef_context(Path::new("/tmp/x"), &root).expect("context");
        let mut lowered_value =
            serde_json::to_value(context.library_dag.raw()).expect("lowered library must encode");
        lowered_value["rootless_defs"] = serde_json::json!(["forged_rootless_def"]);
        let changed_lowering: IrLoweredLibrary =
            serde_json::from_value(lowered_value).expect("changed lowering must decode");
        assert_eq!(
            changed_lowering.library_proof_id(),
            context.library_checked().library_proof_id(),
            "the negative control must retain the checked-library identity",
        );
        let wire = CompiledContextWire {
            source_hash: context.source_hash,
            identity: context.identity.clone(),
            reef_state: context.reef_state.clone(),
            type_env: context.checked_library().type_env().clone(),
            library_checked: context.library_checked().clone(),
            library_dag: changed_lowering,
        };
        let bytes = encode_unchecked_context_wire(&wire);
        let error = CompiledContext::decode(&bytes)
            .expect_err("the cache parser must reject a changed lowered payload");

        assert!(error.contains("lowered library payload"));
    }

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
