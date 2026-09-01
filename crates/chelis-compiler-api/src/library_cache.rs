//! Cross-process dependency-library typecheck cache (chelis#1168).
//!
//! Layer 2 of the build-lane hybrid cache. Where [`crate::stdlib_cache`]
//! caches the typechecked chelis-std sub-context, this module caches the
//! typechecked **dependency** sub-context: the composed
//! `chelis-std ++ dependency-packages` state that a `chelis build` checks
//! the entry file against. Without it, every `chelis build` re-infers the
//! whole dependency (shell) library on each call even though only the
//! entry file changed — the chelis-std cache alone left the deps re-walked
//! (`check_layered_for_build`).
//!
//! ## Layering
//!
//! ```text
//!   Layer 1  StdLibContext     chelis-std                (stdlib_cache.rs)
//!   Layer 2  LibraryContext    stdlib ++ dependencies    (this module)
//!   Layer 3  <fresh>           entry file                (check_layered_for_build)
//! ```
//!
//! A [`LibraryContext`] carries exactly what analyzing the entry against
//! the composed library requires: a proof-bound composed
//! [`crate::pipeline::CheckedLibrary`] over `chelis-std ++ dependencies`.
//! Its `TypeEnv` resolves `(var ...)` references into both chelis-std and
//! the dependency packages, and its checked library program is the exact
//! proof the entry's contextual completion composes onto. It mirrors
//! [`crate::context::build_checked_library_layered`], which stacks the same
//! proof-bound base + extension seam for the whole-package
//! `CompiledContext`; this module lifts it into the `chelis build`
//! front-end and content-addresses it on disk.
//!
//! ## Why the content-addressed key is honest
//!
//! [`library_cache_key`] folds the dependency decls' bytes **with** the
//! Layer-1 stdlib key, so the entry stays out of the key (edit the entry:
//! same key, warm hit) while any dependency edit — or any chelis-std
//! change, which flips the folded stdlib key — is a clean miss. The key
//! also folds the compiler version, so a differently-built binary never
//! stale-hits a sub-context carrying older typecheck semantics.
//!
//! ## warm == cold
//!
//! On a hit the loaded [`LibraryContext`] is the same artifact a cold
//! build produces: the dependency decls are content-addressed, the
//! chelis-std base is itself a byte-identical cached artifact, and the
//! proof-bound library extension over that base is deterministic. Decode
//! reruns the effect and linearity checks to rebind the proof, never
//! trusting the wire bytes. So the entry checked against a warm-loaded context
//! composes to the same whole-program `CheckedProgram` as the cold path —
//! and, per the acceptance oracle, as the monolithic path. The cache is a
//! pure speedup; it never changes an output or a diagnostic.
//!
//! ## Unbounded growth (chelis#1183)
//!
//! Layer 1 (`chelis-std-<ver>-<hash>.tc`) is bounded by toolchain identity,
//! but since chelis#1156 that identity is the compiler BUILD, not the stdlib
//! build: its key folds `build_fingerprint()`, so each locally built compiler
//! mints its own entry and the ones belonging to superseded builds are dead
//! weight (see [`evict_typecheck_cache`], which reclaims them first).
//! Layer 2's key is a
//! hash of USER source, and for a single-package project the "dependency
//! prefix" is the developer's own non-entry modules. So every save of a
//! sibling module mints a new multi-MiB `chelis-lib-*.tc`, and nothing reclaims
//! it (the only `remove_file` in the cache path is tempfile cleanup). The cache
//! lives at `$CHELIS_REEF_HOME/.cache/typecheck` (or the XDG fallback
//! `~/.cache/chelis/typecheck`) and is **unbounded in user edits** until
//! chelis#1183 (a size/age bound or a `chelis cache clear`) lands.
//!
//! ## Disable seam
//!
//! [`crate::stdlib_cache::cache_disabled`] (`CHELIS_STDLIB_CACHE_DISABLE=1`)
//! bypasses this cache too: [`load_or_build_library_context`] then always
//! rebuilds and never touches disk. The CLI additionally routes the whole
//! type-check through the monolithic checker under that flag, so it is the
//! oracle's monolithic-vs-layered seam for both cache layers.

use chelis_types::{CheckedProgram, TypeEnv};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

use crate::cache_envelope;
use crate::compiler::CompilerError;
use crate::stdlib_cache::{StdLibContext, cache_disabled, typecheck_cache_dir};

/// Internal struct-format version. Bumped when [`LibraryContext`]'s shape
/// changes so a stale on-disk entry is a clean miss, not a bad decode.
/// Mixed into the content-addressed key.
///
/// V3 accounts for the serialized type-checker generalization-level state in
/// `TypeEnv`; a V2 dependency-library payload is a clean miss.
/// V4 adds quantified type-variable restrictions and their live substitution
/// ledger.
/// V5 records canonical source positions on deferred positional-expand and
/// reshape obligations inside `TypeEnv`.
/// V6 adds checker-owned nominal parameter kinds and kinded nominal arguments,
/// including dimension-valued applications in dependency-library signatures.
///
/// V2: the sub-context now stores a proof-bound `CheckedLibrary`, and decode
/// reruns effect/linearity checks to rebind the proof (mirroring the stdlib
/// and compiled-context caches). The wire `CheckedProgram` also grew the
/// library-proof-identity fields. A V1 `chelis-lib-*.tc` written by a
/// pre-extraction binary at the same compiler version is a clean miss.
const LIBRARY_CACHE_FORMAT_VERSION: u32 = 6;

/// The typechecked composed `chelis-std ++ dependency-packages`
/// sub-context.
///
/// Built by [`build_library_context`] on top of a [`StdLibContext`],
/// cached under [`library_cache_key`]. Holds the two outer-context
/// products the entry-file analysis needs; it deliberately does NOT carry
/// a lowered library DAG or structural stats, because the `chelis build`
/// path lowers the final composed whole-program `CheckedProgram` and never
/// reads a fitness report off this sub-context.
///
/// Cheap to clone: the heavy state is `Arc`-shared inside `TypeEnv`.
#[derive(Debug, Clone)]
pub struct LibraryContext {
    /// Proof-bound composed `chelis-std ++ dependencies` library: the union
    /// `TypeEnv` scope for analyzing the entry via
    /// `analyze_prepared_with_library`, and the checked library program the
    /// entry's contextual completion composes onto. Dependency-declared
    /// types win on shadow over chelis-std, exactly as the monolithic
    /// whole-library build produces.
    library: crate::pipeline::CheckedLibrary,
    /// Number of expanded Deep exprs the dependency decls contribute.
    ///
    /// The build-lane caller must expand `dependencies ++ entry` as ONE
    /// unit (so macro hygiene counters and the expansion budget match the
    /// monolithic path), then split the expanded Deep at this index to
    /// recover the entry suffix. Stored so a warm hit does not re-derive
    /// it. See [`build_library_context`].
    pub dependency_expanded_len: usize,
    /// SHA-256 of the bincode of the dependency decls' expanded Deep — the
    /// exact expansion this context was type-checked from.
    ///
    /// The caller re-hashes the leading `dependency_expanded_len` exprs of
    /// its `dependencies ++ entry` expansion and compares. A mismatch means
    /// the combined expansion's dependency prefix differs from the cached
    /// one (macro cross-talk: e.g. the entry redefines a macro name a
    /// dependency invokes), so the cached context is not valid for this
    /// program and the caller must fall back to the monolithic path. This
    /// keeps the split byte-identical even in that pathological case.
    pub dependency_deep_digest: [u8; 32],
}

impl LibraryContext {
    /// The proof-bound composed `chelis-std ++ dependencies` library.
    pub fn checked_library(&self) -> &crate::pipeline::CheckedLibrary {
        &self.library
    }

    /// The union `chelis-std ++ dependencies` type environment.
    pub fn type_env(&self) -> &TypeEnv {
        self.library.type_env()
    }

    /// The composed `chelis-std ++ dependencies` checked library program.
    pub fn library_checked(&self) -> &CheckedProgram {
        self.library.program()
    }
}

/// Serde carrier for the cache envelope. Decode revalidates the proof: it
/// checks the type-environment relationship and reruns effect and linearity
/// before it rebinds the `CheckedLibrary`, mirroring the stdlib and
/// compiled-context cache parsers. A forged or mismatched entry is rejected,
/// never trusted.
#[derive(Serialize, Deserialize)]
struct LibraryContextWire {
    type_env: TypeEnv,
    library_checked: CheckedProgram,
    dependency_expanded_len: usize,
    dependency_deep_digest: [u8; 32],
}

impl Serialize for LibraryContext {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        LibraryContextWire {
            type_env: self.library.type_env().clone(),
            library_checked: self.library.program().clone(),
            dependency_expanded_len: self.dependency_expanded_len,
            dependency_deep_digest: self.dependency_deep_digest,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for LibraryContext {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = LibraryContextWire::deserialize(deserializer)?;
        let _linked = chelis_types::install_linked_program_guard();
        let library =
            chelis_pipeline_core::validate_cached_library(wire.type_env, wire.library_checked)
                .map_err(serde::de::Error::custom)?;
        Ok(Self {
            library,
            dependency_expanded_len: wire.dependency_expanded_len,
            dependency_deep_digest: wire.dependency_deep_digest,
        })
    }
}

/// The 32-byte content-addressed cache key for a dependency sub-context.
///
/// Inputs: the struct-format version, the compiler crate version
/// (`COMPILER_VERSION`), the Layer-1 stdlib sub-context key, AND a hash of
/// the linked, internal-name-rewritten dependency `Decl` slice.
///
/// Folding `stdlib_key` is what layers the invalidation: it already
/// encodes the bundled chelis-std version + archive/shell hashes + the
/// actual linked stdlib decls + the compiler version, so ANY chelis-std
/// change flips this key without re-deriving those inputs here. Folding
/// the dependency decl bytes flips it on any dependency edit. The entry
/// file is intentionally absent, so an entry-only edit keeps the key
/// stable and warm-hits. The build fingerprint is also folded directly
/// (in addition to reaching this key inside `stdlib_key`) so the key still
/// pins the compiler build if the stdlib key derivation ever changes.
///
/// chelis#1156: this input used to be `COMPILER_VERSION`, whose whole
/// point as a defence-in-depth layer was void, since a release string does
/// not change between two builds of one unreleased version. It is the
/// build fingerprint now, so the fallback actually holds.
pub fn library_cache_key(
    dependency_decls: &[chelis_surf::ast::Decl],
    stdlib_key: [u8; 32],
) -> [u8; 32] {
    library_cache_key_at_version(dependency_decls, stdlib_key, LIBRARY_CACHE_FORMAT_VERSION)
}

fn library_cache_key_at_version(
    dependency_decls: &[chelis_surf::ast::Decl],
    stdlib_key: [u8; 32],
    format_version: u32,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"chelis_library_typecheck_v");
    hasher.update(format_version.to_le_bytes());
    let compiler_version = crate::build_fingerprint();
    hasher.update(b"compiler_version");
    hasher.update((compiler_version.len() as u64).to_le_bytes());
    hasher.update(compiler_version.as_bytes());
    // The Layer-1 stdlib sub-context identity. Fixed width, appended
    // directly.
    hasher.update(b"stdlib_key");
    hasher.update(stdlib_key);
    // The dependency decls actually being checked. `bincode` is a
    // deterministic encoding, so this is a stable content hash. A `serialize`
    // failure is unreachable for a well-formed `Decl` slice (bincode of `Decl`
    // never fails today), so rather than panic we fold a fixed tag. NOTE this
    // is fail-OPEN: two distinct unserializable decl sets would fold the same
    // tag and collide onto one cache key (a false hit, not a miss). Unreachable
    // for `Decl`; if a future decl type gains a fallible `serialize` this must
    // instead make the caller skip the cache (return `None` / fail closed).
    // chelis#1176 review (F2).
    match bincode::serialize(dependency_decls) {
        Ok(decl_bytes) => {
            hasher.update(b"decls");
            hasher.update((decl_bytes.len() as u64).to_le_bytes());
            hasher.update(&decl_bytes);
        }
        Err(_) => {
            hasher.update(b"decls-unserializable");
        }
    }
    hasher.finalize().into()
}

/// The on-disk path for a dependency sub-context cache entry.
///
/// The `chelis-lib-` prefix keeps it in the same `typecheck` cache
/// directory as the stdlib cache's `chelis-std-` entries without ever
/// colliding on a file name.
fn library_cache_path(cache_dir: &Path, key: [u8; 32]) -> PathBuf {
    cache_dir.join(format!(
        "chelis-lib-{}.tc",
        crate::stdlib_cache::hex_prefix(&key, 8)
    ))
}

/// Load the dependency sub-context from disk if a fresh entry exists, else
/// build it and (best-effort) write it back.
///
/// - `Ok(Some(ctx))` — a warm hit, or a cold build that composed cleanly.
/// - `Ok(None)` — the dependency decls do NOT compose cleanly against the
///   cached chelis-std sub-context (a type, effect, linearity, or macro
///   error). The caller falls back to the monolithic path so the
///   error-path output stays byte-identical — exactly the `Ok(None)`
///   handoff `check_layered_for_build` already documents. Nothing is
///   cached in this case.
/// - `Err(CompilerError)` — reserved for an unexpected failure surfaced by
///   [`build_library_context`]; the dependency-rejection cases are folded
///   into `Ok(None)`.
///
/// `stdlib_key` is the caller-computed Layer-1 key
/// ([`crate::stdlib_cache::stdlib_cache_key`] over the linked stdlib
/// decls); folding it into [`library_cache_key`] is what makes a
/// chelis-std change invalidate this layer.
pub fn load_or_build_library_context(
    stdlib_ctx: &StdLibContext,
    stdlib_key: [u8; 32],
    dependency_decls: &[chelis_surf::ast::Decl],
) -> Result<Option<LibraryContext>, CompilerError> {
    if cache_disabled() {
        return build_library_context(stdlib_ctx, dependency_decls);
    }

    let key = library_cache_key(dependency_decls, stdlib_key);
    let Some(cache_dir) = typecheck_cache_dir() else {
        // No resolvable cache root at all — build uncached.
        return build_library_context(stdlib_ctx, dependency_decls);
    };
    let cache_path = library_cache_path(&cache_dir, key);

    match cache_envelope::load::<LibraryContext>(&cache_path, key) {
        Ok(Some(ctx)) => return Ok(Some(ctx)),
        Ok(None) => {}
        Err(e) => {
            eprintln!(
                "chelis: dependency typecheck cache at {} unusable ({e}); \
                 rebuilding and overwriting",
                cache_path.display()
            );
        }
    }

    let built = build_library_context(stdlib_ctx, dependency_decls)?;
    // Only a clean compose is cacheable; an `Ok(None)` fallback is not an
    // artifact and must not be written (the next build re-derives the
    // byte-identical monolithic error path).
    if let Some(ctx) = &built {
        match cache_envelope::save(&cache_path, key, ctx) {
            Ok(()) => {
                // chelis#1183: keep the cache dir bounded. Only on the miss
                // path, after a successful write, never evicting what we just
                // wrote. Best-effort.
                let live_stdlib = crate::stdlib_cache::stdlib_cache_path(&cache_dir, stdlib_key);
                evict_typecheck_cache(
                    &cache_dir,
                    &cache_path,
                    Some(&live_stdlib),
                    typecheck_cache_max_bytes(),
                );
            }
            Err(e) => {
                eprintln!(
                    "chelis: warning: failed to write dependency typecheck cache to {}: {e}",
                    cache_path.display()
                );
            }
        }
    }
    Ok(built)
}

/// Default size cap on the typecheck cache directory (512 MiB). Overridable via
/// `CHELIS_TYPECHECK_CACHE_MAX_BYTES`. See [`evict_typecheck_cache`].
const DEFAULT_TYPECHECK_CACHE_MAX_BYTES: u64 = 512 * 1024 * 1024;

fn typecheck_cache_max_bytes() -> u64 {
    std::env::var("CHELIS_TYPECHECK_CACHE_MAX_BYTES")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_TYPECHECK_CACHE_MAX_BYTES)
}

/// Best-effort eviction to keep the typecheck cache dir under
/// [`typecheck_cache_max_bytes`] (chelis#1183). Layer 1 (`chelis-std-*.tc`) is
/// bounded by toolchain identity, but Layer 2 (`chelis-lib-*.tc`) is keyed on
/// user source and churns ~MiB per sibling/entry edit with nothing reclaiming
/// it. Called only on the miss path, after a successful write.
///
/// Policy: oldest-first by mtime within three tiers, evicted in this order.
///
/// 1. **Superseded `chelis-std-*`** - Layer-1 entries that are not the running
///    build's. Before chelis#1156 there was only ever one, because the key
///    folded `COMPILER_VERSION`; now it folds `build_fingerprint()`, so every
///    compiler build a developer or CI matrix produces mints its own multi-MiB
///    entry and the previous one is never read again. Reclaiming these first is
///    the only tier where eviction costs nothing at all.
/// 2. **`chelis-lib-*`** - keyed on user source, churns per sibling edit.
/// 3. **The running build's `chelis-std-*`** - written once and hit forever, so
///    evicting it forces a costly re-inference. Last resort.
///
/// Without tier 1 the old two-tier policy would protect every superseded stdlib
/// entry ahead of live Layer-2 entries, which inverts the intent on exactly the
/// machines chelis#1156 serves: those that build the compiler repeatedly.
///
/// The just-written entry is never evicted. Also sweeps `.tmp.*` orphans older
/// than an hour (age-gated so it never races an in-flight `save`'s rename).
/// Every filesystem error is ignored: eviction must never fail a build, and a
/// reader that loses the unlink race falls through to a clean recompute.
///
/// `max_bytes` is passed in rather than read from the environment here so the
/// tiering is testable without mutating process-global state, which
/// `#![forbid(unsafe_code)]` rules out in this crate anyway.
fn evict_typecheck_cache(
    cache_dir: &Path,
    just_written: &Path,
    live_stdlib: Option<&Path>,
    max_bytes: u64,
) {
    let Ok(read_dir) = std::fs::read_dir(cache_dir) else {
        return;
    };
    let now = std::time::SystemTime::now();
    // (path, size, mtime, tier)
    let mut entries: Vec<(PathBuf, u64, std::time::SystemTime, u8)> = Vec::new();
    let mut total: u64 = 0;
    for entry in read_dir.flatten() {
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let mtime = meta.modified().unwrap_or(now);
        if name.contains(".tmp.") || name.ends_with(".tmp") {
            // Age-gated orphan sweep: only reap a temp file too old to be an
            // in-flight `save` about to rename into place.
            if now
                .duration_since(mtime)
                .map(|age| age.as_secs() > 3600)
                .unwrap_or(false)
            {
                let _ = std::fs::remove_file(entry.path());
            }
            continue;
        }
        if !(name.starts_with("chelis-lib-") || name.starts_with("chelis-std-")) {
            continue;
        }
        total += meta.len();
        let path = entry.path();
        // 0 = superseded stdlib entry, 1 = library entry, 2 = the running
        // build's stdlib entry. See the doc comment for why this order.
        let tier: u8 = if name.starts_with("chelis-lib-") {
            1
        } else if live_stdlib.is_some_and(|live| live == path) {
            2
        } else {
            0
        };
        entries.push((path, meta.len(), mtime, tier));
    }
    if total <= max_bytes {
        return;
    }
    // Lowest tier first, then oldest-first by mtime within a tier.
    entries.sort_by(|a, b| a.3.cmp(&b.3).then(a.2.cmp(&b.2)));
    for (path, size, _, _) in entries {
        if total <= max_bytes {
            break;
        }
        if path == just_written {
            continue;
        }
        if std::fs::remove_file(&path).is_ok() {
            total = total.saturating_sub(size);
        }
    }
}

/// Build a [`LibraryContext`] by stacking the dependency decls on the
/// cached chelis-std sub-context.
///
/// Pipeline (mirrors [`crate::context::build_checked_library_layered`], but
/// packaged for the `chelis build` front-end):
/// 1. Surf-desugar + macro-expand the linked dependency decls into Deep.
/// 2. `analyze_prepared_library_with_base(deps, stdlib.checked_library())` ->
///    the dependency type analysis bound to the exact chelis-std proof, over
///    the union chelis-std ++ dependency scope.
/// 3. `complete_context_library_checks` runs effects + linearity over the
///    dependency decls against the bound chelis-std library and composes
///    `chelis-std ++ dependencies` into one proof-bound `CheckedLibrary`.
///
/// Returns `Ok(None)` on a dependency macro, type, effect, or linearity
/// rejection, matching the `Ok(None)` monolithic-fallback contract: the
/// monolithic path then produces the byte-identical diagnostic. A
/// proof-bind mismatch (`ContextMismatch`) is an internal invariant failure
/// and surfaces as an `Err`, never a silent fallback. Only a clean compose
/// yields `Ok(Some(..))`.
pub fn build_library_context(
    stdlib_ctx: &StdLibContext,
    dependency_decls: &[chelis_surf::ast::Decl],
) -> Result<Option<LibraryContext>, CompilerError> {
    // RFC v5: dependency decls are reef-linker output (internal-name
    // mangled); accept the linker name format while building the context.
    let _linked = chelis_types::install_linked_program_guard();

    let prepared = match crate::pipeline::prepare_surf_decls(dependency_decls, None) {
        Ok(prepared) => prepared,
        Err(_) => return Ok(None),
    };

    // Record the dependency expansion boundary + digest BEFORE the analysis
    // consumes `prepared`. The caller expands `dependencies ++ entry` as one
    // unit and splits at `dependency_expanded_len`; the digest lets it
    // verify the combined expansion's dependency prefix matches this one.
    let dependency_expanded_len = prepared.expanded_deep().len();
    let dependency_deep_digest = expanded_deep_digest(prepared.expanded_deep());

    // Type-check + annotate the dependency decls against the exact checked
    // chelis-std proof; the analysis retains that proof for the semantic
    // suffix.
    let analysis = match crate::pipeline::analyze_prepared_library_with_base(
        prepared,
        stdlib_ctx.checked_library(),
    ) {
        Ok(analysis) => analysis,
        Err(_) => return Ok(None),
    };
    let library = match crate::pipeline::complete_context_library_checks(analysis) {
        Ok(library) => library,
        Err(crate::pipeline::LibraryRejection::Effects { .. })
        | Err(crate::pipeline::LibraryRejection::Linearity { .. }) => return Ok(None),
        Err(rejection) => {
            return Err(crate::context::library_rejection_to_compiler_error(
                rejection,
            ));
        }
    };

    Ok(Some(LibraryContext {
        library,
        dependency_expanded_len,
        dependency_deep_digest,
    }))
}

/// SHA-256 of the bincode of an expanded Deep expr slice. Used to pin the
/// exact dependency expansion a [`LibraryContext`] was type-checked from,
/// so the build-lane caller can detect macro cross-talk between the
/// dependency prefix and the entry (see [`LibraryContext`]).
pub fn expanded_deep_digest(exprs: &[chelis_deep::ast::Expr]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"chelis_dep_expanded_deep_v1");
    match bincode::serialize(exprs) {
        Ok(bytes) => {
            hasher.update((bytes.len() as u64).to_le_bytes());
            hasher.update(&bytes);
        }
        // A serialize failure is unreachable for well-formed Deep (bincode of
        // `Expr` never fails today), so we fold a fixed tag rather than panic.
        // NOTE this is fail-OPEN: two distinct unserializable slices fold the
        // same tag, so the digest guard in `layered.rs` would compare EQUAL and
        // ACCEPT the cached context — a false hit, not the "fall-back" an
        // earlier version of this comment wrongly claimed. Unreachable for
        // `Expr`; if a future node type gains a fallible `serialize` this must
        // signal failure so the guard falls back to monolithic (fail closed).
        // chelis#1176 review (F2).
        Err(_) => hasher.update(b"expanded-deep-unserializable"),
    }
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdlib_cache::build_stdlib_context;

    #[test]
    fn cache_format_version_tracks_ordered_constraints_and_nominal_kinds() {
        assert_eq!(LIBRARY_CACHE_FORMAT_VERSION, 6);
    }

    #[test]
    fn preceding_payload_version_is_a_clean_cache_miss() {
        let stdlib_context = build_stdlib_context(&[]).expect("empty stdlib context");
        let decls = sample_decls("preceding_version");
        let stdlib_key = key(5);
        let current_key = library_cache_key(&decls, stdlib_key);
        let preceding_key = library_cache_key_at_version(&decls, stdlib_key, 4);
        assert_ne!(current_key, preceding_key);

        let context = build_library_context(&stdlib_context, &decls)
            .expect("sample context must build")
            .expect("sample dependency must compose");
        let dir = tempfile::tempdir().expect("tempdir");
        let preceding_path = library_cache_path(dir.path(), preceding_key);
        cache_envelope::save(&preceding_path, preceding_key, &context)
            .expect("preceding-version fixture must save");

        let current_path = library_cache_path(dir.path(), current_key);
        let loaded: Option<LibraryContext> = cache_envelope::load(&current_path, current_key)
            .expect("a preceding-version fixture must be a clean miss");
        assert!(loaded.is_none());
        assert!(
            preceding_path.exists(),
            "negative-control fixture must exist"
        );
        assert_ne!(current_path, preceding_path);
    }

    /// chelis#1156 (PR #1161 review, F5): eviction must reclaim a
    /// SUPERSEDED Layer-1 entry before a live Layer-2 entry.
    ///
    /// Layer 1 is now one entry per compiler BUILD, not one per stdlib
    /// build, so a machine that rebuilds the compiler accumulates dead
    /// multi-MiB `chelis-std-*` files. The pre-#1161 two-tier policy
    /// protected all of them ahead of every `chelis-lib-*`, which inverts
    /// the intent on exactly the machines this change serves.
    #[test]
    fn eviction_reclaims_superseded_stdlib_entries_before_library_entries() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let root = dir.path();
        let write = |name: &str, bytes: usize| -> PathBuf {
            let p = root.join(name);
            std::fs::write(&p, vec![b'x'; bytes]).expect("write");
            p
        };
        // The live entry is the NEWEST so that a pure oldest-first policy
        // would not reach it anyway; the discriminator under test is the
        // tier, not the mtime.
        let stale_std = write("chelis-std-0.4.0-aaaaaaaaaaaaaaaa.tc", 4096);
        let live_lib = write("chelis-lib-bbbbbbbbbbbbbbbb.tc", 4096);
        let live_std = write("chelis-std-0.4.0-cccccccccccccccc.tc", 4096);
        let just_written = write("chelis-lib-dddddddddddddddd.tc", 4096);

        // Cap below the total so eviction must free exactly one file.
        evict_typecheck_cache(root, &just_written, Some(&live_std), 12288);

        assert!(
            !stale_std.exists(),
            "the superseded stdlib entry must be reclaimed first"
        );
        assert!(
            live_lib.exists(),
            "a live library entry outranks a dead stdlib entry"
        );
        assert!(
            live_std.exists(),
            "the running build's stdlib entry is the last resort"
        );
        assert!(
            just_written.exists(),
            "the just-written entry is never evicted"
        );
    }

    /// The last-resort tier still holds: with no superseded stdlib entry
    /// to reclaim, library entries go before the running build's Layer-1
    /// entry, which is the pre-#1161 behaviour and must not regress.
    #[test]
    fn eviction_still_protects_the_live_stdlib_entry_over_library_entries() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let root = dir.path();
        let write = |name: &str, bytes: usize| -> PathBuf {
            let p = root.join(name);
            std::fs::write(&p, vec![b'x'; bytes]).expect("write");
            p
        };
        let live_std = write("chelis-std-0.4.0-cccccccccccccccc.tc", 4096);
        let old_lib = write("chelis-lib-bbbbbbbbbbbbbbbb.tc", 4096);
        let just_written = write("chelis-lib-dddddddddddddddd.tc", 4096);

        evict_typecheck_cache(root, &just_written, Some(&live_std), 8192);

        assert!(
            !old_lib.exists(),
            "the library entry is the eviction candidate here"
        );
        assert!(
            live_std.exists(),
            "the running build's stdlib entry must survive"
        );
        assert!(just_written.exists());
    }

    /// A minimal well-formed dependency `Decl` slice. The exact shape is
    /// irrelevant to the key tests; what matters is that the same slice
    /// hashes identically and a different slice hashes differently.
    fn sample_decls(marker: &str) -> Vec<chelis_surf::ast::Decl> {
        chelis_surf::parser::parse_str(&format!(
            "module Dep\nexport ({marker}_value)\ndef {marker}_value() -> int32 = cast(1, int32)\n"
        ))
        .expect("sample decls must parse")
    }

    fn key(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    #[test]
    fn cache_key_is_stable_for_identical_inputs() {
        let decls = sample_decls("a");
        assert_eq!(
            library_cache_key(&decls, key(7)),
            library_cache_key(&decls, key(7))
        );
    }

    #[test]
    fn cache_key_depends_on_the_dependency_decls() {
        // A dependency edit must flip the key: this is the property that
        // makes a changed dependency a clean miss.
        let a = sample_decls("a");
        let b = sample_decls("b");
        assert_ne!(a, b, "test setup: the two decl slices must differ");
        assert_ne!(library_cache_key(&a, key(7)), library_cache_key(&b, key(7)));
    }

    #[test]
    fn cache_key_depends_on_the_stdlib_key() {
        // Folding the Layer-1 stdlib key means a chelis-std change (which
        // flips the stdlib key) invalidates this layer even for unchanged
        // dependency decls.
        let decls = sample_decls("a");
        assert_ne!(
            library_cache_key(&decls, key(1)),
            library_cache_key(&decls, key(2))
        );
    }

    /// Regression for the compiler-build-identity gap, mirroring the stdlib
    /// cache: recompute the key with the compiler-identity component perturbed
    /// and confirm the real key differs.
    ///
    /// chelis#1156 (PR #1161 review, F7): that component used to be
    /// `COMPILER_VERSION`, which made this key's documented defence-in-depth
    /// role ("still pins the compiler build if the stdlib key derivation ever
    /// changes") vacuous, since a release string is shared by every build of
    /// an unreleased version. It is `build_fingerprint()` now.
    #[test]
    fn cache_key_depends_on_the_build_fingerprint() {
        let decls = sample_decls("a");
        let real = library_cache_key(&decls, key(9));

        let recompute_with_compiler_version = |compiler_version: &str| -> [u8; 32] {
            let mut hasher = Sha256::new();
            hasher.update(b"chelis_library_typecheck_v");
            hasher.update(LIBRARY_CACHE_FORMAT_VERSION.to_le_bytes());
            hasher.update(b"compiler_version");
            hasher.update((compiler_version.len() as u64).to_le_bytes());
            hasher.update(compiler_version.as_bytes());
            hasher.update(b"stdlib_key");
            hasher.update(key(9));
            match bincode::serialize(&decls) {
                Ok(decl_bytes) => {
                    hasher.update(b"decls");
                    hasher.update((decl_bytes.len() as u64).to_le_bytes());
                    hasher.update(&decl_bytes);
                }
                Err(_) => hasher.update(b"decls-unserializable"),
            }
            hasher.finalize().into()
        };

        assert_eq!(
            real,
            recompute_with_compiler_version(crate::build_fingerprint()),
            "recompute mirror must match the real key for the running build fingerprint"
        );
        assert_ne!(
            real,
            recompute_with_compiler_version(crate::COMPILER_VERSION),
            "the library cache key must fold the BUILD fingerprint, not the bare \
             release version; two builds of one version must not share this cache"
        );
        assert_ne!(
            real,
            recompute_with_compiler_version("0.0.0-some-other-compiler-build"),
            "a different compiler identity must produce a different library cache key"
        );
    }

    #[test]
    fn cache_path_carries_prefix_and_key() {
        let dir = PathBuf::from("/tmp/tc");
        let path = library_cache_path(&dir, library_cache_key(&sample_decls("a"), key(7)));
        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(
            name.starts_with("chelis-lib-"),
            "must not collide with the stdlib cache's chelis-std- prefix; got {name}"
        );
        assert!(name.ends_with(".tc"), "got {name}");
    }

    /// Order-independent semantic equality of two composed library
    /// programs. `CheckedProgram` and `TypeEnv` carry `HashMap`/`HashSet`
    /// state whose bincode order is nondeterministic, so compare the
    /// substantive typed content (annotated exprs + type env as a set)
    /// rather than raw bytes.
    fn library_checked_semantically_eq(a: &LibraryContext, b: &LibraryContext) -> bool {
        a.library_checked().annotated_exprs() == b.library_checked().annotated_exprs()
            && a.library_checked().type_env() == b.library_checked().type_env()
    }

    #[test]
    fn build_library_context_is_deterministic() {
        // Cold == cold: two independent builds of the same dependency
        // sub-context are semantically identical. This is the property a
        // warm disk hit relies on — a loaded context equals a rebuilt one.
        let stdlib_ctx = build_stdlib_context(&[]).expect("empty stdlib context");
        let deps = sample_decls("a");
        let first = build_library_context(&stdlib_ctx, &deps)
            .expect("build ok")
            .expect("deps compose cleanly");
        let second = build_library_context(&stdlib_ctx, &deps)
            .expect("build ok")
            .expect("deps compose cleanly");
        assert!(
            library_checked_semantically_eq(&first, &second),
            "two cold builds of the same dependency context must be semantically identical"
        );
    }

    #[test]
    fn library_context_round_trips_through_the_cache_envelope() {
        // POSITIVE (chelis#1168): a built LibraryContext writes to disk and
        // loads back under its key with the same composed library — the
        // write+reuse the build-lane cache depends on, exercised at the
        // on-disk envelope level (hermetic tempfile; the end-to-end disk
        // hit/warm-build byte-identity lives in the CLI oracle).
        let stdlib_ctx = build_stdlib_context(&[]).expect("empty stdlib context");
        let deps = sample_decls("a");
        let ctx = build_library_context(&stdlib_ctx, &deps)
            .expect("build ok")
            .expect("deps compose cleanly");
        let k = library_cache_key(&deps, key(5));

        let dir = tempfile::tempdir().expect("tempdir");
        let path = library_cache_path(dir.path(), k);
        cache_envelope::save(&path, k, &ctx).expect("save ok");
        let loaded: LibraryContext = cache_envelope::load(&path, k)
            .expect("load ok")
            .expect("entry present under its key");
        assert!(
            library_checked_semantically_eq(&ctx, &loaded),
            "a reused (loaded) LibraryContext must equal the built one"
        );

        // A different key is a clean miss, never a stale hit against this
        // entry (mirrors the stdlib-mutation negative case one layer up).
        let other: Option<LibraryContext> =
            cache_envelope::load(&path, key(6)).expect("mismatched-key load ok");
        assert!(other.is_none(), "a different key must be a clean miss");
    }
}
