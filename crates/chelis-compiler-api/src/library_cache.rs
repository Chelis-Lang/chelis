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
//! the composed library requires: the composed [`TypeEnv`] (so
//! `(var ...)` references resolve into both chelis-std and the dependency
//! packages) and the composed library [`CheckedProgram`] (the outer
//! `SemanticContext::Library` for the entry's effect / linearity pass, and
//! the left operand of the final [`compose_checked`]). It mirrors
//! [`crate::context::build_library_triple_layered`], which already stacks
//! the same `build_compiled_library_context_with_base` + compose seam for
//! the whole-package `CompiledContext`; this module lifts it into the
//! `chelis build` front-end and content-addresses it on disk.
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
//! chelis-std base is itself a byte-identical cached artifact, and both
//! `build_compiled_library_context_with_base` and `CheckedProgram::compose`
//! are deterministic. So the entry checked against a warm-loaded context
//! composes to the same whole-program `CheckedProgram` as the cold path —
//! and, per the acceptance oracle, as the monolithic path. The cache is a
//! pure speedup; it never changes an output or a diagnostic.
//!
//! ## Unbounded growth (chelis#1183)
//!
//! Unlike Layer 1 — whose cardinality is bounded by toolchain identity
//! (`chelis-std-<ver>-<hash>.tc`, one per stdlib build) — Layer 2's key is a
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
const LIBRARY_CACHE_FORMAT_VERSION: u32 = 1;

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryContext {
    /// IR type-checker snapshot over `chelis-std ++ dependencies` — the
    /// union scope for analyzing the entry via
    /// `analyze_prepared_with_context`. Dependency-declared types win on
    /// shadow over chelis-std, exactly as the monolithic whole-library
    /// build produces.
    pub type_env: TypeEnv,
    /// Composed `chelis-std ++ dependencies` IR + effects + linearity
    /// result. Feeds `SemanticContext::Library(..)` for the entry's
    /// effect / linearity pass and is the left operand of the final
    /// `compose_checked` that produces the whole-program checked state.
    pub library_checked: CheckedProgram,
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
/// stable and warm-hits. `COMPILER_VERSION` is also folded directly (in
/// addition to being inside `stdlib_key`) so the key still pins the
/// compiler build if the stdlib key derivation ever changes.
pub fn library_cache_key(
    dependency_decls: &[chelis_surf::ast::Decl],
    stdlib_key: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"chelis_library_typecheck_v");
    hasher.update(LIBRARY_CACHE_FORMAT_VERSION.to_le_bytes());
    let compiler_version = crate::COMPILER_VERSION;
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
    if let Some(ctx) = &built
        && let Err(e) = cache_envelope::save(&cache_path, key, ctx)
    {
        eprintln!(
            "chelis: warning: failed to write dependency typecheck cache to {}: {e}",
            cache_path.display()
        );
    }
    Ok(built)
}

/// Build a [`LibraryContext`] by stacking the dependency decls on the
/// cached chelis-std sub-context.
///
/// Pipeline (mirrors [`crate::context::build_library_triple_layered`], but
/// packaged for the `chelis build` front-end):
/// 1. Surf-desugar + macro-expand the linked dependency decls into Deep.
/// 2. `build_compiled_library_context_with_base(stdlib.type_env, deps)` ->
///    the composed `(TypeEnv, dependency CheckedProgram)`. The `TypeEnv`
///    is the union chelis-std ++ dependency scope.
/// 3. effects + linearity over the dependency decls against
///    `SemanticContext::Library(stdlib.library_checked)`.
/// 4. `CheckedProgram::compose(stdlib.library_checked, deps_checked)` ->
///    the composed library `CheckedProgram`.
///
/// Returns `Ok(None)` on ANY dependency rejection (macro, type, effect, or
/// linearity), matching the `Ok(None)` monolithic-fallback contract: the
/// monolithic path then produces the byte-identical diagnostic. Only a
/// clean compose yields `Ok(Some(..))`.
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

    // Type-check + annotate the dependency decls stacked on the cached
    // chelis-std sub-context. Returns the union `TypeEnv` and the
    // dependency-only checked type product.
    let (type_env, dependency_type_product) =
        match chelis_types::build_compiled_library_context_with_base(
            &stdlib_ctx.type_env,
            prepared.expanded_deep(),
        ) {
            Ok(pair) => pair,
            Err(_) => return Ok(None),
        };

    let analysis =
        crate::pipeline::prepared_analysis_from_checked(prepared, dependency_type_product);
    let dependency_checked = match crate::pipeline::complete_checks(
        analysis,
        crate::pipeline::SemanticContext::Library(&stdlib_ctx.library_checked),
    ) {
        Ok(checked) => checked,
        Err(crate::pipeline::SemanticRejection::Effects { .. })
        | Err(crate::pipeline::SemanticRejection::Linearity { .. }) => return Ok(None),
    };

    // Compose the cached chelis-std half with the freshly-checked
    // dependency half into the one composed library `CheckedProgram` the
    // entry analysis stacks on.
    let library_checked =
        CheckedProgram::compose(&stdlib_ctx.library_checked, dependency_checked.program());

    Ok(Some(LibraryContext {
        type_env,
        library_checked,
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

    /// A minimal well-formed dependency `Decl` slice. The exact shape is
    /// irrelevant to the key tests; what matters is that the same slice
    /// hashes identically and a different slice hashes differently.
    fn sample_decls(marker: &str) -> Vec<chelis_surf::ast::Decl> {
        chelis_surf::parser::parse_str(&format!(
            "module Dep\nexport ({marker}_value)\ndef {marker}_value -> int32 = cast(1, int32)\n"
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

    #[test]
    fn cache_key_depends_on_the_compiler_version() {
        // Regression for the compiler-build-identity gap, mirroring the
        // stdlib cache: recompute the key with the compiler-version
        // component perturbed and confirm the real key differs.
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
            recompute_with_compiler_version(crate::COMPILER_VERSION),
            "recompute mirror must match the real key for the real compiler version"
        );
        assert_ne!(
            real,
            recompute_with_compiler_version("0.0.0-some-other-compiler-build"),
            "a different compiler version must produce a different library cache key"
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
        a.library_checked.annotated_exprs() == b.library_checked.annotated_exprs()
            && a.library_checked.type_env() == b.library_checked.type_env()
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
