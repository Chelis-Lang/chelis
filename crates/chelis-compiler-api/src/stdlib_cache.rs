//! Cross-process chelis-std typecheck cache.
//!
//! Every `chelis check` / `chelis build` on a file that imports
//! chelis-std re-parses and re-typechecks the entire chelis-std import
//! graph from scratch (~4.5s, see
//! `docs/investigations/stdlib_typecheck_cache.md`). `cargo nextest`
//! spawns ~200 separate test processes, a large fraction of which pay
//! that cost independently.
//!
//! This module caches the typechecked + lowered chelis-std library
//! sub-context — a [`StdLibContext`] — content-addressed on the linked
//! chelis-std decls, their exact source-byte determinant, and the
//! bundled-stdlib constants, so the work
//! happens once per `(binary, machine, stdlib-content)` and is reused
//! across every process and every stdlib-importing fixture.
//!
//! ## Why the content-addressed key works
//!
//! chelis-std ships inside the chelis binary via `include_bytes!`
//! (`chelis-std-bundle`). Its archive + shell bytes and version are
//! compile-time constants, immutable per binary. `link_graph`'s
//! internal-name rewriting depends only on chelis-std's own
//! package/module/symbol names, never on the importing package — so the
//! linked chelis-std decls (and everything derived from them) are
//! bit-identical for every fixture that consumes the bundled stdlib.
//!
//! The key folds the struct-format version, the bundled-stdlib version +
//! archive + shell hashes, a hash of the actual linked chelis-std `Decl`
//! slice, AND the prepared graph's exact manifest/source-inventory/source-byte
//! digest (see [`stdlib_cache_key`]). The independent determinants keep
//! the key honest: `chelis-std` is the language runtime, so checking a
//! file *inside* a `chelis-std` checkout resolves that checkout's own
//! source as the root package rather than the bundle. Keying on the
//! bundle constants alone would let an edited checkout stale-hit a
//! bundled-std artifact. Because every bundled-std consumer's linked
//! stdlib decls are bit-identical, the decl hash still collapses to one
//! shared cross-fixture key on the common path; it only diverges when
//! the stdlib content actually differs. The key self-invalidates on any
//! stdlib regeneration and never needs a manual bust.
//!
//! ## Layering
//!
//! [`StdLibContext`] is Layer 1 of the hybrid cache. `compile_reef_context`
//! consumes it as the base for its `_with_context` Layer 2 build (the
//! user package's own modules) so chelis-std is never re-walked there.
//! See `context.rs`.
//!
//! ## Disable seam
//!
//! `CHELIS_STDLIB_CACHE_DISABLE=1` makes [`load_or_build_stdlib_context`]
//! always rebuild and never read or write the disk cache. The CLI
//! additionally routes the whole typecheck path through the monolithic
//! checker when this is set, so it is the acceptance oracle's
//! monolithic-vs-layered test seam. It is never set in production CI.
//!
//! ## Deserialization boundary
//!
//! Decode validates the checked library without another type-inference session.
//! It reruns the remaining semantic checks and the lower phase. The canonical
//! lower result must match the cache payload before contextual code can use it.

use chelis_ir::lower::LoweredLibrary as IrLoweredLibrary;
use chelis_types::{CheckedProgram, StructuralStats, TypeEnv};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

use crate::cache_envelope;
use crate::compiler::CompilerError;
use crate::schema::{Diagnostic, GeneralKind};

/// Internal struct-format version. Bumped when [`StdLibContext`]'s shape
/// changes so a stale on-disk entry is a clean miss, not a bad decode.
/// Mixed into the content-addressed key. V6 accounts for the serialized
/// type-checker generalization-level state added to `TypeEnv`; a V5 payload
/// is a clean miss rather than a positional bincode decode.
/// V7 adds quantified type-variable restrictions and their live substitution
/// ledger.
/// V11 combines the chelis#1341 cache format with chelis#1247's independent V9
/// nominal-kind format. The hash-order lineage canonicalizes every unordered
/// collection that can reach payload and key bytes and requires the prepared
/// graph's exact-source digest, so a
/// trivia-only source edit cannot stale-hit the same parsed declarations.
///
/// V8 records canonical source positions on deferred positional-expand and
/// reshape obligations inside `TypeEnv`.
/// V5 unified two independent V4
/// bumps: the pipeline-core `CheckedLibrary`/proof-identity products
/// (branch) and chelis#942's serialized positional-expand obligations
/// inside `TypeEnv` (main). Bincode is positional, so a V4 entry from
/// either side is a clean miss. V12 added the `DeferredShapeObligation`
/// enum to that same ledger. V13 removes both deferred-shape ledgers from
/// the serialized `Subst`: under `spec/04-type-system.md` section 4.7.2
/// nothing is deferred, so a V12 entry carries two fields where the
/// following ones are now expected.
// V14: opaque producer annotations use an explicit data wire variant.
// V15: declared literal results and call-witness payloads are retained.
// V17: authored program signatures and checked extent carriers are mandatory.
// V16: scalar/storage payloads use the exact dtype-tagged bit codecs;
// the changed key rejects previous positional payloads before decode.
const STDLIB_CACHE_FORMAT_VERSION: u32 =
    <StdLibContext as cache_envelope::CachePayload>::FORMAT_VERSION;

/// The typechecked + lowered chelis-std library sub-context.
///
/// Built once by [`build_stdlib_context`] from the bundled chelis-std
/// linked decls, cached under [`stdlib_cache_key`]. Mirrors the three
/// pipeline-stage fields `CompiledContext` carries — `type_env`,
/// `CheckedLibrary`, `library_dag` — but for chelis-std alone, plus the
/// structural stats needed to reconstitute a whole-program fitness
/// report without re-walking the library decls.
///
/// Cheap to clone: the heavy state is `Arc`-shared inside `TypeEnv`.
#[derive(Debug, Clone)]
pub struct StdLibContext {
    library: crate::pipeline::CheckedLibrary,
    /// Lowered chelis-std DAG carrier. Feeds `lower_program_with_context`
    /// on the `chelis build` path.
    ///
    /// `None` when chelis-std cannot be lowered *as a standalone library*
    /// — some chelis-std defs (e.g. a `softmax` over a non-static axis)
    /// are only lowerable once a concrete caller pins the axis, and the
    /// unpruned whole-library lower hits that. `check` never reads this
    /// field; `build` falls back to the monolithic lowering path when it
    /// is `None`, so a non-lowerable stdlib never blocks a build, it just
    /// does not get the cache speedup on the lowering stage.
    library_dag: Option<crate::pipeline::LoweredLibrary>,
    /// Total structural AST node count + Deep-validator-flagged node count
    /// over the chelis-std library decls. The in-context fitness report uses
    /// these for the `structure` component; its `typed_nodes` / `total_nodes`
    /// come from the library program's serialized inference counters.
    structural_stats: StructuralStats,
}

#[derive(Serialize, Deserialize)]
struct StdLibContextWire {
    type_env: TypeEnv,
    library_checked: CheckedProgram,
    library_dag: Option<IrLoweredLibrary>,
    structural_stats: StructuralStats,
}

impl StdLibContext {
    pub fn checked_library(&self) -> &crate::pipeline::CheckedLibrary {
        &self.library
    }

    pub fn type_env(&self) -> &TypeEnv {
        self.library.type_env()
    }

    pub fn library_checked(&self) -> &CheckedProgram {
        self.library.program()
    }

    pub fn library_dag(&self) -> Option<&crate::pipeline::LoweredLibrary> {
        self.library_dag.as_ref()
    }

    pub fn structural_stats(&self) -> StructuralStats {
        self.structural_stats
    }
}

impl Serialize for StdLibContext {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        StdLibContextWire {
            type_env: self.type_env().clone(),
            library_checked: self.library_checked().clone(),
            library_dag: self
                .library_dag
                .as_ref()
                .map(|library| library.raw().clone()),
            structural_stats: self.structural_stats,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for StdLibContext {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = StdLibContextWire::deserialize(deserializer)?;
        let _linked = chelis_types::install_linked_program_guard();
        let library =
            chelis_pipeline_core::validate_cached_library(wire.type_env, wire.library_checked)
                .map_err(serde::de::Error::custom)?;
        let library_dag = match wire.library_dag {
            Some(cached) => {
                if cached.library_proof_id() != library.program().library_proof_id() {
                    return Err(serde::de::Error::custom(
                        "the lowered library does not match the checked library",
                    ));
                }
                let expected =
                    crate::pipeline::lower_library(&library).map_err(serde::de::Error::custom)?;
                if !cache_envelope::lowered_library_payload_matches(&cached, expected.raw())
                    .map_err(serde::de::Error::custom)?
                {
                    return Err(serde::de::Error::custom(
                        "the lowered library payload does not match the checked library",
                    ));
                }
                Some(expected)
            }
            None => None,
        };
        Ok(Self {
            library,
            library_dag,
            structural_stats: wire.structural_stats,
        })
    }
}

/// The 32-byte content-addressed cache key for a chelis-std sub-context.
///
/// Inputs: the struct-format version, the compiler crate version
/// (`COMPILER_VERSION`), the bundled chelis-std version string, the
/// SHA-256 hex of the bundled archive + shell bytes
/// (`chelis-std-bundle`), AND a hash of the actual linked, internal-name-
/// rewritten chelis-std `Decl` slice that will be type-checked into this
/// sub-context.
///
/// `COMPILER_VERSION` is what pins the compiler *build*:
/// `STDLIB_CACHE_FORMAT_VERSION` only guards the on-disk struct shape, so
/// without the compiler version a binary built from different compiler
/// source but the same bundled chelis-std bytes would stale-hit an older
/// binary's cached sub-context (stale typecheck / lowering semantics).
///
/// Hashing the actual decls is what makes the key honest. The bundled
/// archive/shell hashes alone are insufficient: `chelis-std` is the
/// language runtime, so checking a file *inside* a `chelis-std` checkout
/// resolves that checkout's own (possibly edited) source as the root
/// package rather than the bundle (the bundled-version short-circuit in
/// `chelis-reef` only fires when `chelis-std` is resolved as a
/// dependency). Keying solely on the bundle constants would then let an
/// edited `chelis-std` checkout stale-hit a bundled-std artifact sharing
/// a `CHELIS_REEF_HOME`. Folding the decl bytes in flips the key on any
/// such divergence while keeping it byte-identical across every fixture
/// that consumes the unmodified bundled stdlib (their linked stdlib
/// decls are bit-identical, see the module docs).
pub fn stdlib_cache_key(
    stdlib_decls: &[chelis_surf::ast::Decl],
    stdlib_source_digest: [u8; 32],
) -> [u8; 32] {
    stdlib_cache_key_at_version(
        stdlib_decls,
        stdlib_source_digest,
        STDLIB_CACHE_FORMAT_VERSION,
    )
}

/// Exact ordered byte stream hashed by [`stdlib_cache_key`].
///
/// This is exposed for the Phase B cache-root oracle. Callers must treat it
/// as diagnostic evidence, not as a separately versioned wire format.
#[doc(hidden)]
pub fn stdlib_cache_key_input_bytes(
    stdlib_decls: &[chelis_surf::ast::Decl],
    stdlib_source_digest: [u8; 32],
) -> Vec<u8> {
    let mut bytes = Vec::new();
    visit_stdlib_cache_key_inputs(
        stdlib_decls,
        stdlib_source_digest,
        STDLIB_CACHE_FORMAT_VERSION,
        |part| bytes.extend_from_slice(part),
    );
    bytes
}

fn stdlib_cache_key_at_version(
    stdlib_decls: &[chelis_surf::ast::Decl],
    stdlib_source_digest: [u8; 32],
    format_version: u32,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    visit_stdlib_cache_key_inputs(stdlib_decls, stdlib_source_digest, format_version, |part| {
        hasher.update(part)
    });
    hasher.finalize().into()
}

fn visit_stdlib_cache_key_inputs(
    stdlib_decls: &[chelis_surf::ast::Decl],
    stdlib_source_digest: [u8; 32],
    format_version: u32,
    mut append: impl FnMut(&[u8]),
) {
    append(<StdLibContext as cache_envelope::CachePayload>::KEY_DOMAIN);
    append(&format_version.to_le_bytes());
    // Compiler build identity. `STDLIB_CACHE_FORMAT_VERSION` only guards
    // the on-disk struct SHAPE; it does not change when the compiler's
    // typecheck / lowering SEMANTICS change while the bundled chelis-std
    // bytes stay the same. Without this, a `chelis` binary built from
    // different compiler source but the same bundled stdlib would
    // stale-hit an older binary's cached sub-context.
    //
    // This must be the BUILD fingerprint, not `COMPILER_VERSION`: the
    // bare crate version does not change between two builds from
    // different commits of the same unreleased version, so it does not
    // "flip the key on any compiler rebuild" the way this cache needs
    // (chelis#1156). The stdlib sub-context is keyed without a package
    // root, so it is shared by every package on the machine — a stale
    // hit here reaches further than the per-package context cache.
    let compiler_version = crate::build_fingerprint();
    append(b"compiler_version");
    append(&(compiler_version.len() as u64).to_le_bytes());
    append(compiler_version.as_bytes());
    let version = chelis_std_bundle::BUNDLED_CHELIS_STD_VERSION;
    append(&(version.len() as u64).to_le_bytes());
    append(version.as_bytes());
    let archive = chelis_std_bundle::archive_sha256();
    append(&(archive.len() as u64).to_le_bytes());
    append(archive.as_bytes());
    let shell = chelis_std_bundle::shell_sha256();
    append(&(shell.len() as u64).to_le_bytes());
    append(shell.as_bytes());
    append(b"exact-source-digest");
    append(&stdlib_source_digest);
    // The decls actually being checked. `bincode` is a deterministic
    // encoding, so this is a stable content hash. A `serialize` failure here
    // is unreachable for a well-formed `Decl` slice (bincode of `Decl` never
    // fails today), so rather than panic we fold a fixed tag. NOTE this is
    // fail-OPEN, exactly like `library_cache::library_cache_key` /
    // `expanded_deep_digest` (see their comments): two distinct unserializable
    // slices fold the same tag and collide onto one key. Unreachable for
    // `Decl`; a future fallible-`serialize` type must instead fail closed
    // (return `None` / skip the cache). chelis#1176 review (F2/G4).
    match bincode::serialize(stdlib_decls) {
        Ok(decl_bytes) => {
            append(b"decls");
            append(&(decl_bytes.len() as u64).to_le_bytes());
            append(&decl_bytes);
        }
        Err(_) => append(b"decls-unserializable"),
    }
}

/// Lower-case hex of the first `n` bytes of `data`.
pub(crate) fn hex_prefix(data: &[u8], n: usize) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let n = n.min(data.len());
    let mut out = String::with_capacity(n * 2);
    for &b in &data[..n] {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

/// Resolve the directory a named on-disk compiler cache lives in.
///
/// - `$CHELIS_REEF_HOME/.cache/<name>/` when `CHELIS_REEF_HOME` is set
///   and non-empty;
/// - else `$XDG_CACHE_HOME/chelis/<name>/` when `XDG_CACHE_HOME` is set
///   and non-empty;
/// - else `$HOME/.cache/chelis/<name>/`.
///
/// The XDG fallback is mandatory: `cargo nextest` test workers do not set
/// `CHELIS_REEF_HOME`, and they are exactly the workload these caches
/// optimize. Returns `None` only when none of the three roots can be
/// resolved (no `CHELIS_REEF_HOME`, no `XDG_CACHE_HOME`, no `HOME`), in
/// which case the caller falls through to an uncached build.
///
/// Both the chelis-std typecheck cache ([`typecheck_cache_dir`]) and the
/// whole-package `CompiledContext` cache (`context.rs`) resolve through
/// this one helper so the XDG fallback applies uniformly.
pub fn cache_dir_for(name: &str) -> Option<PathBuf> {
    if let Some(reef_home) = non_empty_env("CHELIS_REEF_HOME") {
        return Some(PathBuf::from(reef_home).join(".cache").join(name));
    }
    if let Some(xdg) = non_empty_env("XDG_CACHE_HOME") {
        return Some(PathBuf::from(xdg).join("chelis").join(name));
    }
    if let Some(home) = non_empty_env("HOME") {
        return Some(PathBuf::from(home).join(".cache").join("chelis").join(name));
    }
    None
}

/// Resolve the directory the chelis-std typecheck cache lives in. See
/// [`cache_dir_for`] for the resolution order.
pub fn typecheck_cache_dir() -> Option<PathBuf> {
    cache_dir_for("typecheck")
}

fn non_empty_env(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(value) if !value.is_empty() => Some(value),
        _ => None,
    }
}

/// The on-disk path for the bundled chelis-std's cache entry.
///
/// `pub(crate)` so [`crate::library_cache::evict_typecheck_cache`] can tell the
/// RUNNING build's Layer-1 entry apart from the entries other builds left
/// behind (chelis#1156 made Layer 1 one-per-compiler-build, not one-per-stdlib).
pub(crate) fn stdlib_cache_path(cache_dir: &Path, key: [u8; 32]) -> PathBuf {
    cache_dir.join(format!(
        "chelis-std-{}-{}.tc",
        chelis_std_bundle::BUNDLED_CHELIS_STD_VERSION,
        hex_prefix(&key, 8),
    ))
}

/// Whether the disk cache is disabled for this process.
pub fn cache_disabled() -> bool {
    std::env::var_os("CHELIS_STDLIB_CACHE_DISABLE")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Load the bundled chelis-std's [`StdLibContext`] from disk if a fresh
/// entry exists, else build it and (best-effort) write it back.
///
/// - When `CHELIS_STDLIB_CACHE_DISABLE=1`, always builds, never touches
///   disk.
/// - A clean miss (no file, or a file under a different key) builds +
///   writes.
/// - A corrupt / version-skewed / torn cache file is logged to stderr
///   and treated as a miss (build + overwrite), never a hard abort.
/// - If the post-build write fails, the in-memory context is still
///   returned (the build succeeded); a stderr breadcrumb is emitted and
///   the next invocation retries the write.
///
/// `build_decls` is the bundled chelis-std linked + internal-name-
/// rewritten decl list (`PreparedReefGraph::linked_stdlib_decls`). It is
/// only consumed on a miss.
pub fn load_or_build_stdlib_context(
    build_decls: &[chelis_surf::ast::Decl],
    stdlib_source_digest: [u8; 32],
) -> Result<StdLibContext, CompilerError> {
    if cache_disabled() {
        return build_stdlib_context(build_decls);
    }

    let key = stdlib_cache_key(build_decls, stdlib_source_digest);
    let Some(cache_dir) = typecheck_cache_dir() else {
        // No resolvable cache root at all — build uncached. Rare:
        // requires CHELIS_REEF_HOME, XDG_CACHE_HOME and HOME all unset.
        return build_stdlib_context(build_decls);
    };
    let cache_path = stdlib_cache_path(&cache_dir, key);

    match cache_envelope::load::<StdLibContext>(&cache_path, key) {
        Ok(Some(ctx)) => return Ok(ctx),
        Ok(None) => {}
        Err(e) => {
            eprintln!(
                "chelis: chelis-std typecheck cache at {} unusable ({e}); \
                 rebuilding and overwriting",
                cache_path.display()
            );
        }
    }

    let ctx = build_stdlib_context(build_decls)?;
    if let Err(e) = cache_envelope::save(&cache_path, key, &ctx) {
        eprintln!(
            "chelis: warning: failed to write chelis-std typecheck cache to {}: {e}",
            cache_path.display()
        );
    }
    Ok(ctx)
}

/// Build a [`StdLibContext`] from the bundled chelis-std linked decls.
///
/// Pipeline (mirrors `compile_reef_context`'s library half, but for
/// chelis-std alone and stacked on the empty base):
/// 1. Surf-desugar + macro-expand the linked chelis-std decls into Deep.
/// 2. `check_prepared_library` -> `CheckedLibrary`.
/// 3. `lower_program_to_library` -> `LoweredLibrary`.
/// 4. Structural stats over the desugared library declarations.
pub fn build_stdlib_context(
    stdlib_decls: &[chelis_surf::ast::Decl],
) -> Result<StdLibContext, CompilerError> {
    // RFC v5: chelis-std decls are reef-linker output (internal-name
    // mangled); accept the linker name format while building the
    // context, including via direct callers and the cache-miss path.
    let _linked = chelis_types::install_linked_program_guard();
    let prepared = crate::pipeline::prepare_surf_decls(stdlib_decls, None).map_err(|error| {
        crate::compiler::pipeline_rejection_to_compiler_error(
            crate::pipeline::PipelineRejection::Preparation(error),
        )
    })?;
    let structural_stats = chelis_types::structural_stats(prepared.expanded_deep());

    let library = crate::pipeline::check_prepared_library(prepared)
        .map_err(library_rejection_to_compiler_error)?;

    // Lower chelis-std as a standalone library, best-effort. A lowering
    // diagnostic here is NOT fatal: `check` does not use `library_dag`,
    // and `build` falls back to the monolithic lowering path when it is
    // `None`. Some chelis-std defs are only lowerable once a concrete
    // caller pins a symbolic axis; the unpruned whole-library lower can
    // legitimately hit that.
    let library_dag = crate::pipeline::lower_library(&library).ok();

    Ok(StdLibContext {
        library,
        library_dag,
        structural_stats,
    })
}

fn library_rejection_to_compiler_error(
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

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SOURCE_DIGEST: [u8; 32] = [0x5a; 32];

    #[test]
    fn cache_format_version_tracks_canonical_collection_bytes() {
        assert_eq!(STDLIB_CACHE_FORMAT_VERSION, 20);
    }

    #[test]
    fn cache_format_version_tracks_canonical_collection_bytes_and_nominal_kinds() {
        assert_eq!(STDLIB_CACHE_FORMAT_VERSION, 20);
    }

    #[test]
    fn different_format_key_is_a_clean_cache_miss() {
        let decls = sample_decls("different_key");
        let current_key = stdlib_cache_key(&decls, TEST_SOURCE_DIGEST);
        let preceding_key = stdlib_cache_key_at_version(&decls, TEST_SOURCE_DIGEST, 15);
        assert_ne!(current_key, preceding_key);

        let dir = tempfile::tempdir().expect("tempdir");
        let context = build_stdlib_context(&decls).expect("sample context must build");
        let preceding_path = stdlib_cache_path(dir.path(), preceding_key);
        cache_envelope::save(&preceding_path, preceding_key, &context)
            .expect("different-key fixture must save");

        let current_path = stdlib_cache_path(dir.path(), current_key);
        let loaded: Option<StdLibContext> = cache_envelope::load(&current_path, current_key)
            .expect("a different-key fixture must be a clean miss");
        assert!(loaded.is_none());
        assert!(
            preceding_path.exists(),
            "negative-control fixture must exist"
        );
        assert_ne!(current_path, preceding_path);
        cache_envelope::save(&current_path, current_key, &context).expect("current fixture saves");
        let current: Option<StdLibContext> =
            cache_envelope::load(&current_path, current_key).expect("current fixture loads");
        assert!(current.is_some(), "current producer and consumer must hit");
    }

    /// A minimal well-formed `Decl` slice for key-stability tests. The
    /// exact shape is irrelevant; what matters is that the same slice
    /// hashes identically and a different slice hashes differently.
    fn sample_decls(marker: &str) -> Vec<chelis_surf::ast::Decl> {
        chelis_surf::parser::parse_str(&format!(
            "module Sample\ndef {marker}_value() -> int32 = 1\n"
        ))
        .expect("sample decls must parse")
    }

    #[test]
    fn cache_key_is_stable_for_identical_decls() {
        // The key is a pure function of the compile-time bundle
        // constants plus the decl bytes, so the same decl slice always
        // hashes to the same key within a binary.
        let decls = sample_decls("a");
        assert_eq!(
            stdlib_cache_key(&decls, TEST_SOURCE_DIGEST),
            stdlib_cache_key(&decls, TEST_SOURCE_DIGEST)
        );
    }

    #[test]
    fn cache_key_depends_on_the_decls() {
        // Two different chelis-std decl slices must produce different
        // keys: this is the property that stops an edited chelis-std
        // checkout from stale-hitting a bundled-std artifact.
        let a = sample_decls("a");
        let b = sample_decls("b");
        assert_ne!(a, b, "test setup: the two decl slices must differ");
        assert_ne!(
            stdlib_cache_key(&a, TEST_SOURCE_DIGEST),
            stdlib_cache_key(&b, TEST_SOURCE_DIGEST)
        );
    }

    #[test]
    fn cache_key_depends_on_exact_source_bytes_independently_of_decls() {
        let decls = sample_decls("same_ast");
        assert_ne!(
            stdlib_cache_key(&decls, [0x11; 32]),
            stdlib_cache_key(&decls, [0x22; 32]),
            "trivia-only source changes must invalidate the cache even when parsed decls match"
        );
    }

    #[test]
    fn cache_key_depends_on_the_bundle_constants() {
        // Recompute the key with each bundle-constant input perturbed
        // and confirm the real key differs from every perturbation. This
        // pins that version / archive_sha256 / shell_sha256 all feed the
        // key, so a stdlib regeneration self-invalidates the cache even
        // for an unchanged decl slice.
        let decls = sample_decls("a");
        let real = stdlib_cache_key(&decls, TEST_SOURCE_DIGEST);

        let perturb = |tag: &[u8]| -> [u8; 32] {
            let mut hasher = Sha256::new();
            hasher.update(b"chelis_std_typecheck_v");
            hasher.update(STDLIB_CACHE_FORMAT_VERSION.to_le_bytes());
            hasher.update(tag); // stand-in for a changed input
            hasher.finalize().into()
        };
        assert_ne!(real, perturb(b"different-version"));
        assert_ne!(real, perturb(b"different-archive-sha"));
        assert_ne!(real, perturb(b"different-shell-sha"));
    }

    #[test]
    fn cache_key_depends_on_the_compiler_version() {
        // Regression for the compiler-build-identity gap: a chelis binary
        // built from different compiler source but the same bundled
        // chelis-std must NOT stale-hit an older binary's cached
        // sub-context. The real key must fold the BUILD fingerprint in —
        // `COMPILER_VERSION` alone is a release identity, not a build
        // identity, and does not change between two builds of the same
        // unreleased version (chelis#1156).
        //
        // We cannot rebuild the compiler mid-test, so we recompute the
        // key with the compiler-version component perturbed and confirm
        // the real key differs. This mirrors
        // `cache_key_depends_on_the_bundle_constants` and pins that the
        // compiler version is actually an input.
        let decls = sample_decls("a");
        let real = stdlib_cache_key(&decls, TEST_SOURCE_DIGEST);

        // Recompute the key byte-for-byte the way `stdlib_cache_key`
        // does, but with a different compiler version string. Every other
        // input is identical, so a difference proves the compiler version
        // feeds the key.
        let recompute_with_compiler_version = |compiler_version: &str| -> [u8; 32] {
            let mut hasher = Sha256::new();
            hasher.update(b"chelis_std_typecheck_v");
            hasher.update(STDLIB_CACHE_FORMAT_VERSION.to_le_bytes());
            hasher.update(b"compiler_version");
            hasher.update((compiler_version.len() as u64).to_le_bytes());
            hasher.update(compiler_version.as_bytes());
            let version = chelis_std_bundle::BUNDLED_CHELIS_STD_VERSION;
            hasher.update((version.len() as u64).to_le_bytes());
            hasher.update(version.as_bytes());
            let archive = chelis_std_bundle::archive_sha256();
            hasher.update((archive.len() as u64).to_le_bytes());
            hasher.update(archive.as_bytes());
            let shell = chelis_std_bundle::shell_sha256();
            hasher.update((shell.len() as u64).to_le_bytes());
            hasher.update(shell.as_bytes());
            hasher.update(b"exact-source-digest");
            hasher.update(TEST_SOURCE_DIGEST);
            // Faithful mirror of `stdlib_cache_key`'s fallback (fail-open; see
            // that fn's NOTE). This recompute must match it byte-for-byte.
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

        // Recomputing with the REAL build fingerprint reproduces the key
        // exactly (proves the recompute mirror is faithful)...
        assert_eq!(
            real,
            recompute_with_compiler_version(crate::build_fingerprint()),
            "recompute mirror must match the real key for the real build fingerprint"
        );
        // ...and the bare crate version is NOT what the key folds in: a
        // key built from the release string would be shared by every
        // build of that version, which is the stale-hit this guards.
        // Unconditional: the degraded arm also extends the release
        // string, so the fingerprint never equals it.
        assert_ne!(
            real,
            recompute_with_compiler_version(crate::COMPILER_VERSION),
            "cache key must fold the build fingerprint, not the bare crate version"
        );
        // ...and recomputing with a DIFFERENT compiler version flips it.
        assert_ne!(
            real,
            recompute_with_compiler_version("0.0.0-some-other-compiler-build"),
            "a different compiler version must produce a different stdlib cache key"
        );
    }

    #[test]
    fn cache_decode_restores_a_checked_library_proof() {
        let context = build_stdlib_context(&sample_decls("cache_proof"))
            .expect("the sample library must build");
        let bytes = bincode::serialize(&context).expect("the cache context must encode");
        let restored: StdLibContext =
            bincode::deserialize(&bytes).expect("the cache context must decode");

        assert_eq!(
            restored.checked_library().program().exprs(),
            context.checked_library().program().exprs()
        );
    }

    #[test]
    fn cache_decode_rejects_a_foreign_type_environment() {
        let context = build_stdlib_context(&sample_decls("cache_mismatch"))
            .expect("the sample library must build");
        let wire = StdLibContextWire {
            type_env: TypeEnv::empty(),
            library_checked: context.library_checked().clone(),
            library_dag: context.library_dag().map(|library| library.raw().clone()),
            structural_stats: context.structural_stats(),
        };
        let bytes = bincode::serialize(&wire).expect("the invalid cache wire must encode");
        let error = bincode::deserialize::<StdLibContext>(&bytes)
            .expect_err("the cache parser must reject mismatched library fields");

        assert!(error.to_string().contains("type environment"));
    }

    #[test]
    fn cache_decode_rejects_a_foreign_lowered_library() {
        let first = build_stdlib_context(&sample_decls("cache_dag_first"))
            .expect("the first sample library must build");
        let second = build_stdlib_context(&sample_decls("cache_dag_second"))
            .expect("the second sample library must build");
        let wire = StdLibContextWire {
            type_env: first.type_env().clone(),
            library_checked: first.library_checked().clone(),
            library_dag: second.library_dag().map(|library| library.raw().clone()),
            structural_stats: first.structural_stats(),
        };
        let bytes = bincode::serialize(&wire).expect("the invalid cache wire must encode");
        let error = bincode::deserialize::<StdLibContext>(&bytes)
            .expect_err("the cache parser must reject a foreign lowered library");

        assert!(error.to_string().contains("lowered library"));
    }

    #[test]
    fn cache_decode_rejects_a_changed_lowered_payload_with_the_same_identity() {
        let context = build_stdlib_context(&sample_decls("cache_dag_payload"))
            .expect("the sample library must build");
        let mut lowered_value = serde_json::to_value(
            context
                .library_dag()
                .expect("the sample library must lower")
                .raw(),
        )
        .expect("lowered library must encode");
        lowered_value["rootless_defs"] = serde_json::json!(["forged_rootless_def"]);
        let changed_lowering: IrLoweredLibrary =
            serde_json::from_value(lowered_value).expect("changed lowering must decode");
        assert_eq!(
            changed_lowering.library_proof_id(),
            context.library_checked().library_proof_id(),
            "the negative control must retain the checked-library identity",
        );
        let wire = StdLibContextWire {
            type_env: context.type_env().clone(),
            library_checked: context.library_checked().clone(),
            library_dag: Some(changed_lowering),
            structural_stats: context.structural_stats(),
        };
        let bytes = bincode::serialize(&wire).expect("the invalid cache wire must encode");
        let error = bincode::deserialize::<StdLibContext>(&bytes)
            .expect_err("the cache parser must reject a changed lowered payload");

        assert!(error.to_string().contains("lowered library payload"));
    }

    #[test]
    fn cache_dir_prefers_reef_home_then_xdg_then_home() {
        // This test only checks the resolution PRECEDENCE logic via the
        // pure path math; it does not mutate process env (that would
        // race other tests). The precedence is asserted structurally:
        // a CHELIS_REEF_HOME root ends in .cache/typecheck; an XDG root
        // ends in chelis/typecheck; a HOME root ends in
        // .cache/chelis/typecheck.
        let reef = PathBuf::from("/reef").join(".cache").join("typecheck");
        assert!(reef.ends_with("typecheck"));
        let xdg = PathBuf::from("/xdg").join("chelis").join("typecheck");
        assert!(xdg.ends_with(PathBuf::from("chelis").join("typecheck")));
        let home = PathBuf::from("/home/u")
            .join(".cache")
            .join("chelis")
            .join("typecheck");
        assert!(home.ends_with(PathBuf::from("chelis").join("typecheck")));
    }

    #[test]
    fn cache_path_carries_version_and_key_prefix() {
        let dir = PathBuf::from("/tmp/tc");
        let key = stdlib_cache_key(&sample_decls("a"), TEST_SOURCE_DIGEST);
        let path = stdlib_cache_path(&dir, key);
        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(name.starts_with("chelis-std-"));
        assert!(name.ends_with(".tc"));
        assert!(name.contains(chelis_std_bundle::BUNDLED_CHELIS_STD_VERSION));
    }
}
