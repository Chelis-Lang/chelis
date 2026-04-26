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
use chelis_types::{
    CheckedProgram, TypeEnv, build_type_env_from_library, check_linearity,
    check_phase0e_with_context,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

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
    let reef_state = prepare_reef_graph(package_dir).map_err(|e| reef_error(&e))?;
    let digests = reef_state.source_digests().map_err(|e| hash_error(&e))?;
    let source_hash = ContextHash::from_digests(&digests);

    // Surf → Deep desugar + macro expand of the library decls.
    // `linked_library_decls` is already linked + internal-name-rewritten
    // by `prepare_reef_graph`.
    let deep_library_decls = chelis_macros::expand_program(
        &chelis_surf::desugar::desugar_program(&reef_state.linked_library_decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .map_err(|err| stage_error("desugar", err.to_string(), "macro_error"))?
    .into_exprs();

    // Phase C: build the Phase 0e type-env snapshot from the library.
    let type_env =
        build_type_env_from_library(&deep_library_decls).map_err(|report| CompilerError {
            stage: "check".to_string(),
            errors: report.errors.iter().map(check_error_diagnostic).collect(),
        })?;

    // Run the monolithic library check via `check_phase0e_with_context`
    // against an empty outer scope, then layer effects + linearity. This
    // produces the `library_checked` snapshot that
    // `check_effects_with_context` / `check_linearity_with_context`
    // expect as their library argument.
    let checked =
        check_phase0e_with_context(&TypeEnv::empty(), &deep_library_decls).map_err(|report| {
            CompilerError {
                stage: "check".to_string(),
                errors: report.errors.iter().map(check_error_diagnostic).collect(),
            }
        })?;
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
    let library_checked = check_linearity(&checked).map_err(|errors| CompilerError {
        stage: "linearity".to_string(),
        errors: errors.iter().map(check_error_diagnostic).collect(),
    })?;

    // Phase F: lower the library to a `LoweredLibrary` carrier.
    let library_dag = lower_program_to_library(&library_checked);

    Ok(CompiledContext {
        source_hash,
        reef_state,
        type_env,
        library_checked,
        library_dag,
    })
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
