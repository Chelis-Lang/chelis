//! Phase B: skeleton for the chelis Compiled Artifact Caching push.
//!
//! `CompiledContext` is the cacheable artifact a reef package compiles
//! into once. New code (test files, eval inputs) will be checked /
//! lowered against the context's pre-checked, pre-lowered library defs;
//! the context's contents are referenced, not recompiled.
//!
//! Phase B (this file) ships the type skeleton + a stub
//! `compile_reef_context` that internally calls `prepare_reef_graph`
//! (the existing pre-pipeline work) and content-hashes every backed
//! source file. Phases C/D/E/F/G fill in the real pipeline-splitting
//! machinery so that `eval_in_context` only re-compiles the new source.
//!
//! See `/home/jeff/.claude/plans/now-plan-out-the-shimmying-wand.md`
//! for the full plan.

use chelis_reef::{PreparedReefGraph, SourceDigest, prepare_reef_graph};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

use crate::compiler::CompilerError;
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
/// Phase B has the skeleton: `source_hash` (for cache invalidation) and
/// `reef_state` (the existing pre-pipeline graph). Phases C and F will
/// add `library_checked` (pre-checked decls) and `library_dag` (pre-
/// lowered IR); Phase G composes them into the new public APIs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledContext {
    /// Content hash of every backed source file. Stable across calls
    /// on unchanged sources; changes when ANY source byte changes.
    pub source_hash: ContextHash,
    /// The reef state (lockfile-backed package graph + linked library
    /// decls + internal-name maps + dep shells). Phase G will keep
    /// using `compile_with_reef_graph` against this.
    pub(crate) reef_state: PreparedReefGraph,
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
/// Phase B stub: calls `prepare_reef_graph` + content-hashes every
/// backed source file. Phases C/D/E/F will extend this to also produce
/// the pre-checked / pre-lowered fields. The signature is the FINAL one
/// — callers written against this API in parallel won't change.
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
    Ok(CompiledContext {
        source_hash,
        reef_state,
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
