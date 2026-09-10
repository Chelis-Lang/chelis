//! Phase B exact-byte oracle for every persisted compiler-cache root.
//!
//! The parent rewrites one fixed package through several filesystem insertion
//! orders and re-executes this test binary 24 times. Each worker therefore has
//! a fresh randomized hash state while compiling identical source at the same
//! canonical package root. The artifacts are compared as bytes, not by a
//! decoded approximation.

use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use chelis_compiler_api::{
    COMPILER_VERSION, CacheError, CompiledContext, compile_reef_context, library_cache_key,
    library_cache_key_input_bytes, load_or_build_library_context, load_or_build_stdlib_context,
    stdlib_cache_key, stdlib_cache_key_input_bytes,
};
use chelis_reef::{prepare_reef_graph_cached, prepared_graph_cache_key_input_bytes};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

const WORKER_PACKAGE_ROOT: &str = "CHELIS_HASH_ORDER_CACHE_PACKAGE_ROOT";
const WORKER_REEF_HOME: &str = "CHELIS_HASH_ORDER_CACHE_REEF_HOME";
const WORKER_RESULT_DIR: &str = "CHELIS_HASH_ORDER_CACHE_RESULT_DIR";
const PROCESS_BUDGET: usize = 24;
const SYNTHETIC_STDLIB_SOURCE_DIGEST: [u8; 32] = [0x6b; 32];

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Artifact {
    StdlibKey,
    StdlibCacheKeyInputs,
    LibraryKey,
    LibraryCacheKeyInputs,
    StdlibCache,
    LibraryCache,
    CompiledEncode,
    CompiledSave,
    CliWorkerHandoff,
    CompiledCacheKey,
    CompiledCacheKeyInputs,
    PreparedGraphEncode,
    PreparedGraphCache,
    PreparedGraphCacheKey,
    PreparedGraphCacheKeyInputs,
    PreparedGraphStdlibSourceDigest,
}

const ALL_ARTIFACTS: [Artifact; 16] = [
    Artifact::StdlibKey,
    Artifact::StdlibCacheKeyInputs,
    Artifact::LibraryKey,
    Artifact::LibraryCacheKeyInputs,
    Artifact::StdlibCache,
    Artifact::LibraryCache,
    Artifact::CompiledEncode,
    Artifact::CompiledSave,
    Artifact::CliWorkerHandoff,
    Artifact::CompiledCacheKey,
    Artifact::CompiledCacheKeyInputs,
    Artifact::PreparedGraphEncode,
    Artifact::PreparedGraphCache,
    Artifact::PreparedGraphCacheKey,
    Artifact::PreparedGraphCacheKeyInputs,
    Artifact::PreparedGraphStdlibSourceDigest,
];

impl Artifact {
    const fn file_name(self) -> &'static str {
        match self {
            Self::StdlibKey => "stdlib_key.bin",
            Self::StdlibCacheKeyInputs => "stdlib_cache_key_inputs.bin",
            Self::LibraryKey => "library_key.bin",
            Self::LibraryCacheKeyInputs => "library_cache_key_inputs.bin",
            Self::StdlibCache => "stdlib_cache.bin",
            Self::LibraryCache => "library_cache.bin",
            Self::CompiledEncode => "compiled_encode.bin",
            Self::CompiledSave => "compiled_save.bin",
            Self::CliWorkerHandoff => "cli_worker_handoff.bin",
            Self::CompiledCacheKey => "compiled_cache_key.bin",
            Self::CompiledCacheKeyInputs => "compiled_cache_key_inputs.bin",
            Self::PreparedGraphEncode => "prepared_graph_encode.bin",
            Self::PreparedGraphCache => "prepared_graph_cache.bin",
            Self::PreparedGraphCacheKey => "prepared_graph_cache_key.bin",
            Self::PreparedGraphCacheKeyInputs => "prepared_graph_cache_key_inputs.bin",
            Self::PreparedGraphStdlibSourceDigest => "prepared_graph_stdlib_source_digest.bin",
        }
    }
}

struct ArtifactWriter<'a> {
    result_dir: &'a Path,
    produced: BTreeSet<Artifact>,
}

impl<'a> ArtifactWriter<'a> {
    fn new(result_dir: &'a Path) -> Self {
        fs::create_dir_all(result_dir).expect("create worker result directory");
        Self {
            result_dir,
            produced: BTreeSet::new(),
        }
    }

    fn write(&mut self, artifact: Artifact, bytes: impl AsRef<[u8]>) {
        assert!(
            self.produced.insert(artifact),
            "artifact producer wrote {:?} twice",
            artifact
        );
        fs::write(self.result_dir.join(artifact.file_name()), bytes)
            .unwrap_or_else(|error| panic!("write {:?}: {error}", artifact));
    }

    fn copy(&mut self, artifact: Artifact, source: &Path) {
        let bytes = fs::read(source)
            .unwrap_or_else(|error| panic!("read {:?} producer source: {error}", artifact));
        self.write(artifact, bytes);
    }

    fn finish(self) {
        assert_eq!(
            self.produced,
            BTreeSet::from(ALL_ARTIFACTS),
            "the exact artifact registry and executed producers must be bijective"
        );
    }
}

const SOURCE_PERMUTATIONS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

const SOURCES: [(&str, &str); 3] = [
    (
        "main.ch",
        "module HashOrderCache.Main\n\ndef main_value() -> int32 = cast(1, int32)\n",
    ),
    (
        "alpha.ch",
        "module HashOrderCache.Alpha\n\ndef alpha_value() -> int32 = cast(2, int32)\n",
    ),
    (
        "beta.ch",
        "module HashOrderCache.Beta\n\ndef beta_value() -> int32 = cast(3, int32)\n",
    ),
];

fn parse_decls(module: &str, name: &str, value: i32) -> Vec<chelis_surf::ast::Decl> {
    chelis_surf::parser::parse_str(&format!(
        "module {module}\nexport ({name})\ndef {name}() -> int32 = cast({value}, int32)\n"
    ))
    .expect("cache fixture declarations must parse")
}

fn make_package() -> (TempDir, PathBuf) {
    let guard = TempDir::new().expect("package tempdir");
    let root = guard.path().join("hash-order-cache");
    fs::create_dir_all(&root).expect("create package root");
    fs::write(
        root.join("reef.toml"),
        format!(
            "[package]\nname = \"hash-order-cache\"\nversion = \"0.1.0\"\n\
             compiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"HashOrderCache\"\n"
        ),
    )
    .expect("write reef.toml");
    write_sources(&root, 0);
    let root = fs::canonicalize(root).expect("canonicalize package root");
    (guard, root)
}

fn write_sources(root: &Path, permutation: usize) {
    let src = root.join("src");
    if src.exists() {
        fs::remove_dir_all(&src).expect("remove prior source insertion order");
    }
    fs::create_dir(&src).expect("recreate source directory");
    for source_index in SOURCE_PERMUTATIONS[permutation % SOURCE_PERMUTATIONS.len()] {
        let (name, contents) = SOURCES[source_index];
        fs::write(src.join(name), contents).expect("write source fixture");
    }
}

fn unique_cache_file(cache_dir: &Path, prefix: &str) -> PathBuf {
    let mut matches = fs::read_dir(cache_dir)
        .expect("read typecheck cache directory")
        .map(|entry| entry.expect("read cache directory entry").path())
        .filter(|path| {
            path.file_name()
                .and_then(OsStr::to_str)
                .is_some_and(|name| name.starts_with(prefix) && name.ends_with(".tc"))
        })
        .collect::<Vec<_>>();
    matches.sort();
    assert_eq!(
        matches.len(),
        1,
        "expected exactly one {prefix} cache artifact, found {matches:?}"
    );
    matches.pop().expect("one cache artifact")
}

fn unique_prepared_graph_cache_file(reef_home: &Path) -> PathBuf {
    let cache_dir = reef_home.join(".cache/prepared-graphs");
    let mut matches = fs::read_dir(&cache_dir)
        .expect("read prepared-graph cache directory")
        .map(|entry| entry.expect("read prepared-graph cache entry").path())
        .filter(|path| path.extension() == Some(OsStr::new("graph")))
        .collect::<Vec<_>>();
    matches.sort();
    assert_eq!(
        matches.len(),
        1,
        "expected exactly one prepared-graph cache artifact, found {matches:?}"
    );
    matches.pop().expect("one prepared-graph cache artifact")
}

fn write_worker_artifacts(package_root: &Path, reef_home: &Path, result_dir: &Path) {
    let mut artifacts = ArtifactWriter::new(result_dir);

    let stdlib_decls = parse_decls("HashOrderCache.Base", "base_value", 11);
    let dependency_decls = parse_decls("HashOrderCache.Dependency", "dependency_value", 17);
    let stdlib_key = stdlib_cache_key(&stdlib_decls, SYNTHETIC_STDLIB_SOURCE_DIGEST);
    let stdlib_context =
        load_or_build_stdlib_context(&stdlib_decls, SYNTHETIC_STDLIB_SOURCE_DIGEST)
            .expect("build stdlib cache context");
    let library_key = library_cache_key(&dependency_decls, stdlib_key);
    load_or_build_library_context(&stdlib_context, stdlib_key, &dependency_decls)
        .expect("build library cache context")
        .expect("dependency fixture must compose");

    let cache_dir = reef_home.join(".cache/typecheck");
    artifacts.copy(
        Artifact::StdlibCache,
        &unique_cache_file(&cache_dir, "chelis-std-"),
    );
    artifacts.copy(
        Artifact::LibraryCache,
        &unique_cache_file(&cache_dir, "chelis-lib-"),
    );
    artifacts.write(Artifact::StdlibKey, stdlib_key);
    let stdlib_key_inputs =
        stdlib_cache_key_input_bytes(&stdlib_decls, SYNTHETIC_STDLIB_SOURCE_DIGEST);
    let stdlib_key_from_inputs: [u8; 32] = Sha256::digest(&stdlib_key_inputs).into();
    assert_eq!(
        stdlib_key_from_inputs, stdlib_key,
        "the recorded stdlib input root must be the exact key preimage"
    );
    artifacts.write(Artifact::StdlibCacheKeyInputs, stdlib_key_inputs);
    artifacts.write(Artifact::LibraryKey, library_key);
    let library_key_inputs = library_cache_key_input_bytes(&dependency_decls, stdlib_key);
    let library_key_from_inputs: [u8; 32] = Sha256::digest(&library_key_inputs).into();
    assert_eq!(
        library_key_from_inputs, library_key,
        "the recorded library input root must be the exact key preimage"
    );
    artifacts.write(Artifact::LibraryCacheKeyInputs, library_key_inputs);

    let prepared = prepare_reef_graph_cached(package_root).expect("prepare fixed package graph");
    artifacts.write(
        Artifact::PreparedGraphEncode,
        prepared.encode().expect("encode prepared graph"),
    );
    artifacts.write(
        Artifact::PreparedGraphStdlibSourceDigest,
        prepared.stdlib_source_digest(),
    );
    let prepared_cache = unique_prepared_graph_cache_file(reef_home);
    artifacts.copy(Artifact::PreparedGraphCache, &prepared_cache);
    let prepared_key = prepared_cache
        .file_name()
        .and_then(OsStr::to_str)
        .expect("prepared graph cache filename is UTF-8");
    artifacts.write(Artifact::PreparedGraphCacheKey, prepared_key.as_bytes());
    let prepared_key_inputs = prepared_graph_cache_key_input_bytes(package_root)
        .expect("construct the production prepared-graph cache-key preimage");
    let prepared_key_digest: [u8; 32] = Sha256::digest(&prepared_key_inputs).into();
    let expected_prepared_key = format!(
        "{}.graph",
        prepared_key_digest[..16]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    assert_eq!(
        prepared_key, expected_prepared_key,
        "the recorded prepared-graph input root must be the exact filename preimage"
    );
    artifacts.write(Artifact::PreparedGraphCacheKeyInputs, prepared_key_inputs);

    let context =
        compile_reef_context(reef_home, package_root).expect("compile fixed package context");
    let encoded = context.encode().expect("encode compiled context");
    artifacts.write(Artifact::CompiledEncode, &encoded);

    // The CLI parent writes `CompiledContext::encode()` directly to the worker
    // tempfile. This artifact locks that public handoff root under its product
    // name while making the intentional byte identity explicit.
    artifacts.write(Artifact::CliWorkerHandoff, &encoded);

    let compiled_save = result_dir.join("compiled-save-producer.tmp");
    context
        .save(&compiled_save)
        .expect("save compiled-context envelope");
    artifacts.copy(Artifact::CompiledSave, &compiled_save);
    fs::remove_file(compiled_save).expect("remove compiled-save producer temporary");
    let cache_key = CompiledContext::cache_file_name(
        ("hash-order-cache", "0.1.0"),
        context.source_hash,
        &context.identity,
    );
    artifacts.write(Artifact::CompiledCacheKey, cache_key.as_bytes());
    let key_inputs = bincode::serialize(&(
        "hash-order-cache",
        "0.1.0",
        context.source_hash,
        context.identity.clone(),
    ))
    .expect("serialize compiled cache-key inputs");
    artifacts.write(Artifact::CompiledCacheKeyInputs, key_inputs);
    artifacts.finish();
}

#[test]
fn hash_order_cache_worker() {
    let Some(package_root) = std::env::var_os(WORKER_PACKAGE_ROOT) else {
        return;
    };
    let reef_home = PathBuf::from(
        std::env::var_os(WORKER_REEF_HOME).expect("worker reef-home environment variable"),
    );
    let result_dir = PathBuf::from(
        std::env::var_os(WORKER_RESULT_DIR).expect("worker result environment variable"),
    );
    write_worker_artifacts(&PathBuf::from(package_root), &reef_home, &result_dir);
}

fn read_artifacts(result_dir: &Path) -> Vec<(&'static str, Vec<u8>)> {
    ALL_ARTIFACTS
        .iter()
        .map(|artifact| {
            let name = artifact.file_name();
            (
                name,
                fs::read(result_dir.join(name)).unwrap_or_else(|error| {
                    panic!("read worker artifact {name} from {result_dir:?}: {error}")
                }),
            )
        })
        .collect()
}

#[test]
fn cache_roots_are_exact_bytes_across_24_fresh_processes() {
    let (_package_guard, package_root) = make_package();
    let scratch = TempDir::new().expect("worker scratch tempdir");
    let test_binary = std::env::current_exe().expect("resolve current test binary");
    let mut reference: Option<Vec<(&'static str, Vec<u8>)>> = None;

    for worker in 0..PROCESS_BUDGET {
        write_sources(&package_root, worker);
        let reef_home = scratch.path().join(format!("reef-home-{worker}"));
        let result_dir = scratch.path().join(format!("result-{worker}"));
        let output = Command::new(&test_binary)
            .args(["--exact", "hash_order_cache_worker", "--nocapture"])
            .env(WORKER_PACKAGE_ROOT, &package_root)
            .env(WORKER_REEF_HOME, &reef_home)
            .env(WORKER_RESULT_DIR, &result_dir)
            .env("CHELIS_REEF_HOME", &reef_home)
            .env_remove("CHELIS_STDLIB_CACHE_DISABLE")
            .output()
            .expect("run fresh cache worker process");
        assert!(
            output.status.success(),
            "cache worker {worker} failed with {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let observed = read_artifacts(&result_dir);
        if let Some(expected) = &reference {
            for ((expected_name, expected_bytes), (observed_name, observed_bytes)) in
                expected.iter().zip(&observed)
            {
                assert_eq!(expected_name, observed_name);
                assert_eq!(
                    expected_bytes, observed_bytes,
                    "{observed_name} changed in fresh process {worker}"
                );
            }
        } else {
            reference = Some(observed);
        }
    }

    let reference = reference.expect("the process budget is nonzero");
    let saved = reference
        .iter()
        .find(|(name, _)| *name == "compiled_save.bin")
        .expect("compiled saved bytes are part of the artifact set")
        .1
        .clone();
    assert!(saved.starts_with(b"CHELIS_CTX_V19\n"));
    let mut preceding = b"CHELIS_CTX_V18\n".to_vec();
    preceding.extend_from_slice(&saved[b"CHELIS_CTX_V19\n".len()..]);
    let preceding_path = scratch.path().join("compiled-context-v18.ctx");
    fs::write(&preceding_path, preceding).expect("write preceding-format fixture");
    assert!(matches!(
        CompiledContext::load_if_fresh(&preceding_path, scratch.path(), &package_root),
        Err(CacheError::Corrupt(_))
    ));
}

#[test]
fn semantic_cache_input_mutations_change_keys() {
    let stdlib_a = parse_decls("HashOrderCache.Base", "base_value", 11);
    let stdlib_b = parse_decls("HashOrderCache.Base", "base_value", 12);
    let stdlib_key_a = stdlib_cache_key(&stdlib_a, SYNTHETIC_STDLIB_SOURCE_DIGEST);
    let stdlib_key_b = stdlib_cache_key(&stdlib_b, SYNTHETIC_STDLIB_SOURCE_DIGEST);
    assert_ne!(stdlib_key_a, stdlib_key_b);

    let dependency_a = parse_decls("HashOrderCache.Dependency", "dependency_value", 17);
    let dependency_b = parse_decls("HashOrderCache.Dependency", "dependency_value", 18);
    assert_ne!(
        library_cache_key(&dependency_a, stdlib_key_a),
        library_cache_key(&dependency_b, stdlib_key_a)
    );
    assert_ne!(
        library_cache_key(&dependency_a, stdlib_key_a),
        library_cache_key(&dependency_a, stdlib_key_b)
    );
}
