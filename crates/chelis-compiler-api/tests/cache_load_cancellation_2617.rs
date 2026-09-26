//! chelis#2617: a cache load abandoned by cancellation is a cancellation,
//! never an unusable cache.
//!
//! Decoding any of the three persisted compiler caches (the chelis-std and
//! dependency typecheck caches and the compiled package context) revalidates
//! the cached proof, and revalidation polls the cancel token. Each row primes
//! a valid cache in one fresh process, then loads it in another whose cancel
//! token is already tripped. The load must report the cancellation, must not
//! print the "unusable; rebuilding and overwriting" warning or the internal
//! sentinel, and must leave the valid file byte-for-byte in place. The
//! negative control rewrites the payload so it no longer decodes, with a
//! consistent checksum, and loads it without cancellation: that still warns
//! and rebuilds the file.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use chelis_compiler_api::{
    COMPILER_VERSION, CancelToken, EVAL_CANCELLED_MSG, install_cancel_token,
    load_or_build_library_context, load_or_build_stdlib_context, load_or_compile_for_package,
    stdlib_cache_key,
};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

const WORKER_MODE: &str = "CHELIS_2617_WORKER_MODE";
const WORKER_PACKAGE_ROOT: &str = "CHELIS_2617_PACKAGE_ROOT";
const STDLIB_SOURCE_DIGEST: [u8; 32] = [0x26; 32];
/// `CHELIS_CACHE_ENV_V1\n` + version `u32` + key + payload digest + length.
const TYPECHECK_PAYLOAD_DIGEST_AT: usize = 20 + 4 + 32;
const COMPILED_MAGIC: &[u8] = b"CHELIS_CTX_V23\n";

#[derive(Clone, Copy, Debug)]
enum Site {
    Stdlib,
    Library,
    Compiled,
}

impl Site {
    fn name(self) -> &'static str {
        match self {
            Self::Stdlib => "stdlib",
            Self::Library => "library",
            Self::Compiled => "compiled",
        }
    }

    fn from_name(name: &str) -> Self {
        match name {
            "stdlib" => Self::Stdlib,
            "library" => Self::Library,
            "compiled" => Self::Compiled,
            other => panic!("unknown cache site {other}"),
        }
    }

    fn cache_file(self, reef_home: &Path) -> PathBuf {
        let (dir, prefix, extension) = match self {
            Self::Stdlib => (".cache/typecheck", "chelis-std-", "tc"),
            Self::Library => (".cache/typecheck", "chelis-lib-", "tc"),
            Self::Compiled => (".cache/compiled", "", "ctx"),
        };
        let mut matches = fs::read_dir(reef_home.join(dir))
            .expect("read cache directory")
            .map(|entry| entry.expect("read cache entry").path())
            .filter(|path| {
                path.extension().and_then(|ext| ext.to_str()) == Some(extension)
                    && path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with(prefix))
            })
            .collect::<Vec<_>>();
        assert_eq!(matches.len(), 1, "expected one {self:?} cache: {matches:?}");
        matches.pop().expect("one cache file")
    }
}

fn parse_decls(module: &str, name: &str, value: i32) -> Vec<chelis_surf::ast::Decl> {
    chelis_surf::parser::parse_str(&format!(
        "module {module}\nexport ({name})\ndef {name}() -> i32 = cast({value}, i32)\n"
    ))
    .expect("cache fixture declarations must parse")
}

fn make_package(parent: &Path) -> PathBuf {
    let root = parent.join("cancel-2617");
    fs::create_dir_all(root.join("src")).expect("create package source directory");
    fs::write(
        root.join("reef.toml"),
        format!(
            "[package]\nname = \"cancel-2617\"\nversion = \"0.1.0\"\n\
             compiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Cancel\"\n"
        ),
    )
    .expect("write reef.toml");
    fs::write(
        root.join("src/main.ch"),
        "module Cancel.Main\n\ndef main_value() -> i32 = cast(1, i32)\n",
    )
    .expect("write source");
    fs::canonicalize(root).expect("canonicalize package root")
}

/// Load (or build) the cache at `site`. With `cancelled`, the cancel token is
/// tripped first, and the load must fail as a cancellation.
fn load_site(site: Site, package_root: &Path, reef_home: &Path, cancelled: bool) {
    let stdlib_decls = parse_decls("Cancel.Base", "base_value", 11);
    let token = CancelToken::new();
    let result = match site {
        Site::Stdlib => {
            let _guard = install_cancel_token(token.clone());
            if cancelled {
                token.cancel();
            }
            load_or_build_stdlib_context(&stdlib_decls, STDLIB_SOURCE_DIGEST).map(|_| ())
        }
        Site::Library => {
            let stdlib = load_or_build_stdlib_context(&stdlib_decls, STDLIB_SOURCE_DIGEST)
                .expect("the stdlib layer builds before cancellation");
            let stdlib_key = stdlib_cache_key(&stdlib_decls, STDLIB_SOURCE_DIGEST);
            let dependency = parse_decls("Cancel.Dependency", "dependency_value", 17);
            let _guard = install_cancel_token(token.clone());
            if cancelled {
                token.cancel();
            }
            load_or_build_library_context(&stdlib, stdlib_key, &dependency).map(|built| {
                built.expect("the dependency fixture composes");
            })
        }
        Site::Compiled => {
            let _guard = install_cancel_token(token.clone());
            if cancelled {
                token.cancel();
            }
            load_or_compile_for_package(reef_home, package_root, true).map(|_| ())
        }
    };
    match result {
        Ok(()) => assert!(!cancelled, "a cancelled {site:?} load must not succeed"),
        Err(error) => {
            assert!(cancelled, "{site:?} load failed: {error:?}");
            assert!(
                error.is_cancellation(),
                "a cancelled {site:?} load must fail as a cancellation: {error:?}"
            );
        }
    }
}

#[test]
fn cache_cancellation_worker() {
    let Some(mode) = std::env::var_os(WORKER_MODE) else {
        return;
    };
    let mode = mode.to_string_lossy().into_owned();
    let (site, cancelled) = mode.split_once(':').expect("worker mode is site:state");
    let package_root = PathBuf::from(std::env::var_os(WORKER_PACKAGE_ROOT).expect("package"));
    let reef_home = PathBuf::from(std::env::var_os("CHELIS_REEF_HOME").expect("reef home"));
    load_site(
        Site::from_name(site),
        &package_root,
        &reef_home,
        cancelled == "cancelled",
    );
}

fn run_worker(site: Site, state: &str, package_root: &Path, reef_home: &Path) -> Output {
    let output = Command::new(std::env::current_exe().expect("resolve test binary"))
        .args(["--exact", "cache_cancellation_worker", "--nocapture"])
        .env(WORKER_MODE, format!("{}:{state}", site.name()))
        .env(WORKER_PACKAGE_ROOT, package_root)
        .env("CHELIS_REEF_HOME", reef_home)
        .env_remove("CHELIS_STDLIB_CACHE_DISABLE")
        .output()
        .expect("run fresh worker process");
    assert!(
        output.status.success(),
        "{site:?} {state} worker failed with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

/// Offset of the payload length prefix; the payload follows it.
fn payload_length_at(site: Site, bytes: &[u8]) -> usize {
    match site {
        Site::Stdlib | Site::Library => TYPECHECK_PAYLOAD_DIGEST_AT + 32,
        Site::Compiled => {
            assert!(bytes.starts_with(COMPILED_MAGIC), "compiled-context magic");
            // Magic, version `u32`, source hash, then the identity's two
            // length-prefixed strings, then the payload digest.
            let mut at = COMPILED_MAGIC.len() + 4 + 32;
            for _ in 0..2 {
                let len = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) as usize;
                at += 8 + len;
            }
            at + 32
        }
    }
}

/// Replace the payload with its first half and recompute its digest, so the
/// envelope is intact and only the payload decode fails.
fn truncate_payload(site: Site, bytes: &[u8]) -> Vec<u8> {
    let length_at = payload_length_at(site, bytes);
    let digest_at = length_at - 32;
    let len = u64::from_le_bytes(bytes[length_at..length_at + 8].try_into().unwrap()) as usize;
    let payload_at = length_at + 8;
    assert_eq!(
        payload_at + len,
        bytes.len(),
        "{site:?} payload ends the file"
    );
    let payload = &bytes[payload_at..payload_at + len / 2];
    let mut changed = bytes[..digest_at].to_vec();
    changed.extend_from_slice(&Sha256::digest(payload));
    changed.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    changed.extend_from_slice(payload);
    changed
}

fn assert_cancelled_load_leaves_cache_alone(site: Site) {
    let package_dir = TempDir::new().expect("package tempdir");
    let reef_home = TempDir::new().expect("reef home tempdir");
    let package_root = make_package(package_dir.path());
    run_worker(site, "prime", &package_root, reef_home.path());
    let cache = site.cache_file(reef_home.path());
    let valid = fs::read(&cache).expect("read primed cache");

    let output = run_worker(site, "cancelled", &package_root, reef_home.path());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("unusable"),
        "a cancelled {site:?} load must not call the cache unusable: {stderr}"
    );
    assert!(
        !stderr.contains(EVAL_CANCELLED_MSG),
        "the internal cancellation sentinel must not reach stderr: {stderr}"
    );
    assert!(
        fs::read(&cache).expect("reread cache") == valid,
        "a cancelled {site:?} load must leave the valid cache file unchanged"
    );
}

fn assert_undecodable_payload_warns_and_rebuilds(site: Site) {
    let package_dir = TempDir::new().expect("package tempdir");
    let reef_home = TempDir::new().expect("reef home tempdir");
    let package_root = make_package(package_dir.path());
    run_worker(site, "prime", &package_root, reef_home.path());
    let cache = site.cache_file(reef_home.path());
    let valid = fs::read(&cache).expect("read primed cache");
    let broken = truncate_payload(site, &valid);
    fs::write(&cache, &broken).expect("write undecodable payload");

    let output = run_worker(site, "rebuild", &package_root, reef_home.path());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unusable") && stderr.contains("cache decode error"),
        "an undecodable {site:?} payload must warn as a decode failure: {stderr}"
    );
    assert!(
        fs::read(&cache).expect("reread cache") == valid,
        "an undecodable {site:?} cache must be rebuilt and overwritten"
    );
}

#[test]
fn cancelled_stdlib_cache_load_is_a_cancellation() {
    assert_cancelled_load_leaves_cache_alone(Site::Stdlib);
}

#[test]
fn cancelled_library_cache_load_is_a_cancellation() {
    assert_cancelled_load_leaves_cache_alone(Site::Library);
}

#[test]
fn cancelled_compiled_context_load_is_a_cancellation() {
    assert_cancelled_load_leaves_cache_alone(Site::Compiled);
}

#[test]
fn undecodable_stdlib_cache_still_warns_and_rebuilds() {
    assert_undecodable_payload_warns_and_rebuilds(Site::Stdlib);
}

#[test]
fn undecodable_library_cache_still_warns_and_rebuilds() {
    assert_undecodable_payload_warns_and_rebuilds(Site::Library);
}

#[test]
fn undecodable_compiled_context_still_warns_and_rebuilds() {
    assert_undecodable_payload_warns_and_rebuilds(Site::Compiled);
}
