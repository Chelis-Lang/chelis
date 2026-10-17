//! chelis#3744 and chelis#2256: what `chelis test`'s parent hands its workers.
//!
//! The parent only hands the compiled context on. On a warm run it reads the
//! disk cache, verifies it through the untrusted route, and must then seal the
//! worker handoff from the verified bytes rather than encode the decoded
//! context again; on a cold run one encoding serves both the cache file and the
//! handoff. The oracle is the compiler's own count of context payload encodes.
//!
//! The handoff carries the checked parts and no lowering, because the worker
//! re-derives the lowering and would only discard a transmitted one
//! (chelis#2256). The disk payload is the handoff payload followed by the
//! lowering, which is what lets a verified cache entry supply the handoff
//! without an encode.
//!
//! The steps share one test function on purpose: the count is process-wide,
//! and `cargo test` runs the tests of one binary on parallel threads.

use std::fs;
use std::path::{Path, PathBuf};

use bincode::Options;
use chelis_compiler_api::{
    COMPILER_VERSION, CompiledContext, compiled_context_payload_encodes,
    load_or_compile_package_context,
};
use chelis_reef::EntryImports;
use tempfile::TempDir;

fn package_fixture() -> (TempDir, PathBuf) {
    let dir = TempDir::new().expect("tempdir");
    let root = dir.path().join("handoff");
    fs::create_dir_all(root.join("src")).expect("mkdir src");
    fs::write(
        root.join("reef.toml"),
        format!(
            "[package]\nname = \"handoff\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\n\
             module_prefix = \"Handoff\"\n\n[dependencies]\nchelis-std = {{ version = \"{}\" }}\n",
            chelis_std_bundle::BUNDLED_CHELIS_STD_VERSION
        ),
    )
    .expect("write reef.toml");
    let mut source = String::from(
        "module Handoff.Main\nimport Std.Scalar\nexport (scale)\n\
         def scale(x: f32) -> f32 = x * 2.5\n",
    );
    for index in 0..40 {
        source.push_str(&format!(
            "def bulk_{index}(x: i64) -> i64 = add(mul(x, 3i64), {index}i64)\n"
        ));
    }
    fs::write(root.join("src/main.ch"), source).expect("write main.ch");
    let root = fs::canonicalize(root).expect("canonicalize the package root");
    chelis_reef::prepare_program_for_file(
        &root.join("src/main.ch"),
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("the fixture package must resolve and write its lock")
    .expect("the entry file is inside the fixture package");
    (dir, root)
}

/// Mirror of the private envelope.
#[derive(serde::Deserialize)]
struct EnvelopeMirror {
    _version: u32,
    _source_hash: chelis_compiler_api::ContextHash,
    _identity: chelis_compiler_api::CacheIdentity,
    _payload_sha256: [u8; 32],
    payload: Vec<u8>,
}

/// Mirror of the handoff payload: the checked parts, no lowering.
#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct HandoffWireMirror {
    source_hash: chelis_compiler_api::ContextHash,
    identity: chelis_compiler_api::CacheIdentity,
    reef_state: chelis_reef::PreparedReefGraph,
    type_env: chelis_types::TypeEnv,
    library_checked: chelis_types::CheckedProgram,
}

fn envelope_payload(bytes: &[u8]) -> Vec<u8> {
    let magic = bytes
        .iter()
        .position(|byte| *byte == b'\n')
        .expect("the envelope carries its magic prefix")
        + 1;
    bincode::deserialize::<EnvelopeMirror>(&bytes[magic..])
        .expect("the envelope decodes")
        .payload
}

fn load(reef_home: &Path, root: &Path) -> chelis_compiler_api::LoadedContext {
    load_or_compile_package_context(
        reef_home,
        root,
        &EntryImports::none(),
        false,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("the fixture package must compile")
}

fn the_only_cache_file(reef_home: &Path) -> PathBuf {
    let entries: Vec<PathBuf> = fs::read_dir(reef_home.join(".cache/compiled"))
        .expect("the cold load writes the cache directory")
        .map(|entry| entry.expect("cache entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "ctx"))
        .collect();
    assert_eq!(entries.len(), 1, "one cache file: {entries:?}");
    entries.into_iter().next().expect("length checked")
}

#[test]
fn the_parent_hands_workers_verified_cache_bytes_without_the_lowering() {
    let (_dir, root) = package_fixture();
    let reef_home = TempDir::new().expect("reef home");

    // Cold: one encode writes the cache file and supplies the handoff.
    let before = compiled_context_payload_encodes();
    let cold = load(reef_home.path(), &root);
    let (cold_bytes, cold_digest) = cold.encode_for_handoff().expect("cold handoff");
    assert_eq!(
        compiled_context_payload_encodes() - before,
        1,
        "a cold load must encode the context once for both the cache file and the handoff"
    );
    let cache_bytes = fs::read(the_only_cache_file(reef_home.path())).expect("read cache file");

    // Warm: the handoff is sealed from the verified cache payload.
    let before = compiled_context_payload_encodes();
    let warm = load(reef_home.path(), &root);
    let (warm_bytes, warm_digest) = warm.encode_for_handoff().expect("warm handoff");
    assert_eq!(
        compiled_context_payload_encodes() - before,
        0,
        "a warm load re-encoded the context it had just verified; the handoff must be sealed \
         from the verified cache payload (chelis#3744)"
    );
    assert_eq!(
        warm_bytes, cold_bytes,
        "warm and cold runs hand workers different bytes"
    );
    assert_eq!(warm_digest, cold_digest);

    // The equivalence the warm route relies on: what it seals from the cache
    // payload is what encoding the verified context produces.
    let context = warm.into_context();
    let (encoded, encoded_digest) = context
        .encode_for_handoff()
        .expect("encode the verified context");
    assert!(
        encoded == warm_bytes,
        "the handoff sealed from the cache payload differs from encoding the context it decodes to"
    );
    assert_eq!(encoded_digest, warm_digest);

    // chelis#2256: the handoff payload is exactly the checked parts. A
    // lowering left in it would be trailing bytes here.
    let handoff_payload = envelope_payload(&warm_bytes);
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .deserialize::<HandoffWireMirror>(&handoff_payload)
        .expect("the handoff payload is the checked parts and nothing else");
    // The disk payload is those bytes followed by the lowering.
    let disk_payload = envelope_payload(&cache_bytes);
    assert!(
        disk_payload.len() > handoff_payload.len() && disk_payload.starts_with(&handoff_payload),
        "the disk payload must be the handoff payload followed by the lowering ({} vs {} bytes)",
        disk_payload.len(),
        handoff_payload.len()
    );
    let lowering: chelis_ir::lower::LoweredLibrary =
        bincode::deserialize(&disk_payload[handoff_payload.len()..])
            .expect("the disk payload's second section is the lowering");
    assert!(
        bincode::serialize(&lowering).expect("encode the lowering")
            == disk_payload[handoff_payload.len()..],
        "the disk payload ends with exactly the lowering"
    );

    // The worker route accepts the handoff and reconstructs the same context
    // the disk route does.
    let worker = CompiledContext::decode_authenticated(&warm_bytes, &warm_digest)
        .expect("the worker route accepts the handoff");
    assert!(
        bincode::serialize(&worker).expect("encode the worker's context")
            == bincode::serialize(&context).expect("encode the verified context"),
        "the worker reconstructs a different context from the handoff"
    );

    // Each route names its own shape and refuses the other's before decoding.
    let error = CompiledContext::decode(&warm_bytes)
        .expect_err("the disk route must refuse handoff bytes, which carry no lowering");
    assert!(error.contains("magic"), "{error}");
    let error = CompiledContext::decode_authenticated(&cache_bytes, &warm_digest)
        .expect_err("the worker route must refuse disk-cache bytes");
    assert!(error.contains("magic"), "{error}");
}
