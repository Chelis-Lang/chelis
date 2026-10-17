//! chelis#3744 and chelis#2256: what `chelis test`'s parent hands its workers.
//!
//! The parent only hands the compiled context on. On a warm run it reads the
//! disk cache and verifies it through the untrusted route, and must then pass
//! the verified bytes through rather than encode the decoded context again; on
//! a cold run one encoding serves both the cache file and the handoff. The
//! oracle is the compiler's own count of context envelope encodes.
//!
//! The worker re-derives the lowering from the checked library, so its route
//! never decodes the transmitted lowering (chelis#2256): a handoff whose
//! lowering section is cut away still reconstructs the same context there.
//!
//! The steps share one test function on purpose: the count is process-wide,
//! and `cargo test` runs the tests of one binary on parallel threads.

use std::fs;
use std::path::{Path, PathBuf};

use bincode::Options;
use chelis_compiler_api::{
    COMPILER_VERSION, CompiledContext, HandoffDigest, compiled_context_payload_encodes,
    load_or_compile_package_context,
};
use chelis_reef::EntryImports;
use sha2::{Digest, Sha256};
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
#[derive(serde::Serialize, serde::Deserialize)]
struct EnvelopeMirror {
    version: u32,
    source_hash: chelis_compiler_api::ContextHash,
    identity: chelis_compiler_api::CacheIdentity,
    payload_sha256: [u8; 32],
    payload: Vec<u8>,
}

/// Mirror of the checked parts that lead the payload.
#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct CheckedPartsMirror {
    source_hash: chelis_compiler_api::ContextHash,
    identity: chelis_compiler_api::CacheIdentity,
    reef_state: chelis_reef::PreparedReefGraph,
    type_env: chelis_types::TypeEnv,
    library_checked: chelis_types::CheckedProgram,
}

fn magic_len(bytes: &[u8]) -> usize {
    bytes
        .iter()
        .position(|byte| *byte == b'\n')
        .expect("the envelope carries its magic prefix")
        + 1
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
fn the_parent_hands_workers_the_verified_cache_bytes() {
    let (_dir, root) = package_fixture();
    let reef_home = TempDir::new().expect("reef home");

    // Cold: one encode writes the cache file and supplies the handoff.
    let before = compiled_context_payload_encodes();
    let (cold_bytes, cold_digest) = load(reef_home.path(), &root)
        .into_handoff()
        .expect("cold handoff");
    assert_eq!(
        compiled_context_payload_encodes() - before,
        1,
        "a cold load must encode the context once for both the cache file and the handoff"
    );
    let cache_bytes = fs::read(the_only_cache_file(reef_home.path())).expect("read cache file");
    assert!(
        cold_bytes == cache_bytes,
        "the cold handoff differs from the cache file it was saved as"
    );

    // Warm: the handoff is the verified cache file itself.
    let before = compiled_context_payload_encodes();
    let warm = load(reef_home.path(), &root);
    let context = load(reef_home.path(), &root).into_context();
    let (warm_bytes, warm_digest) = warm.into_handoff().expect("warm handoff");
    assert_eq!(
        compiled_context_payload_encodes() - before,
        0,
        "a warm load re-encoded the context it had just verified; the handoff must be the \
         verified cache bytes (chelis#3744)"
    );
    assert!(
        warm_bytes == cache_bytes,
        "the warm handoff is not the cache file"
    );
    assert_eq!(warm_digest, cold_digest);

    // The equivalence the warm route relies on: encoding the context the cache
    // decodes to reproduces the cache bytes and digest.
    let (encoded, encoded_digest) = context
        .encode_for_handoff()
        .expect("encode the verified context");
    assert!(
        encoded == cache_bytes,
        "encoding the decoded cache entry does not reproduce its bytes"
    );
    assert_eq!(encoded_digest, warm_digest);

    // The worker route accepts the handoff and reconstructs the same context
    // the disk route does.
    let worker = CompiledContext::decode_authenticated(&warm_bytes, &warm_digest)
        .expect("the worker route accepts the handoff");
    let expected = bincode::serialize(&context).expect("encode the verified context");
    assert!(
        bincode::serialize(&worker).expect("encode the worker's context") == expected,
        "the worker reconstructs a different context from the handoff"
    );

    // chelis#2256: the worker route never decodes the transmitted lowering.
    // Cut the lowering away, reseal the envelope, and deliver the digest of
    // what is left: the worker still reconstructs the same context, while the
    // disk route, which compares against the lowering, refuses the bytes.
    let magic = magic_len(&warm_bytes);
    let mut envelope: EnvelopeMirror =
        bincode::deserialize(&warm_bytes[magic..]).expect("the envelope decodes");
    let mut rest = envelope.payload.as_slice();
    let _: CheckedPartsMirror =
        serde::Deserialize::deserialize(&mut bincode::Deserializer::with_reader(
            &mut rest,
            bincode::DefaultOptions::new()
                .with_fixint_encoding()
                .allow_trailing_bytes(),
        ))
        .expect("the payload leads with the checked parts");
    let parts_len = envelope.payload.len() - rest.len();
    assert!(
        parts_len < envelope.payload.len(),
        "the payload carries a lowering after the checked parts"
    );
    // Not decoding the lowering does not mean not verifying it: the worker
    // checks the producer's digest over every payload byte first. A byte
    // flipped inside the lowering is refused, whether or not the envelope's
    // own digest is resealed to match.
    let mut tampered = EnvelopeMirror {
        payload: envelope.payload.clone(),
        ..bincode::deserialize(&warm_bytes[magic..]).expect("the envelope decodes")
    };
    let inside_lowering = parts_len + (tampered.payload.len() - parts_len) / 2;
    tampered.payload[inside_lowering] ^= 0x01;
    for reseal in [false, true] {
        if reseal {
            tampered.payload_sha256 = Sha256::digest(&tampered.payload).into();
        }
        let mut bytes = warm_bytes[..magic].to_vec();
        bytes.extend(bincode::serialize(&tampered).expect("encode the tampered envelope"));
        let error = CompiledContext::decode_authenticated(&bytes, &warm_digest)
            .expect_err("a byte flipped inside the lowering must not authenticate");
        let expected_reason = if reseal {
            "digest its producer delivered"
        } else {
            "sha256"
        };
        assert!(
            error.contains(expected_reason),
            "resealed={reseal}: the rejection must name the failed digest: {error}"
        );
    }

    envelope.payload.truncate(parts_len);
    envelope.payload_sha256 = Sha256::digest(&envelope.payload).into();
    let mut unlowered = warm_bytes[..magic].to_vec();
    unlowered.extend(bincode::serialize(&envelope).expect("reseal the envelope"));
    let digest = HandoffDigest::from_hex(&format!("{:x}", Sha256::digest(&envelope.payload)))
        .expect("a freshly computed digest parses");
    let worker = CompiledContext::decode_authenticated(&unlowered, &digest)
        .expect("the worker route reads only the checked parts");
    assert!(
        bincode::serialize(&worker).expect("encode the worker's context") == expected,
        "without the transmitted lowering the worker reconstructs a different context"
    );
    CompiledContext::decode(&unlowered)
        .expect_err("the disk route must refuse a payload without its lowering");
}
