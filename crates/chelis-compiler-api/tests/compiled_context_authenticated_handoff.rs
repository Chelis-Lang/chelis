//! chelis#2211: the authenticated handoff route, and the property it rests on.
//!
//! `CompiledContext` has two reconstruction routes with different trust
//! assumptions. [`CompiledContext::decode`] reads bytes of unknown provenance:
//! every claim available to it comes out of the same bytes, so it re-derives
//! the library from the decoded program and compares the result against the
//! transmitted lowered payload. [`CompiledContext::decode_authenticated`] reads
//! bytes accompanied by a digest that did not travel with them, which settles
//! provenance directly and leaves the re-derivation nothing to establish.
//!
//! That second sentence is only true while re-deriving reproduces exactly what
//! it was handed. `both_decode_routes_reconstruct_identical_contexts` is the
//! test that says so, and it is the reason the fast route is allowed to exist:
//! a normalizing pass added to the effect or linearity checker would make the
//! two routes disagree, and it would fail here rather than diverge in a worker.

use std::fs;
use std::path::{Path, PathBuf};

use chelis_compiler_api::{COMPILER_VERSION, CompiledContext, HandoffDigest, compile_reef_context};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

/// A tempdir-backed reef package with a path dependency, so the compiled
/// context carries a real multi-package library rather than a single module.
fn library_fixture() -> (TempDir, PathBuf) {
    let dir = TempDir::new().expect("tempdir");
    let root = dir.path().join("myapp");
    fs::create_dir_all(root.join("src")).expect("mkdir src");
    fs::create_dir_all(root.join("mylib/src")).expect("mkdir mylib/src");

    fs::write(
        root.join("reef.toml"),
        format!(
            "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"App\"\n\n[dependencies]\nmylib = {{ path = \"./mylib\" }}\n",
        ),
    )
    .expect("write app reef.toml");
    fs::write(
        root.join("src/main.ch"),
        "module App.Main\n\ndef placeholder() -> i32 = cast(0, i32)\n",
    )
    .expect("write main.ch");
    fs::write(
        root.join("mylib/reef.toml"),
        format!(
            "[package]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Mylib\"\n",
        ),
    )
    .expect("write mylib reef.toml");
    // Enough shape that the library exercises linearity and effects rather
    // than a single arithmetic definition: a list consumer, a tensor
    // parameter, and a float constant the lowered payload carries verbatim.
    fs::write(
        root.join("mylib/src/math.ch"),
        "module Mylib.Math\nexport (add, square, host_len, decay, bias)\n\n\
         def add(x: i32, y: i32) -> i32 = x + y\n\
         def square(x: i32) -> i32 = x * x\n\
         def host_len[n](xs: tensor[n, f32]) -> i64 = len(to_list(xs))\n\
         def decay[n](xs: tensor[n, f32]) -> tensor[n, f32] = exp(xs)\n\
         def bias() -> f32 = 0.0\n",
    )
    .expect("write math.ch");
    fs::write(
        root.join("reef.lock"),
        format!(
            "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\n\n[[dependencies]]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\narchive_sha256 = \"\"\nshell_sha256 = \"\"\n\n[dependencies.source]\nkind = \"path\"\npath = \"./mylib\"\n",
        ),
    )
    .expect("write reef.lock");
    (dir, root)
}

fn encoded_fixture() -> (TempDir, Vec<u8>, HandoffDigest) {
    let (dir, root) = library_fixture();
    let context = compile_reef_context(Path::new("/tmp/chelis-2211-unused-reef-home"), &root)
        .expect("the fixture package must compile");
    let (bytes, digest) = context
        .encode_for_handoff()
        .expect("a compiled context must encode");
    (dir, bytes, digest)
}

/// The property the authenticated route depends on.
///
/// One payload, reconstructed both ways, must produce the same value. The
/// untrusted route reruns the effect and linearity checkers over the
/// transmitted program and re-lowers; the authenticated route adopts the
/// transmitted program and re-lowers. They agree only because rerunning those
/// checkers over their own output reproduces it, and nothing else in the design
/// enforces that. If a future change makes either checker normalize, sort,
/// dedup, or intern anything, this test fails and the fast route must be
/// reconsidered rather than silently returning a different library than the
/// producer had.
///
/// The comparison is over serialized bytes. `CompiledContext`'s encoding is
/// already required to be byte-stable across processes with different hash
/// states -- `hash_order_cache_bytes.rs` is that oracle -- so a byte difference
/// here is a value difference, not an ordering artifact.
#[test]
fn both_decode_routes_reconstruct_identical_contexts() {
    let (_dir, bytes, digest) = encoded_fixture();

    let untrusted = CompiledContext::decode(&bytes).expect("the untrusted route must accept");
    let authenticated = CompiledContext::decode_authenticated(&bytes, &digest)
        .expect("the authenticated route must accept");

    let untrusted_bytes = bincode::serialize(&untrusted).expect("re-encode the untrusted route");
    let authenticated_bytes =
        bincode::serialize(&authenticated).expect("re-encode the authenticated route");

    assert_eq!(
        untrusted_bytes.len(),
        authenticated_bytes.len(),
        "the two reconstruction routes produced contexts of different sizes; the transmitted \
         checked program is no longer a fixed point of the effect and linearity checkers, so \
         CompiledContext::decode_authenticated no longer reconstructs the producer's value"
    );
    assert!(
        untrusted_bytes == authenticated_bytes,
        "the two reconstruction routes produced different contexts of the same size; the \
         transmitted checked program is no longer a fixed point of the effect and linearity \
         checkers, so CompiledContext::decode_authenticated no longer reconstructs the \
         producer's value"
    );

    // Same statement against the producer rather than route against route: a
    // round trip through the authenticated route returns the bytes that went
    // in, so what it reconstructs is the value the producer encoded.
    let (re_encoded, re_encoded_digest) = authenticated
        .encode_for_handoff()
        .expect("a reconstructed context must re-encode");
    assert!(
        re_encoded == bytes,
        "the authenticated route did not reproduce the producer's own bytes"
    );
    assert_eq!(re_encoded_digest, digest);
}

/// The negative control for the authentication itself.
///
/// This is the tamper that
/// `cache_wire_compatibility::cache_reconstruction_rejects_changed_numeric_bits_after_checksum_recomputed`
/// defeats by re-derivation, in its general form: change the payload and
/// recompute the embedded digest so the envelope is self-consistent again. The
/// authenticated route catches it without re-deriving anything, because the
/// digest it compares against was delivered by the producer and the rewriter
/// never saw that channel.
#[test]
fn authenticated_decode_rejects_a_payload_rewritten_after_encode() {
    let (_dir, bytes, digest) = encoded_fixture();

    let rewritten = rewrite_payload_and_reseal(&bytes);
    assert_ne!(
        rewritten, bytes,
        "the fixture rewrite must change the bytes"
    );

    // Self-consistent: the envelope's own digest check passes.
    let envelope: ContextFixtureEnvelope =
        bincode::deserialize(&rewritten[magic_len(&rewritten)..])
            .expect("the resealed envelope must decode");
    assert_eq!(
        envelope.payload_sha256,
        <[u8; 32]>::from(Sha256::digest(&envelope.payload)),
        "the fixture must reseal the envelope, or it proves only that a torn write is caught"
    );

    let error = CompiledContext::decode_authenticated(&rewritten, &digest)
        .expect_err("a payload rewritten after encode must not authenticate");
    assert!(
        error.contains("digest its producer delivered"),
        "the rejection must name the out-of-band digest as the reason: {error}"
    );
}

/// A digest that authenticates some other payload does not authenticate this
/// one. Separate from the test above because that one changes the bytes and
/// this one changes the digest; a comparison written the wrong way round could
/// pass one and fail the other.
#[test]
fn authenticated_decode_rejects_a_digest_minted_for_other_bytes() {
    let (_dir, bytes, _digest) = encoded_fixture();
    let (_other_dir, _other_bytes, other_digest) = encoded_fixture();

    let error = CompiledContext::decode_authenticated(&bytes, &other_digest)
        .expect_err("another payload's digest must not authenticate these bytes");
    assert!(
        error.contains("digest its producer delivered"),
        "the rejection must name the out-of-band digest as the reason: {error}"
    );
}

/// The untrusted route is unchanged: it still accepts bytes with no digest
/// beside them, and it still re-derives. Without this, a later change could
/// quietly make `decode` an alias for the authenticated route and delete the
/// coverage that justifies keeping the re-derivation for the disk cache.
#[test]
fn the_untrusted_route_still_accepts_a_payload_with_no_accompanying_digest() {
    let (_dir, bytes, _digest) = encoded_fixture();
    CompiledContext::decode(&bytes)
        .expect("decode must keep working for a caller that has only the bytes");
}

#[test]
fn handoff_digest_hex_round_trips() {
    let (_dir, _bytes, digest) = encoded_fixture();
    let hex = digest.to_hex();
    assert_eq!(hex.len(), 64, "a SHA-256 is 64 hex characters: {hex}");
    assert_eq!(
        HandoffDigest::from_hex(&hex).expect("the hex form must parse"),
        digest
    );
}

/// Every other spelling is refused rather than repaired. A truncated digest
/// that parsed would authenticate bytes it does not cover, and an accepted
/// upper-case spelling would mean two hex forms of one digest compare unequal
/// somewhere else.
#[test]
fn handoff_digest_rejects_every_spelling_but_the_canonical_one() {
    let canonical = "0".repeat(64);
    HandoffDigest::from_hex(&canonical).expect("the canonical spelling must parse");

    for (label, text) in [
        ("empty", String::new()),
        ("truncated", "0".repeat(63)),
        ("over-long", "0".repeat(65)),
        ("upper-case", "A".repeat(64)),
        ("non-hex", "z".repeat(64)),
        ("whitespace-padded", format!(" {} ", "0".repeat(62))),
    ] {
        let error = HandoffDigest::from_hex(&text)
            .expect_err(&format!("the {label} spelling must be refused"));
        assert!(
            error.contains("64 lower-case hex characters"),
            "the {label} rejection must say what was expected: {error}"
        );
    }
}

/// Nothing that ships calls the unauthenticated route.
///
/// `CompiledContext::decode` stays public because an embedder holding bytes of
/// unknown provenance needs it, and because roughly fifteen tests use it as an
/// in-memory round trip to exercise the re-derivation guard -- coverage that
/// would vanish if it started requiring a digest. What must not happen is a
/// production caller quietly appearing beside the authenticated route and
/// paying the re-derivation again, or worse, accepting a handoff nobody
/// authenticated. After chelis#2211 there is no such caller, so this test says
/// so rather than leaving it to the next reader's discipline.
///
/// The check is textual, so a sufficiently determined alias defeats it. It is
/// aimed at the case that actually happens: someone reaching for the obvious
/// name because the authenticated one needs a digest they would have to thread
/// through.
#[test]
fn no_shipped_source_calls_the_unauthenticated_decode_route() {
    let crates_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate sits under crates/");
    let mut sources = Vec::new();
    collect_rust_sources(crates_dir, &mut sources);

    // A scanner that reached nothing would pass without looking, so prove the
    // walk reaches the tree and specifically reaches the one production file
    // that used to hold the call.
    assert!(
        sources.len() > 200,
        "the source walk found only {} files under {}; it is not reaching the tree",
        sources.len(),
        crates_dir.display()
    );
    let worker = crates_dir.join("chelis-cli/src/main.rs");
    let worker_source = sources
        .iter()
        .find(|(path, _)| *path == worker)
        .map(|(_, source)| source.as_str())
        .expect("the walk must reach the CLI worker that owns the handoff");
    assert!(
        worker_source.contains("CompiledContext::decode_authenticated("),
        "the CLI worker must still be the authenticated route's caller, or this test is \
         asserting the absence of a call in a file that no longer decodes at all"
    );

    let offenders: Vec<String> = sources
        .iter()
        .filter(|(_, source)| shipped_part_of(source).contains("CompiledContext::decode("))
        .map(|(path, _)| path.display().to_string())
        .collect();
    assert!(
        offenders.is_empty(),
        "shipped source calls CompiledContext::decode, the route that cannot establish where \
         its bytes came from. Use CompiledContext::decode_authenticated with a digest the \
         producer delivered separately. If the call is test-only, put it in an integration \
         test under tests/ or in a trailing `#[cfg(test)] mod tests`, which this scan \
         recognizes. If the caller genuinely holds bytes of unknown provenance, say so in the \
         pull request and add it here. Offenders: {offenders:?}"
    );
}

/// The part of a source file that ships, which is everything before a trailing
/// `#[cfg(test)] mod tests { .. }`.
///
/// Unit tests live inside `src/`, and `context.rs`'s own tests exercise the
/// unauthenticated route deliberately, so a scan that did not cut them out
/// would report the crate that defines the function. The cut applies only to
/// that one recognized shape: a single column-zero `#[cfg(test)]` opening a
/// `mod tests` that runs to the end of the file.
///
/// Anything else returns the whole file, which is the safe direction. Not
/// recognizing a layout means not being able to prove a region is test code,
/// and the consequence of guessing wrong the other way is exempting shipped
/// code from the check this exists to perform.
fn shipped_part_of(source: &str) -> &str {
    const MARKER: &str = "\n#[cfg(test)]\n";
    let Some(index) = source.find(MARKER) else {
        return source;
    };
    let (shipped, test_part) = source.split_at(index + 1);
    let body = test_part
        .strip_prefix("#[cfg(test)]\n")
        .expect("the marker split leaves the attribute at the front")
        .trim_start();
    let trailing_test_module = body.starts_with("mod tests {")
        && source.trim_end().ends_with('}')
        && !body.contains(MARKER);
    if trailing_test_module {
        shipped
    } else {
        source
    }
}

/// Every `.rs` file under a crate's `src/`, paired with its contents. Test and
/// benchmark trees are deliberately out of scope: the point is what ships.
fn collect_rust_sources(crates_dir: &Path, out: &mut Vec<(PathBuf, String)>) {
    let Ok(entries) = fs::read_dir(crates_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let src = entry.path().join("src");
        if src.is_dir() {
            collect_rust_sources_under(&src, out);
        }
    }
}

fn collect_rust_sources_under(dir: &Path, out: &mut Vec<(PathBuf, String)>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rust_sources_under(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs")
            && let Ok(source) = fs::read_to_string(&path)
        {
            out.push((path, source));
        }
    }
}

/// Mirror of the private envelope shape, so a test can reseal a payload the way
/// an agent with write access to the handoff file would.
#[derive(serde::Serialize, serde::Deserialize)]
struct ContextFixtureEnvelope {
    version: u32,
    source_hash: chelis_compiler_api::ContextHash,
    identity: chelis_compiler_api::CacheIdentity,
    payload_sha256: [u8; 32],
    payload: Vec<u8>,
}

fn magic_len(bytes: &[u8]) -> usize {
    bytes
        .iter()
        .position(|byte| *byte == b'\n')
        .expect("the envelope carries its magic prefix")
        + 1
}

/// Flip one byte deep inside the payload and recompute the envelope's own
/// digest, leaving a file that is internally consistent and not the one the
/// producer wrote.
fn rewrite_payload_and_reseal(bytes: &[u8]) -> Vec<u8> {
    let magic = magic_len(bytes);
    let mut envelope: ContextFixtureEnvelope =
        bincode::deserialize(&bytes[magic..]).expect("the fixture envelope must decode");
    let target = envelope.payload.len() / 2;
    envelope.payload[target] ^= 0x01;
    envelope.payload_sha256 = Sha256::digest(&envelope.payload).into();
    let mut rewritten = bytes[..magic].to_vec();
    rewritten.extend(bincode::serialize(&envelope).expect("the resealed envelope must encode"));
    rewritten
}
