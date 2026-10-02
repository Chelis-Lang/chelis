//! chelis#2211: the authenticated handoff route, and the property every cache
//! route rests on.
//!
//! `CompiledContext` has two reconstruction routes with different trust
//! assumptions. [`CompiledContext::decode`] reads bytes of unknown provenance:
//! every claim available to it comes out of the same bytes, so it re-lowers
//! the decoded program and compares the result against the transmitted
//! lowered payload. [`CompiledContext::decode_authenticated`] reads bytes
//! accompanied by a digest that did not travel with them, which settles
//! provenance directly and leaves the comparison nothing to establish.
//!
//! Both routes, and the stdlib typecheck cache decoder, adopt the effect and
//! linearity results the transmitted program carries instead of rerunning
//! those checkers (chelis#2558). That is only sound while rerunning them
//! reproduces exactly what they were handed.
//! `cached_program_is_a_checker_fixed_point` is the test that says so: a
//! normalizing pass added to the effect or linearity checker, or a layer
//! composition that loses checker results, fails here rather than silently
//! serving a library that differs from a fresh check. The dependency
//! typecheck cache decoder still reruns both checkers, because its wire
//! carries no lowering to compare the program against.

use std::fs;
use std::path::{Path, PathBuf};

use chelis_compiler_api::{
    COMPILER_VERSION, CompiledContext, HandoffDigest, build_library_context, build_stdlib_context,
    compile_reef_context,
};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

/// A tempdir-backed reef package with a path dependency and the bundled
/// standard library, so the compiled context is the shape a real handoff has.
///
/// chelis#2258 review: the first version of this fixture depended on neither,
/// and encoded 34,569 bytes against 6,763,585 for a real package. Almost all of
/// that difference is chelis-std, which every real context carries, so pulling
/// it in is what makes the route-equivalence comparison run over a
/// representative payload rather than a toy one.
///
/// The lock is written by `prepare_program_for_file`, the resolver every
/// source-file command uses, so it records the bundled runtime. chelis#2558
/// review: a hand-written lock naming only the path dependency linked no
/// chelis-std declarations at all, which sent `compile_reef_context` down the
/// monolithic path instead of the layered one every real package takes.
/// `cached_program_is_a_checker_fixed_point` asserts the linked stdlib is
/// non-empty.
///
/// `body` lets a caller ask for a second, genuinely different library. Two
/// copies of the same one would make a swapped type environment a no-op.
/// `std_layer` says how much of the standard library the package imports.
fn library_fixture(body: &str, std_layer: StdLayer) -> (TempDir, PathBuf) {
    let dir = TempDir::new().expect("tempdir");
    let root = dir.path().join("myapp");
    fs::create_dir_all(root.join("src")).expect("mkdir src");
    fs::create_dir_all(root.join("mylib/src")).expect("mkdir mylib/src");

    fs::write(
        root.join("reef.toml"),
        format!(
            "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"App\"\n\n[dependencies]\nmylib = {{ path = \"./mylib\" }}\nchelis-std = {{ version = \"{}\" }}\n",
            chelis_reef::compiler_bundled_chelis_std_version()
        ),
    )
    .expect("write app reef.toml");
    fs::write(
        root.join("src/main.ch"),
        format!(
            "module App.Main\n{}\ndef placeholder() -> i32 = cast(0, i32)\n",
            std_layer.imports()
        ),
    )
    .expect("write main.ch");
    fs::write(
        root.join("mylib/reef.toml"),
        format!(
            "[package]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Mylib\"\n",
        ),
    )
    .expect("write mylib reef.toml");
    fs::write(root.join("mylib/src/math.ch"), body).expect("write math.ch");
    let root = fs::canonicalize(root).expect("canonicalize the package root");
    chelis_reef::prepare_program_for_file(&root.join("src/main.ch"))
        .expect("the fixture package must resolve and write its lock")
        .expect("the entry file is inside the fixture package");
    (dir, root)
}

/// How much of chelis-std a fixture package imports (chelis#2949).
///
/// A package links, and `compile_reef_context` checks, only the chelis-std
/// modules it imports, so this choice sets what each test pays as the library
/// grows. Nextest runs every test in its own process, so nothing compiled for
/// one test is shared with another.
#[derive(Clone, Copy)]
enum StdLayer {
    /// Every chelis-std module. For a property that depends on what the
    /// library contains: route equivalence and the checker fixed point must
    /// hold over every construct the real library uses.
    Whole,
    /// One small chelis-std module. The stdlib layer is still non-empty, so
    /// `compile_reef_context` takes the layered path every real package takes,
    /// and the payload still clears [`FIXTURE_MINIMUM_BYTES`]. For a property
    /// of the handoff envelope, its digest, or the type-environment check,
    /// none of which depends on which declarations the library holds.
    Minimal,
}

impl StdLayer {
    fn imports(self) -> String {
        match self {
            StdLayer::Whole => every_stdlib_module_import(),
            StdLayer::Minimal => "import Std.Scalar\n".to_string(),
        }
    }
}

/// The fixture library: a fixed head that exercises linearity and effects --
/// a list consumer, a tensor parameter, a float constant the lowered payload
/// carries verbatim, and an owned tensor argument the linearity checker
/// records as reusable for its call's result (`settle`) -- followed by `bulk`
/// generated definitions.
///
/// The bulk is what makes the payload representative. A real handoff runs to
/// several megabytes and is dominated by the package's own definitions, so a
/// handful of them compares the two routes over a payload nothing like the one
/// they carry in production.
/// A qualified `import` of every chelis-std module (chelis#2558). A package
/// links only the chelis-std modules it imports, so a fixture that stands
/// for a package linking the whole standard library imports all of them.
fn every_stdlib_module_import() -> String {
    fn collect(dir: &Path, modules: &mut Vec<String>) {
        for entry in fs::read_dir(dir).expect("read chelis-std sources") {
            let path = entry.expect("chelis-std source entry").path();
            if path.is_dir() {
                collect(&path, modules);
            } else if path.extension().and_then(|ext| ext.to_str()) == Some("ch") {
                let source = fs::read_to_string(&path).expect("read chelis-std module");
                let module = source
                    .lines()
                    .find_map(|line| line.strip_prefix("module "))
                    .expect("a chelis-std source declares its module");
                modules.push(module.trim().to_string());
            }
        }
    }
    let mut modules = Vec::new();
    collect(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packages/chelis-std/src"),
        &mut modules,
    );
    modules.sort();
    modules
        .iter()
        .map(|module| format!("import {module}\n"))
        .collect()
}

fn primary_library(bulk: usize) -> String {
    let mut names: Vec<String> = ["add", "square", "host_len", "decay", "bias", "settle"]
        .iter()
        .map(|name| (*name).to_string())
        .collect();
    names.extend((0..bulk).map(|index| format!("bulk_{index}")));
    let mut source = format!("module Mylib.Math\nexport ({})\n\n", names.join(", "));
    source.push_str(
        "def add(x: i32, y: i32) -> i32 = x + y\n\
         def square(x: i32) -> i32 = x * x\n\
         def host_len[n](xs: tensor[n, f32]) -> i64 = len(to_list(xs))\n\
         def decay[n](xs: tensor[n, f32]) -> tensor[n, f32] = exp(xs)\n\
         def bias() -> f32 = 0.0\n\
         def settle(x: tensor[2, f32], w: tensor[2, f32]) -> tensor[2, f32] = {\n  \
         y = realize(x)\n  mul(w, y)\n}\n",
    );
    for index in 0..bulk {
        let scale = index % 7 + 1;
        let shift = index % 3;
        source.push_str(&format!(
            "def bulk_{index}(x: f32) -> f32 = {{\n  a = x * {scale}.5\n  \
             b = a + {shift}.25\n  b - exp(-b * 0.125) * 0.5\n}}\n"
        ));
    }
    source
}

/// How many generated definitions the fixture carries. Sized so the payload
/// stays in the megabytes while the compile stays around a second; the test
/// asserts the resulting floor rather than trusting this number.
const FIXTURE_BULK_DEFINITIONS: usize = 150;

/// The payload must not silently shrink back to a toy. chelis#2258 review found
/// the first version of this fixture at 34,569 bytes against 6,763,585 for a
/// real package, and a tripwire watching 0.5% of its surface is not a tripwire.
const FIXTURE_MINIMUM_BYTES: usize = 1_000_000;

/// A library with different names and signatures, so its type environment
/// genuinely disagrees with [`PRIMARY_LIBRARY`]'s checked program.
const FOREIGN_LIBRARY: &str = "module Mylib.Math\nexport (triple, offset)\n\n\
     def triple(x: i64) -> i64 = x + x + x\n\
     def offset(x: f32, y: f32) -> f32 = x - y\n";

fn encoded_fixture_from(body: &str, std_layer: StdLayer) -> (TempDir, Vec<u8>, HandoffDigest) {
    let (dir, root) = library_fixture(body, std_layer);
    let context = compile_reef_context(Path::new("/tmp/chelis-2211-unused-reef-home"), &root)
        .expect("the fixture package must compile");
    let (bytes, digest) = context
        .encode_for_handoff()
        .expect("a compiled context must encode");
    (dir, bytes, digest)
}

fn encoded_fixture(std_layer: StdLayer) -> (TempDir, Vec<u8>, HandoffDigest) {
    let fixture = encoded_fixture_from(&primary_library(FIXTURE_BULK_DEFINITIONS), std_layer);
    assert!(
        fixture.1.len() >= FIXTURE_MINIMUM_BYTES,
        "the route-equivalence fixture encodes {} bytes, below the {FIXTURE_MINIMUM_BYTES}-byte \
         floor this comparison is supposed to cover",
        fixture.1.len()
    );
    assert!(
        !wire_of(&fixture.1)
            .reef_state
            .linked_stdlib_decls
            .is_empty(),
        "the fixture links no chelis-std declarations, so compile_reef_context took the \
         monolithic path rather than the layered one every real package takes"
    );
    fixture
}

/// One payload, reconstructed both ways, must produce the same value.
///
/// Both routes bind the transmitted program and re-lower it; the untrusted
/// route also compares the re-lowering against the transmitted one. Before
/// chelis#2558 the untrusted route reran the effect and linearity checkers, and
/// this test was the guard that doing so reproduced the transmitted program;
/// `cached_program_is_a_checker_fixed_point` now carries that guard directly,
/// and this test keeps the two routes from drifting apart in what they bind.
///
/// The comparison is over serialized bytes. `CompiledContext`'s encoding is
/// already required to be byte-stable across processes with different hash
/// states -- `hash_order_cache_bytes.rs` is that oracle -- so a byte difference
/// here is a value difference, not an ordering artifact.
#[test]
fn both_decode_routes_reconstruct_identical_contexts() {
    let (_dir, bytes, digest) = encoded_fixture(StdLayer::Whole);

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

/// Adopting the transmitted checker results equals re-deriving them.
///
/// The compiled-context and stdlib typecheck cache decoders bind the
/// transmitted `CheckedProgram` as it stands, trusting the effect rows and
/// linearity facts the producer wrote (chelis#2558). That trust is the claim
/// that the producer's program is a fixed point of both checkers: running
/// `chelis_effects::check_program` and `chelis_types::check_linearity` over it
/// returns it unchanged.
///
/// The claim is checked on each program a cache layer stores, produced by the
/// entry point that layer's writer uses, over a package that links the real
/// bundled standard library and a path dependency:
///
/// - the compiled context (`.ctx`), read back from the disk route. With a
///   linked stdlib, `compile_reef_context` takes the layered path, so this
///   program is the stdlib layer composed with the package layer through
///   `CheckedProgram::compose`;
/// - the stdlib typecheck cache (`StdLibContext`), built by
///   `build_stdlib_context` over the linked stdlib declarations;
/// - the dependency typecheck cache (`LibraryContext`), built by
///   `build_library_context` over the dependency prefix `chelis check` and
///   `chelis build` use, which is the stdlib composed with the dependency.
///
/// The dependency decoder reruns both checkers, so its program being a fixed
/// point is what keeps a warm dependency load equal to a cold one rather than
/// what makes it sound. The two composed programs are where a composition that
/// drops or reorders one layer's checker results shows up.
///
/// The decode routes that adopt no longer run either checker, so this is the
/// only place a checker that starts to normalize, sort, dedup or intern would
/// be noticed. The target is a standing target in
/// `.config/ci-test-targets.toml`, so it runs on every candidate. The
/// comparison is over serialized bytes, which `hash_order_cache_bytes.rs`
/// already requires to be stable across hash states, so a byte difference is
/// a value difference.
#[test]
fn cached_program_is_a_checker_fixed_point() {
    let (_dir, root) = library_fixture(&primary_library(FIXTURE_BULK_DEFINITIONS), StdLayer::Whole);
    let context = compile_reef_context(Path::new("/tmp/chelis-2211-unused-reef-home"), &root)
        .expect("the fixture package must compile");
    let bytes = context.encode().expect("a compiled context must encode");

    // Take the program the disk route adopts: decode through it, then read the
    // bound program back out of its re-encoding.
    let decoded = CompiledContext::decode(&bytes).expect("the disk route must accept");
    let adopted = wire_of(&decoded.encode().expect("a decoded context must re-encode"));
    assert!(
        !adopted.reef_state.linked_stdlib_decls.is_empty(),
        "the fixture links no chelis-std declarations, so compile_reef_context took the \
         monolithic path and this test does not reach the layered program every real package \
         caches"
    );
    assert_checker_fixed_point("the compiled context (.ctx)", &adopted.library_checked);
    assert_carries_reusable_inputs("the compiled context (.ctx)", &adopted.library_checked);

    let prepared = chelis_reef::prepare_program_for_file(&root.join("src/main.ch"))
        .expect("the fixture package must resolve")
        .expect("the entry file is inside the fixture package");
    assert!(
        !prepared.stdlib_decls.is_empty(),
        "the prepared program links no chelis-std declarations, so the stdlib layer checked \
         below would be empty"
    );
    let (dependency_decls, _entry_decls) = prepared.dependency_entry_partition();
    assert!(
        !dependency_decls.is_empty(),
        "the fixture's path dependency must form the dependency prefix, or there is no \
         dependency typecheck cache program to check"
    );
    let stdlib = build_stdlib_context(&prepared.stdlib_decls)
        .expect("the bundled standard library must build");
    assert_checker_fixed_point(
        "the stdlib typecheck cache (StdLibContext)",
        stdlib.library_checked(),
    );
    let dependency = build_library_context(&stdlib, dependency_decls)
        .expect("the dependency layer must build")
        .expect("the fixture's dependency must compose over the standard library");
    assert_checker_fixed_point(
        "the dependency typecheck cache (LibraryContext)",
        dependency.library_checked(),
    );
    assert_carries_reusable_inputs(
        "the dependency typecheck cache (LibraryContext)",
        dependency.library_checked(),
    );
}

/// The composed programs must carry linearity facts from the package's own
/// layer. Without them a composition that drops that layer's facts has
/// nothing to drop, and the fixed-point comparison passes without looking.
/// The fixture's `settle` is the source of those facts.
fn assert_carries_reusable_inputs(layer: &str, stored: &chelis_types::CheckedProgram) {
    assert!(
        *stored.linearity() != chelis_types::LinearityInfo::default(),
        "{layer}: the stored program records no reusable linear inputs, so the fixed-point \
         comparison cannot see a composition that loses them; the fixture's `settle` should \
         produce one"
    );
}

/// Rerun the effect and linearity checkers over `stored` and require the
/// serialized result to equal it.
fn assert_checker_fixed_point(layer: &str, stored: &chelis_types::CheckedProgram) {
    let stored_bytes = bincode::serialize(stored).expect("encode the stored program");
    let _linked = chelis_types::install_linked_program_guard();
    let effected = chelis_effects::check_program(stored)
        .unwrap_or_else(|errors| panic!("{layer}: the effect checker rejects it: {errors:?}"));
    let rederived = chelis_types::check_linearity(&effected)
        .unwrap_or_else(|errors| panic!("{layer}: the linearity checker rejects it: {errors:?}"));
    let rederived_bytes = bincode::serialize(&rederived).expect("encode the re-derived program");

    assert_eq!(
        stored_bytes.len(),
        rederived_bytes.len(),
        "{layer}: rerunning the effect and linearity checkers changed the size of the stored \
         program; it is no longer a fixed point of those checkers, so a cache decoder that \
         adopts the producer's results (chelis#2558) now serves a library that differs from a \
         fresh check"
    );
    assert!(
        stored_bytes == rederived_bytes,
        "{layer}: rerunning the effect and linearity checkers changed the stored program \
         without changing its size; it is no longer a fixed point of those checkers, so a cache \
         decoder that adopts the producer's results (chelis#2558) now serves a library that \
         differs from a fresh check"
    );
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
    let (_dir, bytes, digest) = encoded_fixture(StdLayer::Minimal);

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
    let (_dir, bytes, _digest) = encoded_fixture(StdLayer::Minimal);
    let (_other_dir, _other_bytes, other_digest) =
        encoded_fixture_from(FOREIGN_LIBRARY, StdLayer::Minimal);

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
    let (_dir, bytes, _digest) = encoded_fixture(StdLayer::Minimal);
    CompiledContext::decode(&bytes)
        .expect("decode must keep working for a caller that has only the bytes");
}

#[test]
fn handoff_digest_hex_round_trips() {
    let (_dir, _bytes, digest) = encoded_fixture(StdLayer::Minimal);
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

/// The fast route's one remaining semantic check, locked.
///
/// chelis#2258 review, P2-3: `bind_cached_library`'s
/// `matches_checked_program` is the only semantic check
/// `decode_authenticated` still runs, and
/// `every_cached_library_decoder_rejects_forged_selector_callable_metadata`
/// does not reach it -- that test covers the three `Deserialize`-based
/// decoders, and this route does not go through `CompiledContext`'s
/// `Deserialize`. So the check was present, correct, and untested, which is
/// the state a later optimiser deletes things from, especially since the
/// function's own documentation calls it "a fraction of a percent of the
/// work".
///
/// The forgery pairs one library's checked program with a different library's
/// type environment and then reseals the envelope **and mints a digest over
/// the forged payload**, so the authentication passes and nothing but the
/// semantic check can reject it.
#[test]
fn authenticated_decode_rejects_a_type_environment_from_another_library() {
    let (_dir, bytes, _digest) = encoded_fixture(StdLayer::Minimal);
    let (_foreign_dir, foreign_bytes, _foreign_digest) =
        encoded_fixture_from(FOREIGN_LIBRARY, StdLayer::Minimal);

    let honest = wire_of(&bytes);
    let foreign = wire_of(&foreign_bytes);
    assert_ne!(
        bincode::serialize(&honest.type_env).expect("encode type env"),
        bincode::serialize(&foreign.type_env).expect("encode type env"),
        "the two fixtures must have genuinely different type environments, or this test \
         forges nothing"
    );

    // Positive control first. Taking the payload apart and putting it back
    // together must produce something the route accepts, otherwise the
    // rejection below would prove only that the rebuild is lossy.
    let (rebuilt, rebuilt_digest) = reseal_wire(&bytes, &honest);
    CompiledContext::decode_authenticated(&rebuilt, &rebuilt_digest)
        .expect("an unmodified rebuild must still authenticate and reconstruct");

    let forged = WireMirror {
        type_env: foreign.type_env,
        ..honest
    };
    let (forged_bytes, forged_digest) = reseal_wire(&bytes, &forged);
    // The authentication cannot be what rejects this: the digest was minted
    // over these exact bytes.
    let error = CompiledContext::decode_authenticated(&forged_bytes, &forged_digest)
        .expect_err("a type environment from another library must not reconstruct");
    assert!(
        error.contains("the type environment does not match the checked library"),
        "the rejection must name the type-environment disagreement, not the digest: {error}"
    );
}

/// Mirror of the private `CompiledContextWire`, positionally identical because
/// bincode is positional. If the real wire gains, loses or reorders a field,
/// `wire_of` fails to decode and the tests above fail loudly rather than
/// quietly testing a different shape.
#[derive(serde::Serialize, serde::Deserialize)]
struct WireMirror {
    source_hash: chelis_compiler_api::ContextHash,
    identity: chelis_compiler_api::CacheIdentity,
    reef_state: chelis_reef::PreparedReefGraph,
    type_env: chelis_types::TypeEnv,
    library_checked: chelis_types::CheckedProgram,
    library_dag: chelis_ir::lower::LoweredLibrary,
}

fn wire_of(bytes: &[u8]) -> WireMirror {
    let envelope: ContextFixtureEnvelope =
        bincode::deserialize(&bytes[magic_len(bytes)..]).expect("the fixture envelope must decode");
    bincode::deserialize(&envelope.payload)
        .expect("the payload must decode as the wire this test mirrors")
}

/// Rebuild a self-consistent envelope around `wire` and mint the digest a
/// producer of those bytes would have delivered.
fn reseal_wire(original: &[u8], wire: &WireMirror) -> (Vec<u8>, HandoffDigest) {
    let magic = magic_len(original);
    let mut envelope: ContextFixtureEnvelope =
        bincode::deserialize(&original[magic..]).expect("the fixture envelope must decode");
    envelope.payload = bincode::serialize(wire).expect("the forged wire must encode");
    envelope.payload_sha256 = Sha256::digest(&envelope.payload).into();
    envelope.source_hash = wire.source_hash;
    envelope.identity = wire.identity.clone();
    let digest = HandoffDigest::from_hex(&format!("{:x}", Sha256::digest(&envelope.payload)))
        .expect("a freshly computed digest must parse");
    let mut bytes = original[..magic].to_vec();
    bytes.extend(bincode::serialize(&envelope).expect("the resealed envelope must encode"));
    (bytes, digest)
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
