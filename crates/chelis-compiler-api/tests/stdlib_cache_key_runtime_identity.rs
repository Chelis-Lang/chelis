//! The chelis-std typecheck cache key names the bundled runtime it was linked
//! against.
//!
//! `stdlib_cache_key` folds no runtime constant of its own. The runtime's
//! version and its archive and shell hashes reach the key through the exact
//! chelis-std source digest of the prepared graph, whose chelis-std rows carry
//! the package version and the `<artifact::archive_sha256>` and
//! `<artifact::shell_sha256>` identities of the bundled package. These tests
//! repack the runtime from the same sources with a different archive mtime:
//! the linked declarations and every source row stay the same, so only the
//! artifact rows can tell the two runtimes apart.

use chelis_compiler_api::stdlib_cache_key;
use chelis_reef::{EmbeddedRuntime, pack_runtime_package, prepare_single_file_program};
use chelis_std_bundle::{BUNDLED_CHELIS_STD_VERSION, EMBEDDED_RUNTIME};
use chelis_surf::ast::Decl;
use std::path::PathBuf;

const PROGRAM: &str = "import Std.Text (join)\n\n\
    def probe(parts: List[string]) -> string = join(parts, \",\")\n";

/// The runtime packed from the std sources with every archive member stamped
/// `mtime`, as a binary would embed it.
fn repacked_runtime(mtime: u64) -> &'static EmbeddedRuntime {
    let std_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../packages/chelis-std");
    let packed = pack_runtime_package(&std_root, mtime).expect("pack the std sources");
    assert_eq!(packed.package.version, BUNDLED_CHELIS_STD_VERSION);
    Box::leak(Box::new(EmbeddedRuntime::new(
        BUNDLED_CHELIS_STD_VERSION,
        Box::leak(packed.archive.into_boxed_slice()),
        Box::leak(packed.shell.into_boxed_slice()),
    )))
}

/// The linked chelis-std declarations, the exact source digest, and the cache
/// key of a program that imports the runtime, linked against `runtime`.
fn stdlib_identity(runtime: &'static EmbeddedRuntime) -> (Vec<Decl>, [u8; 32], [u8; 32]) {
    let decls = chelis_surf::parser::parse_str(PROGRAM).expect("parse the probe program");
    let prepared = prepare_single_file_program("probe.ch", &decls, runtime)
        .expect("link the probe program")
        .expect("the probe program imports the runtime");
    let key = stdlib_cache_key(&prepared.stdlib_decls, prepared.stdlib_source_digest);
    (prepared.stdlib_decls, prepared.stdlib_source_digest, key)
}

#[test]
fn a_repacked_runtime_changes_the_stdlib_cache_key() {
    let repacked = repacked_runtime(1);
    assert!(repacked.archive() != EMBEDDED_RUNTIME.archive());
    assert!(repacked.shell() != EMBEDDED_RUNTIME.shell());
    assert_eq!(
        repacked.archive_files().expect("repacked files"),
        EMBEDDED_RUNTIME.archive_files().expect("embedded files"),
        "the repacked runtime carries the same sources"
    );

    let (embedded_decls, embedded_digest, embedded_key) = stdlib_identity(&EMBEDDED_RUNTIME);
    let (repacked_decls, repacked_digest, repacked_key) = stdlib_identity(repacked);
    assert_eq!(
        embedded_decls, repacked_decls,
        "the same sources link the same declarations"
    );
    assert_ne!(
        embedded_digest, repacked_digest,
        "the source digest must carry the runtime's archive and shell identity"
    );
    assert_ne!(
        embedded_key, repacked_key,
        "a cache entry checked against one runtime must not serve another"
    );
}

/// Negative control: the same runtime, packed the same way, keys the same
/// entry, so the inequality above comes from the artifact identity.
#[test]
fn the_same_runtime_keys_the_same_stdlib_cache_entry() {
    let repacked = repacked_runtime(0);
    assert!(repacked.archive() == EMBEDDED_RUNTIME.archive());
    assert_eq!(
        stdlib_identity(&EMBEDDED_RUNTIME),
        stdlib_identity(repacked)
    );
}
