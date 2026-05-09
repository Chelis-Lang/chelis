//! Compile-time guard for the chelis-std bundle.
//!
//! The lib.rs uses `include_bytes!()` against fixed paths under
//! `dist/` — that is what makes the bytes part of the chelis binary.
//! This build script does not produce or copy artifacts; it only:
//!
//! 1. asserts that the expected dist files exist before the lib.rs's
//!    `include_bytes!` runs (a missing file there would surface as
//!    a less-readable cargo error pointing at the macro site), and
//! 2. emits `cargo:rerun-if-changed=dist/...` so cargo invalidates
//!    the bundle crate (and every crate that depends on it) when
//!    the artifacts change on disk. Re-bundling is the standard
//!    "regenerate artifacts → commit → build" pipeline; this script
//!    just wires cargo's incremental story to it.
//!
//! Artifact regeneration is performed by `scripts/regenerate_chelis_std_bundle.py`
//! (see top-level repo). The build.rs intentionally does NOT run the
//! chelis CLI or shell out — the artifacts live in the repo as
//! committed binaries.

use std::path::PathBuf;

const BUNDLED_VERSION: &str = "0.3.0";

fn main() {
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is always set by cargo"),
    );
    let dist = manifest_dir.join("dist");
    let archive = dist.join(format!("chelis-std-{BUNDLED_VERSION}.tar.zst"));
    let shell = dist.join(format!("chelis-std-{BUNDLED_VERSION}.chb"));

    for path in [&archive, &shell] {
        if !path.exists() {
            panic!(
                "chelis-std bundle artifact missing at {}.\n\
                 Run `python3 scripts/regenerate_chelis_std_bundle.py` from the\n\
                 repository root to rebuild and copy the bytes into\n\
                 crates/chelis-std-bundle/dist/, then commit them.",
                path.display()
            );
        }
        println!("cargo:rerun-if-changed={}", path.display());
    }

    // Also rerun if build.rs itself or the version constant moves —
    // the `include_bytes!` paths are derived from the version in
    // lib.rs, which is independent of this script but conceptually
    // co-evolves with it.
    println!("cargo:rerun-if-changed=build.rs");
}
