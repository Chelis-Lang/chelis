//! Ratchets for the properties `#![no_std]` alone does not defend.
//!
//! Most of this crate's purity is enforced by the compiler and needs no test.
//! With `#![no_std]` and no `alloc`, a `std::` path does not resolve, and
//! `String` / `Vec` / `Box` / `format!` do not exist — so "performs no
//! allocation" and "contains no source-text generator" are not assertions this
//! file could make more strongly than the build already does.
//!
//! What the compiler does *not* prevent is someone restoring those
//! capabilities: adding `extern crate alloc` brings back the allocating types,
//! and adding a manifest dependency can pull in `std` transitively. Both are
//! one-line changes that would compile cleanly and silently undo the
//! boundary. Those are what this file guards.
//!
//! Reading the manifest and the source as text is deliberate. The property is
//! about what the crate *declares*, which is not observable from inside the
//! compiled crate.

use std::fs;
use std::path::PathBuf;

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn manifest() -> String {
    fs::read_to_string(crate_root().join("Cargo.toml")).expect("read chelis-vocab Cargo.toml")
}

fn lib_source() -> String {
    fs::read_to_string(crate_root().join("src").join("lib.rs")).expect("read chelis-vocab lib.rs")
}

/// The dependency graph stays empty. A dependency is the ordinary way `std`
/// re-enters a `no_std` crate, and it is invisible from the source.
#[test]
fn manifest_declares_no_dependencies() {
    let manifest = manifest();
    for section in [
        "[dependencies]",
        "[dev-dependencies]",
        "[build-dependencies]",
    ] {
        let Some(start) = manifest.find(section) else {
            continue;
        };
        let rest = &manifest[start + section.len()..];
        let body = rest.split("\n[").next().unwrap_or(rest);
        let entries: Vec<&str> = body
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect();
        assert!(
            entries.is_empty(),
            "chelis-vocab is the dependency-bottom crate and must declare none; \
             {section} contains {entries:?}. Adding one is a deliberate change to \
             that property, not a routine edit."
        );
    }
}

#[test]
fn crate_declares_no_std() {
    assert!(
        lib_source().contains("#![no_std]"),
        "chelis-vocab must declare #![no_std]; without it the standard library \
         returns and every purity property below it becomes unenforced"
    );
}

/// `extern crate alloc` would restore `String`, `Vec`, `Box`, and `format!`,
/// which is exactly what let a C source-text generator live in this crate
/// before it moved to `chelis-runtime`.
#[test]
fn crate_does_not_opt_back_into_alloc() {
    let source = lib_source();
    assert!(
        !source.contains("extern crate alloc"),
        "chelis-vocab must not depend on alloc. Its absence is what makes \
         'the decode path performs no allocation' and 'no source-text \
         generator lives here' compiler-enforced rather than reviewed"
    );
    assert!(
        !source.contains("alloc::"),
        "chelis-vocab must not reference alloc paths"
    );
}

/// `unsafe` is forbidden by attribute; this pins the attribute itself, since
/// deleting it is what would make the ban stop applying.
#[test]
fn crate_forbids_unsafe() {
    assert!(
        lib_source().contains("#![forbid(unsafe_code)]"),
        "chelis-vocab must forbid unsafe; it is a closed vocabulary of plain \
         data and has no call for it"
    );
}

/// The lint table is declared rather than inherited. Neither a workspace table
/// nor clippy's default groups enable these, so without the manifest entry the
/// classes are simply off — which is how they were before this crate declared
/// them.
#[test]
fn manifest_declares_the_lint_table() {
    let manifest = manifest();
    assert!(
        manifest.contains("[lints.clippy]"),
        "chelis-vocab must declare its lint table; the cast and arithmetic \
         classes are off by default across this workspace"
    );
    for lint in ["cast_possible_truncation", "cast_possible_wrap"] {
        assert!(
            manifest.contains(lint),
            "chelis-vocab's lint table must enable {lint}: it describes an ABI, \
             where a lossy numeric conversion is a wire-format defect"
        );
    }
}
