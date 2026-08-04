//! Acceptance oracle for the cross-process dependency-library typecheck
//! cache (chelis#1168 — Layer 2 of the build-lane hybrid cache).
//!
//! Layer 1 (the chelis-std sub-context) has its own oracle in
//! `stdlib_typecheck_cache_oracle.rs`. That oracle's non-std fixture
//! (`pseudo_nautilus`) depends only on chelis-std, so the dependency layer
//! there is EMPTY and the two-layer path is what runs. This file adds a
//! fixture with a real NON-chelis-std path dependency, so the new
//! [`chelis_compiler_api::library_cache`] layer actually engages, and pins
//! the non-negotiable properties for it.
//!
//! **These oracles compare the EMITTED C (`main.c` + `main.h` bytes), not
//! `chelis build` stdout.** Build stdout carries only "Wrote …" paths, so a
//! stdout compare is blind to a miscompile that changes the C body while
//! the path lines stay equal (chelis#1168 review, HIGH-2). The emitted C is
//! out-dir-independent (relative `#include`), so separate out dirs compare
//! directly.
//!
//! Properties:
//! 1. `cold_vs_warm_build_c_identical` — cold and warm builds emit
//!    byte-identical C. The dependency cache must not perturb output.
//! 2. `monolithic_vs_layered_build_c_identical` — the monolithic checker
//!    (`CHELIS_STDLIB_CACHE_DISABLE=1`, both cache layers + the
//!    `_with_context` layering bypassed) and the three-layer path emit
//!    byte-identical C. A divergence is a compiler-correctness bug in the
//!    dependency-layer split.
//! 3. `macro_hygiene_monolithic_vs_layered_c_identical` — the same, with a
//!    binder-minting macro in the dependency and a colliding binder in the
//!    entry: the split must preserve macro hygiene (chelis#1168 regression).
//! 4. `dependency_edit_misses_and_rebuilds` — a dependency byte edit flips
//!    the key (clean miss); the layered C equals the monolithic C of the
//!    mutated program (never a stale hit).
//! 5. `entry_only_edit_reuses_dependency_cache` — an entry-only edit leaves
//!    the dependency key unchanged (warm hit), writes no new dependency
//!    artifact, and the layered C equals the monolithic C of the edited
//!    program.
//! 6. `dependency_build_writes_library_cache_artifact` — a build with a
//!    dependency writes a `chelis-lib-*.tc` file (the Layer-2 cache
//!    engaged).

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

/// Pinned compiler version string for fixture manifests.
const COMPILER_VERSION: &str = chelis_compiler_api::COMPILER_VERSION;

fn write(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent dirs");
    }
    fs::write(path, contents).expect("write fixture file");
}

/// Materialize a two-package reef fixture under `scratch`: a root package
/// `pseudo-app` with a NON-chelis-std path dependency `azdep`. `azdep`
/// sorts before `pseudo-app` alphabetically, so it lands in the stable
/// dependency prefix and the Layer-2 cache engages. Returns the entry file
/// path (`src/main.ch`).
///
/// `dep_body` / `entry_body` supply the two module bodies so tests can vary
/// them (plain vs macro-bearing). Programs are compiled-lane-friendly
/// (int32 + the `add` RISC primitive, no host-only builtins, no chelis-std)
/// so `chelis build` emits C cleanly.
fn stage_dep_fixture(scratch: &Path, dep_body: &str, entry_body: &str) -> PathBuf {
    let root = scratch.join("pseudo-app");
    write(
        &root.join("reef.toml"),
        &format!(
            "[package]\nname = \"pseudo-app\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"PseudoApp\"\n\n[dependencies]\nazdep = {{ path = \"./azdep\" }}\n"
        ),
    );
    write(&root.join("src/main.ch"), entry_body);
    write(
        &root.join("azdep/reef.toml"),
        &format!(
            "[package]\nname = \"azdep\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Azdep\"\n"
        ),
    );
    write(&root.join("azdep/src/math.ch"), dep_body);
    write(
        &root.join("reef.lock"),
        &format!(
            "[package]\nname = \"pseudo-app\"\nversion = \"0.1.0\"\n\n[[dependencies]]\nname = \"azdep\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\narchive_sha256 = \"\"\nshell_sha256 = \"\"\n\n[dependencies.source]\nkind = \"path\"\npath = \"./azdep\"\n"
        ),
    );
    root.join("src/main.ch")
}

/// A plain (macro-free) dependency + entry pair that exercises the cache.
fn plain_bodies() -> (&'static str, &'static str) {
    (
        "module Azdep.Math\nexport (az_add)\n\ndef az_add(x: int32, y: int32) -> int32 = add(x, y)\n",
        "module PseudoApp.Main\nimport Azdep.Math (az_add)\n\ndef main_value -> int32 = az_add(cast(3, int32), cast(4, int32))\n",
    )
}

/// A macro-bearing pair: the dependency defines and uses a binder-minting
/// macro (advancing the shared hygiene counter), and the entry binds
/// `v_macro_0` — the name a restarted counter would collide with.
fn macro_bodies() -> (&'static str, &'static str) {
    (
        "module Azdep.Math\nexport (dep_val)\n\nmacro dmk(a) = {\n  q = a\n  add(q, q)\n}\n\ndef dep_val(x: int32) -> int32 = dmk(x)\n",
        "module PseudoApp.Main\nimport Azdep.Math (dep_val)\n\nmacro emk(a) = {\n  v = cast(7, int32)\n  add(v, a)\n}\n\ndef main_value -> int32 = {\n  v_macro_0 = dep_val(cast(5, int32))\n  emk(v_macro_0)\n}\n",
    )
}

/// Build `entry` into a fresh out dir and return the emitted
/// `(main.c, main.h)` bytes. `extra_env` carries the optional
/// `CHELIS_STDLIB_CACHE_DISABLE` monolithic seam.
fn build_c(entry: &Path, cache_home: &Path, extra_env: &[(&str, &str)]) -> (Vec<u8>, Vec<u8>) {
    let out = tempdir().expect("out dir");
    let mut cmd = Command::cargo_bin("chelis").expect("chelis binary");
    cmd.env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", cache_home);
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    cmd.arg("build").arg(entry).arg("-o").arg(out.path());
    cmd.assert().success();
    let c = fs::read(out.path().join("main.c")).expect("main.c emitted");
    let h = fs::read(out.path().join("main.h")).expect("main.h emitted");
    (c, h)
}

/// Fresh, empty cache directory (its own `CHELIS_REEF_HOME`).
fn fresh_cache_home() -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let home = dir.path().join("reef-home");
    fs::create_dir_all(&home).expect("mkdir reef-home");
    (dir, home)
}

/// `chelis-lib-*.tc` artifacts under a `CHELIS_REEF_HOME` cache dir.
fn library_cache_artifacts(cache_home: &Path) -> Vec<PathBuf> {
    let dir = cache_home.join(".cache").join("typecheck");
    let Ok(entries) = fs::read_dir(&dir) else {
        return Vec::new();
    };
    entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("chelis-lib-") && n.ends_with(".tc"))
                .unwrap_or(false)
        })
        .collect()
}

#[test]
fn cold_vs_warm_build_c_identical() {
    let (scratch, cache_home) = fresh_cache_home();
    let (dep, entry_body) = plain_bodies();
    let entry = stage_dep_fixture(scratch.path(), dep, entry_body);

    // Cold: empty cache dir — computes + writes the dependency artifact.
    // Warm: same cache dir — must hit.
    let cold = build_c(&entry, &cache_home, &[]);
    let warm = build_c(&entry, &cache_home, &[]);
    assert_eq!(
        cold, warm,
        "cold vs warm emitted C diverged: a dependency cache hit must never \
         perturb the compiled output"
    );
}

#[test]
fn monolithic_vs_layered_build_c_identical() {
    let (scratch, cache_home) = fresh_cache_home();
    let (dep, entry_body) = plain_bodies();
    let entry = stage_dep_fixture(scratch.path(), dep, entry_body);

    let monolithic = build_c(&entry, &cache_home, &[("CHELIS_STDLIB_CACHE_DISABLE", "1")]);
    let layered = build_c(&entry, &cache_home, &[]);
    assert_eq!(
        monolithic, layered,
        "monolithic vs three-layer emitted C diverged: a compiler-correctness \
         divergence in the dependency-layer split"
    );
}

#[test]
fn macro_hygiene_monolithic_vs_layered_c_identical() {
    let (scratch, cache_home) = fresh_cache_home();
    let (dep, entry_body) = macro_bodies();
    let entry = stage_dep_fixture(scratch.path(), dep, entry_body);

    // The dependency's macro advances the hygiene counter; the entry's
    // `v_macro_0` binding is exactly what a restarted counter would capture.
    // The three-layer split must expand dependency+entry as one unit.
    let monolithic = build_c(&entry, &cache_home, &[("CHELIS_STDLIB_CACHE_DISABLE", "1")]);
    let layered = build_c(&entry, &cache_home, &[]);
    // Confirm the Layer-2 cache actually engaged for this fixture.
    assert!(
        !library_cache_artifacts(&cache_home).is_empty(),
        "the macro fixture must engage the dependency cache"
    );
    assert_eq!(
        monolithic, layered,
        "macro hygiene diverged across the dependency/entry split: the \
         layered path renamed the entry's binders (miscompile)"
    );
}

#[test]
fn dependency_edit_misses_and_rebuilds() {
    let (scratch, cache_home) = fresh_cache_home();
    let (dep, entry_body) = plain_bodies();
    let entry = stage_dep_fixture(scratch.path(), dep, entry_body);

    // Warm the dependency cache.
    let _ = build_c(&entry, &cache_home, &[]);

    // Mutate the dependency: `add(x, y)` -> `add(y, x)` changes both the
    // source bytes and the emitted body, so the key must flip to a miss.
    let dep_path = scratch.path().join("pseudo-app/azdep/src/math.ch");
    fs::write(
        &dep_path,
        "module Azdep.Math\nexport (az_add)\n\ndef az_add(x: int32, y: int32) -> int32 = add(y, x)\n",
    )
    .expect("rewrite dep");

    let layered = build_c(&entry, &cache_home, &[]);
    let monolithic = build_c(&entry, &cache_home, &[("CHELIS_STDLIB_CACHE_DISABLE", "1")]);
    assert_eq!(
        layered, monolithic,
        "after a dependency edit the layered C must match the monolithic C of \
         the mutated program (clean miss, not a stale hit)"
    );
}

#[test]
fn entry_only_edit_reuses_dependency_cache() {
    let (scratch, cache_home) = fresh_cache_home();
    let (dep, entry_body) = plain_bodies();
    let entry = stage_dep_fixture(scratch.path(), dep, entry_body);

    // Warm the dependency cache with the original entry.
    let _ = build_c(&entry, &cache_home, &[]);
    let deps_before = library_cache_artifacts(&cache_home);
    assert!(
        !deps_before.is_empty(),
        "the first build must have written a dependency cache artifact"
    );

    // Edit ONLY the entry (3+4 -> 5+6). The dependency is untouched, so its
    // key is unchanged: a warm dependency hit.
    fs::write(
        &entry,
        "module PseudoApp.Main\nimport Azdep.Math (az_add)\n\ndef main_value -> int32 = az_add(cast(5, int32), cast(6, int32))\n",
    )
    .expect("rewrite entry");

    let layered = build_c(&entry, &cache_home, &[]);
    let monolithic = build_c(&entry, &cache_home, &[("CHELIS_STDLIB_CACHE_DISABLE", "1")]);
    assert_eq!(
        layered, monolithic,
        "an entry-only edit must reflect in the layered C and equal the \
         monolithic C of the edited program (warm dependency hit)"
    );

    // No new dependency artifact was written for the entry edit.
    let mut before = deps_before;
    let mut after = library_cache_artifacts(&cache_home);
    before.sort();
    after.sort();
    assert_eq!(
        before, after,
        "an entry-only edit must not write a new dependency cache artifact"
    );
}

#[test]
fn dependency_build_writes_library_cache_artifact() {
    let (scratch, cache_home) = fresh_cache_home();
    let (dep, entry_body) = plain_bodies();
    let entry = stage_dep_fixture(scratch.path(), dep, entry_body);

    assert!(
        library_cache_artifacts(&cache_home).is_empty(),
        "cold cache must start with no dependency artifacts"
    );
    let _ = build_c(&entry, &cache_home, &[]);
    assert!(
        !library_cache_artifacts(&cache_home).is_empty(),
        "building a package with a dependency must write a chelis-lib-*.tc \
         Layer-2 artifact"
    );
}
