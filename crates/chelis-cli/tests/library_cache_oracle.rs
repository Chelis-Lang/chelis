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
//! 7. `pruning_fires_monolithic_vs_layered_build_c_identical` — with an unused
//!    dependency def (so build-time pruning fires), the layered path — which
//!    now RUNS under pruning (chelis#1168 engage-under-pruning) and subsumes
//!    the cross-module full-program check — emits C byte-identical to the
//!    monolithic path.
//! 8. `pruning_fires_macro_hygiene_monolithic_vs_layered_c_identical` — the
//!    same with an unused macro-minting dependency def: expansion advances the
//!    shared hygiene counter over the full program before pruning drops the
//!    def, so the entry's binders survive the split identically.
//! 9. `eval_only_wrapper_build_accept_reject_parity` — a well-typed package
//!    with an unreachable eval-only def AND an unreachable wrapper of it builds
//!    identically (accept) in both cache regimes: the transitive eval-only drop
//!    leaves no dangling reference for the monolithic check to spuriously
//!    reject. Negative parity — the C-bytes oracles assert success on both arms
//!    and are blind to an accept/reject flip.
//! 10. `unreachable_dep_type_error_rejected_in_both_cache_regimes` — the
//!     rejection direction: a violation in an unreachable dependency def is
//!     rejected byte-identically in both cache regimes (layered `Ok(None)` →
//!     monolithic fallback).

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
            "schema = \"1\"\n\n[package]\nname = \"pseudo-app\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"PseudoApp\"\n\n[dependencies]\nazdep = {{ path = \"./azdep\" }}\n"
        ),
    );
    write(&root.join("src/main.ch"), entry_body);
    write(
        &root.join("azdep/reef.toml"),
        &format!(
            "schema = \"1\"\n\n[package]\nname = \"azdep\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Azdep\"\n"
        ),
    );
    write(&root.join("azdep/src/math.ch"), dep_body);
    write(
        &root.join("reef.lock"),
        &format!(
            "schema = \"1\"\n\n[package]\nname = \"pseudo-app\"\nversion = \"0.1.0\"\n\n[[dependencies]]\nname = \"azdep\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\narchive_sha256 = \"\"\nshell_sha256 = \"\"\n\n[dependencies.source]\nkind = \"path\"\npath = \"./azdep\"\n"
        ),
    );
    root.join("src/main.ch")
}

/// A plain (macro-free) dependency + entry pair that exercises the cache.
fn plain_bodies() -> (&'static str, &'static str) {
    (
        "module Azdep.Math\nexport (az_add)\n\ndef az_add(x: int32, y: int32) -> int32 = add(x, y)\n",
        "module PseudoApp.Main\nimport Azdep.Math (az_add)\n\ndef main_value() ->int32 = az_add(cast(3, int32), cast(4, int32))\n",
    )
}

/// A macro-bearing pair: the dependency defines and uses a binder-minting
/// macro (advancing the shared hygiene counter), and the entry binds
/// `v_macro_0` — the name a restarted counter would collide with.
fn macro_bodies() -> (&'static str, &'static str) {
    (
        "module Azdep.Math\nexport (dep_val)\n\nmacro dmk(a) = {\n  q = a\n  add(q, q)\n}\n\ndef dep_val(x: int32) -> int32 = dmk(x)\n",
        "module PseudoApp.Main\nimport Azdep.Math (dep_val)\n\nmacro emk(a) = {\n  v = cast(7, int32)\n  add(v, a)\n}\n\ndef main_value() ->int32 = {\n  v_macro_0 = dep_val(cast(5, int32))\n  emk(v_macro_0)\n}\n",
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
        "module PseudoApp.Main\nimport Azdep.Math (az_add)\n\ndef main_value() ->int32 = az_add(cast(5, int32), cast(6, int32))\n",
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

// ── Pruning-FIRES oracles (chelis#1168 engage-under-pruning) ─────────
//
// Every oracle above uses fixtures whose defs are all reachable, so
// `pruned_deep_exprs.len() == full_deep_exprs.len()` and build-time pruning
// never fires. That is exactly the case the #1168 cache was ORIGINALLY gated
// to (`layered_full_checked` was `None` whenever pruning fired). These
// fixtures add an UNUSED dependency def so pruning drops it — the path where
// `cmd_build` runs the cross-module full-program check. With the gate removed,
// the layered check runs under pruning and subsumes that check; the emitted C
// must still match the monolithic path byte-for-byte.

/// Like `plain_bodies`, but the dependency carries an extra `az_unused` def
/// that the entry never calls. The entry matches `plain_bodies`' entry, so
/// after pruning drops `az_unused` the program equals the `plain_bodies`
/// program.
fn plain_bodies_with_unused_dep() -> (&'static str, &'static str) {
    (
        "module Azdep.Math\nexport (az_add, az_unused)\n\ndef az_add(x: int32, y: int32) -> int32 = add(x, y)\ndef az_unused(x: int32, y: int32) -> int32 = add(add(x, y), y)\n",
        "module PseudoApp.Main\nimport Azdep.Math (az_add)\n\ndef main_value() ->int32 = az_add(cast(3, int32), cast(4, int32))\n",
    )
}

/// Like `macro_bodies`, but the dependency also carries an UNUSED
/// macro-minting def (`dep_unused`). Its `dmk` expansion still advances the
/// shared hygiene counter over the full program (expansion precedes pruning),
/// then pruning drops it — so the entry's `v_macro_0` binding must survive the
/// dependency/entry split identically under pruning.
fn macro_bodies_with_unused_dep() -> (&'static str, &'static str) {
    (
        "module Azdep.Math\nexport (dep_val, dep_unused)\n\nmacro dmk(a) = {\n  q = a\n  add(q, q)\n}\n\ndef dep_val(x: int32) -> int32 = dmk(x)\ndef dep_unused(x: int32) -> int32 = dmk(add(x, x))\n",
        "module PseudoApp.Main\nimport Azdep.Math (dep_val)\n\nmacro emk(a) = {\n  v = cast(7, int32)\n  add(v, a)\n}\n\ndef main_value() ->int32 = {\n  v_macro_0 = dep_val(cast(5, int32))\n  emk(v_macro_0)\n}\n",
    )
}

#[test]
fn pruning_fires_monolithic_vs_layered_build_c_identical() {
    let (scratch, cache_home) = fresh_cache_home();
    let (dep, entry_body) = plain_bodies_with_unused_dep();
    let entry = stage_dep_fixture(scratch.path(), dep, entry_body);

    let monolithic = build_c(&entry, &cache_home, &[("CHELIS_STDLIB_CACHE_DISABLE", "1")]);
    let layered = build_c(&entry, &cache_home, &[]);
    assert_eq!(
        monolithic, layered,
        "under build-time pruning, the layered path's emitted C must match the \
         monolithic path (chelis#1168 engage-under-pruning)"
    );

    // Confirm pruning ACTUALLY fired: `az_unused` is unreachable from the
    // entry, so its symbol must be absent from the compiled C. The emitted C
    // carries the un-mangled def name (`az_add` appears), so `az_unused` would
    // too if pruning had not dropped it — i.e. this fixture genuinely exercises
    // `pruned_deep_exprs.len() != full_deep_exprs.len()`.
    let layered_c = String::from_utf8_lossy(&layered.0);
    assert!(
        layered_c.contains("az_add"),
        "sanity: the reachable dependency def must appear in the emitted C"
    );
    assert!(
        !layered_c.contains("az_unused"),
        "pruning must drop the unused dependency def `az_unused`: this fixture \
         must exercise the pruning-fires path"
    );
}

#[test]
fn pruning_fires_macro_hygiene_monolithic_vs_layered_c_identical() {
    let (scratch, cache_home) = fresh_cache_home();
    let (dep, entry_body) = macro_bodies_with_unused_dep();
    let entry = stage_dep_fixture(scratch.path(), dep, entry_body);

    let monolithic = build_c(&entry, &cache_home, &[("CHELIS_STDLIB_CACHE_DISABLE", "1")]);
    let layered = build_c(&entry, &cache_home, &[]);
    assert!(
        !library_cache_artifacts(&cache_home).is_empty(),
        "the macro fixture must engage the dependency cache"
    );
    assert_eq!(
        monolithic, layered,
        "macro hygiene diverged under pruning: the layered path renamed the \
         entry's binders (miscompile)"
    );
}

// ── Accept/reject parity under the eval-only transitive drop (chelis#1168) ──
//
// `cmd_build` drops UNREACHABLE defs that (transitively) reference an eval-only
// host builtin BEFORE the monolithic full-program check, but the layered cache
// path checks the intact pre-drop decls. If the drop were non-transitive, an
// unreachable wrapper of a dropped def would keep a DANGLING reference: the
// monolithic path manufactures an unbound-variable error the layered path never
// sees, so build accept/reject would flip on cache state. The existing C-bytes
// oracles all assert success on both arms, so they are structurally blind to
// this — hence a dedicated negative parity oracle (chelis#1168 fable-verify).

/// Build `entry` and capture `(success, stderr)` WITHOUT asserting the outcome,
/// so the monolithic and cache-warm paths can be compared for accept/reject
/// parity.
fn build_capture(entry: &Path, cache_home: &Path, extra_env: &[(&str, &str)]) -> (bool, String) {
    let out = tempdir().expect("out dir");
    let mut cmd = Command::cargo_bin("chelis").expect("chelis binary");
    cmd.env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", cache_home);
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    cmd.arg("build").arg(entry).arg("-o").arg(out.path());
    let output = cmd.output().expect("run chelis build");
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// The dependency has an unreachable eval-only CHAIN: `dep_runner` uses the
/// eval-only `round_to` builtin, `dep_wrapper` calls `dep_runner`, and
/// `dep_outer` calls `dep_wrapper`. The entry uses only `az_add`. The transitive
/// drop must remove ALL THREE so no dangling reference reaches the monolithic
/// check. The chain is depth-3 on purpose: a single-pass "direct users + their
/// direct dependents" drop would leave `dep_outer` dangling, so this fixture
/// pins the FIXPOINT, not just one hop.
fn eval_only_wrapper_bodies() -> (&'static str, &'static str) {
    (
        "module Azdep.Math\nexport (az_add)\n\ndef az_add(x: int32, y: int32) -> int32 = add(x, y)\ndef dep_runner(x: f64) -> f64 = round_to(x, cast(2, int32))\ndef dep_wrapper(x: f64) -> f64 = dep_runner(x)\ndef dep_outer(x: f64) -> f64 = dep_wrapper(x)\n",
        "module PseudoApp.Main\nimport Azdep.Math (az_add)\n\ndef main_value() ->int32 = az_add(cast(3, int32), cast(4, int32))\n",
    )
}

#[test]
fn eval_only_wrapper_build_accept_reject_parity() {
    let (scratch, cache_home) = fresh_cache_home();
    let (dep, entry_body) = eval_only_wrapper_bodies();
    let entry = stage_dep_fixture(scratch.path(), dep, entry_body);

    let (mono_ok, mono_err) =
        build_capture(&entry, &cache_home, &[("CHELIS_STDLIB_CACHE_DISABLE", "1")]);
    let _ = build_capture(&entry, &cache_home, &[]); // cold: warm the cache
    let (warm_ok, warm_err) = build_capture(&entry, &cache_home, &[]);

    assert_eq!(
        mono_ok, warm_ok,
        "build accept/reject must not flip on cache state\n \
         monolithic: ok={mono_ok} stderr={mono_err:?}\n \
         cache-warm: ok={warm_ok} stderr={warm_err:?}"
    );
    assert_eq!(
        mono_err, warm_err,
        "build stderr must be byte-identical across cache regimes"
    );
    // The package is well-typed (`chelis check` accepts it): the transitive drop
    // removes the unreachable eval-only chain, so it builds in BOTH regimes.
    assert!(
        mono_ok,
        "well-typed package must build in both cache regimes; stderr={mono_err:?}"
    );
}

/// An unreachable dependency def with a plain type error. It is NOT eval-only,
/// so it survives the eval-only drop and reaches the monolithic full-program
/// check; the layered check also sees it and returns `Ok(None)`, so the
/// monolithic fallback produces the diagnostic. Both regimes must REJECT.
fn unreachable_type_error_bodies() -> (&'static str, &'static str) {
    (
        "module Azdep.Math\nexport (az_add)\n\ndef az_add(x: int32, y: int32) -> int32 = add(x, y)\ndef dep_broken(x: int32) -> int32 = add(x, cast(1, f64))\n",
        "module PseudoApp.Main\nimport Azdep.Math (az_add)\n\ndef main_value() ->int32 = az_add(cast(3, int32), cast(4, int32))\n",
    )
}

#[test]
fn unreachable_dep_type_error_rejected_in_both_cache_regimes() {
    // The rejection direction of accept/reject parity: a violation in an
    // unreachable dependency def must surface in BOTH cache regimes (the layered
    // check returns `Ok(None)` on the error and the monolithic fallback emits
    // the diagnostic), byte-identical (chelis#1168).
    let (scratch, cache_home) = fresh_cache_home();
    let (dep, entry_body) = unreachable_type_error_bodies();
    let entry = stage_dep_fixture(scratch.path(), dep, entry_body);

    let (mono_ok, mono_err) =
        build_capture(&entry, &cache_home, &[("CHELIS_STDLIB_CACHE_DISABLE", "1")]);
    let _ = build_capture(&entry, &cache_home, &[]);
    let (warm_ok, warm_err) = build_capture(&entry, &cache_home, &[]);

    assert!(
        !mono_ok && !warm_ok,
        "an unreachable dependency type error must be rejected in BOTH cache \
         regimes: monolithic ok={mono_ok}, cache-warm ok={warm_ok}"
    );
    assert_eq!(
        mono_err, warm_err,
        "the rejection stderr must be byte-identical across cache regimes"
    );
}

// ── Differential cache-parity sweep (chelis#1176) ───────────────────
//
// The removed monolithic full-program check only ran on reef packages, so the
// property "layered-clean ⟹ monolithic-clean" (and byte-identical emitted C)
// must hold across many entry shapes. A sweep over the loose `examples/` corpus
// would NOT exercise this: those files are not in a reef package, so the layered
// cache never engages and `CHELIS_STDLIB_CACHE_DISABLE` is a no-op. Instead this
// sweeps varied shapes staged as reef packages (dep sorts before the root, an
// unused dep def forces pruning), asserting the monolithic and cache-warm builds
// AGREE on accept/reject + stderr, and emit byte-identical C when both succeed.

/// `(success, stderr, Some((main.c, main.h) bytes) on success)`.
type BuildProbe = (bool, String, Option<(Vec<u8>, Vec<u8>)>);

/// Build `entry` and capture its outcome for cross-cache-regime comparison.
fn build_probe(entry: &Path, cache_home: &Path, extra_env: &[(&str, &str)]) -> BuildProbe {
    let out = tempdir().expect("out dir");
    let mut cmd = Command::cargo_bin("chelis").expect("chelis binary");
    cmd.env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", cache_home);
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    cmd.arg("build").arg(entry).arg("-o").arg(out.path());
    let output = cmd.output().expect("run chelis build");
    let ok = output.status.success();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    // The emitted C is `<file-stem>.c` (+ `.h`); the runtime files are `.h`/`.a`.
    // Find the single `.c` so this works for any entry stem, not just `main.ch`.
    let files = ok
        .then(|| {
            let c_path = fs::read_dir(out.path())
                .ok()?
                .flatten()
                .map(|e| e.path())
                .find(|p| p.extension().and_then(|e| e.to_str()) == Some("c"))?;
            let c = fs::read(&c_path).ok()?;
            let h = fs::read(c_path.with_extension("h")).ok()?;
            Some((c, h))
        })
        .flatten();
    (ok, stderr, files)
}

#[test]
fn differential_cache_parity_sweep() {
    // (label, dependency module body, entry module body). Each dependency has an
    // UNUSED def so pruning fires. Shapes: int32 scalar, f64 scalar, tensor
    // entry, multi-def entry, macro-bearing, plus the two accept/reject-parity
    // witnesses.
    let cases: &[(&str, &str, &str)] = &[
        (
            "int32_scalar",
            "module Azdep.Math\nexport (az_add, az_unused)\n\ndef az_add(x: int32, y: int32) -> int32 = add(x, y)\ndef az_unused(x: int32) -> int32 = add(x, x)\n",
            "module PseudoApp.Main\nimport Azdep.Math (az_add)\n\ndef main_value() ->int32 = az_add(cast(3, int32), cast(4, int32))\n",
        ),
        (
            "f64_scalar",
            "module Azdep.Math\nexport (az_fadd, az_unused)\n\ndef az_fadd(x: f64, y: f64) -> f64 = add(x, y)\ndef az_unused(x: f64) -> f64 = add(x, x)\n",
            "module PseudoApp.Main\nimport Azdep.Math (az_fadd)\n\ndef main_value() ->f64 = az_fadd(cast(1.0, f64), cast(2.0, f64))\n",
        ),
        (
            "tensor_entry",
            "module Azdep.Math\nexport (az_vadd, az_unused)\n\ndef az_vadd(v: tensor[3, f32], w: tensor[3, f32]) -> tensor[3, f32] = add(v, w)\ndef az_unused(v: tensor[3, f32]) -> tensor[3, f32] = add(v, v)\n",
            "module PseudoApp.Main\nimport Azdep.Math (az_vadd)\n\ndef main(v: tensor[3, f32], w: tensor[3, f32]) -> tensor[3, f32] = az_vadd(v, w)\n",
        ),
        (
            "multi_def_entry",
            "module Azdep.Math\nexport (d1, d2, az_unused)\n\ndef d1(x: int32, y: int32) -> int32 = add(x, y)\ndef d2(x: int32) -> int32 = add(x, x)\ndef az_unused(x: int32) -> int32 = add(add(x, x), x)\n",
            "module PseudoApp.Main\nimport Azdep.Math (d1, d2)\n\ndef h(x: int32) -> int32 = d2(x)\ndef main_value() ->int32 = h(d1(cast(1, int32), cast(2, int32)))\n",
        ),
        (
            "macro_bearing",
            "module Azdep.Math\nexport (dep_val, dep_unused)\n\nmacro dmk(a) = {\n  q = a\n  add(q, q)\n}\n\ndef dep_val(x: int32) -> int32 = dmk(x)\ndef dep_unused(x: int32) -> int32 = dmk(add(x, x))\n",
            "module PseudoApp.Main\nimport Azdep.Math (dep_val)\n\ndef main_value() ->int32 = dep_val(cast(5, int32))\n",
        ),
    ];

    for (label, dep, entry_body) in cases {
        let (scratch, cache_home) = fresh_cache_home();
        let entry = stage_dep_fixture(scratch.path(), dep, entry_body);

        let (mono_ok, mono_err, mono_c) =
            build_probe(&entry, &cache_home, &[("CHELIS_STDLIB_CACHE_DISABLE", "1")]);
        let _ = build_probe(&entry, &cache_home, &[]); // cold: warm the cache
        let (warm_ok, warm_err, warm_c) = build_probe(&entry, &cache_home, &[]);

        assert_eq!(
            mono_ok, warm_ok,
            "[{label}] build accept/reject must not flip on cache state\n \
             monolithic: ok={mono_ok} stderr={mono_err:?}\n \
             cache-warm: ok={warm_ok} stderr={warm_err:?}"
        );
        assert_eq!(
            mono_err, warm_err,
            "[{label}] build stderr must be byte-identical across cache regimes"
        );
        if mono_ok {
            assert_eq!(
                mono_c, warm_c,
                "[{label}] emitted C must be byte-identical across cache regimes"
            );
        }
    }
}

#[test]
fn chelis_std_declares_no_macros() {
    // The layered split expands chelis-std SEPARATELY from `deps ++ entry`
    // (`check_layered_for_build`), relying on chelis-std minting no macros --
    // else the separate expansion would restart the per-`expand_program`
    // hygiene counter and rename non-stdlib binders, silently changing emitted
    // C. The dep/entry seam is digest-guarded against exactly this; the
    // stdlib/non-stdlib seam is not, so it rests on this invariant. Lock it
    // (chelis#1176 review).
    let (scratch, _cache_home) = fresh_cache_home();
    let (dep, entry_body) = plain_bodies();
    let entry = stage_dep_fixture(scratch.path(), dep, entry_body);
    let prepared = chelis_reef::prepare_program_for_file(&entry)
        .expect("prepare_program_for_file")
        .expect("entry resolves inside a reef package");
    let macro_count = prepared
        .stdlib_decls
        .iter()
        .filter(|d| matches!(d, chelis_surf::ast::Decl::MacroDef { .. }))
        .count();
    assert_eq!(
        macro_count, 0,
        "chelis-std must declare no macros: the layered build cache expands it \
         separately from deps++entry, so a chelis-std macro would desync the \
         hygiene counter and rename non-stdlib binders (chelis#1176)"
    );
}

/// Like `stage_dep_fixture` but with explicit package names, so a test can put
/// the dependency package AFTER the root in name order (the chelis#1182 case).
fn stage_named_dep_fixture(
    scratch: &Path,
    root_name: &str,
    root_prefix: &str,
    dep_name: &str,
    dep_prefix: &str,
    dep_body: &str,
    entry_body: &str,
) -> PathBuf {
    let root = scratch.join(root_name);
    write(
        &root.join("reef.toml"),
        &format!(
            "schema = \"1\"\n\n[package]\nname = \"{root_name}\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"{root_prefix}\"\n\n[dependencies]\n{dep_name} = {{ path = \"./{dep_name}\" }}\n"
        ),
    );
    write(&root.join("src/main.ch"), entry_body);
    write(
        &root.join(format!("{dep_name}/reef.toml")),
        &format!(
            "schema = \"1\"\n\n[package]\nname = \"{dep_name}\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"{dep_prefix}\"\n"
        ),
    );
    write(&root.join(format!("{dep_name}/src/math.ch")), dep_body);
    write(
        &root.join("reef.lock"),
        &format!(
            "schema = \"1\"\n\n[package]\nname = \"{root_name}\"\nversion = \"0.1.0\"\n\n[[dependencies]]\nname = \"{dep_name}\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\narchive_sha256 = \"\"\nshell_sha256 = \"\"\n\n[dependencies.source]\nkind = \"path\"\npath = \"./{dep_name}\"\n"
        ),
    );
    root.join("src/main.ch")
}

#[test]
fn dependency_after_root_engages_cache_and_matches_monolithic() {
    // chelis#1182 acceptance: a dependency package whose name sorts AFTER the
    // root's used to land in the entry suffix (re-inferred every build, no
    // artifact written). With the root emitted last, it lands in the cached
    // prefix. Assert the artifact IS written AND warm C == monolithic C.
    let (scratch, cache_home) = fresh_cache_home();
    let entry = stage_named_dep_fixture(
        scratch.path(),
        "pseudo-app",
        "PseudoApp",
        "zzdep", // "zzdep" > "pseudo-app": sorts AFTER the root
        "Zzdep",
        "module Zzdep.Math\nexport (zz_add)\n\ndef zz_add(x: int32, y: int32) -> int32 = add(x, y)\n",
        "module PseudoApp.Main\nimport Zzdep.Math (zz_add)\n\ndef main_value() ->int32 = zz_add(cast(3, int32), cast(4, int32))\n",
    );
    let monolithic = build_c(&entry, &cache_home, &[("CHELIS_STDLIB_CACHE_DISABLE", "1")]);
    let _cold = build_c(&entry, &cache_home, &[]);
    let warm = build_c(&entry, &cache_home, &[]);
    assert!(
        !library_cache_artifacts(&cache_home).is_empty(),
        "chelis#1182: a dependency sorting after the root must now be cached \
         (a chelis-lib-*.tc must be written)"
    );
    assert_eq!(
        monolithic, warm,
        "after-root dependency: warm C must equal monolithic C"
    );
}

/// Total bytes of files under `dir` (0 if unreadable).
fn dir_size(dir: &Path) -> u64 {
    fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.metadata().ok())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum()
}

/// Total bytes of `chelis-std-*.tc` entries under `dir`.
fn std_artifact_bytes(dir: &Path) -> u64 {
    fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("chelis-std-"))
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum()
}

#[test]
fn library_cache_evicts_under_cap_and_protects_stdlib() {
    // chelis#1183: the Layer-2 cache is bounded. Build several distinct-dep
    // packages with a cap set to force eviction; assert chelis-lib entries are
    // bounded (older ones evicted), and chelis-std survives (it is evicted only
    // after all chelis-lib, so a once-written-hit-forever std entry is not the
    // first casualty).
    let (scratch, cache_home) = fresh_cache_home();
    let dep = |i: usize| {
        format!(
            "module Azdep.Math\nexport (az_add)\n\ndef az_add(x: int32, y: int32) -> int32 = add(add(x, y), cast({i}, int32))\n"
        )
    };
    let entry = "module PseudoApp.Main\nimport Azdep.Math (az_add)\n\ndef main_value() ->int32 = az_add(cast(3, int32), cast(4, int32))\n";
    let td = cache_home.join(".cache").join("typecheck");

    // First build (default cap): writes chelis-std + one chelis-lib.
    let f0 = stage_dep_fixture(&scratch.path().join("p0"), &dep(0), entry);
    let _ = build_c(&f0, &cache_home, &[]);
    assert!(
        std_artifact_bytes(&td) > 0,
        "chelis-std artifact must exist after the first build"
    );
    // Cap = current dir size, so each new chelis-lib pushes over → eviction.
    let cap_env = dir_size(&td).to_string();
    for i in 1..=4 {
        let f = stage_dep_fixture(&scratch.path().join(format!("p{i}")), &dep(i), entry);
        let _ = build_c(
            &f,
            &cache_home,
            &[("CHELIS_TYPECHECK_CACHE_MAX_BYTES", &cap_env)],
        );
    }

    assert!(
        std_artifact_bytes(&td) > 0,
        "chelis-std must survive eviction (chelis-lib is evicted first)"
    );
    let lib_count = library_cache_artifacts(&cache_home).len();
    assert!(
        lib_count < 5,
        "eviction must bound chelis-lib entries under the cap; found {lib_count}"
    );
}

#[test]
fn chelis_std_importing_build_monolithic_vs_cache_warm_c_identical() {
    // The other oracles use no-chelis-std fixtures. This one imports chelis-std,
    // so the build pulls in Std.Process (eval-only, entry-unreachable) and
    // `drop_unreachable_eval_only_defs` actually fires against REAL chelis-std --
    // the motivating path (chelis#1176 review, F4). Reuses the committed
    // `pseudo_nautilus` fixture (an erf approximation over chelis-std scalars).
    let scratch = tempdir().expect("scratch");
    let pkg = scratch.path().join("pseudo-nautilus");
    write(
        &pkg.join("reef.toml"),
        include_str!("fixtures/pseudo_nautilus/reef.toml"),
    );
    write(
        &pkg.join("src/special.ch"),
        include_str!("fixtures/pseudo_nautilus/src/special.ch"),
    );
    let entry = pkg.join("src/special.ch");
    let (_guard, cache_home) = fresh_cache_home();

    // Settle reef.lock (the first build resolves + writes it, which fixes the
    // chelis-std set folded into the key) so the compared builds are stable.
    let _ = build_probe(&entry, &cache_home, &[]);

    let (mono_ok, _mono_err, mono_c) =
        build_probe(&entry, &cache_home, &[("CHELIS_STDLIB_CACHE_DISABLE", "1")]);
    let (warm_ok, _warm_err, warm_c) = build_probe(&entry, &cache_home, &[]);

    assert!(
        mono_ok && warm_ok,
        "a chelis-std-importing package must build in both cache regimes"
    );
    // Confirm the stdlib layer engaged (i.e. real chelis-std was linked and the
    // eval-only drop path ran over it, not the no-chelis-std shortcut).
    let std_artifacts = fs::read_dir(cache_home.join(".cache").join("typecheck"))
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("chelis-std-"))
        .count();
    assert!(
        std_artifacts > 0,
        "the chelis-std sub-context must have been cached (the drop ran over real chelis-std)"
    );
    assert_eq!(
        mono_c, warm_c,
        "chelis-std-importing build C must be byte-identical, monolithic vs cache-warm"
    );
}
