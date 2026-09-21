//! chelis#2318 — an exported `def` carrying an undischarged `Random` compiled
//! into a public entry that cannot honour it. `chelis check` scored 1.0, the
//! build succeeded, and the compiled behaviour then depended on which lane the
//! draw routed to: the tensor-DAG lane returned values drawn against an
//! INACTIVE RNG (silent, unreproducible from any `with seed` the author
//! wrote), while the C host lane aborted at the first call.
//!
//! ## What was actually wrong
//!
//! NOT the effect checker. The issue reported the checker as the root cause,
//! but a definition whose inferred effect set contains `Random` is an ordinary
//! function whose CALLER supplies the handler — that is how the stdlib's own
//! random helpers work (`spec/04-type-system.md` §7.1: `normal_like` and the
//! Kaiming/Xavier initializers "inherit that effect through calls"). Rejecting
//! every such definition would reject the documented stdlib pattern.
//! `random_carrying_def_discharged_by_its_caller_still_builds` is the guard
//! that keeps that pattern legal.
//!
//! The defect is at the BUILD BOUNDARY. The public entry ABI carries no RNG
//! frame, so `host_emit.rs`'s authored-export wrapper opened one inactive
//! (`chelis_rng_state __chelis_rng_local = {0ULL, 0ULL, 0}`) and handed it to
//! the body. The rule that forbids exactly this already existed for ONE
//! emission path — `emit.rs`'s `public tensor entry cannot receive inherited
//! Random`, chelis#1872 — but it lives in `emit_evaluation`, whose only caller
//! is the fixed-control path (`lib.rs` `codegen_evaluation_with_options`). The
//! ordinary export build structurally never reached it.
//!
//! ## Demote, then reject
//!
//! The remedy is not a blanket rejection. A program that merely DEFINES a
//! random helper and uses it correctly under `with seed(...)` published a
//! broken wrapper for that helper too — `lib_ok`'s header really did declare
//! `chelis_tensor* my_noise(chelis_tensor* x)` over an inactive RNG — but the
//! program itself is fine, because `main` bypasses the wrapper and calls the
//! owned body inside the seeded scope. Rejecting it would make the ordinary
//! library pattern uncompilable.
//!
//! So a `Random`-carrying definition is DEMOTED: its body is still emitted for
//! the program's own seeded callers, and only the unusable public wrapper and
//! its header declaration are withheld. Demotion is rejected instead when the
//! program has no globals, because `emit_main` is emitted exactly when globals
//! exist — with none, there is no internal caller, the demoted definition is
//! unreachable, and withholding it would silently delete the only thing the
//! author asked to build. A publishable sibling does not excuse that deletion:
//! `e2` would still have exported `bc`, but `draw` is the point of `e2`.
//!
//! ## Spec authority
//!
//! `spec/04-type-system.md` [04-EFF-3], authored with this change: the build
//! boundary SHALL reject every public entry whose body can perform a `Random`
//! effect that no handler inside that body discharges, identically on every
//! emission path, and a path that instead emits an entry holding an inactive
//! RNG state is non-conforming "whether the resulting draw traps or returns a
//! value" — which is precisely the two behaviours this issue reported.
//!
//! [04-EFF-3] is the missing normative text: before it, `emit.rs`'s guard
//! enforced a rule the spec never stated, on one path only.
//!
//! ## Test roles
//!
//! * `dag_lane_*` and `host_lane_*` are the two NEGATIVE cases, one per lane —
//!   the silent draw and the abort respectively. Both are rejected: neither
//!   program opens an RNG scope, so neither has any internal caller that could
//!   discharge the effect.
//! * `random_carrying_def_discharged_by_its_caller_still_builds` and
//!   `def_that_discharges_its_own_random_is_a_valid_public_entry` are POSITIVE
//!   PARITY: the rejection must not widen to legal programs.
//! * `check_still_scores_one_on_an_exported_random_def` is a DISPOSITION LOCK
//!   asserting the fix did NOT land in the checker. Those programs are well
//!   typed; the fix belongs at the build boundary and nowhere else.
//! * `top_level_binding_random_is_still_a_check_error` locks the pre-existing
//!   checker behaviour that the issue's `caught` case exercises.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::{gcc_available, link_generated, parse_tensor_data, write_file};

/// The issue's `e1`: a foldable template, so the draw routes to the
/// tensor-DAG lane. This one returned values against an inactive RNG.
const DAG_LANE_EXPORT: &str =
    "def draw(c: f32) -> tensor[4, f32] = uniform_like(to_tensor([c, c, c, c]), 2.0f32, 5.0f32)\n";

/// The issue's `e2`: a runtime-derived template, so the draw routes to the C
/// host lane. This one aborted at the first call.
const HOST_LANE_EXPORT: &str = "def bc(c: f32) -> tensor[4, f32] = to_tensor([c, c, c, c])\n\
     def draw(c: f32) -> tensor[4, f32] = uniform_like(bc(c), 2.0f32, 5.0f32)\n";

/// The stdlib pattern: a definition carrying `Random`, discharged by its
/// CALLER. This must stay legal and must keep building.
const DISCHARGED_BY_CALLER: &str = "def my_noise(x: tensor[4, f32]) -> tensor[4, f32] = uniform_like(x, 2.0f32, 5.0f32)\n\
     sampled = with seed(42i64) \
     { my_noise(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32])) }\n";

/// A public entry that discharges its OWN `Random`. Its effect row is closed,
/// so [04-EFF-3] admits it.
const SELF_HANDLED_ENTRY: &str = "def draw(c: f32) -> tensor[4, f32] = \
     with seed(42i64) { uniform_like(to_tensor([c, c, c, c]), 2.0f32, 5.0f32) }\n\
     sampled = draw(0.5f32)\n";

/// The issue's `caught` case: the same effect on a top-level VALUE binding,
/// which the checker has always rejected.
const TOP_LEVEL_BINDING: &str = "def bc(c: f32) -> tensor[4, f32] = to_tensor([c, c, c, c])\n\
     sampled = uniform_like(bc(cast(0.5, f32)), 2.0f32, 5.0f32)\n";

/// Run `chelis build --target c` and return (success, combined stderr).
fn build_result(program: &str, name: &str) -> (bool, String) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("chelis build should run");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Run `chelis check` and return the reported score.
fn check_score(program: &str, name: &str) -> f64 {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("chelis check should run");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("check output not JSON: {e}\n{stdout}"));
    parsed["score"]
        .as_f64()
        .unwrap_or_else(|| panic!("check output has no numeric score: {stdout}"))
}

/// Assert a build was rejected by [04-EFF-3], naming the rule and the entry.
fn assert_eff3_rejection(program: &str, name: &str) {
    let (ok, stderr) = build_result(program, name);
    assert!(
        !ok,
        "[{name}] build succeeded; [04-EFF-3] requires a public entry with an \
         undischarged Random to be rejected before any artifact is produced",
    );
    assert!(
        stderr.contains("[04-EFF-3]"),
        "[{name}] rejection does not cite [04-EFF-3]: {stderr}",
    );
    assert!(
        stderr.contains("draw"),
        "[{name}] rejection does not name the offending entry `draw`: {stderr}",
    );
}

/// NEGATIVE: the tensor-DAG lane. Before the fix this built and returned
/// `2, 4.64993, 3.29458, 2.0793` against an inactive RNG, exit 0 — the silent
/// half, and the worse of the two.
#[test]
fn dag_lane_export_with_undischarged_random_is_rejected() {
    assert_eff3_rejection(DAG_LANE_EXPORT, "dag_lane");
}

/// NEGATIVE: the C host lane. Before the fix this built and aborted at the
/// first call with `uniform_like requires an active host RNG scope`, exit 134.
///
/// Like `e1` this is rejected, and for the same reason: the program opens no
/// `with seed(...)` region anywhere, so nothing in it can call `draw` under a
/// handler. A publishable sibling (`bc`, which carries no `Random`) does not
/// change that — `draw` is the point of the program, and withholding it while
/// exporting `bc` would delete what the author asked for.
#[test]
fn host_lane_export_with_undischarged_random_is_rejected() {
    assert_eff3_rejection(HOST_LANE_EXPORT, "host_lane");
}

/// The rejection must explain itself: why the entry cannot exist, and what to
/// do instead. A bare "unsupported" leaves the author guessing which of the two
/// remedies applies.
#[test]
fn the_rejection_names_the_cause_and_both_remedies() {
    let (ok, stderr) = build_result(DAG_LANE_EXPORT, "message");
    assert!(!ok, "expected a rejection");
    for fragment in [
        "nothing inside it handles that effect",
        "no RNG frame",
        "opens no `with seed(...)` region anywhere",
        "wrap the body in `with seed(...) { ... }`",
        "call it from a seeded region",
    ] {
        assert!(
            stderr.contains(fragment),
            "rejection should say {fragment:?}; got:\n{stderr}",
        );
    }
}

/// POSITIVE PARITY: the stdlib pattern. `my_noise` carries `Random` and is
/// discharged by its caller, and the program has a global, so `my_noise` is
/// DEMOTED rather than rejected and the build succeeds. If this fails, the fix
/// landed too wide and the ordinary library pattern no longer compiles.
#[test]
fn random_carrying_def_discharged_by_its_caller_still_builds() {
    if !gcc_available() {
        eprintln!("skipping: no C compiler available");
        return;
    }
    let (ok, stderr) = build_result(DISCHARGED_BY_CALLER, "discharged");
    assert!(
        ok,
        "a Random-carrying def discharged by its caller must still build \
         (spec/04-type-system.md §7.1 stdlib helper pattern): {stderr}",
    );
}

/// POSITIVE PARITY: a public entry whose body discharges its own `Random` has
/// a closed effect row, so [04-EFF-3] admits it. Built, linked, and run, so
/// this is an execution claim rather than a build-only one.
#[test]
fn def_that_discharges_its_own_random_is_a_valid_public_entry() {
    if !gcc_available() {
        eprintln!("skipping: no C compiler available");
        return;
    }
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("selfhandled.ch");
    let out_dir = dir.path().join("selfhandled-out");
    write_file(&path, SELF_HANDLED_ENTRY);
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    let status = link_generated(&out_dir, "selfhandled.c", "selfhandled");
    assert!(status.success(), "link of generated C failed: {status}");
    let run = std::process::Command::new(out_dir.join("selfhandled"))
        .output()
        .expect("compiled binary should run");
    assert!(
        run.status.success(),
        "a self-handled Random entry must run, not abort: {}",
        String::from_utf8_lossy(&run.stderr),
    );
    let values = parse_tensor_data(&String::from_utf8_lossy(&run.stdout), "sampled");
    assert_eq!(values.len(), 4, "expected a 4-element draw, got {values:?}");
    for (i, v) in values.iter().enumerate() {
        assert!(
            (2.0..5.0).contains(v),
            "elem[{i}] = {v} is outside the declared [2, 5) bounds",
        );
    }
}

/// DISPOSITION LOCK: the fix is at the build boundary, NOT the checker. These
/// programs are well typed — a definition carrying `Random` is an ordinary
/// function — so `check` must keep scoring 1.0. A failure here means the
/// rejection was added to the checker, which would also reject the stdlib
/// pattern above.
#[test]
fn check_still_scores_one_on_an_exported_random_def() {
    assert_eq!(
        check_score(DAG_LANE_EXPORT, "dag_check"),
        1.0,
        "an exported def carrying Random is well typed; the [04-EFF-3] \
         rejection belongs to the build boundary, not `check`",
    );
    assert_eq!(
        check_score(HOST_LANE_EXPORT, "host_check"),
        1.0,
        "an exported def carrying Random is well typed; the [04-EFF-3] \
         rejection belongs to the build boundary, not `check`",
    );
}

/// DISPOSITION LOCK: the pre-existing checker rejection for a top-level VALUE
/// binding is unchanged. Green before and after.
#[test]
fn top_level_binding_random_is_still_a_check_error() {
    let score = check_score(TOP_LEVEL_BINDING, "binding_check");
    assert!(
        score < 1.0,
        "an unhandled Random on a top-level value binding is a check error \
         (spec/04-type-system.md §7.1); got score {score}",
    );
}

/// DEMOTION LOCK: this is the behaviour change, and it is observable in the
/// published header. Before the fix `lib_ok.h` declared
/// `chelis_tensor* my_noise(chelis_tensor* x);` and `lib_ok.c` defined it with
/// external linkage over an inactive RNG — a symbol any C caller could link
/// against and get draws no `with seed` could reproduce. After the fix the
/// definition is still emitted for the program's own seeded `main`, but the
/// public wrapper and its declaration are gone.
#[test]
fn demoted_definition_is_not_published_or_externally_linkable() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("demoted.ch");
    let out_dir = dir.path().join("demoted-out");
    write_file(&path, DISCHARGED_BY_CALLER);
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let header = std::fs::read_to_string(out_dir.join("demoted.h")).expect("generated header");
    assert!(
        !header.contains("my_noise"),
        "[04-EFF-3] a demoted definition must not be declared in the published \
         header; no caller could supply its seed. Header was:\n{header}",
    );

    // The wrapper definition itself must be gone too: a symbol absent from the
    // header but present with external linkage in the object is still linkable.
    let source = std::fs::read_to_string(out_dir.join("demoted.c")).expect("generated source");
    assert!(
        !source.contains("\nchelis_tensor* my_noise("),
        "[04-EFF-3] the externally-linkable wrapper must not be emitted; \
         withholding only the header declaration leaves the symbol linkable",
    );
}

/// EXECUTION PARITY for the demotion: withholding the wrapper must not change
/// what the program computes. The demoted definition is still called by the
/// program's own seeded `main`, so the draw must land inside the declared
/// bounds rather than against an inactive RNG.
#[test]
fn demotion_does_not_change_what_the_program_computes() {
    if !gcc_available() {
        eprintln!("skipping: no C compiler available");
        return;
    }
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("demoted_run.ch");
    let out_dir = dir.path().join("demoted_run-out");
    write_file(&path, DISCHARGED_BY_CALLER);
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    let status = link_generated(&out_dir, "demoted_run.c", "demoted_run");
    assert!(status.success(), "link of generated C failed: {status}");
    let run = std::process::Command::new(out_dir.join("demoted_run"))
        .output()
        .expect("compiled binary should run");
    assert!(
        run.status.success(),
        "demoted program must still run: {}",
        String::from_utf8_lossy(&run.stderr),
    );
    let values = parse_tensor_data(&String::from_utf8_lossy(&run.stdout), "sampled");
    assert_eq!(values.len(), 4, "expected a 4-element draw, got {values:?}");
    for (i, v) in values.iter().enumerate() {
        assert!(
            (2.0..5.0).contains(v),
            "elem[{i}] = {v} is outside the declared [2, 5) bounds; the draw \
             did not run under the program's seeded scope",
        );
    }
}
