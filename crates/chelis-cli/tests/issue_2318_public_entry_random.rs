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
//! So the decision is made PER ENTRY, at each publication site:
//!
//! * A path that publishes one entry per definition withholds the offending
//!   one — from the header and from external linkage — and the program's other
//!   entries are unaffected. That is what keeps a library-shaped module legal.
//! * A path that publishes a single entry for the whole program has no sibling
//!   to keep, so withholding and emitting nothing are the same thing, and the
//!   build is rejected instead of silently producing an empty artifact.
//!
//! Two earlier revisions decided this program-wide instead, and both were
//! rejected by review. `globals.is_empty()` rejected a seeded helper beside its
//! seeded caller. A whole-program "does any `with seed(...)` appear" test then
//! both over-rejected — it failed the repository's own `chelis-std` random,
//! kaiming and xavier modules, a regression against base — and under-protected,
//! because a seeded draw in the SAME definition as an unseeded one satisfied it
//! and left the tensor-DAG path publishing the defect verbatim.
//! `a_seeded_sibling_draw_does_not_admit_an_unseeded_one` and
//! `a_library_shaped_module_with_pure_siblings_still_builds` pin both halves.
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
//! * `dag_lane_*` and `host_lane_*` are the two NEGATIVE cases, one per
//!   emission path — the silent draw and the abort respectively. They differ
//!   in OUTCOME because the paths differ: the tensor-DAG path publishes one
//!   entry for the whole program, so withholding leaves nothing and the build
//!   is rejected; the host path publishes one entry per definition, so the
//!   offender is withheld and its siblings survive.
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

/// A single TENSOR-parameter entry, which routes to the tensor-DAG emitter.
///
/// The parameter type is the whole point. `chelis build` selects its emitter by
/// the entry's SIGNATURE SHAPE, not by template foldability: a tensor parameter
/// goes to `emit_dag_with_options`, a scalar one to host-ABI emission. An
/// earlier revision of these tests used `c: f32` here and called it the "DAG
/// lane", so both negatives exercised the SAME path and this one was never
/// covered — which is how the reported defect survived a fix and a review
/// round. Changing this line back to a scalar silently retires the coverage.
const DAG_LANE_EXPORT: &str =
    "def draw(x: tensor[4, f32]) -> tensor[4, f32] = uniform_like(x, 2.0f32, 5.0f32)\n";

/// The same lane, reached through a program that DOES open a seed scope — in
/// the very same definition as the unseeded draw.
///
/// A whole-program "does any `with seed(...)` appear" precondition admits this
/// and then leaves the tensor-DAG emitter unguarded, which is how red-team
/// round 2 reproduced the defect after round 1's repair. The admission
/// decision has to be per entry.
const DAG_LANE_PARTIALLY_SEEDED: &str = "def draw(x: tensor[4, f32]) -> tensor[4, f32] = \
     add(with seed(7i64) { uniform_like(x, 1.0f32, 2.0f32) }, \
     uniform_like(x, 2.0f32, 5.0f32))\n";

/// The issue's `e2`: two scalar-parameter defs. This one aborted at the first
/// call before the fix.
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
/// Unlike `e1`, this program has a publishable sibling (`bc`, which carries no
/// `Random`), and this emission path publishes one entry per definition. So
/// `draw` is WITHHELD and `bc` is published, rather than the whole build being
/// rejected. What [04-EFF-3] forbids is emitting `draw` over an inactive RNG,
/// and that is what must be gone.
///
/// This program is structurally identical to a library module — one
/// `Random`-carrying definition beside pure ones — so rejecting it would
/// reject `packages/chelis-std/src/init/`. The two cannot be distinguished
/// from source, which is why withholding is reported rather than fatal
/// (`a_withheld_entry_is_reported_on_stderr`).
#[test]
fn host_lane_export_with_undischarged_random_is_withheld_not_published() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("host_lane.ch");
    let out_dir = dir.path().join("host_lane-out");
    write_file(&path, HOST_LANE_EXPORT);
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
    let header = std::fs::read_to_string(out_dir.join("host_lane.h")).expect("generated header");
    assert!(
        !header.contains("chelis_fn_64726177"),
        "[04-EFF-3] `draw` carries an undischarged Random and must not be \
         published. Header:\n{header}",
    );
    assert!(
        header.contains("chelis_fn_6263"),
        "`bc` carries no Random and must stay published; withholding must not \
         widen to its siblings. Header:\n{header}",
    );
}

/// The rejection must explain itself: why the entry cannot exist, and what to
/// do instead. A bare "unsupported" leaves the author guessing which of the two
/// remedies applies.
#[test]
fn the_rejection_names_the_cause_and_both_remedies() {
    let (ok, stderr) = build_result(DAG_LANE_EXPORT, "message");
    assert!(!ok, "expected a rejection");
    for fragment in [
        // Cause, then why no caller can help, then why withholding is not an
        // option here, then both remedies. An earlier revision asserted
        // "opens no `with seed(...)` region anywhere" — the whole-program
        // precondition that red-team round 2 showed was both too broad and
        // too weak, and which this revision deleted.
        "nothing inside it handles that effect",
        "no RNG frame",
        "no other definition left to publish in its place",
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

/// DEMOTION LOCK: the withheld entry must be gone from the published header
/// AND from the object's external symbols.
///
/// Emitted entry symbols are hex-mangled (`chelis_fn_<utf8-hex>`), so an
/// earlier version of this test — which grepped the header and the `.c` for
/// the SOURCE spelling `my_noise` — was vacuous: it passed with the fix
/// reverted, because the broken wrapper is published as
/// `chelis_fn_6d795f6e6f697365` and the string `my_noise` appears in neither
/// artifact either way. Red-team round 1 caught it. Assert the mangled symbol,
/// and assert against `nm` rather than the header alone: a symbol absent from
/// the header can still be externally linkable.
#[test]
fn demoted_definition_is_not_published_or_externally_linkable() {
    if !gcc_available() {
        eprintln!("skipping: no C compiler available");
        return;
    }
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

    // `my_noise` encoded with the `chelis_fn_<utf8-hex>` scheme in
    // `host_emit::emitted_function_name`.
    let mangled = format!(
        "chelis_fn_{}",
        "my_noise"
            .bytes()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );
    let header = std::fs::read_to_string(out_dir.join("demoted.h")).expect("generated header");
    assert!(
        !header.contains(&mangled),
        "[04-EFF-3] the demoted definition must not be declared in the published \
         header; no caller could supply its seed. Looked for {mangled}. Header:\n{header}",
    );

    let source = std::fs::read_to_string(out_dir.join("demoted.c")).expect("generated source");
    // A declaration or definition of the public entry lives at FILE SCOPE, so
    // only unindented lines can carry one. Indented mentions are inside a
    // function body — calls, and `fprintf` diagnostics that embed the symbol
    // name in a string literal — and say nothing about linkage.
    //
    // A file-scope mention must therefore be `static`. An earlier revision
    // emitted a non-`static` forward declaration for the withheld wrapper,
    // which is exactly what this catches.
    for line in source.lines() {
        if line.contains(&mangled) && !line.starts_with(char::is_whitespace) {
            assert!(
                line.starts_with("static"),
                "[04-EFF-3] a file-scope mention of the withheld entry must have \
                 internal linkage, found: {line}",
            );
        }
    }

    // The object is the real test: a symbol absent from the header can still
    // be linkable, and that is exactly what a C caller would reach for.
    let object = out_dir.join("demoted.o");
    let compiled = std::process::Command::new("clang")
        .args(["-O2", "-c"])
        .arg(out_dir.join("demoted.c"))
        .arg("-I")
        .arg(&out_dir)
        .arg("-o")
        .arg(&object)
        .status()
        .expect("clang should run");
    assert!(compiled.success(), "generated C must compile");
    let symbols = std::process::Command::new("nm")
        .args(["-gU"])
        .arg(&object)
        .output()
        .expect("nm should run");
    let symbols = String::from_utf8_lossy(&symbols.stdout);
    assert!(
        !symbols.contains(&mangled),
        "[04-EFF-3] the demoted definition must not be externally linkable; \
         `nm -gU` still lists {mangled}:\n{symbols}",
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

/// REGRESSION, chelis#2318 round 2: a seeded region in the SAME definition as
/// an unseeded draw must not admit the entry.
///
/// Round 1's repair used a whole-program precondition — "does any
/// `with seed(...)` appear anywhere" — and this program satisfies it from
/// inside the offending definition itself. The tensor-DAG emitter was then
/// completely unguarded: `check` scored 1.0, the build exited 0, and the
/// published entry baked `CHELIS_EFFECTIVE_UNIFORM_SEED(0ULL)`, reproducing
/// the reported defect verbatim.
#[test]
fn a_seeded_sibling_draw_does_not_admit_an_unseeded_one() {
    assert_eff3_rejection(DAG_LANE_PARTIALLY_SEEDED, "partial_seed");
}

/// POSITIVE PARITY, chelis#2318 round 2: the repository's own stdlib shape.
///
/// `packages/chelis-std/src/init/random.ch` declares `normal_like ! { Random }`
/// beside pure helpers and contains NO `with seed` anywhere — the §7.1
/// caller-supplies-the-handler pattern this change argues must stay legal.
/// Round 1's repair rejected that file outright, a regression against base
/// that the in-repo suites missed because nothing built a library-shaped file.
#[test]
fn a_library_shaped_module_with_pure_siblings_still_builds() {
    let program = "def noise(x: tensor[4, f32]) -> tensor[4, f32] = \
                   uniform_like(x, 0.0f32, 1.0f32)\n\
                   def square(x: tensor[4, f32]) -> tensor[4, f32] = mul(x, x)\n\
                   def double(x: tensor[4, f32]) -> tensor[4, f32] = add(x, x)\n";
    let (ok, stderr) = build_result(program, "libshape");
    // Distinguish "rejected by [04-EFF-3]" from "this fixture stopped type
    // checking": an earlier version used `mul(x, 2.0f32)`, which [05-OP] does
    // not admit, so the test failed for a reason that had nothing to do with
    // the rule under test.
    // Match the ERROR, not the string: a withheld helper now emits a
    // `warning: [04-EFF-3]` line on this very program, and matching the bare
    // atom id would trip on it.
    assert!(
        !stderr.contains("error: [04-EFF-3]") && !stderr.contains("unsupported:"),
        "a library-shaped module must not be REJECTED by [04-EFF-3]; being \
         warned about its withheld helper is correct. stderr:\n{stderr}",
    );
    assert!(
        ok,
        "a library-shaped module (a Random-carrying helper beside pure \
         siblings, no seed anywhere) must keep building; its helper is \
         withheld, not the whole file rejected. stderr:\n{stderr}",
    );
}

/// Withholding must not be silent.
///
/// A withheld entry is invisible in the artifact — the caller just gets a
/// header that lacks it, and finds out at link time. Before this warning, the
/// only trace was a comment buried in the generated `.c`, and `e2` built with
/// zero bytes on stderr. The build still succeeds: the artifact is valid and
/// its other entries are usable.
#[test]
fn a_withheld_entry_is_reported_on_stderr() {
    let (ok, stderr) = build_result(HOST_LANE_EXPORT, "warned");
    assert!(ok, "the program must still build: {stderr}");
    assert!(
        stderr.contains("warning: [04-EFF-3]"),
        "withholding must be reported, not silent; stderr:\n{stderr}",
    );
    assert!(
        stderr.contains("`draw`"),
        "the warning must name the withheld definition; stderr:\n{stderr}",
    );
    assert!(
        stderr.contains("no RNG frame"),
        "the warning must say why it could not be published; stderr:\n{stderr}",
    );
}

/// The converse: a program with nothing withheld must warn about nothing.
/// A warning that fires on every build teaches nobody anything.
#[test]
fn a_program_with_nothing_withheld_does_not_warn() {
    let (ok, stderr) = build_result(SELF_HANDLED_ENTRY, "unwarned");
    assert!(ok, "the self-handled entry must build: {stderr}");
    assert!(
        !stderr.contains("[04-EFF-3]"),
        "an entry that discharges its own Random is published, so nothing was \
         withheld and nothing should be reported; stderr:\n{stderr}",
    );
}
