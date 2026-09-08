//! chelis#1506: the spelling the rejection tells users to write, run on the
//! lanes that must execute it.
//!
//! `[05-OP-36]` makes a scalar beside a tensor a type error, and the
//! diagnostic names the replacement rather than leaving the user to invent
//! one, following the `div`/`floor_div` precedent. A named replacement is only
//! worth naming if it runs, so these rows run it.
//!
//! # Why the diagnostic names the SYMBOLIC form
//!
//! Two spellings exist. `expand(to_tensor([1.5f32]), 0i32, 3i64)` gives the
//! size as a literal and works on every lane today. `expand(to_tensor([1.5f32]),
//! 0i32, shape(xs, 0i32))` reads the size off the operand and is the only one
//! that expresses a RUNTIME extent, which is what a `sig f: tensor[n, f32]`
//! signature and Coral's real call sites need. The diagnostic names the
//! symbolic form for that reason, and this file records that the symbolic form
//! does not yet execute on C.

mod common;

use assert_cmd::Command;
use common::{gcc_available, link_generated};
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::{TempDir, tempdir};

/// Size given as a literal. Both lanes execute it.
const LITERAL_SIZE: &str = "module Repro.Issue1506Literal\n\
     def f(xs: tensor[3, f32]) -> tensor[3, bool] = \
     gt(xs, expand(to_tensor([1.5f32]), 0i32, 3i64))\n\
     out = f(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]))\n";

/// Size read off the operand, under a symbolic signature. The def form: the
/// C lane emits an entry-prologue guard over `xs` for this shape.
const SYMBOLIC_SIZE_DEF: &str = "module Repro.Issue1506Symbolic\n\
     sig f: tensor[n, f32] -> tensor[n, bool]\n\
     def f(xs) = gt(xs, expand(to_tensor([1.5f32]), 0i32, shape(xs, 0i32)))\n\
     out = f(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]))\n";

/// The same computation with no `def`, so `shape(xs, 0)` folds to `Lit(3)`.
/// A different symptom of the same defect.
const SYMBOLIC_SIZE_FOLDED: &str = "module Repro.Issue1506Folded\n\
     xs = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)])\n\
     out = gt(xs, expand(to_tensor([1.5f32]), 0i32, shape(xs, 0i32)))\n";

fn fixture(dir: &TempDir, stem: &str, source: &str) -> std::path::PathBuf {
    let path = dir.path().join(format!("{stem}.ch"));
    fs::write(&path, source).expect("fixture");
    path
}

fn chelis(args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(args)
        .output()
        .expect("run chelis")
}

fn eval(path: &Path) -> String {
    let out = chelis(&["eval", "--file", path.to_str().expect("UTF-8 path")]);
    assert!(
        out.status.success(),
        "eval must succeed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).to_string()
}

/// Build to C, link, run. Returns whether it exited zero plus combined output.
fn build_link_run(dir: &TempDir, stem: &str, source: &str) -> (bool, String) {
    let path = fixture(dir, stem, source);
    let out_dir = dir.path().join(format!("{stem}-out"));
    let build = chelis(&[
        "build",
        "--allow-style-violations",
        path.to_str().expect("UTF-8 path"),
        "--target",
        "c",
        "-o",
        out_dir.to_str().expect("UTF-8 path"),
    ]);
    assert!(
        build.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let linked = link_generated(&out_dir, &format!("{stem}.c"), stem);
    assert!(linked.success(), "link failed: {linked}");
    let run = StdCommand::new(out_dir.join(stem))
        .output()
        .expect("run the linked binary");
    let mut combined = String::from_utf8_lossy(&run.stdout).to_string();
    combined.push_str(&String::from_utf8_lossy(&run.stderr));
    (run.status.success(), combined)
}

/// The executing form: literal size, both lanes, same answer.
///
/// DISPOSITION LOCK. chelis#1277 S2b made `expand` over a unit-extent operand
/// the same-rank broadcast on every lane, with the oracle receipts
/// `expand.positional.replacement.c` and `expand.positional.replacement.eval`
/// in `runtime_extent_slice_b.rs`. This row is the comparison consumer of that
/// property, which those receipts do not cover.
#[test]
fn the_literal_size_replacement_executes_on_both_lanes() {
    let dir = tempdir().expect("tempdir");
    let path = fixture(&dir, "literal", LITERAL_SIZE);
    assert!(
        eval(&path).contains("shape=[3], data=[false, true, true]"),
        "the evaluator broadcasts the bound and compares elementwise"
    );
    if !gcc_available() {
        return;
    }
    let (ran, output) = build_link_run(&dir, "literal", LITERAL_SIZE);
    assert!(ran, "the literal-size replacement must run on C: {output}");
    assert!(
        output.contains("shape=[3], data=[false, true, true]"),
        "the C lane must agree with the evaluator: {output}"
    );
}

/// KNOWN GAP, chelis#1619. The symbolic-size replacement runs on the evaluator
/// and TRAPS on C.
///
/// The unit-extent claim belongs to the `expand` operand, `to_tensor([1.5f32])`,
/// whose extent at axis 0 is 1. When `size` is `shape(xs, 0i32)` the derivation
/// attributes the claim to the axis the size expression READS, so the emitted C
/// guards `xs` instead: `if (chelis_tensor_shape(inputs[0], 0) != 1)`. With
/// `xs` at extent 3 that fires, and with `xs` at extent 1 the program runs,
/// which is exactly when broadcasting is pointless.
///
/// This is chelis#1277's S2b derivation, not chelis#1506's checker change: the
/// trap reproduces with `app.rs`, `app_post.rs` and `host_ops.rs` reverted to
/// the base.
///
/// WHEN chelis#1619 IS FIXED this row goes RED. Replace the trap assertions
/// with the same execution assertions the literal-size row makes: both lanes
/// print `[false, true, true]`. Do not soften it in any other direction; the
/// acceptance of chelis#1506's diagnostic depends on this spelling running.
#[test]
fn the_symbolic_size_replacement_runs_on_eval_and_traps_on_c_pending_1619() {
    let dir = tempdir().expect("tempdir");
    let path = fixture(&dir, "symbolic", SYMBOLIC_SIZE_DEF);
    assert!(
        eval(&path).contains("shape=[3], data=[false, true, true]"),
        "the evaluator already executes the symbolic-size replacement"
    );
    if !gcc_available() {
        return;
    }
    let (ran, output) = build_link_run(&dir, "symbolic", SYMBOLIC_SIZE_DEF);
    assert!(
        !ran,
        "chelis#1619: the C entry guard reads the consumer's axis, so this \
         traps today. If it now RUNS, chelis#1619 is fixed: flip this row to \
         assert `[false, true, true]` on both lanes: {output}"
    );
    assert!(
        output.contains("numeric trap: domain in load at int64"),
        "the trap is the [04-NUM-9] rendering of the misattributed claim: {output}"
    );
    assert!(
        output.contains("claimed = 1") && output.contains("xs axis 0 = 3"),
        "and it names `xs`, the CONSUMER, rather than the expand operand, \
         which is the defect chelis#1619 records: {output}"
    );
}

/// KNOWN GAP, chelis#1619, second symptom. With no `def`, `shape(xs, 0)` folds
/// to `Lit(3)` and the misattributed claim becomes a static disagreement, so
/// the program is refused at BUILD instead of trapping at run time. It still
/// checks clean, which is why the refusal is a lowering invariant rather than
/// a type error.
///
/// WHEN chelis#1619 IS FIXED this row goes RED: the build succeeds and the
/// binary prints `[false, true, true]`. Flip it the same way.
#[test]
fn the_folded_symbolic_size_replacement_is_refused_at_build_pending_1619() {
    let dir = tempdir().expect("tempdir");
    let path = fixture(&dir, "folded", SYMBOLIC_SIZE_FOLDED);
    let checked = chelis(&["check", path.to_str().expect("UTF-8 path")]);
    let report = String::from_utf8_lossy(&checked.stdout);
    assert!(
        checked.status.success() && report.contains("\"score\": 1"),
        "the program is well typed; the defect is below the checker: {report}"
    );

    let out_dir = dir.path().join("folded-out");
    let build = chelis(&[
        "build",
        "--allow-style-violations",
        path.to_str().expect("UTF-8 path"),
        "--target",
        "c",
        "-o",
        out_dir.to_str().expect("UTF-8 path"),
    ]);
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr)
    );
    assert!(
        text.contains("ownership lowering invariant failed")
            && text.contains("mismatched dimension at axis 0: Lit(1) vs Lit(3)"),
        "chelis#1619 folded: the expand output keeps the operand's Lit(1). If \
         this now BUILDS, chelis#1619 is fixed: flip this row to assert \
         execution on both lanes: {text}"
    );
}
