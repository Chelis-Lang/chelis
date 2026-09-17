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
//! that expresses a RUNTIME extent, which is what a `sig f[n]: tensor[n, f32]`
//! signature and Coral's real call sites need. The diagnostic names the
//! symbolic form for that reason. These tests require the literal, symbolic,
//! and folded forms to execute with exact shapes and values on both lanes.

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

/// Size read off the consumer, under a symbolic signature. The singleton
/// operand and the size source are different tensors.
const SYMBOLIC_SIZE_DEF: &str = "module Repro.Issue1506Symbolic\n\
     sig f[n]: tensor[n, f32] -> tensor[n, bool]\n\
     def f(xs) = gt(xs, expand(to_tensor([1.5f32]), 0i32, shape(xs, 0i32)))\n\
     out = f(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]))\n";

/// The same computation with no `def`, so `shape(xs, 0)` folds to `Lit(3)`.
/// Its broadcast axis must carry the folded result extent.
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
    assert!(
        gcc_available(),
        "C compiler required for both-lane acceptance"
    );
    let (ran, output) = build_link_run(&dir, "literal", LITERAL_SIZE);
    assert!(ran, "the literal-size replacement must run on C: {output}");
    assert!(
        output.contains("shape=[3], data=[false, true, true]"),
        "the C lane must agree with the evaluator: {output}"
    );
}

/// [05-MOV-1], chelis#1619: a shape-derived size replaces the unit axis.
#[test]
fn the_symbolic_size_replacement_executes_on_both_lanes() {
    assert!(
        gcc_available(),
        "C compiler required for both-lane acceptance"
    );
    let dir = tempdir().expect("tempdir");
    let path = fixture(&dir, "symbolic", SYMBOLIC_SIZE_DEF);
    let evaluated = eval(&path);
    assert!(
        evaluated.contains("shape=[3], data=[false, true, true]"),
        "{evaluated}"
    );
    let (ran, output) = build_link_run(&dir, "symbolic", SYMBOLIC_SIZE_DEF);
    assert!(
        ran && output.contains("shape=[3], data=[false, true, true]"),
        "{output}"
    );
}

/// The same shape read folded at a top-level binding must also execute.
#[test]
fn the_folded_symbolic_size_replacement_executes_on_both_lanes() {
    assert!(
        gcc_available(),
        "C compiler required for both-lane acceptance"
    );
    let dir = tempdir().expect("tempdir");
    let path = fixture(&dir, "folded", SYMBOLIC_SIZE_FOLDED);
    let checked = chelis(&["check", path.to_str().expect("UTF-8 path")]);
    let report = String::from_utf8_lossy(&checked.stdout);
    assert!(
        checked.status.success() && report.contains("\"score\": 1"),
        "{report}"
    );
    let evaluated = eval(&path);
    assert!(
        evaluated.contains("shape=[3], data=[false, true, true]"),
        "{evaluated}"
    );
    let (ran, output) = build_link_run(&dir, "folded", SYMBOLIC_SIZE_FOLDED);
    assert!(
        ran && output.contains("shape=[3], data=[false, true, true]"),
        "{output}"
    );
}
