//! chelis#859 -- `block` implemented end-to-end.
//!
//! spec/03-deep-syntax.md §2.3: `(block {} expr₁ ... exprₙ)` is sequenced
//! expressions whose value is the last child. Before this, the checker
//! rejected every expression-position `block` loudly (and before chelis#731
//! Phase 3 it fell through the unknown-tag wildcard), so the one §2.3
//! expression form without a checker case was unusable from `.dp`. The
//! full pipeline now agrees: check accepts and types the block as its last
//! child, eval sequences children (transcript effects included) and
//! returns the last value, and the C build lane emits the same semantics
//! through the host `Let` encoding -- verified here so a
//! check-passes-build-fails gap (the chelis#850 shape) cannot open.
//!
//! Both polarities per the negative-test-parity rule: the well-typed block
//! scores 1.0 and evaluates; the ill-typed and childless forms are
//! rejected with the right kinds.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

fn run_chelis(args: &[&str], path: &std::path::Path) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(args.iter().copied().chain([path.to_str().unwrap()]))
        .output()
        .expect("chelis should run")
}

fn check_score(program: &str) -> f64 {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.dp");
    write_file(&path, program);
    let out = run_chelis(&["check"], &path);
    let parsed: serde_json::Value =
        serde_json::from_slice(&out.stdout).unwrap_or_else(|e| panic!("check must emit JSON: {e}"));
    parsed["score"].as_f64().expect("numeric score")
}

const WELL_TYPED_BLOCK: &str = "(def {} out (block {} \
     (app {} (var {} print) (lit {type: (t-prim {} f32)} 1.5)) \
     (lit {type: (t-prim {} f32)} 2.5)))\n";

#[test]
fn well_typed_block_checks_at_score_one() {
    let score = check_score(WELL_TYPED_BLOCK);
    assert!(
        (score - 1.0).abs() < f64::EPSILON,
        "a well-typed block must score 1.0, got {score}"
    );
}

#[test]
fn block_evaluates_children_in_order_and_returns_last() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.dp");
    write_file(&path, WELL_TYPED_BLOCK);
    let out = run_chelis(&["eval", "--file"], &path);
    assert!(
        out.status.success(),
        "block eval must succeed; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let print_line = stdout
        .lines()
        .position(|line| line.trim() == "1.5")
        .expect("the discarded print child must run first");
    let value_line = stdout
        .lines()
        .position(|line| line.trim().ends_with("2.5"))
        .expect("the block's value is its last child");
    assert!(
        print_line < value_line,
        "children evaluate in order; got:\n{stdout}"
    );
}

#[test]
fn block_builds_through_the_c_lane() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.dp");
    write_file(&path, WELL_TYPED_BLOCK);
    let out_dir = dir.path().join("out");
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
    assert!(
        out.status.success(),
        "a checked block must BUILD (a check-passes-build-fails gap is the \
         chelis#850 shape); stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn ill_typed_discarded_block_child_scores_below_one() {
    let score = check_score(
        "(def {} out (block {} \
         (app {} (var {} add) (lit {type: (t-prim {} f32)} 1.0) \
         (lit {type: (t-prim {} int64)} 2)) \
         (lit {type: (t-prim {} f32)} 2.5)))\n",
    );
    assert!(
        score < 1.0,
        "an ill-typed discarded child must be caught, got {score}"
    );
}

#[test]
fn childless_block_scores_below_one_and_fails_eval() {
    let score = check_score("(def {} out (block {}))\n");
    assert!(
        score < 1.0,
        "a childless block must be rejected, got {score}"
    );
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.dp");
    write_file(&path, "(def {} out (block {}))\n");
    let out = run_chelis(&["eval", "--file"], &path);
    assert!(
        !out.status.success(),
        "a childless block must not evaluate to a value"
    );
}
