//! Named regressions from the PR #1215 adversarial review (chelis#1201
//! follow-ups). Two shapes the base revision lowered that the first head
//! rejected:
//!
//! - A generic callee with a CALLABLE parameter is inlined (a callable
//!   argument has no host-type key to specialize on), and the inline path
//!   must thread the call's checked result type through the body — a
//!   generic ADT constructor there is only resolvable from it ->
//!   `callable_param_generic_callee_resolves_its_adt_result`
//! - A NON-recursive generic call whose instantiation never resolves
//!   (`pick[a, b]` applied to `Empty`: `b` unconstrained, the parameter
//!   untouched) falls back to guarded value-level inlining instead of dying
//!   on the specialization residue with a diagnostic that mislabels the
//!   call recursive -> `non_recursive_unconstrained_generic_call_compiles`
//!
//! Both assert full build+link+run parity against the eval lane, so the
//! compiled artifact — not just the absence of a rejection — is the
//! evidence. The fail-closed residue itself (a genuinely unresolvable
//! call) remains pinned by `recursive_generic_monomorphization.rs`, whose
//! diagnostic is now recursion-neutral and cites the open chelis#1226.
//!
//! Run: `cargo nextest run -p chelis-cli --test issue_1201_review_regressions
//! --no-fail-fast`

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use std::path::PathBuf;
use std::process::Command as StdCommand;
use tempfile::{TempDir, tempdir};

#[path = "common/mod.rs"]
mod common;

use common::{link_generated, write_file};

/// PR #1215 review P1 reproducer, verbatim shape: the callee takes a
/// callable AND returns a generic ADT, so the inline path must carry the
/// checked `Box[int64]` result type into the body for `Full { item: .. }`
/// to resolve. Prints `1`.
const CALLABLE_PARAM_GENERIC_RESULT: &str = "\
type Box[a] =
  | Empty
  | Full { item: a }
def apply[a](f: (a) -> a, x: a) -> Box[a] = Full { item: f(x) }
def bump(n: int64) -> int64 = n
def read() -> int64 = match apply(bump, cast(1, int64)) with {
  | Empty => cast(0, int64)
  | Full { item: i } => i
}
out = print(read())
";

/// PR #1215 review P2 reproducer, verbatim shape: `b` never resolves
/// (`Empty` constrains nothing) and the `y` parameter is untouched, so the
/// call has no concrete type application to specialize — but it is not
/// recursive, and value-level inlining lowers it exactly as the base
/// revision did. Prints `1`.
const NON_RECURSIVE_UNCONSTRAINED: &str = "\
type Box[a] =
  | Empty
  | Full { value: a }
def pick[a, b](x: a, y: Box[b]) -> a = x
def main() -> int32 = pick(cast(1, int32), Empty)
out = print(main())
";

fn chelis() -> Command {
    let mut cmd = Command::cargo_bin("chelis").expect("chelis binary");
    cmd.env("CHELIS_STYLE_GATE_DISABLE", "1");
    cmd
}

fn build_ok(source: &str, stem: &str) -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    let out_dir = dir.path().join("out");
    write_file(&path, source);
    chelis()
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
    (dir, out_dir)
}

fn eval_first_line(source: &str, stem: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    write_file(&path, source);
    let output = chelis()
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    String::from_utf8_lossy(&output)
        .lines()
        .next()
        .unwrap_or_default()
        .to_string()
}

fn run_first_line(out_dir: &std::path::Path, stem: &str) -> String {
    let source_file = format!("{stem}.c");
    let status = link_generated(out_dir, &source_file, "run");
    assert!(status.success(), "generated C must link: {status}");
    let output = StdCommand::new(out_dir.join("run"))
        .output()
        .expect("run compiled binary");
    assert!(
        output.status.success(),
        "compiled binary failed: {}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .unwrap_or_default()
        .to_string()
}

#[test]
fn callable_param_generic_callee_resolves_its_adt_result() {
    let eval = eval_first_line(CALLABLE_PARAM_GENERIC_RESULT, "apply_eval");
    let (_dir, out_dir) = build_ok(CALLABLE_PARAM_GENERIC_RESULT, "apply");
    let compiled = run_first_line(&out_dir, "apply");
    assert_eq!(eval, "1");
    assert_eq!(compiled, eval, "compiled output must match the eval lane");
}

#[test]
fn non_recursive_unconstrained_generic_call_compiles() {
    let eval = eval_first_line(NON_RECURSIVE_UNCONSTRAINED, "pick_eval");
    let (_dir, out_dir) = build_ok(NON_RECURSIVE_UNCONSTRAINED, "pick");
    let compiled = run_first_line(&out_dir, "pick");
    assert_eq!(eval, "1");
    assert_eq!(compiled, eval, "compiled output must match the eval lane");
}
