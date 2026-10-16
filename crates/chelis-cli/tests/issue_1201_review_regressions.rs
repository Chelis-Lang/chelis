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
/// checked `Box[i64]` result type into the body for `Full { item: .. }`
/// to resolve. Prints `1`.
const CALLABLE_PARAM_GENERIC_RESULT: &str = "\
type Box[a] =
  | Empty
  | Full { item: a }
def apply[a](f: (a) -> a, x: a) -> Box[a] = Full { item: f(x) }
def bump(n: i64) -> i64 = n
def read() -> i64 = match apply(bump, cast(1, i64)) with {
  | Empty => cast(0, i64)
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
def main() -> i32 = pick(cast(1, i32), Empty)
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
            "--emit-c",
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

// [04-ADT-1], [04-INF-6] and §4.7: discard of an unobserved,
// payload-free result cannot discard the computation which produces it.
fn retained_nullary_source(producer: &str, actual: &str) -> String {
    format!(
        "type Box[a] =\n  | Empty\n  | Full {{ value: a }}\n\
         {producer}\n\
         def pick[a, b](x: a, y: Box[b]) -> a ! {{IO}} = {{\n\
         _ = print(\"body\")\n x\n}}\n\
         def main() -> i32 ! {{IO}} = pick({{\n\
         _ = print(\"actual-x\")\n 1i32\n}}, {{\n\
         _ = print(\"actual-y\")\n {actual}\n}})\n\
         out = print(main())\n"
    )
}

fn assert_retained_nullary_execution(source: &str, stem: &str, markers: &[&str], trap: bool) {
    let dir = tempdir().unwrap();
    let path = dir.path().join(format!("{stem}.ch"));
    write_file(&path, source);
    for native in [false, true] {
        let output = if native {
            let (_build_dir, out_dir) = build_ok(source, stem);
            assert!(link_generated(&out_dir, &format!("{stem}.c"), "run").success());
            StdCommand::new(out_dir.join("run")).output().unwrap()
        } else {
            chelis()
                .args(["eval", "--file", path.to_str().unwrap()])
                .output()
                .unwrap()
        };
        let lane = if native { "native" } else { "Eval" };
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            output.status.success(),
            !trap,
            "{lane} {stem}: {source}\n{text}"
        );
        let mut previous = 0;
        for marker in markers {
            assert_eq!(
                text.matches(marker).count(),
                1,
                "{lane} {stem} {marker}: {text}"
            );
            let position = text.find(marker).unwrap();
            assert!(position >= previous, "{lane} {stem} order: {text}");
            previous = position + marker.len();
        }
        if trap {
            assert!(!text.contains("body"), "{lane} {stem}: {text}");
            assert!(!text.contains("out ="), "{lane} {stem}: {text}");
            assert!(
                text.contains("trap") || text.contains("Domain"),
                "{lane} {stem}: {text}"
            );
        } else {
            assert!(
                text.lines().any(|line| line == "1"),
                "{lane} {stem}: {text}"
            );
        }
    }
}

#[test]
fn retained_unused_nullary_actual_keeps_eager_effects_once() {
    let source = retained_nullary_source("", "Empty");
    assert_retained_nullary_execution(
        &source,
        "unused_empty",
        &["actual-x", "actual-y", "body"],
        false,
    );
}

#[test]
fn retained_unused_nullary_named_producer_keeps_its_boundary() {
    let source = retained_nullary_source(
        "def empty[b]() -> Box[b] ! {IO} = {\n _ = print(\"producer\")\n Empty\n}",
        "empty()",
    );
    assert_retained_nullary_execution(
        &source,
        "unused_named_empty",
        &["actual-x", "actual-y", "producer", "body"],
        false,
    );
}

#[test]
fn retained_unused_nullary_actual_keeps_producer_trap_before_body() {
    let source = retained_nullary_source(
        "def empty[b](zero: i32) -> Box[b] ! {IO} = {\n _ = print(\"producer\")\n _ = trunc_div(1i32, zero)\n Empty\n}",
        "empty(0i32)",
    );
    assert_retained_nullary_execution(
        &source,
        "unused_empty_trap",
        &["actual-x", "actual-y", "producer"],
        true,
    );
}

#[test]
fn retained_unused_nullary_actual_keeps_named_input_claim() {
    // Precision polymorphism selects the retained entry owner. The ordinary
    // monomorphized-call validator independently rejects known mismatches
    // before their runtime named-input claim; that residual has its own repro.
    let source = retained_nullary_source(
        "def required[n, p: Float](value: tensor[n, p], ignored: tensor[n, p]) -> tensor[n, p] ! {IO} = {\n _ = print(\"producer\")\n relu(value)\n}\ndef empty[b](value: tensor[*, f64], ignored: tensor[*, f64]) -> Box[b] ! {IO} = {\n _ = required(value, ignored)\n Empty\n}",
        "empty(to_tensor([1.0f64, 2.0f64]), to_tensor([1.0f64, 2.0f64, 3.0f64]))",
    );
    assert_retained_nullary_execution(
        &source,
        "unused_empty_input_claim",
        &["actual-x", "actual-y"],
        true,
    );
}

#[test]
fn retained_unused_nullary_actual_keeps_prefix_result_claim() {
    let source = retained_nullary_source(
        "def broken(value: tensor[*, f64]) -> tensor[2, f64] ! {IO} = {\n _ = print(\"producer\")\n relu(value)\n}",
        "{\n _ = broken(to_tensor([1.0f64, 2.0f64, 3.0f64]))\n Empty\n}",
    );
    assert_retained_nullary_execution(
        &source,
        "unused_empty_result_claim",
        &["actual-x", "actual-y", "producer"],
        true,
    );
}

#[test]
fn retained_unused_payload_constructor_keeps_its_producer_trap() {
    let source = retained_nullary_source(
        "def payload(zero: i32) -> i32 ! {IO} = {\n _ = print(\"producer\")\n trunc_div(1i32, zero)\n}",
        "Full { value: payload(0i32) }",
    );
    assert_retained_nullary_execution(
        &source,
        "unused_payload_trap",
        &["actual-x", "actual-y", "producer"],
        true,
    );
}

#[test]
fn retained_observed_unconstrained_nullary_result_stays_unresolved() {
    let source = "type Box[a] =\n  | Empty\n  | Full { value: a }\ndef observe[a, b](x: a, y: Box[b]) -> Box[b] = y\nout = print(observe(1i32, Empty))\n";
    let dir = tempdir().unwrap();
    let path = dir.path().join("observed_empty.ch");
    let out_dir = dir.path().join("out");
    write_file(&path, source);
    let output = chelis()
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let text = String::from_utf8_lossy(&output.stderr);
    assert!(
        text.contains("unresolved host type variable") && text.contains("[05-UNS-1]"),
        "{text}"
    );
    assert!(!out_dir.join("observed_empty.c").exists());
}
