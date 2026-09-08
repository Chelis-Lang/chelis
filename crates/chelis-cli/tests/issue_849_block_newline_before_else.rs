//! chelis#849: inside a `{ }` block, an `if ... then X` whose `else` began on a
//! later line failed to parse with `expected Else, found Eof`, pointing at the
//! start of the `if` rather than at the newline that truncated it.
//!
//! The parser-level cause and its unit coverage live in
//! `chelis-surf::parser` (`block_expr_end` ended a block statement at any
//! depth-0 newline, with a carve-out for `|>` but not for `else`). These
//! tests pin the user-visible symptom instead: the layout the issue reported
//! goes through the real CLI, on the same `fmt` / `check` commands it was
//! reported against.

use std::fs;

use assert_cmd::Command;
use tempfile::tempdir;

/// The issue's reproduction, adapted to canonical Surf v0.19: a `{ }` block
/// with a binding (a one-expression block is separately rejected by the
/// v0.19 tail-expression rule, which is unrelated to this defect).
const BLOCK_ELSE_ON_LATER_LINE: &str = concat!(
    "def f(a: f32, b: f32) -> bool = {\n",
    "  c = neq(a, a)\n",
    "  if c then true\n",
    "  else lte(a, b)\n",
    "}\n",
);

#[test]
fn fmt_accepts_a_block_if_whose_else_is_on_a_later_line() {
    let directory = tempdir().expect("tempdir");
    let source = directory.path().join("repro.ch");
    fs::write(&source, BLOCK_ELSE_ON_LATER_LINE).expect("write fixture");

    // `fmt --inplace` succeeds and canonicalises to the one-line form. Before
    // the fix this failed with `expected Else, found Eof`.
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", "--inplace"])
        .arg(&source)
        .assert()
        .success();

    let formatted = fs::read_to_string(&source).expect("read back");
    assert!(
        formatted.contains("if c then true else lte(a, b)"),
        "fmt must canonicalise the split if/else onto one line, got:\n{formatted}"
    );
}

#[test]
fn check_accepts_a_block_if_whose_else_is_on_a_later_line() {
    let directory = tempdir().expect("tempdir");
    let source = directory.path().join("repro.ch");
    fs::write(&source, BLOCK_ELSE_ON_LATER_LINE).expect("write fixture");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", "--allow-style-violations"])
        .arg(&source)
        .output()
        .expect("run chelis check");
    assert!(output.status.success(), "a parsing source must check clean");
    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("check emits JSON");
    assert_eq!(
        report["errors"].as_array().map(Vec::len),
        Some(0),
        "no diagnostics; got {report}"
    );
}

/// The same defect one token earlier, and the shape that motivated
/// admitting `then` as a continuation (PR #1369 review): a multiline
/// `else if` chain puts a `then` and an `else` on later lines.
const MULTILINE_ELSE_IF_CHAIN: &str = concat!(
    "def f(a: f32, b: f32) -> f32 = {\n",
    "  c = neq(a, a)\n",
    "  if c\n",
    "  then a\n",
    "  else if lt(a, b)\n",
    "  then b\n",
    "  else mul(a, b)\n",
    "}\n",
);

#[test]
fn a_multiline_else_if_chain_checks_clean() {
    let directory = tempdir().expect("tempdir");
    let source = directory.path().join("chain.ch");
    fs::write(&source, MULTILINE_ELSE_IF_CHAIN).expect("write fixture");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", "--allow-style-violations"])
        .arg(&source)
        .output()
        .expect("run chelis check");
    assert!(output.status.success(), "the chain must check clean");
    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("check emits JSON");
    assert_eq!(
        report["errors"].as_array().map(Vec::len),
        Some(0),
        "no diagnostics; got {report}"
    );
}

#[test]
fn canonical_output_still_comes_from_the_shared_printer() {
    // Accepting authored layout freedom must not create a second producer
    // form: `fmt` collapses every accepted spelling to the one canonical
    // line, and is idempotent on its own output.
    let directory = tempdir().expect("tempdir");
    let source = directory.path().join("chain.ch");
    fs::write(&source, MULTILINE_ELSE_IF_CHAIN).expect("write fixture");

    for _ in 0..2 {
        Command::cargo_bin("chelis")
            .expect("binary")
            .args(["fmt", "--inplace"])
            .arg(&source)
            .assert()
            .success();
    }
    let formatted = fs::read_to_string(&source).expect("read back");
    assert!(
        formatted.contains("if c then a else if lt(a, b) then b else mul(a, b)"),
        "fmt must collapse the chain to the canonical line, got:\n{formatted}"
    );
}

#[test]
fn a_block_if_with_no_else_is_still_rejected() {
    // Negative parity: the newline carve-out must not make `else` optional.
    let directory = tempdir().expect("tempdir");
    let source = directory.path().join("no_else.ch");
    fs::write(
        &source,
        concat!(
            "def f(a: f32, b: f32) -> bool = {\n",
            "  c = neq(a, a)\n",
            "  if c then true\n",
            "}\n",
        ),
    )
    .expect("write fixture");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", "--allow-style-violations"])
        .arg(&source)
        .output()
        .expect("run chelis check");
    assert!(
        !output.status.success(),
        "a missing `else` must be rejected"
    );
    assert_rejected_for(&output.stdout, "Else");
}

/// The parity negative. `then` is newly admitted as a continuation on the
/// same public path, and admitting a keyword after a newline must not make
/// it OPTIONAL. Without this, the `else` case alone would leave the other
/// half of the `if` contract unpinned at the CLI surface.
#[test]
fn a_block_if_with_no_then_is_still_rejected() {
    let dir = tempdir().expect("tempdir");
    let source = dir.path().join("no_then.ch");
    fs::write(
        &source,
        "def f(a: f32, b: f32) -> f32 = {\n  c = neq(a, a)\n  if c\n  else b\n}\n",
    )
    .expect("write fixture");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", "--allow-style-violations"])
        .arg(&source)
        .output()
        .expect("run chelis check");
    assert!(
        !output.status.success(),
        "a missing `then` must be rejected"
    );
    assert_rejected_for(&output.stdout, "Then");
}

/// Assert the report rejects for the RIGHT reason.
///
/// Deserializes stdout rather than substring-matching a concatenation of
/// stdout and stderr: that form stays green on an unrelated rejection, on
/// incidental stderr text, and on an output-shape regression that emits no
/// report at all. chelis#886 is also reshaping this document, so the check
/// reads the `errors` array structurally instead of its formatting.
fn assert_rejected_for(stdout: &[u8], expected_token: &str) {
    let rendered = String::from_utf8_lossy(stdout);
    let report: serde_json::Value = serde_json::from_slice(stdout)
        .unwrap_or_else(|e| panic!("check must still emit a JSON report; {e}\n{rendered}"));
    let errors = report["errors"]
        .as_array()
        .unwrap_or_else(|| panic!("`errors` must be an array; got:\n{rendered}"));
    assert!(
        !errors.is_empty(),
        "a rejected document must carry at least one diagnostic; got:\n{rendered}"
    );
    let needle = format!("expected {expected_token}");
    assert!(
        errors.iter().any(|error| error["message"]
            .as_str()
            .is_some_and(|message| message.contains(&needle))),
        "no diagnostic reports `{needle}`; got:\n{errors:#?}"
    );
}
