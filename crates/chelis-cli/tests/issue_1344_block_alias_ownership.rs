//! chelis#1344 (parent chelis#1286): a `let` binding inside a compiled
//! function whose value is a bare pointer copy of an allocation the block
//! does not own was released at the block close, dropping a reference the
//! block never acquired.
//!
//! Root cause: the emitter's transfer-leaf retain fired only when the
//! copied source was a *tracked block binding*, so a copy of a function
//! parameter or a captured top-level binding into an owned slot took no
//! retain while the slot's block-close release stayed unconditional. The
//! same provenance guard lived on the call-escape retain, so a call whose
//! result may alias a parameter (`d = pass_through(text)`) under-retained
//! the same way. The stdlib carrier was
//! `Std.Io.Json.canonical_bigint_text`'s `digits = if negative then
//! string_slice(text, ..) else text`, which freed the caller's string on
//! every non-negative bigint token and made compiled `parse_json` of any
//! out-of-int64 integer corrupt the heap (PR #1302 red-team finding P0-1).
//!
//! The fix drops the source-provenance half of both guards for binding
//! VALUE TEMPS: their release (through the binding name at the block
//! close) is unconditional, so a bare copy into one always retains, and
//! the source's owner (block, caller, or enclosing scope) keeps its own
//! release path. A block's RESULT TARGET keeps the tracked-binding rule:
//! its release path is the caller's alias-aware machinery (`main`'s root
//! ledger, or the caller's own call-escape retain), so retaining a
//! returned parameter there would double-count and leak per call - the
//! `issue_1222` binder-key parameter test pins that side. Retain and
//! block-close release cancel, so the fix adds no per-call leak - the
//! failure mode that forced PR #1340 to revert its release-suppression
//! variant of this change (see chelis#1344's measurements).
//!
//! Residual chelis#1344 scope, deliberately not covered here because the
//! fix does not reach it: a callee that returns a *captured* top-level
//! binding it was never passed (no transfer leaf exists in the caller),
//! and a transfer that reaches a binding through a call-argument temp
//! (`__call_argN` is not an owned destination). Both remain recorded on
//! the issue.
//!
//! Oracle: each program is built to C, linked, and RUN, its stdout is
//! compared against `chelis eval` on the same source, and the emitted-C
//! assertions pin *why* it passes - the alias arm retains exactly once -
//! so a future change cannot restore the crash while keeping the run
//! green by suppressing the release instead.

use std::fs;
use std::process::Command as StdCommand;

use assert_cmd::Command;
use tempfile::tempdir;

mod common;

/// Build `source` to C, link it, run it, and return the run's stdout
/// together with the emitted C translation unit. Same pipeline as
/// `common::build_and_run`, which discards the emitted source these tests
/// also assert on.
fn build_run_and_emit(source: &str, stem: &str) -> (String, String) {
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("{stem}.ch"));
    let out_dir = dir.path().join(format!("{stem}-out"));
    fs::write(&src_path, source).expect("write .ch source");

    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            src_path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let emitted = fs::read_to_string(out_dir.join(format!("{stem}.c"))).expect("emitted C");
    let status = common::link_generated(&out_dir, &format!("{stem}.c"), stem);
    assert!(status.success(), "linking emitted C failed: {status}");

    let run = StdCommand::new(out_dir.join(stem))
        .output()
        .expect("compiled binary should run");
    assert!(
        run.status.success(),
        "chelis#1344: the compiled binary must exit 0, not abort while \
         releasing a block binding that aliases a value the block does not \
         own. status={} stdout=\n{}\nstderr=\n{}",
        run.status,
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr),
    );
    (
        String::from_utf8(run.stdout).expect("utf-8 stdout"),
        emitted,
    )
}

/// `chelis eval` on the same source: aliasing is a legitimate source form,
/// so the fix may not change what either lane observes, only how many
/// times the allocation is released.
fn eval_stdout(source: &str, stem: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("{stem}.ch"));
    fs::write(&src_path, source).expect("write .ch source");
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", src_path.to_str().unwrap()])
        .output()
        .expect("eval should run");
    assert!(
        output.status.success(),
        "eval failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("utf-8 stdout")
}

/// The emitted body of one compiled function, sliced from its defining
/// `signature {` to the closing brace, so a retain or release in another
/// function (or `main`'s root cleanup) cannot satisfy an assertion here.
fn emitted_function<'a>(emitted: &'a str, signature: &str) -> &'a str {
    let definition = format!("{signature} {{");
    let start = emitted
        .find(&definition)
        .unwrap_or_else(|| panic!("emitted C defines `{definition}`:\n{emitted}"));
    let rest = &emitted[start..];
    let end = rest.find("\n}").expect("function is closed");
    &rest[..end]
}

fn count_in(body: &str, call: &str) -> usize {
    body.matches(call).count()
}

fn skip_without_cc() -> bool {
    if common::gcc_available() {
        return false;
    }
    eprintln!("skipping: no host C compiler on PATH");
    true
}

fn shared_line<'a>(stdout: &'a str, name: &str) -> &'a str {
    let prefix = format!("{name} = ");
    stdout
        .lines()
        .find(|line| line.starts_with(&prefix))
        .unwrap_or_else(|| panic!("stdout does not contain `{prefix}` line:\n{stdout}"))
}

// ---------------------------------------------------------------------------
// 1. chelis#1344's reported shape: a function-body copy of a captured
//    top-level binding. Pre-fix: `my_take` released `y` (which IS `g`) at
//    its block close and `main` released `g` again - exit 134 under a
//    checked allocator, silent use-after-free otherwise.
// ---------------------------------------------------------------------------

const CAPTURED_COPY: &str = "g = [1i64, 2i64]\n\
def my_take() -> int64 = {\n\
  y = g\n\
  len(y)\n\
}\n\
n = my_take()\n";

#[test]
fn captured_top_level_copy_in_function_body_frees_once() {
    if skip_without_cc() {
        return;
    }
    let (stdout, emitted) = build_run_and_emit(CAPTURED_COPY, "captured_copy");
    assert_eq!(
        shared_line(&stdout, "n"),
        shared_line(&eval_stdout(CAPTURED_COPY, "captured_copy"), "n"),
    );
    let body = emitted_function(&emitted, "int64_t my_take()");
    assert_eq!(
        count_in(body, "chelis_list_retain("),
        1,
        "the copy of the captured binding must retain so the block-close \
         release does not drop main's reference:\n{body}"
    );
    assert_eq!(
        count_in(body, "chelis_list_release("),
        1,
        "the block still releases its binding exactly once:\n{body}"
    );
}

// ---------------------------------------------------------------------------
// 2. The PR #1302 P0 shape, distilled from Std.Io.Json's
//    `canonical_bigint_text`: one arm of a conditional binding aliases the
//    function's parameter while the other allocates fresh. The retain on
//    the alias arm is what makes the unconditional block-close release
//    correct on both paths.
// ---------------------------------------------------------------------------

const PARAM_ALIAS_ARM: &str = "def gate(text: string, negative: bool) -> int64 = {\n\
  digits = if negative then string_slice(text, 1i64, sub(string_len(text), 1i64)) else text\n\
  string_len(digits)\n\
}\n\
raw = string_concat(\"12\", \"34\")\n\
n = gate(raw, false)\n";

#[test]
fn parameter_alias_arm_retains_before_block_release() {
    if skip_without_cc() {
        return;
    }
    let (stdout, emitted) = build_run_and_emit(PARAM_ALIAS_ARM, "param_alias_arm");
    assert_eq!(
        shared_line(&stdout, "n"),
        shared_line(&eval_stdout(PARAM_ALIAS_ARM, "param_alias_arm"), "n"),
    );
    let body = emitted_function(&emitted, "int64_t gate(chelis_string text, bool negative)");
    assert_eq!(
        count_in(body, "chelis_string_retain("),
        1,
        "the parameter-aliasing arm must retain into the owned slot; the \
         fresh string_slice arm already owns its result:\n{body}"
    );
    assert_eq!(
        count_in(body, "chelis_string_release("),
        1,
        "the block still releases `digits` exactly once:\n{body}"
    );
}

// ---------------------------------------------------------------------------
// 3. The call-escape sibling: the callee returns its parameter, so the
//    caller's binding shares the allocation of a value the caller was
//    handed. Pre-fix this underflowed the string refcount when the outer
//    caller released its own binding.
// ---------------------------------------------------------------------------

const CALL_ESCAPE_PARAM: &str = "def pass_through(s: string) -> string = s\n\
def wrap(text: string) -> int64 = {\n\
  d = pass_through(text)\n\
  string_len(d)\n\
}\n\
def outer(s0: string) -> int64 = {\n\
  raw = string_concat(s0, s0)\n\
  n = wrap(raw)\n\
  add(n, string_len(raw))\n\
}\n\
val = outer(\"hi\")\n";

#[test]
fn call_escape_of_parameter_retains_the_result() {
    if skip_without_cc() {
        return;
    }
    let (stdout, emitted) = build_run_and_emit(CALL_ESCAPE_PARAM, "call_escape_param");
    assert_eq!(
        shared_line(&stdout, "val"),
        shared_line(&eval_stdout(CALL_ESCAPE_PARAM, "call_escape_param"), "val"),
    );
    let body = emitted_function(&emitted, "int64_t wrap(chelis_string text)");
    assert_eq!(
        count_in(body, "chelis_string_retain("),
        1,
        "the may-return-its-argument call result must retain before the \
         block-close release of `d`:\n{body}"
    );
    assert_eq!(
        count_in(body, "chelis_string_release("),
        1,
        "the block still releases `d` exactly once:\n{body}"
    );
}

// ---------------------------------------------------------------------------
// 4. Negative parity: a binding whose value is freshly allocated takes no
//    alias retain. This pins that the fix is the targeted transfer-leaf
//    retain, not a blanket retain that would leak one reference per call.
// ---------------------------------------------------------------------------

const FRESH_ONLY: &str = "def fresh_only() -> int64 = {\n\
  y = string_concat(\"a\", \"b\")\n\
  string_len(y)\n\
}\n\
n = fresh_only()\n";

#[test]
fn fresh_binding_takes_no_alias_retain() {
    if skip_without_cc() {
        return;
    }
    let (stdout, emitted) = build_run_and_emit(FRESH_ONLY, "fresh_only");
    assert_eq!(
        shared_line(&stdout, "n"),
        shared_line(&eval_stdout(FRESH_ONLY, "fresh_only"), "n"),
    );
    let body = emitted_function(&emitted, "int64_t fresh_only()");
    assert_eq!(
        count_in(body, "chelis_string_retain("),
        0,
        "a fresh allocation already owns its reference; retaining it here \
         would leak once per call:\n{body}"
    );
    assert_eq!(
        count_in(body, "chelis_string_release("),
        1,
        "the block releases its fresh binding exactly once:\n{body}"
    );
}
