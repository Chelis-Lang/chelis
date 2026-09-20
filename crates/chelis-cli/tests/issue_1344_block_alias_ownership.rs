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
//! out-of-i64 integer corrupt the heap (PR #1302 red-team finding P0-1).
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
//! fix does not reach it: a transfer that reaches a binding through a
//! call-argument temp (`__call_argN` is not an owned destination). The
//! returns-captured shape's block-caller variant IS closed here (test 6
//! below: the escape retain fires for a may-return-outer callee into a
//! value temp); a top-level `b = retg()` stays abstain-borrowed through
//! `main`'s provenance, by design. The call-arg-temp shape remains
//! recorded on the issue.
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
    let (head, _params) = signature
        .split_once('(')
        .expect("function signature has parameters");
    let (_ret, name) = head
        .rsplit_once(' ')
        .expect("function signature has a return type and name");
    // chelis#1820: located by NAME, not by the full signature. chelis#1799
    // added a `chelis_rng_state` parameter to every host body, and the old
    // full-signature needle then missed the definition and failed before this
    // row counted anything. The parameter list is not what the row asserts.
    let rest = common::host_body_definition(
        emitted,
        &format!("{}__chelis_owned_body", common::authored_c_symbol(name)),
    );
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
def my_take() -> i64 = {\n\
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
        0,
        "the verified body borrows the captured binding; the local spelling \
         must not mint an owner:\n{body}"
    );
    assert_eq!(
        count_in(body, "chelis_list_release("),
        0,
        "a borrowed capture has no function-body release:\n{body}"
    );
}

// ---------------------------------------------------------------------------
// 2. The PR #1302 P0 shape, distilled from Std.Io.Json's
//    `canonical_bigint_text`: one arm of a conditional binding aliases the
//    function's parameter while the other allocates fresh. The retain on
//    the alias arm is what makes the unconditional block-close release
//    correct on both paths.
// ---------------------------------------------------------------------------

const PARAM_ALIAS_ARM: &str = "def gate(text: string, negative: bool) -> i64 = {\n\
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
        count_in(body, "chelis_string_release(text);"),
        2,
        "the body-owned parameter must have one release site in each mutually \
         exclusive branch:\n{body}"
    );
    assert_eq!(
        count_in(body, "chelis_string_release(__let_0);"),
        1,
        "the joined branch-result owner must be released once after use:\n{body}"
    );
    assert_eq!(
        count_in(body, "chelis_string_release("),
        3,
        "the two exclusive parameter-release sites plus the joined-result \
         release are the complete source-level release plan:\n{body}"
    );

    let alias_retain = body
        .find("chelis_string_retain(text);")
        .expect("parameter-aliasing arm retain");
    let alias_release = alias_retain
        + body[alias_retain..]
            .find("chelis_string_release(text);")
            .expect("parameter-aliasing arm release");
    let joined_release = body
        .find("chelis_string_release(__let_0);")
        .expect("joined-result release");
    assert!(
        alias_retain < alias_release && alias_release < joined_release,
        "the alias arm must retain before releasing its parameter owner, and \
         the joined result must be released afterward:\n{body}"
    );
}

// ---------------------------------------------------------------------------
// 3. The call-escape sibling: the callee returns its parameter, so the
//    caller's binding shares the allocation of a value the caller was
//    handed. Pre-fix this underflowed the string refcount when the outer
//    caller released its own binding.
// ---------------------------------------------------------------------------

const CALL_ESCAPE_PARAM: &str = "def pass_through(s: string) -> string = s\n\
def wrap(text: string) -> i64 = {\n\
  d = pass_through(text)\n\
  string_len(d)\n\
}\n\
def outer(s0: string) -> i64 = {\n\
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
        2,
        "the call-result owner and the body-owned parameter are each \
         released once:\n{body}"
    );
}

// ---------------------------------------------------------------------------
// 4. Negative parity: a binding whose value is freshly allocated takes no
//    alias retain. This pins that the fix is the targeted transfer-leaf
//    retain, not a blanket retain that would leak one reference per call.
// ---------------------------------------------------------------------------

const FRESH_ONLY: &str = "def fresh_only() -> i64 = {\n\
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
        count_in(body, "chelis_string_release(__let_0);"),
        1,
        "the block releases its fresh binding exactly once:\n{body}"
    );
    assert_eq!(
        count_in(body, "chelis_string_concat_owned("),
        1,
        "the uniquely owned lhs literal must move into concat:\n{body}"
    );
    assert_eq!(
        count_in(body, "chelis_string_concat("),
        0,
        "falling back to the cloning concat would hide a stale ownership oracle:\n{body}"
    );
    assert_eq!(
        count_in(body, "chelis_string_release(__arg0_1);"),
        0,
        "the lhs literal owner is consumed by concat and must not be released again:\n{body}"
    );
    assert_eq!(
        count_in(body, "chelis_string_release(__arg1_2);"),
        1,
        "the borrowed rhs literal remains independently owned and releases once:\n{body}"
    );
    assert_eq!(
        count_in(body, "chelis_string_release("),
        2,
        "only the rhs literal and the fresh result owner release independently:\n{body}"
    );
}

// ---------------------------------------------------------------------------
// 5. Exactly-once composition (round-2 red-team finding on PR #1302): a
//    binding-mediated parameter return is OWNED - the copy retains at the
//    value temp and the result leaf retains the escaping binding - so the
//    caller must claim it rather than compensate it. The first cut left
//    `analyze_returns_arg` flagging `f` as may-return-its-argument, and
//    the caller's call-escape retain then triple-counted: three retains
//    against two releases, one leaked string per call. The counts below
//    pin each side of the composition.
// ---------------------------------------------------------------------------

const BINDING_MEDIATED_RETURN: &str = "def f(p: string) -> string = {\n\
  d = p\n\
  d\n\
}\n\
def g2() -> i64 = {\n\
  raw = string_concat(\"ab\", \"cd\")\n\
  out = f(raw)\n\
  add(string_len(out), string_len(raw))\n\
}\n\
n = g2()\n";

#[test]
fn binding_mediated_parameter_return_composes_exactly_once() {
    if skip_without_cc() {
        return;
    }
    let (stdout, emitted) = build_run_and_emit(BINDING_MEDIATED_RETURN, "binding_mediated");
    assert_eq!(
        shared_line(&stdout, "n"),
        shared_line(
            &eval_stdout(BINDING_MEDIATED_RETURN, "binding_mediated"),
            "n"
        ),
    );
    let f_body = emitted_function(&emitted, "chelis_string f(chelis_string p)");
    assert_eq!(
        count_in(f_body, "chelis_string_retain("),
        0,
        "the consuming body transfers its owned parameter through the local \
         alias into the return without another retain:\n{f_body}"
    );
    assert_eq!(
        count_in(f_body, "chelis_string_release("),
        0,
        "the transferred parameter is returned, not released in `f`:\n{f_body}"
    );
    let g2_body = emitted_function(&emitted, "int64_t g2()");
    assert_eq!(
        count_in(g2_body, "chelis_string_retain("),
        1,
        "the internal consuming call receives exactly one clone of `raw`; \
         its returned owner is then claimed without a second retain:\n{g2_body}"
    );
    for (owner, label) in [("__let_3", "out"), ("__let_0", "raw")] {
        assert_eq!(
            count_in(g2_body, &format!("chelis_string_release({owner});")),
            1,
            "the caller releases `{label}` exactly once through its verified \
            expression owner `{owner}`:\n{g2_body}"
        );
    }
    assert_eq!(
        count_in(g2_body, "chelis_string_concat_owned("),
        1,
        "the uniquely owned lhs literal must move into concat:\n{g2_body}"
    );
    assert_eq!(
        count_in(g2_body, "chelis_string_concat("),
        0,
        "falling back to the cloning concat would hide a stale ownership oracle:\n{g2_body}"
    );
    assert_eq!(
        count_in(g2_body, "chelis_string_release(__arg0_1);"),
        0,
        "the lhs literal owner is consumed by concat and must not be released again:\n{g2_body}"
    );
    assert_eq!(
        count_in(g2_body, "chelis_string_release(__arg1_2);"),
        1,
        "the borrowed rhs literal remains independently owned and releases once:\n{g2_body}"
    );
    assert_eq!(
        count_in(g2_body, "chelis_string_release("),
        3,
        "only the rhs literal plus `out` and `raw` release independently:\n{g2_body}"
    );
}

// ---------------------------------------------------------------------------
// 6. Outer-returning call into a value temp (round-3 red-team finding on
//    PR #1302): a callee that returns a CAPTURED top-level binding hands
//    back a borrowed reference through no argument at all, so the
//    call-escape retain's argument scan never fired and the binding held
//    a reference it did not own. That falsified the owned-return
//    summary's precondition, and `main` claiming the result over-released
//    the captured allocation (abort). The escape retain now also fires
//    for a may-return-outer callee when the destination is a value temp,
//    which additionally closes chelis#1344's block-caller variant of the
//    returns-captured shape.
// ---------------------------------------------------------------------------

const OUTER_RETURNING_CALL: &str = "gcap = string_concat(\"ab\", \"cd\")\n\
def retg() -> string = gcap\n\
def f() -> string = {\n\
  d = retg()\n\
  d\n\
}\n\
b = f()\n";

#[test]
fn outer_returning_call_into_a_value_temp_is_retained_and_claimed() {
    if skip_without_cc() {
        return;
    }
    let (stdout, emitted) = build_run_and_emit(OUTER_RETURNING_CALL, "outer_return_call");
    assert_eq!(
        shared_line(&stdout, "b"),
        shared_line(&eval_stdout(OUTER_RETURNING_CALL, "outer_return_call"), "b"),
    );
    let f_body = emitted_function(&emitted, "chelis_string f()");
    assert_eq!(
        count_in(f_body, "chelis_string_retain("),
        0,
        "`retg` returns an already-owned result and `f` transfers it through \
         the local alias without another retain:\n{f_body}"
    );
    assert_eq!(
        count_in(f_body, "chelis_string_release("),
        0,
        "the result owner leaves through `f`'s return:\n{f_body}"
    );
}

/// Disposition lock for `common::host_body_definition`, the shared locator this
/// file and three siblings reach through `emitted_function`. It runs on
/// synthetic C rather than on an emission, because neither case it holds is
/// reachable from today's emitter -- which is exactly why an emitted-code test
/// could not hold them, and why the locator's original comment asserted they
/// could not happen instead of checking.
///
/// Two ways the locator can misread. A match with no left word boundary accepts
/// `g_run__chelis_owned_body(` as `run__chelis_owned_body`, and the decoy's
/// body then satisfies a caller's assertions for the wrong reason. And taking
/// the FIRST `)` as the end of the parameter list mistakes a nested parenthesis
/// for the end of the signature: the `{` test then fails, the loop runs out,
/// and the helper panics that the body is not emitted here -- a confident,
/// loud, wrong diagnosis about a body that is present. That is chelis#1808's
/// own misdiagnosis one layer down.
///
/// Both measured RED against the pre-fold helper, which took the first `)` and
/// had no boundary check: the suffix case returned the decoy's body, and the
/// nested case panicked with "the body is not emitted here".
#[test]
fn the_shared_host_body_locator_reads_the_definition_and_not_a_look_alike() {
    let suffix_decoy = concat!(
        "void g_run__chelis_owned_body(chelis_tensor* x) { decoy; }\n",
        "void run__chelis_owned_body(chelis_tensor* x);\n",
        "void run__chelis_owned_body(chelis_tensor* x) { real; }\n",
    );
    let body = common::host_body_definition(suffix_decoy, "run__chelis_owned_body");
    assert!(
        body.contains("real;") && !body.contains("decoy;"),
        "a symbol ending in the searched name is not the searched body:\n{body}"
    );

    let nested_parameter = concat!(
        "void run__chelis_owned_body(void (*emit)(int), chelis_tensor* x);\n",
        "void run__chelis_owned_body(void (*emit)(int), chelis_tensor* x) { real; }\n",
    );
    let body = common::host_body_definition(nested_parameter, "run__chelis_owned_body");
    assert!(
        body.contains("real;"),
        "a nested parenthesis in the parameter list does not end the signature:\n{body}"
    );
}
