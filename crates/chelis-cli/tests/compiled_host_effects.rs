//! chelis#1297: compiled C executes the host-runtime operations of
//! [05-HOST-1], [05-HOST-2] and [05-HOST-3] with the same typed result or
//! language trap, in the same effect order, as `chelis eval`.
//!
//! Every program runs on both lanes through [`parity::assert_lanes_agree`],
//! which compares stdout, exit status and the failure message modulo eval's
//! `error: ` presentation prefix. Each accepted program has a failing twin
//! that fails for its semantic reason on both lanes, never for a target
//! gate. `process_run` and the clock reads keep their own files
//! (`process_run_builtin.rs`, `host_clock_builtins.rs`).

#[path = "common/host_effect_parity.rs"]
mod parity;

use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

/// `chelis build` fails for `source`; returns stderr, which must not carry a
/// target-support rejection.
fn build_rejection(source: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("rejected.ch");
    fs::write(&path, source).expect("write source");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", path.to_str().unwrap(), "--target", "c", "--emit-c"])
        .arg("--output")
        .arg(dir.path().join("out"))
        .output()
        .expect("chelis build runs");
    assert!(!output.status.success(), "{source} must not build");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        !stderr.contains("unsupported:"),
        "a legal operation must not meet a target gate: {stderr}"
    );
    stderr
}

// ---------------------------------------------------------------------------
// [05-OP-1] round_to
// ---------------------------------------------------------------------------

/// `round_to` rounds the exact binary value at the operand's own width, ties
/// to even, for every signed-integer `places` dtype, inside and outside a
/// function, with non-finite operands passing through.
#[test]
fn round_to_matches_eval_at_each_width() {
    let source = "def at(x: f64, places: i64) -> f64 = round_to(x, places)\n\
                  a = round_to(2.675f64, 2i64)\n\
                  b = round_to(0.125f64, 2i32)\n\
                  c = round_to(0.375f32, 2i8)\n\
                  d = round_to(2.675f32, 2i16)\n\
                  e = at(-0.001f64, 1i64)\n\
                  f = at(div(1.0f64, 0.0f64), 3i64)\n\
                  g = round_to(1.1f32, 8i64)\n\
                  h = at(123.456f64, 0i64)\n";
    let run = parity::assert_lanes_agree(source, "rounding");
    assert_eq!(run.status, Some(0), "{run:?}");
    for expected in [
        "a = 2.67\n",
        "b = 0.12\n",
        "c = 0.38\n",
        "d = 2.67\n",
        "e = -0.0\n",
        "f = inf\n",
        "g = 1.1\n",
        "h = 123.0\n",
    ] {
        assert!(
            run.stdout.contains(expected),
            "missing `{expected}`: {run:?}"
        );
    }
}

/// The failing twins: `places` outside the admitted domain is the same
/// language trap on both lanes, after the effects that precede it.
#[test]
fn round_to_domain_failures_match_eval() {
    for (name, source, failure) in [
        (
            "negative",
            "a = print(\"before\")\nb = round_to(1.25, -1i64)\n",
            "round_to: places must be in 0..=100, got -1",
        ),
        (
            "toolarge",
            "b = round_to(1.25f32, 101i32)\n",
            "round_to: places must be in 0..=100, got 101",
        ),
    ] {
        let run = parity::assert_lanes_agree(source, name);
        assert_eq!(run.status, Some(1), "{name}: {run:?}");
        assert!(run.failure.starts_with(failure), "{name}: {run:?}");
    }
}

/// A dtype outside the admitted set is a type error at check time, which
/// the build reports as a type error rather than a target gate.
#[test]
fn round_to_rejects_an_unadmitted_dtype_by_type() {
    let stderr = build_rejection("a = round_to(3i64, 2i64)\n");
    assert!(stderr.contains("round_to"), "{stderr}");
}

// ---------------------------------------------------------------------------
// [05-OP-61], [05-OP-2..3], [05-OP-5] CSV text tables
// ---------------------------------------------------------------------------

const CSV_TABLE: &str =
    "t = parse_csv(\"b,a,note\\n9007199254740993,1.5,\\\"x,y\\\"\\n-2, 2e3 ,plain\\n\")\n";

/// Parsing keeps cells as text in header order; every accessor reads them
/// exactly, and serialization requotes them.
#[test]
fn csv_matches_eval_for_every_accessor() {
    let source = format!(
        "{CSV_TABLE}\
         def cell(table: List[Dict[string, string]], row: i64) -> i64 = csv_int(table, row, \"b\")\n\
         cols = csv_cols(t)\n\
         rows = csv_nrows(t)\n\
         notes = csv_strs(t, \"note\")\n\
         prices = csv_f64s(t, \"a\")\n\
         ids = csv_ints(t, \"b\")\n\
         note = csv_str(t, 0i64, \"note\")\n\
         price = csv_f64(t, 1i64, \"a\")\n\
         id = cell(t, 0i64)\n\
         text = to_csv(t)\n\
         empty = to_csv(parse_csv(\"only\\n\"))\n"
    );
    let run = parity::assert_lanes_agree(&source, "csvread");
    assert_eq!(run.status, Some(0), "{run:?}");
    for expected in [
        "rows = 2\n",
        "notes = [x,y, plain]\n",
        "prices = [1.5, 2000.0]\n",
        "ids = [9007199254740993, -2]\n",
        "note = x,y\n",
        "price = 2000.0\n",
        "id = 9007199254740993\n",
    ] {
        assert!(
            run.stdout.contains(expected),
            "missing `{expected}`: {run:?}"
        );
    }
}

/// The failing twins: each malformed table or cell is the same named
/// failure on both lanes, never a default, NaN, or skipped cell.
#[test]
fn csv_failures_match_eval() {
    for (name, body, failure) in [
        (
            "missing",
            "x = csv_f64(t, 0i64, \"zz\")\n",
            "csv_f64: column `zz` not found; available columns: b, a, note",
        ),
        (
            "floatsyntax",
            "x = csv_ints(t, \"a\")\n",
            "csv_ints: cell `1.5` uses float syntax; use the corresponding csv_f64 accessor",
        ),
        (
            "textcell",
            "x = csv_f64s(t, \"note\")\n",
            "csv_f64s: cell `x,y` is not a finite f64: invalid number (expected a digit)",
        ),
        (
            "range",
            "x = csv_str(t, 2i64, \"b\")\n",
            "csv_str: data row 2 out of range for 2 rows",
        ),
        (
            "negative",
            "x = csv_int(t, -1i64, \"b\")\n",
            "csv_int: row index must be nonnegative, got -1",
        ),
        (
            "ragged",
            "x = parse_csv(\"a,b\\n1\\n\")\n",
            "parse_csv: row 2 has 1 fields, expected 2",
        ),
        (
            "overflow",
            "x = csv_ints(parse_csv(\"n\\n9223372036854775808\\n\"), \"n\")\n",
            "csv_ints: cell `9223372036854775808` overflows i64",
        ),
    ] {
        let source = format!("{CSV_TABLE}{body}");
        let run = parity::assert_lanes_agree(&source, name);
        assert_eq!(run.status, Some(1), "{name}: {run:?}");
        assert_eq!(run.failure, failure, "{name}: {run:?}");
    }
}

// ---------------------------------------------------------------------------
// [05-HOST-3] the `test_*` assertion family
// ---------------------------------------------------------------------------

/// Passing assertions of every identity return unit on both lanes, at
/// top level and inside a `Test` function, over scalars, strings, recursive
/// values, signed zeros, and tensors at their own dtype.
#[test]
fn passing_assertions_match_eval() {
    let source = "type Pick = | Left(i64) | Right(string)\n\
                  def test_values(n: i64) -> unit ! {Test} = {\n\
                  _ = test_assert(gt(n, 0i64), \"positive\")\n\
                  _ = test_assert_eq([n, 2i64], [n, 2i64], \"list\")\n\
                  _ = test_assert_eq((n, \"s\"), (n, \"s\"), \"tuple\")\n\
                  _ = test_assert_eq(Some(n), Some(n), \"option\")\n\
                  test_assert_eq(Right(\"r\"), Right(\"r\"), \"adt\")\n\
                  }\n\
                  a = test_values(3i64)\n\
                  b = test_assert_eq(0.0f64, -0.0f64, \"signed zero\")\n\
                  c = test_assert_eq(\"x\", \"x\", \"string\")\n\
                  d = test_assert_eq_tensor(to_tensor([1i32, 2i32]), to_tensor([1i32, 2i32]), \"ints\")\n\
                  e = test_assert_close_tensor(to_tensor([1.0f32, 2.0f32]), to_tensor([1.05f32, 2.0f32]), 0.1f32, \"close\")\n\
                  f = test_assert_close_tensor(to_tensor([1.0f64]), to_tensor([1.0f64]), 0.0f64, \"exact\")\n\
                  after = print(\"done\")\n";
    let run = parity::assert_lanes_agree(source, "passing");
    assert_eq!(run.status, Some(0), "{run:?}");
    assert!(run.stdout.contains("done\n"), "{run:?}");
}

/// The failing twins: each assertion traps `Test` with its label and the
/// eval-identical message, after the effects that precede it and before any
/// that follow.
#[test]
fn failing_assertions_match_eval() {
    for (name, assertion, failure) in [
        (
            "bool",
            "test_assert(false, \"flag\")",
            "assert failed: flag",
        ),
        (
            "scalar",
            "test_assert_eq(3i64, 4i64, \"n\")",
            "assert_eq (n): expected 4, got 3",
        ),
        (
            "nan",
            "test_assert_eq(div(0.0f64, 0.0f64), div(0.0f64, 0.0f64), \"nan\")",
            "assert_eq (nan): expected NaN, got NaN",
        ),
        (
            "list",
            "test_assert_eq([1i64, 2i64], [1i64, 3i64], \"xs\")",
            "assert_eq (xs): expected [1, 3], got [1, 2]",
        ),
        (
            "option",
            "test_assert_eq(Some(\"a\"), None, \"o\")",
            "assert_eq (o): expected None, got Some(a)",
        ),
        (
            "tensor",
            "test_assert_eq_tensor(to_tensor([1i8, 2i8]), to_tensor([1i8, 5i8]), \"t\")",
            "assert_eq_tensor (t): first mismatch at row-major index 1",
        ),
        (
            "close",
            "test_assert_close_tensor(to_tensor([1.0f32, 2.0f32]), to_tensor([1.0f32, 2.5f32]), 0.1f32, \"c\")",
            "assert_close_tensor (c): at index 1 expected 2.5, got 2.0, tol 0.1",
        ),
        (
            "closenan",
            "test_assert_close_tensor(to_tensor([div(0.0f64, 0.0f64)]), to_tensor([1.0f64]), 1.0f64, \"cn\")",
            "assert_close_tensor (cn): at index 0 expected 1.0, got NaN, tol 1.0 (NaN is never close)",
        ),
        (
            "tolerance",
            "test_assert_close_tensor(to_tensor([1.0f32]), to_tensor([1.0f32]), -1.0f32, \"tl\")",
            "assert_close_tensor (tl): invalid tolerance -1.0 (must be finite and non-negative)",
        ),
    ] {
        let source = format!(
            "before = print(\"before\")\nchecked = {assertion}\nafter = print(\"after\")\n"
        );
        let run = parity::assert_lanes_agree(&source, name);
        assert_eq!(run.status, Some(1), "{name}: {run:?}");
        assert_eq!(run.failure, failure, "{name}: {run:?}");
        assert!(
            run.stdout.contains("before") && !run.stdout.contains("after"),
            "{name}: {run:?}"
        );
    }
}

/// An uncalled assertion definition is ordinary compiled code: the program
/// builds without an abort stub, and its executable asserts nothing.
#[test]
fn an_uncalled_assertion_definition_compiles_without_a_stub() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("dead.ch");
    fs::write(
        &path,
        "def test_dead() -> unit ! {Test} = test_assert(false, \"unreachable\")\nout = 7i32\n",
    )
    .expect("write source");
    let out = dir.path().join("out");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", path.to_str().unwrap(), "--target", "c", "--emit-c"])
        .arg("--output")
        .arg(&out)
        .assert()
        .success();
    let emitted = fs::read_to_string(out.join("dead.c")).expect("emitted C");
    assert!(!emitted.contains("unsupported:"), "{emitted}");
    assert!(emitted.contains("chelis_test_assert_fail"), "{emitted}");
    let run = parity::assert_lanes_agree(
        "def test_dead() -> unit ! {Test} = test_assert(false, \"unreachable\")\nout = 7i32\n",
        "dead",
    );
    assert_eq!(run.stdout, "out = 7\n", "{run:?}");
}

// ---------------------------------------------------------------------------
// [05-HOST-1] tensor_scan over scalar states
// ---------------------------------------------------------------------------

/// `tensor_scan` stacks the `n` states at the state's own dtype, runs the
/// callback once per index in order with its effects, evaluates `initial`
/// before `n`, and invokes nothing when `n` is zero.
#[test]
fn tensor_scan_matches_eval_for_scalar_states() {
    let source = "def noisy(label: string, value: i64) -> i64 ! {IO} = {\n\
                  _ = print(label)\n\
                  value\n\
                  }\n\
                  def step(prev: i32, i: i64) -> i32 ! {IO} = {\n\
                  _ = print(to_string(i))\n\
                  add(prev, cast(i, i32))\n\
                  }\n\
                  ints = tensor_scan(1i64, fn (prev: i64, i: i64) -> add(prev, i), 4i64)\n\
                  narrow = tensor_scan(10i32, step, noisy(\"n\", 3i64))\n\
                  ordered = tensor_scan(noisy(\"init\", 5i64), fn (prev: i64, i: i64) -> mul(prev, 2i64), noisy(\"length\", 2i64))\n\
                  halves = tensor_scan(1.0f32, fn (prev: f32, i: i64) -> mul(prev, 0.5f32), 3i64)\n\
                  wide = tensor_scan(0.1f64, fn (prev: f64, i: i64) -> add(prev, 0.1f64), 2i64)\n\
                  flips = tensor_scan(true, fn (prev: bool, i: i64) -> not(prev), 3i64)\n\
                  halfs = tensor_scan(1.5f16, fn (prev: f16, i: i64) -> mul(prev, 2.0f16), 2i64)\n\
                  brains = tensor_scan(1.5bf16, fn (prev: bf16, i: i64) -> add(prev, 1.0bf16), 2i64)\n\
                  bytes = tensor_scan(-3i8, fn (prev: i8, i: i64) -> add(prev, 1i8), 2i64)\n\
                  shorts = tensor_scan(1i16, fn (prev: i16, i: i64) -> mul(prev, 3i16), 2i64)\n\
                  empty = tensor_scan(7i8, step8, 0i64)\n\
                  def step8(prev: i8, i: i64) -> i8 ! {IO} = {\n\
                  _ = print(\"never\")\n\
                  prev\n\
                  }\n";
    let run = parity::assert_lanes_agree(source, "scans");
    assert_eq!(run.status, Some(0), "{run:?}");
    for expected in [
        "ints = tensor(shape=[4], data=[1, 2, 4, 7])\n",
        "n\n0\n1\n2\ninit\nlength\n",
        "narrow = tensor(shape=[3], data=[10, 11, 13])\n",
        "ordered = tensor(shape=[2], data=[10, 20])\n",
        "halves = tensor(shape=[3], data=[0.5, 0.25, 0.125])\n",
        "flips = tensor(shape=[3], data=[false, true, false])\n",
        "empty = tensor(shape=[0], data=[])\n",
        "halfs = tensor(shape=[2], data=[3.0, 6.0])\n",
        "brains = tensor(shape=[2], data=[2.5, 3.5])\n",
        "bytes = tensor(shape=[2], data=[-2, -1])\n",
        "shorts = tensor(shape=[2], data=[3, 9])\n",
    ] {
        assert!(
            run.stdout.contains(expected),
            "missing `{expected}`: {run:?}"
        );
    }
    assert!(!run.stdout.contains("never"), "{run:?}");
}

/// The failing twins: a negative length and an overflowing integer callback
/// are the same trap on both lanes, after the iterations before it.
#[test]
fn tensor_scan_failures_match_eval() {
    let run = parity::assert_lanes_agree(
        "xs = tensor_scan(1i64, fn (prev: i64, i: i64) -> prev, -1i64)\n",
        "negative",
    );
    assert_eq!(run.status, Some(1), "{run:?}");
    assert_eq!(
        run.failure, "tensor_scan requires a non-negative length, got -1",
        "{run:?}"
    );
    let run = parity::assert_lanes_agree_on_numeric_trap(
        "def step(prev: i32, i: i64) -> i32 ! {IO} = {\n_ = print(to_string(i))\nadd(prev, 1i32)\n}\nxs = tensor_scan(2147483646i32, step, 3i64)\n",
        "overflow",
    );
    assert_eq!(run.stdout, "0\n1\n", "{run:?}");
    assert_eq!(
        run.failure, "numeric trap: overflow in add at i32",
        "{run:?}"
    );
}

// ---------------------------------------------------------------------------
// The comparator rejects every divergence #1297 names
// ---------------------------------------------------------------------------

fn run(status: i32, stdout: &str, failure: &str) -> parity::LaneRun {
    parity::LaneRun {
        status: Some(status),
        stdout: stdout.to_string(),
        failure: failure.to_string(),
    }
}

/// A compiled lane that reorders effects, reads a default clock, erases a
/// dtype, makes an assertion inert, or changes a failure's text disagrees
/// with eval; only eval's `error: ` presentation prefix is ignored.
#[test]
fn the_comparator_rejects_each_named_divergence() {
    let eval = run(0, "first\nsecond\nw.0 = 1791074996\nd = 2.67\n", "");
    for (divergence, compiled) in [
        (
            "reordered effects",
            run(0, "second\nfirst\nw.0 = 1791074996\nd = 2.67\n", ""),
        ),
        (
            "default clock reading",
            run(0, "first\nsecond\nw.0 = 0\nd = 2.67\n", ""),
        ),
        (
            "dtype erasure",
            run(
                0,
                "first\nsecond\nw.0 = 1791074996\nd = 2.6700000762939453\n",
                "",
            ),
        ),
        (
            "dropped effect",
            run(0, "first\nw.0 = 1791074996\nd = 2.67\n", ""),
        ),
    ] {
        assert!(!parity::lanes_agree(&eval, &compiled), "{divergence}");
    }
    let failing = run(1, "before\n", "assert failed: flag");
    assert!(
        !parity::lanes_agree(&failing, &run(0, "before\nafter\n", "")),
        "an inert assertion"
    );
    assert!(
        !parity::lanes_agree(&failing, &run(1, "before\n", "assert failed")),
        "a changed failure message"
    );
    assert_eq!(
        parity::eval_failure_body("before\nerror: assert failed: flag\n"),
        parity::compiled_failure_body("assert failed: flag\n"),
        "the presentation prefix alone is not a divergence"
    );
}
