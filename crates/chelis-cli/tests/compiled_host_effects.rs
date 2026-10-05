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
    assert_eq!(run.context, "tensor_scan length is negative: -1", "{run:?}");
    assert_eq!(
        run.failure, "numeric trap: domain in tensor_scan at i64",
        "{run:?}"
    );
    let run = parity::assert_lanes_agree(
        "def step(prev: i32, i: i64) -> i32 ! {IO} = {\n_ = print(to_string(i))\nadd(prev, 1i32)\n}\nxs = tensor_scan(2147483646i32, step, 3i64)\n",
        "overflow",
    );
    assert_eq!(run.status, Some(1), "{run:?}");
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
        context: String::new(),
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
    let with_context = |context: &str| parity::LaneRun {
        context: context.to_string(),
        ..run(1, "", "numeric trap: domain in add at i64")
    };
    assert!(
        !parity::lanes_agree(
            &with_context("add operands disagree at axis 0: lhs [2] has 2, rhs [3] has 3"),
            &with_context("Domain: add operand shape mismatch"),
        ),
        "a changed trap context"
    );
    assert_eq!(
        parity::failure_context(
            "error: add operands disagree\nnumeric trap: domain in add at i64\n"
        ),
        parity::failure_context("add operands disagree\nnumeric trap: domain in add at i64\n"),
        "the presentation prefix alone is not a context divergence"
    );
    assert_eq!(
        parity::eval_failure_body("before\nerror: assert failed: flag\n"),
        parity::compiled_failure_body("assert failed: flag\n"),
        "the presentation prefix alone is not a divergence"
    );
}

// ---------------------------------------------------------------------------
// [04-NUM-10] a numeric trap ends a compiled program as it ends eval
// ---------------------------------------------------------------------------

/// Every numeric trap class exits with eval's status 1 and message, after
/// the output that precedes it; no compiled trap aborts (chelis#3107).
#[test]
fn numeric_traps_exit_like_eval() {
    for (name, body, failure) in [
        (
            "overflow",
            "x = f(2147483647i32)\n",
            "numeric trap: overflow in add at i32",
        ),
        (
            "divide",
            "x = floor_div(7i64, g(0i64))\n",
            "numeric trap: division by zero in floor_div at i64",
        ),
        (
            "tensor",
            "x = add(to_tensor([2147483647i32]), to_tensor([1i32]))\n",
            "",
        ),
    ] {
        let source = format!(
            "def f(a: i32) -> i32 = add(a, 1i32)\ndef g(a: i64) -> i64 = a\nbefore = print(\"before\")\n{body}"
        );
        let run = parity::assert_lanes_agree(&source, name);
        assert_eq!(run.status, Some(1), "{name}: {run:?}");
        if !failure.is_empty() {
            assert_eq!(run.failure, failure, "{name}: {run:?}");
        }
        assert_eq!(run.stdout, "before\n", "{name}: {run:?}");
    }
}

// ---------------------------------------------------------------------------
// [05-HOST-1] tensor_scan over tensor states (chelis#2999)
// ---------------------------------------------------------------------------

/// A tensor state stacks to `[n] ++ state_shape` at the state's dtype: the
/// issue's f16 rows, a rank-0 state, a 2x2 i64 state, a bool state, an
/// `n = 0` scan that keeps every state extent without running the
/// callback, and a zero-sized state whose callback still runs `n` times.
#[test]
fn tensor_scan_matches_eval_for_tensor_states() {
    let source = "def doubled(prev: tensor[2, f16], i: i64) -> tensor[2, f16] = mul(prev, to_tensor([2.0f16, 2.0f16]))\n\
                  def loud(prev: tensor[2, f16], i: i64) -> tensor[2, f16] ! {IO} = {\n\
                  _ = print(\"never\")\n\
                  prev\n\
                  }\n\
                  def tick(prev: tensor[*, f32], i: i64) -> tensor[*, f32] ! {IO} = {\n\
                  _ = print(to_string(i))\n\
                  prev\n\
                  }\n\
                  rows = tensor_scan(to_tensor([1.0f16, 2.0f16]), doubled, 3i64)\n\
                  none = tensor_scan(to_tensor([1.0f16, 2.0f16]), loud, 0i64)\n\
                  hollow = tensor_scan(to_tensor([]: List[f32]), tick, 3i64)\n\
                  point = tensor_scan(scalar_to_tensor(3i64), fn (prev: tensor[i64], i: i64) -> add(prev, scalar_to_tensor(i)), 3i64)\n\
                  grid = tensor_scan(to_tensor([[1i64, 2i64], [3i64, 4i64]]), fn (prev: tensor[2, 2, i64], i: i64) -> add(prev, prev), 2i64)\n\
                  masks = tensor_scan(to_tensor([true, false]), fn (prev: tensor[2, bool], i: i64) -> not(prev), 2i64)\n";
    let run = parity::assert_lanes_agree(source, "tensorscans");
    assert_eq!(run.status, Some(0), "{run:?}");
    for expected in [
        "0\n1\n2\n",
        "rows = tensor(shape=[3, 2], data=[2.0, 4.0, 4.0, 8.0, 8.0, 16.0])\n",
        "none = tensor(shape=[0, 2], data=[])\n",
        "hollow = tensor(shape=[3, 0], data=[])\n",
        "point = tensor(shape=[3], data=[3, 4, 6])\n",
        "grid = tensor(shape=[2, 2, 2], data=[2, 4, 6, 8, 4, 8, 12, 16])\n",
        "masks = tensor(shape=[2, 2], data=[false, true, true, false])\n",
    ] {
        assert!(
            run.stdout.contains(expected),
            "missing `{expected}`: {run:?}"
        );
    }
    assert!(!run.stdout.contains("never"), "{run:?}");
}

/// The failing twin: a callback that changes the state's shape or dtype is
/// a `tensor_scan` type error at check time, on every lane.
#[test]
fn tensor_scan_rejects_a_state_changing_callback_by_type() {
    for body in [
        "x = tensor_scan(to_tensor([1.0f32, 2.0f32]), fn (prev: tensor[2, f32], i: i64) -> to_tensor([1.0f32]), 2i64)\n",
        "x = tensor_scan(to_tensor([1.0f32, 2.0f32]), fn (prev: tensor[2, f32], i: i64) -> to_tensor([1.0f64, 2.0f64]), 2i64)\n",
    ] {
        let stderr = build_rejection(body);
        assert!(stderr.contains("tensor_scan"), "{stderr}");
    }
}

/// [05-HOST-1] makes the state's shape invariant across every callback
/// application. A callback whose declared state type admits another shape
/// is checked when each application returns: the first changed state traps
/// `Domain` in `tensor_scan` in spec/04-type-system.md section 4.7's form,
/// before the next application's effects, identically on both lanes.
#[test]
fn tensor_scan_traps_a_state_whose_shape_changes_at_run_time() {
    for (name, source, effects, context) in [
        (
            "scangrow",
            "def step(prev: tensor[*, f32], i: i64) -> tensor[*, f32] ! { IO } = {\n\
             _ = print(to_string(i))\n\
             concat([prev, prev], 0i32)\n\
             }\n\
             xs = tensor_scan(to_tensor([1.0f32]), step, 3i64)\n",
            "0\n",
            "tensor_scan state changed at axis 0: initial state [1] has 1, callback result [2] has 2",
        ),
        (
            "scanrank2grow",
            "def step(prev: tensor[*, *, i32], i: i64) -> tensor[*, *, i32] = concat([prev, prev], 1i32)\n\
             xs = tensor_scan(to_tensor([[1i32], [2i32]]), step, 2i64)\n",
            "",
            "tensor_scan state changed at axis 1: initial state [2, 1] has 1, callback result [2, 2] has 2",
        ),
        (
            "scantranspose",
            "def step(prev: tensor[*, *, i32], i: i64) -> tensor[*, *, i32] ! { IO } = {\n\
             _ = print(to_string(i))\n\
             permute(prev, 1i32, 0i32)\n\
             }\n\
             xs = tensor_scan(to_tensor([[1i32], [2i32]]), step, 2i64)\n",
            "0\n",
            "tensor_scan state changed at axis 0: initial state [2, 1] has 2, callback result [1, 2] has 1",
        ),
        (
            "scaninlinegrow",
            "xs = tensor_scan(to_tensor([1.0f32]), fn (prev: tensor[*, f32], i: i64) -> concat([prev, prev], 0i32), 2i64)\n",
            "",
            "tensor_scan state changed at axis 0: initial state [1] has 1, callback result [2] has 2",
        ),
    ] {
        let run = parity::assert_lanes_agree(source, name);
        assert_eq!(run.status, Some(1), "{name}: {run:?}");
        assert_eq!(run.stdout, effects, "{name}: {run:?}");
        assert_eq!(run.context, context, "{name}: {run:?}");
        assert_eq!(
            run.failure, "numeric trap: domain in tensor_scan at i64",
            "{name}: {run:?}"
        );
    }
}

/// The passing twin: the same `*`-typed callbacks that keep the state's
/// shape run every application on both lanes.
#[test]
fn tensor_scan_runs_a_wildcard_callback_that_keeps_the_state_shape() {
    let source = "def step(prev: tensor[*, f32], i: i64) -> tensor[*, f32] ! { IO } = {\n\
                  _ = print(to_string(i))\n\
                  add(prev, prev)\n\
                  }\n\
                  def turn(prev: tensor[*, *, i32], i: i64) -> tensor[*, *, i32] = reshape(prev, [shape(prev, 0i32), shape(prev, 1i32)])\n\
                  xs = tensor_scan(to_tensor([1.0f32]), step, 3i64)\n\
                  ys = tensor_scan(to_tensor([[1i32], [2i32]]), turn, 2i64)\n";
    let run = parity::assert_lanes_agree(source, "scankeep");
    assert_eq!(run.status, Some(0), "{run:?}");
    assert_eq!(
        run.stdout,
        "0\n1\n2\nxs = tensor(shape=[3, 1], data=[2.0, 4.0, 8.0])\n\
         ys = tensor(shape=[2, 2, 1], data=[1, 2, 1, 2])\n",
        "{run:?}"
    );
}

// ---------------------------------------------------------------------------
// [05-OP-1] round_to at every active float dtype and every places (#1295)
// ---------------------------------------------------------------------------

/// `round_to` admits f16 and bf16, rounds left of the point for negative
/// `places`, is the identity once the quantum is finer than the value, and
/// overflows to infinity, identically on both lanes.
#[test]
fn round_to_matches_eval_for_every_float_dtype_and_places() {
    let source = "h = round_to(cast(2.675f32, f16), 2i64)\n\
                  b = round_to(cast(2.675f32, bf16), 1i8)\n\
                  left = round_to(1234.5f64, -2i64)\n\
                  tie = round_to(1250.0f32, -2i16)\n\
                  zero = round_to(-49.0f64, -2i32)\n\
                  fine = round_to(0.1f64, 1000i64)\n\
                  coarse = round_to(1.0f64, -400i64)\n\
                  huge = round_to(cast(65504.0f32, f16), -5i64)\n";
    let run = parity::assert_lanes_agree(source, "roundwide");
    assert_eq!(run.status, Some(0), "{run:?}");
    for expected in [
        "h = 2.68\n",
        "left = 1200.0\n",
        "tie = 1200.0\n",
        "zero = -0.0\n",
        "fine = 0.1\n",
        "coarse = 0.0\n",
        "huge = inf\n",
    ] {
        assert!(
            run.stdout.contains(expected),
            "missing `{expected}`: {run:?}"
        );
    }
}
// ---------------------------------------------------------------------------
// [04-NUM-10] language failures end a compiled program as they end eval
// ---------------------------------------------------------------------------

/// A runtime disagreement between elementwise operand shapes is a `Domain`
/// trap in the operation (spec/04-type-system.md section 4.7), an index
/// outside a sparse axis is a `Domain` trap in the sparse primitive
/// ([05-SPARSE-1], spec/05 section 3.5), and an index outside a List fails
/// loudly ([05-OP-54]). Each failure ends both lanes with status 1, the same
/// context and failure line, and the output before it; no compiled failure
/// aborts.
#[test]
fn language_failures_exit_like_eval() {
    let runtime_list = |items: &str| format!("map(fn (i: i64) -> {items}, range(0i64, 2i64))");
    for (name, body, context, failure) in [
        (
            "shape",
            "a = to_tensor(map(fn (i: i64) -> 1.0f32, range(0i64, 2i64)))\n\
             b = to_tensor(map(fn (i: i64) -> 1.0f32, range(0i64, 3i64)))\n\
             c = add(a, b)\n"
                .to_string(),
            "add operands disagree at axis 0: lhs [2] has 2, rhs [3] has 3",
            "numeric trap: domain in add at i64",
        ),
        (
            "compare",
            "a = to_tensor(map(fn (i: i64) -> 1i64, range(0i64, 3i64)))\n\
             b = to_tensor(map(fn (i: i64) -> 1i64, range(0i64, 2i64)))\n\
             c = lt(a, b)\n"
                .to_string(),
            "lt operands disagree at axis 0: lhs [3] has 3, rhs [2] has 2",
            "numeric trap: domain in lt at i64",
        ),
        (
            "gather",
            format!(
                "t = to_tensor([1i64, 2i64, 3i64])\ng = gather(t, to_tensor({}), 0i32)\n",
                runtime_list("add(i, 2i64)")
            ),
            "gather index 3 out of bounds at axis 0 of extent 3",
            "numeric trap: domain in gather at i64",
        ),
        (
            "index",
            "xs = [1i64, 2i64]\ny = index(xs, add(1i64, 4i64))\n".to_string(),
            "",
            "index 5 out of bounds for list of len 2",
        ),
        (
            "take",
            "k = sub(tensor_to_scalar(sum(to_tensor([1i64, 1i64]), 0)), 5i64)\n\
             ys = take([1i64, 2i64], k)\n"
                .to_string(),
            "",
            "take requires non-negative count, got -3",
        ),
        (
            "tensor_index",
            "k = tensor_to_scalar(sum(to_tensor([2i64, 3i64]), 0))\n\
             t = to_tensor(index([[1.0f32], [2.0f32]], k))\n"
                .to_string(),
            "",
            "index 5 out of bounds for list of len 2",
        ),
    ] {
        let source = format!("before = print(\"before\")\n{body}");
        let run = parity::assert_lanes_agree(&source, name);
        assert_eq!(run.status, Some(1), "{name}: {run:?}");
        assert_eq!(run.failure, failure, "{name}: {run:?}");
        assert_eq!(run.stdout, "before\n", "{name}: {run:?}");
        assert_eq!(run.context, context, "{name}: {run:?}");
    }
}

/// A runtime disagreement between matmul's operands is a `Domain` trap in
/// `matmul` itself (spec/04-type-system.md section 4.7: the trap names the
/// operation that introduces the guarded extent), never in the `mul` that the
/// lowering decomposes it into, and never a non-trap failure. The context
/// line names both operands as written. The rows cover an IO-effect caller,
/// a pure callee reached from `main`, and a batched rank-3 product.
#[test]
fn matmul_operand_disagreements_trap_in_matmul_on_both_lanes() {
    let runtime = |count: &str| format!("tensor_to_scalar(sum(to_tensor({count}), 0))");
    let ones = |count: &str| format!("to_tensor(map(fn (i: i64) -> 1.0f32, range(0i64, {count})))");
    let two = runtime("[1i64, 1i64]");
    let three = runtime("[1i64, 2i64]");
    for (name, source, context) in [
        (
            "matmul_io",
            format!(
                "def run(a: tensor[*, *, f32], b: tensor[*, *, f32]) -> tensor[*, *, f32] ! {{IO}} = {{\n\
                 _ = print(\"in\")\n\
                 matmul(a, b)\n\
                 }}\n\
                 a = reshape({}, [{two}, 3i64])\n\
                 b = reshape({}, [{two}, 2i64])\n\
                 c = run(a, b)\n",
                ones("6i64"),
                ones("4i64")
            ),
            "matmul shared axis disagrees: lhs [2, 3] has 3 at axis 1, rhs [2, 2] has 2 at axis 0",
        ),
        (
            "matmul_main",
            format!(
                "def mm(a: tensor[*, *, f32], b: tensor[*, *, f32]) -> tensor[*, *, f32] = matmul(a, b)\n\
                 def main() -> tensor[*, *, f32] = mm(reshape({}, [{two}, 3i64]), reshape({}, [{two}, 2i64]))\n",
                ones("6i64"),
                ones("4i64")
            ),
            "matmul shared axis disagrees: lhs [2, 3] has 3 at axis 1, rhs [2, 2] has 2 at axis 0",
        ),
        (
            "matmul_batched",
            format!(
                "def mm(a: tensor[*, *, *, f32], b: tensor[*, *, *, f32]) -> tensor[*, *, *, f32] = matmul(a, b)\n\
                 def main() -> tensor[*, *, *, f32] = mm(reshape({}, [2i64, 2i64, {three}]), reshape({}, [2i64, {two}, 2i64]))\n",
                ones("12i64"),
                ones("8i64")
            ),
            "matmul shared axis disagrees: lhs [2, 2, 3] has 3 at axis 2, rhs [2, 2, 2] has 2 at axis 1",
        ),
        (
            "matmul_claimed",
            "sig f[n]: tensor[2, n, f32] -> tensor[n, 2, f32] -> tensor[2, 2, f32]\n\
             def f(x, y) = {\n\
             k = sub(shape(x, cast(1, i32)), cast(1, i64))\n\
             a = shrink(x, [[cast(0, i64), cast(2, i64)], [cast(0, i64), k]])\n\
             matmul(a, y)\n\
             }\n\
             out = f(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]), \
             to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32], [5.0f32, 6.0f32]]))\n"
                .to_string(),
            "matmul shared axis disagrees: lhs [2, 2] has 2 at axis 1, rhs [3, 2] has 3 at axis 0",
        ),
        (
            "matmul_vmap",
            "sig g[n]: tensor[2, n, f32] -> tensor[n, 2, f32] -> tensor[2, 2, f32]\n\
             def g(x, y) = {\n\
             k = sub(shape(x, cast(1, i32)), cast(1, i64))\n\
             a = shrink(x, [[cast(0, i64), cast(2, i64)], [cast(0, i64), k]])\n\
             matmul(a, y)\n\
             }\n\
             out = vmap(g)(to_tensor([[[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]]), \
             to_tensor([[[1.0f32, 2.0f32], [3.0f32, 4.0f32], [5.0f32, 6.0f32]]]))\n"
                .to_string(),
            "matmul shared axis disagrees: lhs [1, 2, 2] has 2 at axis 2, rhs [1, 3, 2] has 3 at axis 1",
        ),
    ] {
        let run = parity::assert_lanes_agree(&source, name);
        assert_eq!(run.status, Some(1), "{name}: {run:?}");
        assert_eq!(
            run.failure, "numeric trap: domain in matmul at i64",
            "{name}: {run:?}"
        );
        assert_eq!(run.context, context, "{name}: {run:?}");
    }
}

/// The lanes recognise matmul's decomposed product by its two expands
/// carrying the matmul call's one span (`tier2::is_matmul_product`), a
/// heuristic until chelis#3125 marks the product structurally. This pins
/// where that heuristic must still answer correctly:
/// - two matmuls nested in one expression each report `matmul`;
/// - an authored `mul` of two expands at matmul's axes, on one line, reports
///   `mul`, because each authored expand has its own span.
#[test]
fn matmul_trap_naming_survives_nested_and_look_alike_products() {
    let ones = |count: &str| format!("to_tensor(map(fn (i: i64) -> 1.0f32, range(0i64, {count})))");
    let three = "tensor_to_scalar(sum(to_tensor([1i64, 2i64]), 0))";
    for (name, source, op, context) in [
        (
            "matmul_nested",
            format!(
                "def mm(a: tensor[*, *, f32], b: tensor[*, *, f32], c: tensor[*, *, f32]) -> tensor[*, *, f32] = matmul(matmul(a, b), c)\n\
                 def main() -> tensor[*, *, f32] = mm(reshape({}, [2i64, 2i64]), reshape({}, [2i64, {three}]), reshape({}, [2i64, 2i64]))\n",
                ones("4i64"),
                ones("6i64"),
                ones("4i64")
            ),
            "matmul",
            "matmul shared axis disagrees: lhs [2, 3] has 3 at axis 1, rhs [2, 2] has 2 at axis 0",
        ),
        (
            "matmul_lookalike",
            format!(
                "def f(p: tensor[*, f32], q: tensor[*, f32], n: i64) -> tensor[*, *, *, f32] = mul(expand(reshape(p, [2i64, n, 1i64]), 2i32, 2i64), expand(reshape(q, [1i64, 2i64, 2i64]), 0i32, 2i64))\n\
                 def main() -> tensor[*, *, *, f32] = f({}, {}, {three})\n",
                ones("6i64"),
                ones("4i64")
            ),
            "mul",
            "mul operands disagree at axis 1: lhs [2, 3, 2] has 3, rhs [2, 2, 2] has 2",
        ),
    ] {
        let run = parity::assert_lanes_agree(&source, name);
        assert_eq!(run.status, Some(1), "{name}: {run:?}");
        assert_eq!(
            run.failure,
            format!("numeric trap: domain in {op} at i64"),
            "{name}: {run:?}"
        );
        assert_eq!(run.context, context, "{name}: {run:?}");
    }
}
// ---------------------------------------------------------------------------
// [05-OP-9], [05-OP-10] pad_sequences and pad_sequences_to
// ---------------------------------------------------------------------------

/// Both operations admit every active data element dtype, `bool` included,
/// and move its bits unchanged on both lanes (an `f64` element keeps its
/// width), inside and outside a function, padding short
/// rows and truncating long ones.
#[test]
fn pad_sequences_moves_bool_elements_like_eval() {
    let source = "def padded(rows: List[List[bool]]) -> tensor[*, *, bool] = pad_sequences(rows, false)\n\
                  a = pad_sequences([[true], [false, true]], false)\n\
                  b = pad_sequences_to([[true, true, true], [false]], 2i64, true)\n\
                  c = padded([[false, true], [true]])\n\
                  d = pad_sequences_to([[1i8], [2i8, 3i8]], 3i64, 0i8)\n\
                  e = pad_sequences([[0.1f64], [0.2f64, 0.3f64]], 0.5f64)\n\
                  f = pad_sequences_to([[cast(0.1f32, f16)]], 2i64, cast(0.0f32, f16))\n";
    let run = parity::assert_lanes_agree(source, "padbool");
    assert_eq!(run.status, Some(0), "{run:?}");
    for expected in [
        "a = tensor(shape=[2, 2], data=[true, false, false, true])\n",
        "b = tensor(shape=[2, 2], data=[true, true, false, true])\n",
        "c = tensor(shape=[2, 2], data=[false, true, true, false])\n",
        "d = tensor(shape=[2, 3], data=[1, 0, 0, 2, 3, 0])\n",
        "e = tensor(shape=[2, 2], data=[0.1, 0.5, 0.2, 0.3])\n",
    ] {
        assert!(
            run.stdout.contains(expected),
            "missing `{expected}`: {run:?}"
        );
    }
}

/// `string` and `key` are not data element dtypes, so nested lists of them
/// are type errors, reported as such rather than as a target gate.
#[test]
fn pad_sequences_rejects_a_non_data_element_dtype_by_type() {
    for source in [
        "a = pad_sequences([[\"x\"], [\"y\", \"z\"]], \"\")\n",
        "a = pad_sequences_to([[\"x\"]], 2i64, \"\")\n",
        "k = key_from_seed(7i64)\na = pad_sequences([[k]], k)\n",
    ] {
        let stderr = build_rejection(source);
        assert!(stderr.contains("pad_sequences"), "{source}: {stderr}");
    }
}

/// A negative `width` is a negative result extent, so spec/04-type-system.md
/// section 4.7's non-negativity guard traps `Domain` in `pad_sequences_to`,
/// with one context line, on both lanes.
#[test]
fn pad_sequences_to_traps_a_negative_width_like_eval() {
    let width = "tensor_to_scalar(sum(to_tensor(map(fn (i: i64) -> -1i64, range(0i64, 2i64))), 0))";
    let source = format!(
        "def run(w: i64) -> tensor[*, *, i8] ! {{IO}} = {{\n\
         _ = print(\"before\")\n\
         pad_sequences_to([[1i8]], w, 0i8)\n\
         }}\n\
         a = run({width})\n"
    );
    let run = parity::assert_lanes_agree(&source, "padneg");
    assert_eq!(run.status, Some(1), "{run:?}");
    assert_eq!(
        run.failure, "numeric trap: domain in pad_sequences_to at i64",
        "{run:?}"
    );
    assert_eq!(
        run.context, "pad_sequences_to target extent at axis 1 is negative: -2",
        "{run:?}"
    );
}

// ---------------------------------------------------------------------------
// [05-OP-8] uniform_like bounds at the template dtype
// ---------------------------------------------------------------------------

/// `low` and `high` have the template's dtype `p` at every active float dtype,
/// and both lanes draw the same stored values from a folded template (the DAG
/// lane) and from a runtime-derived one (the host lane), with folded and
/// computed bounds alike.
#[test]
fn uniform_like_takes_bounds_at_the_template_dtype_like_eval() {
    let source = "def bc16(c: f16) -> tensor[3, f16] = to_tensor([c, c, c])\n\
                  def bcb(c: bf16) -> tensor[2, bf16] = to_tensor([c, c])\n\
                  def bc64(c: f64) -> tensor[2, f64] = to_tensor([c, c])\n\
                  a = uniform_like(key_from_seed(7i64), cast(to_tensor([0.0f32, 0.0f32, 0.0f32]), f16), cast(2.0f32, f16), cast(5.0f32, f16))\n\
                  b = uniform_like(key_from_seed(7i64), bc16(cast(0.0f32, f16)), cast(2.0f32, f16), cast(5.0f32, f16))\n\
                  c = uniform_like(key_from_seed(7i64), cast(to_tensor([0.0f32, 0.0f32]), bf16), cast(-1.0f32, bf16), cast(1.0f32, bf16))\n\
                  d = uniform_like(key_from_seed(7i64), bcb(cast(0.0f32, bf16)), cast(-1.0f32, bf16), cast(1.0f32, bf16))\n\
                  e = uniform_like(key_from_seed(7i64), to_tensor([0.0f64, 0.0f64]), 0.1f64, 0.9f64)\n\
                  f = uniform_like(key_from_seed(7i64), bc64(0.0f64), 0.1f64, 0.9f64)\n\
                  g = uniform_like(key_from_seed(7i64), to_tensor([0.0f32, 0.0f32]), 0.0f32, 1.0f32)\n\
                  def at16(x: f16) -> f16 = x\n\
                  def at64(x: f64) -> f64 = x\n\
                  h = uniform_like(key_from_seed(7i64), bc16(cast(0.0f32, f16)), at16(cast(2.0f32, f16)), at16(cast(5.0f32, f16)))\n\
                  i = uniform_like(key_from_seed(7i64), bc64(0.0f64), at64(0.1f64), at64(0.9f64))\n";
    let run = parity::assert_lanes_agree(source, "uniformwide");
    assert_eq!(run.status, Some(0), "{run:?}");
    let line = |name: &str| {
        run.stdout
            .lines()
            .find(|line| line.starts_with(&format!("{name} = ")))
            .unwrap_or_else(|| panic!("no `{name}` line: {run:?}"))
            .split_once(" = ")
            .unwrap()
            .1
            .to_string()
    };
    // The folded and runtime templates draw the same stream.
    assert_eq!(line("a"), line("b"), "{run:?}");
    assert_eq!(line("c"), line("d"), "{run:?}");
    assert_eq!(line("e"), line("f"), "{run:?}");
    // A bound the emitter cannot fold reaches the draw at the template dtype.
    assert_eq!(line("b"), line("h"), "{run:?}");
    assert_eq!(line("f"), line("i"), "{run:?}");
}

/// A bound at another dtype than the template's is a type error at every
/// float template dtype, including the f32 bounds the old signature fixed.
#[test]
fn uniform_like_rejects_a_bound_of_another_dtype_by_type() {
    for source in [
        "a = uniform_like(key_from_seed(7i64), to_tensor([0.0f64]), 0.0f32, 1.0f32)\n",
        "a = uniform_like(key_from_seed(7i64), cast(to_tensor([0.0f32]), f16), 0.0f32, 1.0f32)\n",
        "a = uniform_like(key_from_seed(7i64), to_tensor([0.0f32]), 0.0f64, 1.0f64)\n",
        "a = uniform_like(key_from_seed(7i64), to_tensor([0.0f32]), 0.0f32, 1.0f64)\n",
    ] {
        let stderr = build_rejection(source);
        assert!(
            stderr.contains("PrecisionMismatch: `uniform_like` argument")
                || stderr.contains("uniform_like bounds must have the template's"),
            "{source}: {stderr}"
        );
    }
}

// ---------------------------------------------------------------------------
// [05-OP-37] dropout through an alias, at bf16, and its pathwise adjoint
// ---------------------------------------------------------------------------

/// chelis#1295's dropout residue, under explicit keys:
/// - a function reached through a local alias dispatches `dropout` as a direct
///   call does (the evaluator once failed with `unknown runtime name dropout`);
/// - bf16 draws at its own dtype;
/// - the pathwise adjoint of `sum(dropout(k, x, r))` is the forward's scaled
///   mask, so it equals the draw over ones under the same key.
///
/// Both lanes print the same values.
#[test]
fn dropout_alias_bf16_and_adjoint_match_eval() {
    let source = "def thin(k: key, x: tensor[4, f32], rate: f32) -> tensor[4, f32] = dropout(k, x, rate)\n\
                  def aliased(k: key, x: tensor[4, f32]) -> tensor[4, f32] = {\n\
                  f = thin\n\
                  f(k, x, 0.25f32)\n\
                  }\n\
                  def loss(x: tensor[4, f32]) -> f32 = tensor_to_scalar(sum(dropout(key_from_seed(7i64), x, 0.25f32), 0))\n\
                  def loss16(x: tensor[4, bf16]) -> bf16 = tensor_to_scalar(sum(dropout(key_from_seed(7i64), x, cast(0.25f32, bf16)), 0))\n\
                  ones = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])\n\
                  ones16 = cast(ones, bf16)\n\
                  a = aliased(key_from_seed(7i64), ones)\n\
                  b = thin(key_from_seed(7i64), ones, 0.25f32)\n\
                  c = dropout(key_from_seed(7i64), cast(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]), bf16), cast(0.25f32, bf16))\n\
                  d = dropout(key_from_seed(7i64), ones16, cast(0.25f32, bf16))\n\
                  g = grad(loss)(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]))\n\
                  gb = grad(loss16)(cast(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]), bf16))\n";
    let run = parity::assert_lanes_agree(source, "dropoutwide");
    assert_eq!(run.status, Some(0), "{run:?}");
    let line = |name: &str| {
        run.stdout
            .lines()
            .find_map(|line| line.strip_prefix(&format!("{name} = ")))
            .unwrap_or_else(|| panic!("no `{name}` line: {run:?}"))
            .to_string()
    };
    // Non-vacuity: the key drops some elements and keeps others.
    assert_eq!(
        line("a"),
        "tensor(shape=[4], data=[0.0, 1.3333334, 1.3333334, 1.3333334])",
        "{run:?}"
    );
    assert_eq!(line("a"), line("b"), "{run:?}");
    assert_eq!(
        line("c"),
        "tensor(shape=[4], data=[0.0, 2.67, 4.0, 5.34])",
        "{run:?}"
    );
    assert_eq!(line("g"), line("a"), "{run:?}");
    assert_eq!(line("gb"), line("d"), "{run:?}");
}

/// A rate of another dtype than the input's is a type error, as [05-OP-37]
/// requires a same-dtype rate.
#[test]
fn dropout_rejects_a_rate_of_another_dtype_by_type() {
    let stderr = build_rejection(
        "d = dropout(key_from_seed(7i64), cast(to_tensor([1.0f32, 2.0f32]), bf16), 0.25f32)\n",
    );
    assert!(stderr.contains("PrecisionMismatch"), "{stderr}");
}

// ---------------------------------------------------------------------------
// spec/04 section 4.7: operand extents fixed by literals after inlining
// ---------------------------------------------------------------------------

/// `g` adds two `List` arguments through `to_tensor`, so their lengths are
/// visible only once a call with literal Lists is inlined. chelis#3115's two
/// root forms: a `def main` root and a top-level binding after an effect.
fn inlined_operand_lengths(lhs: &str, rhs: &str) -> [String; 2] {
    [
        format!(
            "def g(a: List[f32], b: List[f32]) -> tensor[*, f32] = add(to_tensor(a), to_tensor(b))\n\
             def main() -> tensor[*, f32] = g({lhs}, {rhs})\n"
        ),
        format!(
            "def g(a: List[f32], b: List[f32]) = add(to_tensor(a), to_tensor(b))\n\
             before = print(\"before\")\n\
             out = g({lhs}, {rhs})\n"
        ),
    ]
}

/// spec/04-type-system.md section 4.7: "Literals that become visible only
/// when a call is inlined prove it just the same when the lowered graph fixes
/// the claimed axis to a different extent: the program is rejected before
/// any execution". The compiled lane rejects at build with the checker's
/// `DimensionMismatch`, naming `add`, the operand and both extents, never an
/// internal invariant. (`chelis eval` still executes and traps; its
/// pre-execution rejection is chelis#3115's open half.)
#[test]
fn literal_operand_extents_fixed_after_inlining_are_rejected_at_build() {
    for source in inlined_operand_lengths("[1.0f32, 2.0f32]", "[1.0f32, 2.0f32, 3.0f32]") {
        let stderr = build_rejection(&source);
        assert!(
            stderr.contains(
                "DimensionMismatch: `add` argument 2, axis 0: expected 2, got 3 \
                 (extents fixed by literals after inlining)"
            ),
            "{source}: {stderr}"
        );
        assert!(!stderr.contains("invariant"), "{source}: {stderr}");
    }
}

/// The negative twins. Agreeing literal lengths build and compute on both
/// lanes, and a length the graph does not fix keeps section 4.7's run-time
/// `Domain` trap in `add` on both lanes.
#[test]
fn agreeing_or_runtime_operand_lengths_are_not_refuted_at_build() {
    for (index, source) in inlined_operand_lengths("[1.0f32, 2.0f32]", "[3.0f32, 4.0f32]")
        .iter()
        .enumerate()
    {
        let run = parity::assert_lanes_agree(source, &format!("agreeing{index}"));
        assert_eq!(run.status, Some(0), "{run:?}");
        assert!(run.stdout.contains("data=[4.0, 6.0]"), "{run:?}");
    }
    let runtime = "map(fn (i: i64) -> 1.0f32, range(0i64, tensor_to_scalar(sum(to_tensor([1i64, 1i64]), 0))))";
    for (index, source) in inlined_operand_lengths(runtime, "[1.0f32, 2.0f32, 3.0f32]")
        .iter()
        .enumerate()
    {
        let run = parity::assert_lanes_agree(source, &format!("runtimelen{index}"));
        assert_eq!(run.status, Some(1), "{run:?}");
        assert_eq!(run.failure, "numeric trap: domain in add at i64", "{run:?}");
        assert_eq!(
            run.context, "add operands disagree at axis 0: lhs [2] has 2, rhs [3] has 3",
            "{run:?}"
        );
    }
}
