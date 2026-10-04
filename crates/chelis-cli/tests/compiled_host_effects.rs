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
