//! chelis#3265: a by-name record literal evaluates its field expressions in
//! written order in every lane (spec/03-deep-syntax.md §4.4), while the
//! constructor stores its slots in declared order.
//!
//! The compiled C lane evaluated the fields in declared order, so a record
//! literal written in a different order trapped on a different field, and
//! printed its effects in a different order, than `chelis eval`.
//!
//! Every test runs `eval --file` and `build --target c`, executes the built
//! program, and requires both lanes to report exactly the expected trap or
//! output. The controls are the negative direction: a record literal written
//! in declared order traps on its first-written (and first-declared) field,
//! tuples keep their order, and a permuted literal without traps or effects
//! still fills each slot with its own field's value.

#[path = "common/mod.rs"]
mod common;

use assert_cmd::Command;
use std::fs;
use std::process::Command as StdCommand;
use tempfile::tempdir;

/// Exit code, stdout and stderr of one lane.
type Outcome = (i32, String, String);

fn outcome(output: std::process::Output) -> Outcome {
    (
        output.status.code().expect("exit code"),
        String::from_utf8(output.stdout).expect("UTF-8 stdout"),
        String::from_utf8(output.stderr).expect("UTF-8 stderr"),
    )
}

/// Run `source` in both lanes: `eval --file`, then the built C program.
fn run_lanes(source: &str, name: &str) -> (Outcome, Outcome) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    fs::write(&path, source).expect("source");
    let interpreted = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("eval runs");
    Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
    let native = StdCommand::new(out_dir.join(name))
        .output()
        .expect("compiled program runs");
    (outcome(interpreted), outcome(native))
}

/// Both lanes must fail with exactly the `index` trap of list index `index`.
fn assert_lanes_trap_on(source: &str, name: &str, index: i64) {
    let message = format!("index {index} out of bounds for list of len 2\n");
    let (interpreted, native) = run_lanes(source, name);
    assert_eq!(
        interpreted,
        (1, String::new(), format!("error: {message}")),
        "{name}: eval"
    );
    assert_eq!(native, (1, String::new(), message), "{name}: compiled C");
}

/// Both lanes must succeed and print exactly `expected`.
fn assert_lanes_print(source: &str, name: &str, expected: &[&str]) {
    let (interpreted, native) = run_lanes(source, name);
    let expected = (0, format!("{}\n", expected.join("\n")), String::new());
    assert_eq!(interpreted, expected, "{name}: eval");
    assert_eq!(native, expected, "{name}: compiled C");
}

const S_F32: &str = "type S =\n  | S { lo: f32, hi: f32 }\n";
const S_STRING: &str = "type S =\n  | S { lo: string, hi: string }\n";
const SHOW: &str = "def show(s: S) -> string =\n  match s with {\n    \
                    | S { lo, hi } => string_concat(lo, hi)\n  }\n";

/// `h` builds an `S` whose fields index a two-element list at `i` and `j`,
/// with `literal` naming them, then returns `result` from it.
fn trap_program(literal: &str, result: &str) -> String {
    format!(
        "{S_F32}def h(x: f32, i: i64, j: i64) -> f32 = {{\n  \
         xs: List[f32] = [x, mul(x, x)]\n{result}\n}}\nbad = h(3.0f32, 5i64, 7i64)\n",
        result = result.replace("LITERAL", literal)
    )
}

const WRITTEN_HI_FIRST: &str = "S { hi: index(xs, i), lo: index(xs, j) }";

#[test]
fn the_first_written_field_traps_first() {
    // The issue's witness: `hi` is written first (index 5), `lo` is declared
    // first (index 7). Before the fix the C lane trapped on 7.
    let source = trap_program(
        WRITTEN_HI_FIRST,
        "  match LITERAL with {\n    | S { lo, hi } => lo\n  }",
    );
    assert_lanes_trap_on(&source, "record_order", 5);
}

#[test]
fn three_fields_trap_in_written_not_declared_or_alphabetical_order() {
    // Declared `m, z, a`; written `z, a, m`; alphabetical `a, m, z`. Written
    // order traps on 5, declared order on 7, alphabetical order on 6.
    let source = "type T =\n  | T { m: f32, z: f32, a: f32 }\n\
                  def g(x: f32, i: i64, j: i64, k: i64) -> f32 = {\n  \
                  xs: List[f32] = [x, mul(x, x)]\n  \
                  match T { z: index(xs, i), a: index(xs, j), m: index(xs, k) } with {\n    \
                  | T { m, z, a } => m\n  }\n}\nbad3 = g(3.0f32, 5i64, 6i64, 7i64)\n";
    assert_lanes_trap_on(source, "record_order3", 5);
}

#[test]
fn written_order_holds_however_the_record_is_read() {
    for (name, result) in [
        (
            "positional_pattern",
            "  match LITERAL with {\n    | S(lo, hi) => lo\n  }",
        ),
        ("field_access", "  s = LITERAL\n  s.lo"),
        (
            "lo_never_read",
            "  match LITERAL with {\n    | S { lo, hi } => hi\n  }",
        ),
    ] {
        assert_lanes_trap_on(&trap_program(WRITTEN_HI_FIRST, result), name, 5);
    }
}

#[test]
fn a_declared_order_literal_traps_on_its_first_field() {
    // Control: written in declared order, `lo` (index 7) comes first in both
    // orders, so both lanes trap on 7, not on the second field's 5.
    let source = trap_program(
        "S { lo: index(xs, j), hi: index(xs, i) }",
        "  match LITERAL with {\n    | S { lo, hi } => lo\n  }",
    );
    assert_lanes_trap_on(&source, "declared_order_trap", 7);
}

#[test]
fn field_effects_happen_in_written_order() {
    // The issue's output witness: the value is unchanged, and the effect of
    // the field written first is printed first.
    let source = format!(
        "{S_STRING}{SHOW}def h() -> string ! {{ IO }} = \
         show(S {{ hi: debug(\"hi-written-first\"), lo: debug(\"lo-declared-first\") }})\n\
         out = h()\n"
    );
    assert_lanes_print(
        &source,
        "record_order_debug",
        &[
            "hi-written-first",
            "lo-declared-first",
            "out = lo-declared-firsthi-written-first",
        ],
    );
}

#[test]
fn nested_and_sequential_permuted_literals_each_keep_written_order() {
    // Each permuted literal binds its own field locals: two in sequence, two
    // nested inside an outer permuted literal, and one in a match arm.
    let source = format!(
        "{S_STRING}type P =\n  | P {{ a: S, b: S }}\n{SHOW}\
         def h() -> string ! {{ IO }} = {{\n  \
         s1 = S {{ hi: debug(\"1\"), lo: debug(\"2\") }}\n  \
         s2 = S {{ hi: debug(\"3\"), lo: debug(\"4\") }}\n  \
         p = P {{ b: S {{ hi: debug(\"5\"), lo: debug(\"6\") }}, a: S {{ hi: debug(\"7\"), lo: s1.lo }} }}\n  \
         r = match p with {{\n    | P {{ a, b }} => S {{ hi: show(b), lo: show(a) }}\n  }}\n  \
         string_concat(string_concat(show(r), show(s2)), show(s1))\n}}\nout = h()\n"
    );
    assert_lanes_print(
        &source,
        "record_order_nested",
        &["1", "2", "3", "4", "5", "6", "7", "out = 27654321"],
    );
}

#[test]
fn field_effects_keep_written_order_under_access_and_positional_patterns() {
    let source = format!(
        "{S_STRING}def by_access() -> string ! {{ IO }} = {{\n  \
         s = S {{ hi: debug(\"hi-1\"), lo: debug(\"lo-1\") }}\n  s.lo\n}}\n\
         def by_position() -> string ! {{ IO }} =\n  \
         match S {{ hi: debug(\"hi-2\"), lo: debug(\"lo-2\") }} with {{\n    \
         | S(lo, hi) => hi\n  }}\nout = by_access()\nout2 = by_position()\n"
    );
    assert_lanes_print(
        &source,
        "record_order_debug_reads",
        &["hi-1", "lo-1", "hi-2", "lo-2", "out = lo-1", "out2 = hi-2"],
    );
}

#[test]
fn declared_order_literals_and_tuples_keep_their_order() {
    // The issue's controls, unchanged by the fix.
    let source = format!(
        "{S_STRING}{SHOW}def h() -> string ! {{ IO }} = \
         show(S {{ lo: debug(\"lo-written-first\"), hi: debug(\"hi-written-second\") }})\n\
         def t() -> string ! {{ IO }} = {{\n  \
         p = (debug(\"tuple-slot0-written-first\"), debug(\"tuple-slot1-written-second\"))\n  \
         string_concat(p.0, p.1)\n}}\nout = h()\nout2 = t()\n"
    );
    assert_lanes_print(
        &source,
        "record_order_debug_control",
        &[
            "lo-written-first",
            "hi-written-second",
            "tuple-slot0-written-first",
            "tuple-slot1-written-second",
            "out = lo-written-firsthi-written-second",
            "out2 = tuple-slot0-written-firsttuple-slot1-written-second",
        ],
    );
}

#[test]
fn a_permuted_literal_fills_each_declared_slot_with_its_own_value() {
    // No traps or effects: every slot must hold its own field's value,
    // whether read by name, by position or as a printed global, and an
    // owned argument bound into a permuted field stays usable afterwards.
    let source = "type T =\n  | T { m: f32, z: f32, a: string }\n\
                  def by_name(x: f32) -> f32 = {\n  \
                  t = T { z: mul(x, 2.0f32), a: \"aa\", m: add(x, 1.0f32) }\n  \
                  match t with {\n    | T { m, z, a } => sub(z, m)\n  }\n}\n\
                  def by_position(x: f32) -> string = {\n  \
                  t = T { a: \"aa\", z: mul(x, 2.0f32), m: add(x, 1.0f32) }\n  \
                  match t with {\n    | T(m, z, a) => string_concat(a, \"!\")\n  }\n}\n\
                  def owned(s: string) -> string = {\n  \
                  t = T { a: s, z: 2.0f32, m: 1.0f32 }\n  \
                  string_concat(t.a, s)\n}\n\
                  d = by_name(3.0f32)\nout = by_position(3.0f32)\nown = owned(\"o\")\n\
                  v = T { a: \"s\", z: 2.0f32, m: 1.0f32 }\n";
    assert_lanes_print(
        source,
        "record_order_values",
        &[
            "d = 2.0",
            "out = aa!",
            "own = oo",
            "v.m = 1.0",
            "v.z = 2.0",
            "v.a = s",
        ],
    );
}
