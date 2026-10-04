//! chelis#3128: the shared Surf printer parenthesizes every pipe operand and
//! ascription operand whose bare text would re-parse as a different program.
//!
//! Each case is authored with the grouping it needs. The formatter must print
//! it as `expected`, that text must be a formatter fixed point, and both the
//! formatter path (Surf to Surf) and the resugar path (Deep to Surf) must
//! re-parse to the authored program's Deep.

use chelis_deep::printer::print_canonical;
use chelis_surf::desugar::desugar_program;
use chelis_surf::format::{format_program, format_source};
use chelis_surf::parser::parse_str;
use chelis_surf::resugar::{normalize_deep_for_surface_roundtrip, resugar_program};

fn canonical_deep(source: &str) -> String {
    let decls = parse_str(source).unwrap_or_else(|error| panic!("{error}\n{source}"));
    let deep = desugar_program(&decls).unwrap_or_else(|error| panic!("{error}\n{source}"));
    print_canonical(
        &normalize_deep_for_surface_roundtrip(&deep).expect("valid round-trip metadata"),
    )
}

fn assert_round_trips(body: &str, expected: &str) {
    let source = format!("def main(x: f32) -> f32 = {body}\n");
    let authored = canonical_deep(&source);

    let formatted = format_source(&source).unwrap_or_else(|error| panic!("{error}\n{source}"));
    assert_eq!(
        formatted,
        format!("def main(x: f32) -> f32 = {expected}\n"),
        "canonical text for `{body}`"
    );
    assert_eq!(
        format_source(&formatted).expect("formatted text reparses"),
        formatted,
        "formatted text is a fixed point for `{body}`"
    );
    assert_eq!(
        canonical_deep(&formatted),
        authored,
        "formatting changed the Deep of `{body}`"
    );

    let deep = desugar_program(&parse_str(&source).expect("parse")).expect("desugar");
    let resugared = format_program(&resugar_program(&deep).expect("resugar"));
    assert_eq!(
        canonical_deep(&resugared),
        authored,
        "resugaring changed the Deep of `{body}`: {resugared}"
    );
}

#[test]
fn an_open_tailed_seed_is_parenthesized() {
    for (body, expected) in [
        (
            "(if x > 0.0 then x else 1.0) |> f",
            "(if (x > 0.0) then x else 1.0) |> f",
        ),
        ("(fn (v: f32) -> v) |> g(x)", "(fn (v: f32) -> v) |> g(x)"),
        ("(x |> f) |> g", "(x |> f) |> g"),
    ] {
        assert_round_trips(body, expected);
    }
}

#[test]
fn an_open_tailed_stage_before_another_stage_is_parenthesized() {
    for (body, expected) in [
        (
            "x |> (if x > 0.0 then f else g) |> h",
            "x |> (if (x > 0.0) then f else g) |> h",
        ),
        (
            "x |> (fn (v: f32) -> v) |> h",
            "x |> (fn (v: f32) -> v) |> h",
        ),
    ] {
        assert_round_trips(body, expected);
    }
}

#[test]
fn a_stage_the_parser_reads_as_a_prefix_is_parenthesized_in_any_position() {
    for (body, expected) in [
        ("x |> (y |> f)", "x |> (y |> f)"),
        ("x |> (y |> f) |> h", "x |> (y |> f) |> h"),
        ("x |> (r.f)", "x |> (r.f)"),
        ("x |> ((f, g).0) |> h", "x |> ((f, g).0) |> h"),
        ("x |> (M.f(y))", "x |> (M.f(y))"),
        ("x |> (r with { f: g })", "x |> (r with { f: g })"),
    ] {
        assert_round_trips(body, expected);
    }
}

#[test]
fn an_open_tailed_ascription_operand_is_parenthesized() {
    for (body, expected) in [
        (
            "((if x > 0.0 then x else 1.0) : f32)",
            "((if (x > 0.0) then x else 1.0) : f32)",
        ),
        (
            "((fn (v: f32) -> v) : f32 -> f32)(x)",
            "((fn (v: f32) -> v) : f32 -> f32)(x)",
        ),
        ("((-x) : f32)", "((-x) : f32)"),
        ("((&x) : f32)", "((&x) : f32)"),
        (
            "((x |> (fn (v: f32) -> v)) : f32)",
            "((x |> (fn (v: f32) -> v)) : f32)",
        ),
    ] {
        assert_round_trips(body, expected);
    }
}

#[test]
fn closed_operands_stay_bare() {
    // Negative controls: none of these needs grouping, and the printer adds
    // none.
    for body in [
        "(-x) |> f",
        "(x + 1.0) |> f",
        "x |> f |> (fn (v: f32) -> v)",
        "x |> (if (x > 0.0) then f else g)",
        "x |> f(y) |> g",
        "((x |> f) : f32)",
        // A pipe whose last stage is grouped is closed, so it needs no
        // further grouping as an ascription operand.
        "((x |> (y |> (fn (v: f32) -> v))) : f32)",
        "(x : f32) |> f",
        "x |> realize |> f",
        "x |> cast(f64) |> f",
        "(r with { f: x }) |> g",
        "(x, x) |> f",
        "do { x; x } |> f",
    ] {
        assert_round_trips(body, body);
    }
}
