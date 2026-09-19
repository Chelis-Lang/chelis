//! chelis#1537 PP9 / [04-TOT-5]: the four public checker entries apply the
//! same semantic pass set, in the same diagnostic order, on both admitted
//! program carriers.
//!
//! The first carrier is the ordinary unstamped tree produced by Surf
//! desugaring or Deep parsing. Every row admitted by the stamped file boundary
//! is also printed canonically and rerun in that carrier. The one legacy
//! malformed-form control runs through all four checker entries and separately
//! proves the stamped constructor rejects it at admission. No row is ignored.

use chelis_deep::{
    Atom, DeepTag, Expr, List, Metadata, Span, parse_and_stamp_file,
    parser::parse_str as parse_deep, printer,
};
use chelis_surf::{desugar::desugar_program, parser::parse_str as parse_surf};
use chelis_types::errors::CheckError;
use chelis_types::{
    check_ir_fitness, check_ir_program, check_program, check_typed_program, infer_ir_program,
    infer_program,
};

#[derive(Clone, Copy)]
enum Source {
    Surf(&'static str),
    Deep(&'static str),
    LegacyMalformedDef,
}

#[derive(Clone, Copy)]
enum Expected {
    Accept,
    Reject(&'static str),
}

#[derive(Clone, Copy)]
struct Row {
    name: &'static str,
    source: Source,
    expected: Expected,
}

fn unstamped(source: Source) -> Vec<Expr> {
    match source {
        Source::Surf(source) => {
            let declarations = parse_surf(source)
                .unwrap_or_else(|error| panic!("Surf fixture must parse:\n{source}\n{error}"));
            desugar_program(&declarations).expect("Surf fixture must desugar")
        }
        Source::Deep(source) => parse_deep(source)
            .unwrap_or_else(|error| panic!("Deep fixture must parse:\n{source}\n{error}")),
        Source::LegacyMalformedDef => {
            let span = Span::new(0, 0);
            vec![Expr::List(
                List {
                    elements: vec![
                        Expr::Atom(Atom::Tag(DeepTag::Def), span),
                        Expr::Map(Metadata::default(), span),
                        Expr::Atom(Atom::Name("only_name".to_string()), span),
                    ],
                },
                span,
            )]
        }
    }
}

fn stamped(exprs: &[Expr]) -> Vec<Expr> {
    let canonical = printer::print_canonical(exprs);
    parse_and_stamp_file(&canonical)
        .unwrap_or_else(|error| panic!("canonical fixture must stamp:\n{canonical}\n{error}"))
}

fn diagnostics(errors: &[CheckError]) -> Vec<String> {
    errors
        .iter()
        .map(|error| {
            format!(
                "{}|{}|{}",
                error.kind.diagnostic_name(),
                error.message,
                error.suggestions.join("\u{1f}")
            )
        })
        .collect()
}

fn check_ir_diagnostics(exprs: &[Expr]) -> Vec<String> {
    check_ir_program(exprs)
        .err()
        .map(|result| diagnostics(&result.errors))
        .unwrap_or_default()
}

fn check_typed_diagnostics(exprs: &[Expr]) -> Vec<String> {
    check_typed_program(exprs)
        .err()
        .map(|result| diagnostics(&result.errors))
        .unwrap_or_default()
}

fn infer_ir_diagnostics(exprs: &[Expr]) -> Vec<String> {
    diagnostics(&infer_ir_program(exprs).errors)
}

fn infer_diagnostics(exprs: &[Expr]) -> Vec<String> {
    diagnostics(&infer_program(exprs).errors)
}

fn assert_entry_parity(row: Row, carrier: &str, exprs: &[Expr]) -> Vec<String> {
    let check_ir = check_ir_diagnostics(exprs);
    let check_typed = check_typed_diagnostics(exprs);
    let infer_ir = infer_ir_diagnostics(exprs);
    let infer = infer_diagnostics(exprs);
    for (entry, actual) in [
        ("check_typed_program", &check_typed),
        ("infer_ir_program", &infer_ir),
        ("infer_program", &infer),
    ] {
        assert_eq!(
            actual, &check_ir,
            "{} ({carrier}): {entry} must match check_ir_program's ordered diagnostics\n\
             check_ir={check_ir:#?}\ncheck_typed={check_typed:#?}\n\
             infer_ir={infer_ir:#?}\ninfer={infer:#?}",
            row.name,
        );
    }

    let check_report = check_program(exprs);
    let ir_report = check_ir_fitness(exprs);
    assert_eq!(
        diagnostics(&check_report.errors),
        infer,
        "{} ({carrier}): check_program must expose infer_program's diagnostics",
        row.name
    );
    assert_eq!(
        diagnostics(&ir_report.errors),
        infer_ir,
        "{} ({carrier}): check_ir_fitness must expose infer_ir_program's diagnostics",
        row.name
    );
    if check_ir.is_empty() {
        assert_eq!(check_report.score, 1.0, "{} ({carrier})", row.name);
        assert_eq!(ir_report.score, 1.0, "{} ({carrier})", row.name);
    } else {
        assert!(check_report.score < 1.0, "{} ({carrier})", row.name);
        assert!(ir_report.score < 1.0, "{} ({carrier})", row.name);
    }
    check_ir
}

fn assert_row(row: Row) {
    let plain = unstamped(row.source);
    let plain_diagnostics = assert_entry_parity(row, "unstamped", &plain);
    match row.source {
        Source::LegacyMalformedDef => {
            let canonical = printer::print_canonical(&plain);
            let error = parse_and_stamp_file(&canonical)
                .expect_err("the stamped constructor must reject wrong known-tag arity");
            assert!(
                error.to_string().contains("wrong child count"),
                "unexpected stamped-admission error: {error}"
            );
        }
        Source::Surf(_) | Source::Deep(_) => {
            let stamped = stamped(&plain);
            let stamped_diagnostics = assert_entry_parity(row, "stamped", &stamped);
            assert_eq!(
                stamped_diagnostics, plain_diagnostics,
                "{}: admitted carriers must preserve ordered diagnostics",
                row.name
            );
        }
    }
    match row.expected {
        Expected::Accept => assert!(
            plain_diagnostics.is_empty(),
            "{} must remain accepted: {plain_diagnostics:#?}",
            row.name
        ),
        Expected::Reject(needle) => assert!(
            plain_diagnostics
                .iter()
                .any(|diagnostic| diagnostic.contains(needle)),
            "{} must reject with {needle:?}: {plain_diagnostics:#?}",
            row.name
        ),
    }
}

const ROWS: &[Row] = &[
    Row {
        name: "direct recursion is a language-level function",
        source: Source::Surf("def recurse(x: i32) -> i32 = recurse(x)\n"),
        expected: Expected::Accept,
    },
    Row {
        name: "mutual recursion is a language-level function group",
        source: Source::Surf(
            "def left(x: i32) -> i32 = right(x)\n\
             def right(x: i32) -> i32 = left(x)\n",
        ),
        expected: Expected::Accept,
    },
    Row {
        name: "recursive function with an authored binder",
        source: Source::Surf("def recurse[a](x: a) -> a = recurse(x)\n"),
        expected: Expected::Accept,
    },
    Row {
        name: "module-wrapped recursion",
        source: Source::Surf(
            "module Recursive\n\
             def recurse(x: i32) -> i32 = recurse(x)\n",
        ),
        expected: Expected::Accept,
    },
    Row {
        name: "defsig-less forward function uses its compiler-authored stamp",
        source: Source::Deep(
            "(def {} caller (fn {} (params {}) (app {} (var {} helper))))\n\n\
             (def {} helper (fn {type: (t-fn {} (t-prim {} i32))} (params {}) \
             (lit {type: (t-prim {} i32)} 1)))\n",
        ),
        expected: Expected::Accept,
    },
    Row {
        name: "conv rejects a statically nonpositive stride",
        source: Source::Surf(
            "def f(x: tensor[1, 1, 4, f32], k: tensor[1, 1, 2, f32]) = \
             conv(x, k, [0i64], [(0i64, 0i64)])\n",
        ),
        expected: Expected::Reject("positive stride"),
    },
    Row {
        name: "conv rejects statically negative padding",
        source: Source::Surf(
            "def f(x: tensor[1, 1, 4, f32], k: tensor[1, 1, 2, f32]) = \
             conv(x, k, [1i64], [(-1i64, 0i64)])\n",
        ),
        expected: Expected::Reject("non-negative padding"),
    },
    Row {
        name: "conv static range checks survive an untracked let shape",
        source: Source::Surf(
            "def f(x: tensor[1, 1, 4, f32], k: tensor[1, 1, 2, f32]) = {\n\
             \x20 y = expand(x, 0i32, 2i64)\n\
             \x20 conv(y, k, [0i64], [(0i64, 0i64)])\n\
             }\n",
        ),
        expected: Expected::Reject("positive stride"),
    },
    Row {
        name: "conv admits a runtime stride",
        source: Source::Surf(
            "def f(x: tensor[1, 1, 4, f32], k: tensor[1, 1, 2, f32], \
             strides: List[i64]) = conv(x, k, strides, [(0i64, 0i64)])\n",
        ),
        expected: Expected::Accept,
    },
    Row {
        name: "conv admits runtime padding",
        source: Source::Surf(
            "def f(x: tensor[1, 1, 4, f32], k: tensor[1, 1, 2, f32], \
             padding: List[(i64, i64)]) = conv(x, k, [1i64], padding)\n",
        ),
        expected: Expected::Accept,
    },
    Row {
        name: "vmap rejects an element-derived movement extent",
        source: Source::Surf(
            "def g[n, m](x: tensor[n, f32]) -> tensor[m, f32] = {\n\
             \x20 end = cast(tensor_to_scalar(sum(x, 0i32)), i64)\n\
             \x20 shrink(x, [[0i64, end]])\n\
             }\n\
             out = vmap(g)(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]))\n",
        ),
        expected: Expected::Reject("batch_varying_extent"),
    },
    Row {
        name: "mean admits a symbolic selected extent",
        source: Source::Surf(
            "def f[n](x: tensor[32, n, f32]) -> tensor[32, f32] = mean(x, 1i32)\n",
        ),
        expected: Expected::Accept,
    },
    Row {
        name: "layer_norm admits a symbolic hidden extent",
        source: Source::Surf(
            "def f(x: tensor[batch, hidden, f32], gamma: tensor[hidden, f32], \
             beta: tensor[hidden, f32]) -> tensor[batch, hidden, f32] = \
             layer_norm(x, gamma, beta, 0.00001f32)\n",
        ),
        expected: Expected::Accept,
    },
    Row {
        name: "elementwise rank mismatch remains a type error",
        source: Source::Surf("def f(a: tensor[2, 3, f32], b: tensor[3, f32]) = add(a, b)\n"),
        expected: Expected::Reject("rank mismatch"),
    },
    Row {
        name: "unknown Deep tag has one ordinary checker owner",
        source: Source::Deep("(def {} bad (bogus-tag {}))\n"),
        expected: Expected::Reject("unknown Deep tag"),
    },
    Row {
        name: "top-level eager binding cycle",
        source: Source::Surf("a = b\nb = a\n"),
        expected: Expected::Reject("CycleDetected"),
    },
    Row {
        name: "malformed Deep arity remains rejected",
        source: Source::LegacyMalformedDef,
        expected: Expected::Reject("MalformedForm"),
    },
    Row {
        name: "literal reduction axis bounds remain checked",
        source: Source::Surf("def f(x: tensor[2, f32]) -> f32 = mean(x, 4i32)\n"),
        expected: Expected::Reject("out of bounds"),
    },
    Row {
        name: "defsig and body mismatch remains rejected",
        source: Source::Deep(
            "(defsig {} k (t-prim {} f32))\n\n\
             (def {} k (lit {type: (t-prim {} i32)} 1))\n",
        ),
        expected: Expected::Reject("body doesn't match declared signature"),
    },
    Row {
        name: "ascribed self-reference remains an external input",
        source: Source::Surf("x = (x : tensor[4, f32])\n"),
        expected: Expected::Accept,
    },
    Row {
        name: "literal conv metadata remains accepted",
        source: Source::Surf(
            "def f(x: tensor[1, 1, 4, f32], k: tensor[1, 1, 2, f32]) \
             -> tensor[1, 1, 3, f32] = conv(x, k, [1i64], [(0i64, 0i64)])\n",
        ),
        expected: Expected::Accept,
    },
    Row {
        name: "ordinary scalar control",
        source: Source::Surf("answer: i32 = add(40, 2)\n"),
        expected: Expected::Accept,
    },
    Row {
        name: "ordinary tensor control",
        source: Source::Surf("def f[n](x: tensor[n, f32]) -> tensor[n, f32] = relu(x)\n"),
        expected: Expected::Accept,
    },
];

#[test]
fn pp9_named_corpus_has_four_entry_two_carrier_ordered_parity() {
    assert_eq!(
        ROWS.len(),
        23,
        "the PP9 corpus must not silently lose a row"
    );
    for row in ROWS {
        assert_row(*row);
    }
}
