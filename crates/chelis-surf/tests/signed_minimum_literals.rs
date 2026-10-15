//! [04-LIT-2], Surf P10: direct unary-minus signed minima finalize at their
//! own dtype, without binding the positive magnitude or changing the unary AST.
use chelis_deep::printer::print_canonical;
use chelis_surf::desugar::desugar_program;
use chelis_surf::format::{format_program, format_source};
use chelis_surf::parser::parse_str;
use chelis_surf::resugar::{normalize_deep_for_surface_roundtrip, resugar_program};

fn deep(source: &str) -> Vec<chelis_deep::ast::Expr> {
    desugar_program(&parse_str(source).expect("Surf parses")).expect("Surf desugars")
}

#[test]
fn every_signed_minimum_is_one_exact_literal_in_an_ordinary_operand() {
    for (dtype, literal, value) in [
        ("i8", "-128i8", "-128"),
        ("i16", "-32768i16", "-32768"),
        ("i32", "-2147483648i32", "-2147483648"),
        ("i64", "-9223372036854775808i64", "-9223372036854775808"),
        ("i32", "-2147483648", "-2147483648"),
    ] {
        let syntax = parse_str(&format!("answer = {literal}\n")).unwrap();
        assert!(
            matches!(
                &syntax[0],
                chelis_surf::ast::Decl::LetDef {
                    value: chelis_surf::ast::Expr::Unary(chelis_surf::ast::UnaryOp::Neg, _, _),
                    ..
                }
            ),
            "{syntax:?}"
        );
        let text = print_canonical(&deep(&format!("answer = identity({literal})\n")));
        assert!(text.contains(&format!(" {value})")), "{text}");
        assert!(text.contains(&format!("(t-prim {{}} {dtype})")), "{text}");
        assert!(!text.contains("(var {} neg)"), "{text}");
        if dtype != "i64" {
            assert!(
                !text.contains("(t-prim {} i64)"),
                "no implicit widening: {text}"
            );
        }
    }
}

#[test]
fn grouping_casts_and_normalized_roundtrip_preserve_each_minimum() {
    for (dtype, magnitude) in [
        ("i8", "128"),
        ("i16", "32768"),
        ("i32", "2147483648"),
        ("i64", "9223372036854775808"),
    ] {
        // Grouping the complete expression or only its magnitude preserves
        // the same direct unary operand, including the i64 parser sentinel.
        for literal in [
            format!("(-{magnitude}{dtype})"),
            format!("-(({magnitude}{dtype}))"),
            format!("-((\n{magnitude}{dtype}\n))"),
            format!("cast(-{magnitude}, {dtype})"),
            format!("cast(-{magnitude}{dtype}, {dtype})"),
        ] {
            let source = format!("answer = identity({literal})\n");
            let formatted = format_source(&source).unwrap();
            assert_eq!(formatted, format_source(&formatted).unwrap());
            let before = deep(&source);
            let resugared = format_program(&resugar_program(&before).unwrap());
            let after = deep(&resugared);
            assert_eq!(
                print_canonical(&normalize_deep_for_surface_roundtrip(&before).unwrap()),
                print_canonical(&normalize_deep_for_surface_roundtrip(&after).unwrap()),
                "{source}\n{resugared}"
            );
        }
    }
    let text = print_canonical(&deep("answer = identity(-((2147483648i32)))\n"));
    assert!(text.contains(" -2147483648)"), "{text}");
}

#[test]
fn positive_magnitudes_explicit_neg_calls_and_below_minimum_are_not_folded() {
    for (dtype, magnitude, below) in [
        ("i8", "128", "129"),
        ("i16", "32768", "32769"),
        ("i32", "2147483648", "2147483649"),
    ] {
        let positive = print_canonical(&deep(&format!("answer = identity({magnitude}{dtype})\n")));
        assert!(positive.contains(&format!(" {magnitude})")), "{positive}");
        let named = print_canonical(&deep(&format!("answer = neg({magnitude}{dtype})\n")));
        assert!(
            named.contains("(var {") && named.contains(" neg)"),
            "{named}"
        );
        assert!(!named.contains(&format!(" -{magnitude})")), "{named}");
        let below = print_canonical(&deep(&format!("answer = identity(-{below}{dtype})\n")));
        assert!(below.contains("(var {} neg)"), "{below}");
    }
    let nested = print_canonical(&deep("answer = -(-128i8)\n"));
    assert!(
        nested.contains("(var {} neg)") && nested.contains(" -128)"),
        "{nested}"
    );
}

#[test]
fn i64_magnitude_grouping_cannot_cross_expression_barriers() {
    for expression in [
        "(9223372036854775808i64)",
        "neg((9223372036854775808i64))",
        "-((9223372036854775808i64 + 0i64))",
        "-((9223372036854775808i64 : i64))",
        "-((identity(9223372036854775808i64)))",
        "-((9223372036854775809i64))",
        "-((9223372036854775808i64)",
        "-((9223372036854775808i64)))",
    ] {
        let source = format!("answer = {expression}\n");
        assert!(parse_str(&source).is_err(), "{source}");
    }
}
