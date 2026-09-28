//! Span-survival S1 tests for `Expr::span_id()`.
//!
//! Per `spec/design/chelis_span_survival.md` §3 (S1) and
//! `spec/03-deep-syntax.md` §1.1.1: the chelis Deep parser/printer must
//! preserve the `span` metadata key end-to-end through parse → reprint →
//! re-parse, and the `Expr::span_id()` accessor reads `meta["span"]` from
//! a node's metadata map. These tests lock that contract and exercise the
//! negative parity required by the repo CLAUDE.md.

use chelis_deep::parser::parse_str;
use chelis_deep::printer::print_canonical;
use chelis_deep::{Atom, Expr};
use std::collections::BTreeSet;
use std::time::Instant;

/// Walk the AST in pre-order and collect every span ID present on a List
/// node via `Expr::span_id()`.
fn collect_span_ids(exprs: &[Expr]) -> Vec<String> {
    let mut acc = Vec::new();
    for expr in exprs {
        collect_span_ids_one(expr, &mut acc);
    }
    acc
}

fn collect_span_ids_one(expr: &Expr, acc: &mut Vec<String>) {
    if let Some(id) = expr.span_id() {
        acc.push(id.to_string());
    }
    match expr {
        Expr::Node(node, _) => {
            // Recurse into metadata values
            node.meta()
                .visit_expressions(&mut |value, _| collect_span_ids_one(value, acc));
            // Recurse into children
            for child in node.children_slice() {
                collect_span_ids_one(child, acc);
            }
        }
        Expr::BareList(elems, _) => {
            for child in elems {
                collect_span_ids_one(child, acc);
            }
        }
        _ => {}
    }
}

// ── Black-Scholes fixture round-trip (S1 oracle) ─────────────────────

const BLACK_SCHOLES_DP: &str =
    include_str!("../../chelis-cli/tests/fixtures/octant/black_scholes/call_price.dp");

#[test]
fn roundtrip_span_metadata_black_scholes_fixture() {
    // Parse → walk → assert 20 expected n_001..n_020 span IDs are present.
    let exprs1 = parse_str(BLACK_SCHOLES_DP).expect("first parse failed");
    let ids1 = collect_span_ids(&exprs1);

    let expected: Vec<String> = (1..=20).map(|i| format!("n_{i:03}")).collect();
    assert_eq!(
        ids1.len(),
        20,
        "expected 20 spans in first parse, got {}: {:?}",
        ids1.len(),
        ids1
    );
    let ids1_set: BTreeSet<&String> = ids1.iter().collect();
    let expected_set: BTreeSet<&String> = expected.iter().collect();
    assert_eq!(
        ids1_set, expected_set,
        "first parse: span ID set mismatch (got {ids1:?})"
    );

    // Reprint → re-parse → walk again → same 20 spans (same order).
    let printed = print_canonical(&exprs1);
    let exprs2 = parse_str(&printed).expect("second parse failed");
    let ids2 = collect_span_ids(&exprs2);
    assert_eq!(
        ids2, ids1,
        "second parse: span ID list (with order) differs from first parse"
    );

    // Third cycle: reprint → re-parse → same set (fixed-point check).
    let reprinted = print_canonical(&exprs2);
    let exprs3 = parse_str(&reprinted).expect("third parse failed");
    let ids3 = collect_span_ids(&exprs3);
    assert_eq!(ids3, ids1, "third parse: span ID list drifted from first");
    assert_eq!(
        printed, reprinted,
        "printer is not a fixed point on span-attributed input"
    );
}

#[test]
fn black_scholes_fixture_span_ids_are_strings() {
    // Sanity: the fixture's `span` values really are string literals,
    // exercising the `Expr::Atom(Atom::Str, _)` branch of `span_id()`.
    let exprs = parse_str(BLACK_SCHOLES_DP).expect("parse failed");
    // Top-level def carries n_001.
    assert_eq!(exprs[0].span_id(), Some("n_001"));
}

// ── Negative / edge-case round-trip parity ───────────────────────────

#[test]
fn span_id_returns_none_for_atom() {
    // Atoms do not carry metadata.
    let exprs = parse_str("42").expect("parse failed");
    assert_eq!(exprs.len(), 1);
    assert!(matches!(&exprs[0], Expr::Atom(Atom::Int(42), _)));
    assert_eq!(exprs[0].span_id(), None);
}

#[test]
fn span_id_returns_none_when_meta_lacks_span_key() {
    // List with non-empty meta but no `span` key.
    let exprs = parse_str("(def {type: (t-prim {} f32)} c (lit {} 0))").expect("parse failed");
    assert_eq!(exprs[0].span_id(), None);
}

#[test]
fn span_id_returns_none_for_empty_meta() {
    let exprs = parse_str("(def {} c (lit {} 0))").expect("parse failed");
    assert_eq!(exprs[0].span_id(), None);
}

#[test]
fn non_string_span_cannot_enter_an_annotation_carrier() {
    assert!(parse_str("(def {span: 42} c (lit {} 0))").is_err());
    let raw = serde_json::json!({"entries": [["span", Expr::Atom(chelis_deep::Atom::Int(42), chelis_deep::Span::new(0, 0))]]});
    assert!(serde_json::from_value::<chelis_deep::Metadata>(raw).is_err());
}

#[test]
fn span_id_empty_string_is_some_empty() {
    // Empty string is a valid (though unusual) span ID. Lock the
    // convention: `Some("")`, NOT `None`. This catches accidental
    // is_empty() coercions in future refactors.
    let src = r#"(def {span: ""} c (lit {} 0))"#;
    let exprs = parse_str(src).expect("parse failed");
    assert_eq!(exprs[0].span_id(), Some(""));

    // Round-trip preserves the empty string.
    let printed = print_canonical(&exprs);
    let exprs2 = parse_str(&printed).expect("second parse failed");
    assert_eq!(exprs2[0].span_id(), Some(""));
    let reprinted = print_canonical(&exprs2);
    assert_eq!(printed, reprinted, "empty-span printer not fixed point");
}

#[test]
fn span_id_unicode_preserved_bit_for_bit() {
    // Greek letters + ASCII dot path. Exact bytes preserved through
    // parse → print → parse.
    let original = "α.β.γ";
    let src = format!(r#"(def {{span: "{original}"}} c (lit {{}} 0))"#);
    let exprs = parse_str(&src).expect("parse failed");
    assert_eq!(exprs[0].span_id(), Some(original));

    let printed = print_canonical(&exprs);
    let exprs2 = parse_str(&printed).expect("second parse failed");
    assert_eq!(exprs2[0].span_id(), Some(original));
    assert_eq!(
        print_canonical(&exprs2),
        printed,
        "Unicode span printer not fixed point"
    );
}

#[test]
fn span_id_large_2kb_preserved_through_roundtrip() {
    // 2000-char span ID — well past any reasonable producer length.
    let big: String = "a".repeat(2000);
    let src = format!(r#"(def {{span: "{big}"}} c (lit {{}} 0))"#);
    let exprs = parse_str(&src).expect("parse failed");
    assert_eq!(exprs[0].span_id(), Some(big.as_str()));

    let printed = print_canonical(&exprs);
    let exprs2 = parse_str(&printed).expect("second parse failed");
    assert_eq!(exprs2[0].span_id(), Some(big.as_str()));
    assert_eq!(
        print_canonical(&exprs2),
        printed,
        "2KB span printer not fixed point"
    );
}

#[test]
#[ignore = "Pre-existing O(n²) in printer clone path for deeply nested Expr::Node (chelis#908 follow-up)"]
fn span_id_thousand_nested_spans_complete_in_under_a_minute() {
    // This oracle measures round-trip complexity, not deliberately constrained
    // stack behavior. Keep its stack explicit so platform-specific libtest
    // thread defaults do not change the oracle as the Expr representation grows.
    std::thread::Builder::new()
        .name("thousand-nested-spans".to_string())
        .stack_size(8 * 1024 * 1024)
        .spawn(run_thousand_nested_spans_oracle)
        .expect("spawn thousand-nested-spans oracle")
        .join()
        .expect("thousand-nested-spans oracle panicked");
}

fn run_thousand_nested_spans_oracle() {
    // Build a deeply nested app chain with 1000 spans. Catches accidental
    // O(n^2) behavior in metadata handling: parse, reprint, re-parse, then
    // assert all 1000 IDs survive both reprints. Runtime budgeted well
    // under 60s on default hardware (typically <1s).
    const N: usize = 1000;

    // Construct: (app {span: "n_0000"} (var {} f) (app {span: "n_0001"} (var {} f) ... (var {} x) ...))
    // simpler shape: chain of unary apps.
    // Build inside-out as a string.
    let mut src = String::from("(var {} x)");
    for i in (0..N).rev() {
        src = format!(r#"(app {{span: "n_{i:04}"}} (var {{}} f) {src})"#);
    }

    let t0 = Instant::now();
    let exprs1 = parse_str(&src).expect("first parse failed");
    let ids1 = collect_span_ids(&exprs1);
    assert_eq!(ids1.len(), N, "first parse: span count mismatch");

    let printed = print_canonical(&exprs1);
    let exprs2 = parse_str(&printed).expect("second parse failed");
    let ids2 = collect_span_ids(&exprs2);
    assert_eq!(ids2, ids1, "reprint: span list differs from first parse");

    let reprinted = print_canonical(&exprs2);
    let exprs3 = parse_str(&reprinted).expect("third parse failed");
    let ids3 = collect_span_ids(&exprs3);
    assert_eq!(ids3, ids1, "third parse: span list drifted");
    assert_eq!(printed, reprinted, "printer not fixed point on N=1000");
    let elapsed = t0.elapsed();

    // Loose budget — well under a minute. Local runs typically <1s.
    assert!(
        elapsed.as_secs() < 30,
        "1000-span round-trip took {elapsed:?}; suggests quadratic behavior"
    );
    eprintln!("1000-span round-trip elapsed: {elapsed:?}");
}
