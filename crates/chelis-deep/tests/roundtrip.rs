use chelis_deep::DeepTag;
use chelis_deep::parser::parse_str;
use chelis_deep::printer::{print_canonical, print_canonical_flat};
use chelis_deep::validate::validate;
use chelis_deep::{Atom, Expr};

fn roundtrip(source: &str) {
    let exprs1 = parse_str(source).expect("first parse failed");
    let printed = print_canonical(&exprs1);
    let exprs2 = parse_str(&printed).expect("second parse failed (printed output was invalid)");
    let reprinted = print_canonical(&exprs2);
    assert_eq!(
        printed, reprinted,
        "round-trip failed: print(parse(print(parse(s)))) != print(parse(s))"
    );
}

// ── Fixture round-trip tests ──────────────────────────────────────

#[test]
fn roundtrip_hello_tensor() {
    roundtrip(include_str!("fixtures/hello_tensor.dp"));
}

#[test]
fn roundtrip_simple_def() {
    roundtrip(include_str!("fixtures/simple_def.dp"));
}

#[test]
fn roundtrip_pattern_match() {
    roundtrip(include_str!("fixtures/pattern_match.dp"));
}

#[test]
fn roundtrip_metadata() {
    roundtrip(include_str!("fixtures/metadata.dp"));
}

#[test]
fn roundtrip_pipeline() {
    roundtrip(include_str!("fixtures/pipeline.dp"));
}

// ── AST structure tests ───────────────────────────────────────────

#[test]
fn post_sprint_def_ast_structure() {
    // Post-sprint 3-tuple: (def {} square (fn {} (params {} x) (app {} (var {} mul) (var {} x) (var {} x))))
    let src = "(def {} square (fn {} (params {} x) (app {} (var {} mul) (var {} x) (var {} x))))";
    let exprs = parse_str(src).expect("parse failed");
    assert_eq!(exprs.len(), 1);

    match &exprs[0] {
        Expr::List(list, _) => {
            // element 0: tag "def"
            match &list.elements[0] {
                Expr::Atom(Atom::Tag(t), _) => assert_eq!(*t, DeepTag::Def),
                other => panic!("expected Symbol(def), got {:?}", other),
            }
            // element 1: metadata map {}
            match &list.elements[1] {
                Expr::Map(m, _) => assert!(m.entries.is_empty()),
                other => panic!("expected empty Map, got {:?}", other),
            }
            // element 2: name "square"
            match &list.elements[2] {
                Expr::Atom(Atom::Name(s), _) => assert_eq!(s, "square"),
                other => panic!("expected Symbol(square), got {:?}", other),
            }
            // element 3: fn node
            match &list.elements[3] {
                Expr::List(func, _) => match &func.elements[0] {
                    Expr::Atom(Atom::Tag(t), _) => assert_eq!(*t, DeepTag::Fn),
                    other => panic!("expected Symbol(fn), got {:?}", other),
                },
                other => panic!("expected fn list, got {:?}", other),
            }
        }
        other => panic!("expected top-level List, got {:?}", other),
    }
    roundtrip(src);
}

#[test]
fn multiple_top_level_exprs() {
    let source = "(deftype {} Foo) (def {} bar (fn {} (params {}) (var {} nop)))";
    let exprs = parse_str(source).expect("parse failed");
    assert_eq!(exprs.len(), 2);
    match &exprs[0] {
        Expr::List(list, _) => match &list.elements[0] {
            Expr::Atom(Atom::Tag(t), _) => assert_eq!(*t, DeepTag::Deftype),
            other => panic!("expected Symbol(deftype), got {:?}", other),
        },
        other => panic!("expected List(deftype), got {:?}", other),
    }
    match &exprs[1] {
        Expr::List(list, _) => match &list.elements[0] {
            Expr::Atom(Atom::Tag(t), _) => assert_eq!(*t, DeepTag::Def),
            other => panic!("expected Symbol(def), got {:?}", other),
        },
        other => panic!("expected List(def), got {:?}", other),
    }
}

// ── Parse error tests ─────────────────────────────────────────────

#[test]
fn error_unmatched_open_paren() {
    let result = parse_str("(add 1 2");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("unexpected end of input"),
        "expected EOF error, got: {msg}"
    );
}

#[test]
fn error_unexpected_close_paren() {
    let result = parse_str(")");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("found )"), "expected ')' error, got: {msg}");
}

#[test]
fn empty_list_now_valid() {
    // Empty lists () are valid in the new spec (e.g., empty guard)
    let exprs = parse_str("()").unwrap();
    assert_eq!(exprs.len(), 1);
}

#[test]
fn error_incomplete_metadata() {
    let result = parse_str("^{:type}");
    assert!(result.is_err(), "incomplete metadata should fail to parse");
}

// ── Map parsing tests ────────────────────────────────────────────

#[test]
fn parse_empty_map() {
    let exprs = parse_str("{}").unwrap();
    assert_eq!(exprs.len(), 1);
    match &exprs[0] {
        Expr::Map(m, _) => assert!(m.entries.is_empty()),
        other => panic!("expected empty Map, got {:?}", other),
    }
}

#[test]
fn parse_map_with_entries() {
    let exprs = parse_str("{type: f32}").unwrap();
    assert_eq!(exprs.len(), 1);
    match &exprs[0] {
        Expr::Map(m, _) => {
            assert_eq!(m.entries.len(), 1);
            assert_eq!(m.entries[0].0, "type");
        }
        other => panic!("expected Map with entries, got {:?}", other),
    }
}

#[test]
fn parse_3tuple_node() {
    let exprs = parse_str("(app {} (var {} f) (var {} x))").unwrap();
    assert_eq!(exprs.len(), 1);
    roundtrip("(app {} (var {} f) (var {} x))");
}

#[test]
fn parse_node_with_typed_metadata() {
    roundtrip(
        "(def {type: (t-fn {} (t-prim {} f32) (t-prim {} f32))} square (fn {} (params {} x) (var {} x)))",
    );
}

// ── Spec conformance tests ───────────────────────────────────────

#[test]
fn spec_colon_as_list_head_parser_leniency() {
    // Parser accepts (: 42 i32) — bare list, not a 3-tuple node.
    // Validator only checks 3-tuple nodes (sym + Map + children),
    // so bare lists like this pass without warnings.
    let exprs = parse_str("(: 42 i32)").expect("parse failed");
    assert_eq!(exprs.len(), 1);
    match &exprs[0] {
        Expr::List(list, _) => {
            assert_eq!(list.elements.len(), 3);
            match &list.elements[0] {
                Expr::Atom(Atom::Name(s), _) => assert_eq!(s, ":"),
                other => panic!("expected Symbol(:), got {:?}", other),
            }
        }
        other => panic!("expected List, got {:?}", other),
    }
    // Bare list — no Map at [1], so validator skips it (not a tagged node)
    let warnings = validate(&exprs);
    assert!(
        warnings.is_empty(),
        "bare list should not trigger validation"
    );
    roundtrip("(: 42 i32)");
}

#[test]
fn spec_nested_lists_post_sprint() {
    // Post-sprint form: (deftype {} Option (t-var {} a) (variant {} Some (t-var {} a)) (variant {} None))
    let src = "(deftype {} Option (t-var {} a) (variant {} Some (t-var {} a)) (variant {} None))";
    let exprs = parse_str(src).expect("parse failed");
    assert_eq!(exprs.len(), 1);
    match &exprs[0] {
        Expr::List(list, _) => {
            // tag = deftype, meta = {}, name = Option, then type-var and variants
            match &list.elements[0] {
                Expr::Atom(Atom::Tag(t), _) => assert_eq!(*t, DeepTag::Deftype),
                other => panic!("expected Symbol(deftype), got {:?}", other),
            }
        }
        other => panic!("expected List, got {:?}", other),
    }
    let warnings = validate(&exprs);
    assert!(
        warnings.is_empty(),
        "expected no warnings, got: {warnings:?}"
    );
    roundtrip(src);
}

#[test]
fn spec_metadata_on_list_items() {
    // (fn (^{:type f32} x) body) — metadata on items within a list
    let exprs = parse_str("(fn (^{:type f32} x) body)").expect("parse failed");
    assert_eq!(exprs.len(), 1);
    match &exprs[0] {
        Expr::List(outer, _) => {
            assert_eq!(outer.elements.len(), 3);
            match &outer.elements[1] {
                Expr::List(params, _) => {
                    assert_eq!(params.elements.len(), 1);
                    match &params.elements[0] {
                        Expr::MetaExpr(meta, _) => {
                            assert_eq!(meta.entries.len(), 1);
                            assert_eq!(meta.entries[0].0, "type");
                        }
                        other => panic!("expected MetaExpr, got {:?}", other),
                    }
                }
                other => panic!("expected params list, got {:?}", other),
            }
        }
        other => panic!("expected List, got {:?}", other),
    }
    roundtrip("(fn (^{:type f32} x) body)");
}

#[test]
fn spec_float_exponent_only() {
    // -1e-5 as a single Float atom
    let exprs = parse_str("-1e-5").expect("parse failed");
    assert_eq!(exprs.len(), 1);
    match &exprs[0] {
        Expr::Atom(Atom::Float(f), _) => assert!((f - (-1e-5)).abs() < 1e-15),
        other => panic!("expected Float(-1e-5), got {:?}", other),
    }

    // 1e10 as Float
    let exprs = parse_str("1e10").expect("parse failed");
    assert_eq!(exprs.len(), 1);
    match &exprs[0] {
        Expr::Atom(Atom::Float(f), _) => assert!((f - 1e10).abs() < 1.0),
        other => panic!("expected Float(1e10), got {:?}", other),
    }

    // 5E3 as Float
    let exprs = parse_str("5E3").expect("parse failed");
    assert_eq!(exprs.len(), 1);
    match &exprs[0] {
        Expr::Atom(Atom::Float(f), _) => assert!((f - 5e3).abs() < 1e-10),
        other => panic!("expected Float(5E3), got {:?}", other),
    }
}

// ── Fixture validation tests ─────────────────────────────────────

fn assert_fixture_valid(name: &str, source: &str) {
    let exprs = parse_str(source).unwrap_or_else(|e| panic!("fixture {name} failed to parse: {e}"));
    let warnings = validate(&exprs);
    assert!(
        warnings.is_empty(),
        "fixture {name} has validation warnings: {warnings:?}"
    );
}

#[test]
fn all_fixtures_pass_tag_validator() {
    assert_fixture_valid("hello_tensor.dp", include_str!("fixtures/hello_tensor.dp"));
    assert_fixture_valid("simple_def.dp", include_str!("fixtures/simple_def.dp"));
    assert_fixture_valid(
        "pattern_match.dp",
        include_str!("fixtures/pattern_match.dp"),
    );
    assert_fixture_valid("metadata.dp", include_str!("fixtures/metadata.dp"));
    assert_fixture_valid("pipeline.dp", include_str!("fixtures/pipeline.dp"));
}

#[test]
fn pretty_and_flat_printers_round_trip_to_same_ast() {
    let source = include_str!("fixtures/pattern_match.dp");
    let exprs = parse_str(source).expect("parse fixture");

    let pretty = print_canonical(&exprs);
    let flat = print_canonical_flat(&exprs);

    let pretty_exprs = parse_str(&pretty).expect("parse pretty");
    let flat_exprs = parse_str(&flat).expect("parse flat");

    assert_eq!(print_canonical(&pretty_exprs), pretty);
    assert_eq!(print_canonical_flat(&pretty_exprs), flat);
    assert_eq!(print_canonical(&flat_exprs), pretty);
    assert_eq!(print_canonical_flat(&flat_exprs), flat);
}
