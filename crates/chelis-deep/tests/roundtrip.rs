use chelis_deep::parser::parse_str;
use chelis_deep::printer::print_canonical;
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
fn simple_def_ast_structure() {
    let exprs = parse_str("(def square (sig (-> f32 f32)) (fn (x) (apply mul x x)))")
        .expect("parse failed");
    assert_eq!(exprs.len(), 1);

    // Top level is a list with tag "def"
    let top = &exprs[0];
    match top {
        Expr::List(list, _) => {
            assert_eq!(list.elements.len(), 4);

            // element 0: the tag "def"
            match &list.elements[0] {
                Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "def"),
                other => panic!("expected Symbol(def), got {:?}", other),
            }

            // element 1: the name symbol
            match &list.elements[1] {
                Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "square"),
                other => panic!("expected Symbol(square), got {:?}", other),
            }

            // element 2: the sig
            match &list.elements[2] {
                Expr::List(sig, _) => {
                    assert_eq!(sig.elements.len(), 2);
                    match &sig.elements[0] {
                        Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "sig"),
                        other => panic!("expected Symbol(sig), got {:?}", other),
                    }
                    match &sig.elements[1] {
                        Expr::List(arrow, _) => {
                            assert_eq!(arrow.elements.len(), 3);
                            match &arrow.elements[0] {
                                Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "->"),
                                other => panic!("expected Symbol(->), got {:?}", other),
                            }
                        }
                        other => panic!("expected arrow list, got {:?}", other),
                    }
                }
                other => panic!("expected sig list, got {:?}", other),
            }

            // element 3: the fn body
            match &list.elements[3] {
                Expr::List(func, _) => {
                    assert_eq!(func.elements.len(), 3);
                    match &func.elements[0] {
                        Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "fn"),
                        other => panic!("expected Symbol(fn), got {:?}", other),
                    }
                    // params list
                    match &func.elements[1] {
                        Expr::List(params, _) => {
                            assert_eq!(params.elements.len(), 1);
                            match &params.elements[0] {
                                Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "x"),
                                other => panic!("expected Symbol(x), got {:?}", other),
                            }
                        }
                        other => panic!("expected params list, got {:?}", other),
                    }
                    // body: (apply mul x x)
                    match &func.elements[2] {
                        Expr::List(apply, _) => {
                            assert_eq!(apply.elements.len(), 4);
                            match &apply.elements[0] {
                                Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "apply"),
                                other => panic!("expected Symbol(apply), got {:?}", other),
                            }
                        }
                        other => panic!("expected apply list, got {:?}", other),
                    }
                }
                other => panic!("expected fn list, got {:?}", other),
            }
        }
        other => panic!("expected top-level List, got {:?}", other),
    }
}

#[test]
fn multiple_top_level_exprs() {
    let source = "(type Foo) (def bar (fn (params) (nop)))";
    let exprs = parse_str(source).expect("parse failed");
    assert_eq!(exprs.len(), 2);
    match &exprs[0] {
        Expr::List(list, _) => match &list.elements[0] {
            Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "type"),
            other => panic!("expected Symbol(type), got {:?}", other),
        },
        other => panic!("expected List(type), got {:?}", other),
    }
    match &exprs[1] {
        Expr::List(list, _) => match &list.elements[0] {
            Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "def"),
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

// ── Spec conformance tests ───────────────────────────────────────

#[test]
fn spec_colon_as_list_head() {
    // (: 42 i32) — colon as list head element
    let exprs = parse_str("(: 42 i32)").expect("parse failed");
    assert_eq!(exprs.len(), 1);
    match &exprs[0] {
        Expr::List(list, _) => {
            assert_eq!(list.elements.len(), 3);
            match &list.elements[0] {
                Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, ":"),
                other => panic!("expected Symbol(:), got {:?}", other),
            }
            match &list.elements[1] {
                Expr::Atom(Atom::Int(42), _) => {}
                other => panic!("expected Int(42), got {:?}", other),
            }
            match &list.elements[2] {
                Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "i32"),
                other => panic!("expected Symbol(i32), got {:?}", other),
            }
        }
        other => panic!("expected List, got {:?}", other),
    }
    roundtrip("(: 42 i32)");
}

#[test]
fn spec_nested_lists_as_elements() {
    // (type Option (a) ((Some a) (None))) — nested lists as elements
    let exprs = parse_str("(type Option (a) ((Some a) (None)))").expect("parse failed");
    assert_eq!(exprs.len(), 1);
    match &exprs[0] {
        Expr::List(list, _) => {
            assert_eq!(list.elements.len(), 4);
            // element 3 is a list whose elements are themselves lists
            match &list.elements[3] {
                Expr::List(variants, _) => {
                    assert_eq!(variants.elements.len(), 2);
                    match &variants.elements[0] {
                        Expr::List(some, _) => {
                            assert_eq!(some.elements.len(), 2);
                            match &some.elements[0] {
                                Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "Some"),
                                other => panic!("expected Symbol(Some), got {:?}", other),
                            }
                        }
                        other => panic!("expected List(Some ...), got {:?}", other),
                    }
                    match &variants.elements[1] {
                        Expr::List(none, _) => {
                            assert_eq!(none.elements.len(), 1);
                            match &none.elements[0] {
                                Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "None"),
                                other => panic!("expected Symbol(None), got {:?}", other),
                            }
                        }
                        other => panic!("expected List(None), got {:?}", other),
                    }
                }
                other => panic!("expected variants list, got {:?}", other),
            }
        }
        other => panic!("expected List, got {:?}", other),
    }
    roundtrip("(type Option (a) ((Some a) (None)))");
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
