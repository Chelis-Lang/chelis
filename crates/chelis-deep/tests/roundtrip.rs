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
            assert_eq!(list.tag, "def");
            assert_eq!(list.children.len(), 3);

            // First child is the name symbol
            match &list.children[0] {
                Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "square"),
                other => panic!("expected Symbol(square), got {:?}", other),
            }

            // Second child is the sig
            match &list.children[1] {
                Expr::List(sig, _) => {
                    assert_eq!(sig.tag, "sig");
                    assert_eq!(sig.children.len(), 1);
                    match &sig.children[0] {
                        Expr::List(arrow, _) => {
                            assert_eq!(arrow.tag, "->");
                            assert_eq!(arrow.children.len(), 2);
                        }
                        other => panic!("expected arrow list, got {:?}", other),
                    }
                }
                other => panic!("expected sig list, got {:?}", other),
            }

            // Third child is the fn body
            match &list.children[2] {
                Expr::List(func, _) => {
                    assert_eq!(func.tag, "fn");
                    assert_eq!(func.children.len(), 2);
                    // params list
                    match &func.children[0] {
                        Expr::List(params, _) => {
                            assert_eq!(params.tag, "x");
                            assert!(params.children.is_empty());
                        }
                        other => panic!("expected params list, got {:?}", other),
                    }
                    // body: (apply mul x x)
                    match &func.children[1] {
                        Expr::List(apply, _) => {
                            assert_eq!(apply.tag, "apply");
                            assert_eq!(apply.children.len(), 3);
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
        Expr::List(list, _) => assert_eq!(list.tag, "type"),
        other => panic!("expected List(type), got {:?}", other),
    }
    match &exprs[1] {
        Expr::List(list, _) => assert_eq!(list.tag, "def"),
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
fn error_empty_list() {
    let result = parse_str("()");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("empty list"),
        "expected empty-list error, got: {msg}"
    );
}

#[test]
fn error_list_with_non_symbol_tag() {
    let result = parse_str("(42 a b)");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("symbol"),
        "expected 'symbol' in error, got: {msg}"
    );
}

#[test]
fn error_incomplete_metadata() {
    let result = parse_str("^{:type}");
    assert!(result.is_err(), "incomplete metadata should fail to parse");
}
