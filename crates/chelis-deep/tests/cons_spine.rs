//! One borrowed canonical Cons/Nil walk serves all compiler readers.

use chelis_deep::cons_spine::{ConsSpine, ConsSpineTail};
use chelis_deep::{Atom, DeepTag, Expr, MetaExpr, Metadata, Span};

fn var(name: &str) -> Expr {
    Expr::node(
        DeepTag::Var,
        Metadata::default(),
        vec![Expr::Atom(Atom::Name(name.to_string()), Span::new(0, 0))],
        Span::new(0, 0),
    )
}

fn cons(head: Expr, tail: Expr) -> Expr {
    Expr::node(
        DeepTag::App,
        Metadata::default(),
        vec![var("Cons"), head, tail],
        Span::new(0, 0),
    )
}

#[test]
fn terminal_name_reader_preserves_qualified_cons_without_widening_exact_readers() {
    let chain = Expr::node(
        DeepTag::App,
        Metadata::default(),
        vec![var("Library__Cons"), var("head"), var("Nil")],
        Span::new(0, 0),
    );
    assert!(ConsSpine::new(&chain).next().is_none());
    let mut qualified = ConsSpine::with_terminal_names(&chain);
    assert!(qualified.next().is_some());
    qualified.require_nil().expect("qualified Cons, exact Nil");
}

#[test]
fn metadata_wrapped_list_cursor_is_an_opt_in_reader_shape() {
    let wrapped_nil = Expr::MetaExpr(
        MetaExpr {
            metadata: Metadata::default(),
            expr: Box::new(var("Nil")),
        },
        Span::new(0, 0),
    );
    let chain = cons(var("head"), wrapped_nil);
    assert!(ConsSpine::new(&chain).require_nil().is_err());
    let mut wrapped = ConsSpine::with_metadata_wrappers(&chain);
    assert!(wrapped.next().is_some());
    wrapped.require_nil().expect("wrapped Nil");
}

#[test]
fn canonical_spine_visits_thousands_of_heads_in_order_without_native_recursion() {
    for depth in [2_000, 5_000, 16_686] {
        let mut chain = var("Nil");
        for index in (0..depth).rev() {
            chain = cons(var(&format!("head_{index}")), chain);
        }
        let mut spine = ConsSpine::new(&chain);
        for index in 0..depth {
            let cell = spine.next().expect("Cons cell");
            let chelis_deep::ExprCarrier::DecodedNode(
                DeepTag::Var,
                _,
                [Expr::Atom(Atom::Name(name), _)],
            ) = cell.head.carrier()
            else {
                panic!("head must be a variable");
            };
            assert_eq!(name, &format!("head_{index}"));
        }
        assert!(spine.next().is_none());
        assert!(matches!(spine.tail(), Some(ConsSpineTail::Nil(_))));
        // Deep AST teardown is separate from the borrowed spine traversal.
        std::mem::forget(chain);
    }
}

#[test]
fn canonical_spine_reports_an_improper_tail_without_fabricating_nil() {
    let chain = cons(var("head"), var("dynamic_tail"));
    let mut spine = ConsSpine::new(&chain);
    assert!(spine.next().is_some());
    assert!(spine.next().is_none());
    assert!(matches!(spine.tail(), Some(ConsSpineTail::Other(_))));
    let improper = spine.require_nil().expect_err("dynamic tail is not Nil");
    assert!(matches!(
        improper.tail.carrier(),
        chelis_deep::ExprCarrier::DecodedNode(
            DeepTag::Var,
            _,
            [Expr::Atom(Atom::Name(name), _)]
        ) if name == "dynamic_tail"
    ));
}
