//! LU3: semantic validation must not recurse through a canonical Cons spine.

use chelis_deep::{Atom, DeepTag, Expr, Metadata, Span};
use chelis_types::check_ir_program;

const SPAN: Span = Span { offset: 0, len: 0 };

fn node(tag: DeepTag, children: Vec<Expr>) -> Expr {
    Expr::node(tag, Metadata::default(), children, SPAN)
}

fn var(name: &str) -> Expr {
    node(
        DeepTag::Var,
        vec![Expr::Atom(Atom::Name(name.to_string()), SPAN)],
    )
}

fn int(value: i64) -> Expr {
    node(DeepTag::Lit, vec![Expr::Atom(Atom::Int(value), SPAN)])
}

fn cons_chain(count: usize, mut tail: Expr) -> Expr {
    for value in (0..count).rev() {
        tail = node(DeepTag::App, vec![var("Cons"), int(value as i64), tail]);
    }
    tail
}

fn list_program(value: Expr) -> Vec<Expr> {
    vec![node(
        DeepTag::Def,
        vec![Expr::Atom(Atom::Name("xs".to_string()), SPAN), value],
    )]
}

#[test]
fn semantic_validator_accepts_a_long_canonical_cons_spine() {
    let program = list_program(cons_chain(4_096, var("Nil")));
    let result = check_ir_program(&program);
    assert!(result.is_ok(), "long canonical list: {result:?}");
}

#[test]
fn semantic_validator_keeps_an_improper_cons_tail_loud() {
    let program = list_program(cons_chain(3, int(7)));
    let report = check_ir_program(&program).expect_err("an integer tail is not a List");
    assert!(
        !report.errors.is_empty(),
        "improper tail must report a type error"
    );
}
