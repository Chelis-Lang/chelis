//! Node identities: distinct per node, kept by copies, invisible as content.

use chelis_deep::annotations::{MetadataValue, Spanned};
use chelis_deep::{
    Atom, DeepTag, Expr, List, Metadata, NodeIdentity, Span, assign_node_identities,
};

const ZERO: Span = Span { offset: 0, len: 0 };

fn form(stamped: bool, tag: DeepTag, meta: Metadata, children: Vec<Expr>) -> Expr {
    if stamped {
        Expr::node(tag, meta, children, ZERO)
    } else {
        let mut elements = vec![Expr::Atom(Atom::Tag(tag), ZERO), Expr::Map(meta, ZERO)];
        elements.extend(children);
        Expr::List(List { elements }, ZERO)
    }
}

fn var(stamped: bool, name: &str) -> Expr {
    form(
        stamped,
        DeepTag::Var,
        Metadata::default(),
        vec![Expr::Atom(Atom::Name(name.to_string()), ZERO)],
    )
}

/// `(app {doc: "call"} (var {} f) (var {} x) (var {} x))`: the two argument
/// nodes are equal values.
fn call(stamped: bool) -> Expr {
    let doc = Metadata::from(MetadataValue::Doc(Spanned::new("call".into(), ZERO)));
    form(
        stamped,
        DeepTag::App,
        doc,
        vec![var(stamped, "f"), var(stamped, "x"), var(stamped, "x")],
    )
}

fn identities(expr: &Expr, out: &mut Vec<Option<NodeIdentity>>) {
    let chelis_deep::ExprCarrier::DecodedNode(_, _, children) = expr.carrier() else {
        return;
    };
    out.push(expr.node_identity());
    for child in children {
        identities(child, out);
    }
}

fn all_identities(program: &[Expr]) -> Vec<Option<NodeIdentity>> {
    let mut out = Vec::new();
    for expr in program {
        identities(expr, &mut out);
    }
    out
}

fn children(expr: &Expr) -> &[Expr] {
    match expr.carrier() {
        chelis_deep::ExprCarrier::DecodedNode(_, _, children) => children,
        _ => panic!("a decoded node"),
    }
}

#[test]
fn equal_subtrees_get_distinct_identities_in_both_carriers() {
    for stamped in [true, false] {
        let mut program = vec![call(stamped)];
        assert_eq!(assign_node_identities(&mut program), 4);
        let args = children(&program[0]);
        assert_eq!(args[1], args[2], "the arguments are equal values");
        assert_ne!(args[1].node_identity(), args[2].node_identity());
        let found = all_identities(&program);
        let mut distinct = found.clone();
        distinct.sort();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            4,
            "every node has its own identity: {found:?}"
        );
        assert!(found.iter().all(Option::is_some), "{found:?}");
    }
}

#[test]
fn copies_and_the_list_view_keep_the_identity_of_their_node() {
    let mut program = vec![call(true)];
    assign_node_identities(&mut program);
    let original = all_identities(&program);
    assert_eq!(all_identities(&program.clone()), original);

    let Expr::Node(node, span) = &program[0] else {
        panic!("stamped carrier")
    };
    let view = node.to_list(*span);
    assert_eq!(view.node_identity(), program[0].node_identity());
    let viewed: Vec<_> = view.elements[2..].iter().map(Expr::node_identity).collect();
    assert_eq!(viewed, original[1..].to_vec());
}

#[test]
fn identities_are_not_content() {
    for stamped in [true, false] {
        let plain = vec![call(stamped)];
        let mut numbered = plain.clone();
        assign_node_identities(&mut numbered);
        assert_eq!(numbered, plain);
        assert_eq!(format!("{numbered:?}"), format!("{plain:?}"));
        assert_eq!(format!("{numbered:#?}"), format!("{plain:#?}"));
        assert_eq!(
            serde_json::to_string(&numbered).unwrap(),
            serde_json::to_string(&plain).unwrap()
        );
        let argument = &children(&numbered[0])[1];
        let chelis_deep::ExprCarrier::DecodedNode(_, meta, _) = argument.carrier() else {
            panic!("a decoded node")
        };
        assert!(argument.node_identity().is_some());
        assert!(meta.is_empty(), "an identity is not an annotation");
        assert_eq!(meta.values().count(), 0);
    }
}

#[test]
fn renumbering_ignores_identities_copied_from_another_program() {
    let mut first = vec![call(false)];
    assign_node_identities(&mut first);
    let reused = children(&first[0])[2].clone();
    assert!(reused.node_identity().is_some());
    let tuple = |last| {
        vec![form(
            false,
            DeepTag::Tuple,
            Metadata::default(),
            vec![var(false, "y"), last],
        )]
    };
    let mut second = tuple(reused);
    let mut fresh = tuple(var(false, "x"));
    assert_eq!(second, fresh);
    assign_node_identities(&mut second);
    assign_node_identities(&mut fresh);
    assert_eq!(all_identities(&second), all_identities(&fresh));
}
