//! Transitional bridge helpers for chelis#908 (Expr::List removal).

use chelis_deep::Span;
use chelis_deep::ast::{Atom, Expr, List};

/// Reconstruct a `List` from a `Node` (bridge, chelis#908).
pub(crate) fn node_to_list(node: &chelis_deep::node::Node, span: Span) -> List {
    use chelis_deep::node::ChildRef;
    let mut elements = Vec::with_capacity(node.child_count() + 2);
    elements.push(Expr::Atom(Atom::Name(node.tag().as_str().to_string()), span));
    elements.push(Expr::Map(node.meta().clone(), span));
    for child_ref in node.children_iter() {
        match child_ref {
            ChildRef::Expr(e)
            | ChildRef::Syntax(e)
            | ChildRef::Type(e)
            | ChildRef::EffectHandler(e)
            | ChildRef::Bypass(e) => elements.push(e.clone()),
            ChildRef::Binder(s) => {
                elements.push(Expr::Atom(Atom::Name(s.to_string()), span));
            }
            ChildRef::Selector(s) => {
                elements.push(Expr::Atom(Atom::Name(s.to_string()), span));
            }
        }
    }
    List { elements }
}
