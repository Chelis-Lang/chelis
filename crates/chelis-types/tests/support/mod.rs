use chelis_deep::{Atom, DeepTag, Expr, Span};

/// Mutate the last stamped node with `tag` in reverse source order. This is a
/// controlled adversarial seam for [04-TOT-3]: production parsing cannot
/// construct wrong-arity nodes, but the transitional mutable Node API can
/// still hand one to the checker until #1023 closes that public surface.
pub fn append_name_to_last_node(exprs: &mut [Expr], tag: DeepTag, name: &str) {
    let child = Expr::Atom(Atom::Name(name.to_string()), Span::new(0, 0));
    assert!(
        exprs
            .iter_mut()
            .rev()
            .any(|expr| append_to_last(expr, tag, &child)),
        "fixture must contain a stamped `{}` node",
        tag.as_str()
    );
}

fn append_to_last(expr: &mut Expr, tag: DeepTag, child: &Expr) -> bool {
    match expr {
        Expr::Atom(..) => false,
        Expr::List(list, _) => list
            .elements
            .iter_mut()
            .rev()
            .any(|expr| append_to_last(expr, tag, child)),
        Expr::Map(meta, _) => meta
            .entries
            .iter_mut()
            .rev()
            .any(|(_, expr)| append_to_last(expr, tag, child)),
        Expr::MetaExpr(meta, _) => {
            append_to_last(&mut meta.expr, tag, child)
                || meta
                    .entries
                    .iter_mut()
                    .rev()
                    .any(|(_, expr)| append_to_last(expr, tag, child))
        }
        Expr::Node(node, _) => {
            if node
                .children_slice_mut()
                .iter_mut()
                .rev()
                .any(|expr| append_to_last(expr, tag, child))
                || node
                    .meta_mut()
                    .entries
                    .iter_mut()
                    .rev()
                    .any(|(_, expr)| append_to_last(expr, tag, child))
            {
                return true;
            }
            if node.tag() == tag {
                node.children_vec_mut().push(child.clone());
                return true;
            }
            false
        }
        Expr::BareList(elements, _) => elements
            .iter_mut()
            .rev()
            .any(|expr| append_to_last(expr, tag, child)),
        Expr::UnknownForm(data) => {
            data.children
                .iter_mut()
                .rev()
                .any(|expr| append_to_last(expr, tag, child))
                || data
                    .meta
                    .entries
                    .iter_mut()
                    .rev()
                    .any(|(_, expr)| append_to_last(expr, tag, child))
        }
    }
}
