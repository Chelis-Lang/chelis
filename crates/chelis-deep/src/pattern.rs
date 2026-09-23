//! Shared structural operations over Deep patterns.

use crate::{Atom, DeepTag, Expr};

/// Return every value binder introduced by a Deep pattern.
///
/// This is the single policy owner used by authoring, macro hygiene, and
/// checker scope analysis. The tag match is deliberately exhaustive: adding a
/// new Deep form requires deciding whether and how it binds names instead of
/// silently treating it as binder-free.
pub fn pattern_binder_names(expr: &Expr) -> Vec<String> {
    let mut names = Vec::new();
    collect_pattern_binder_names(expr, &mut names);
    names
}

fn collect_pattern_binder_names(expr: &Expr, out: &mut Vec<String>) {
    let Some((tag, children)) = parts(expr) else {
        return;
    };
    match tag {
        DeepTag::PatVar => {
            if let Some(name) = children.first().and_then(symbol) {
                out.push(name.to_string());
            }
        }
        DeepTag::PatAs => {
            if let Some(name) = children.first().and_then(symbol) {
                out.push(name.to_string());
            }
            if let Some(inner) = children.get(1) {
                collect_pattern_binder_names(inner, out);
            }
        }
        DeepTag::PatTuple => {
            for child in children {
                collect_pattern_binder_names(child, out);
            }
        }
        DeepTag::PatCtor => {
            for child in children.iter().skip(1) {
                collect_pattern_binder_names(child, out);
            }
        }
        DeepTag::PatRecord => {
            for field in children.iter().skip(1) {
                if let Some((DeepTag::Kv, kv_children)) = parts(field)
                    && let Some(field_pattern) = kv_children.get(1)
                {
                    collect_pattern_binder_names(field_pattern, out);
                }
            }
        }
        DeepTag::PatLit | DeepTag::PatWild => {}
        DeepTag::Module
        | DeepTag::Import
        | DeepTag::ImportAll
        | DeepTag::Export
        | DeepTag::Defsig
        | DeepTag::Def
        | DeepTag::Deftype
        | DeepTag::Typealias
        | DeepTag::Variant
        | DeepTag::Field
        | DeepTag::Defdim
        | DeepTag::If
        | DeepTag::Match
        | DeepTag::Arm
        | DeepTag::Fn
        | DeepTag::Let
        | DeepTag::Bind
        | DeepTag::Var
        | DeepTag::Lit
        | DeepTag::App
        | DeepTag::Record
        | DeepTag::Access
        | DeepTag::Pipe
        | DeepTag::Block
        | DeepTag::Tuple
        | DeepTag::TupleGet
        | DeepTag::RecordUpdate
        | DeepTag::Par
        | DeepTag::Borrow
        | DeepTag::TPrim
        | DeepTag::TFn
        | DeepTag::TTensor
        | DeepTag::TRef
        | DeepTag::TAdt
        | DeepTag::TVar
        | DeepTag::TUnit
        | DeepTag::TTuple
        | DeepTag::DName
        | DeepTag::DVar
        | DeepTag::DLit
        | DeepTag::DRank
        | DeepTag::Effects
        | DeepTag::Resource
        | DeepTag::HandleEffect
        | DeepTag::Grad
        | DeepTag::Vmap
        | DeepTag::Jit
        | DeepTag::Realize
        | DeepTag::Cast
        | DeepTag::Copy
        | DeepTag::Quote
        | DeepTag::Unquote
        | DeepTag::Splice
        | DeepTag::Params
        | DeepTag::Kv => {}
    }
}

fn parts(expr: &Expr) -> Option<(DeepTag, &[Expr])> {
    match expr {
        Expr::Node(node, _) => Some((node.tag(), node.children_slice())),
        _ => None,
    }
}

fn symbol(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Name(name), _) => Some(name),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_node_patterns_collect_all_binders() {
        let exprs = crate::parser::parse_str(
            "(pat-as {} whole (pat-tuple {} (pat-var {} left) \
             (pat-ctor {} Pair (pat-var {} right) (pat-wild {}))))",
        )
        .expect("pattern fixture must parse");
        assert_eq!(pattern_binder_names(&exprs[0]), ["whole", "left", "right"]);
    }
}
