//! Checker annotation writeback may only preserve or increase information.

use chelis_deep::{Atom, DeepTag, Expr, parse_and_stamp};
use chelis_types::check_ir_program;

fn find_matmul_type(expr: &Expr) -> Option<Expr> {
    let (tag, meta, children): (DeepTag, &chelis_deep::Metadata, &[Expr]) = match expr {
        Expr::Node(node, _) => (node.tag(), node.meta(), node.children_slice()),
        Expr::List(list, _) => (
            list.tag()?,
            match list.elements.get(1)? {
                Expr::Map(meta, _) => meta,
                _ => return None,
            },
            list.elements.get(2..).unwrap_or_default(),
        ),
        _ => return None,
    };
    if tag == DeepTag::App
        && children.first().is_some_and(|callee| match callee {
            Expr::Node(node, _) if node.tag() == DeepTag::Var => matches!(node.children_slice().first(), Some(Expr::Atom(Atom::Name(name), _)) if name == "matmul"),
            Expr::List(list, _) if list.tag() == Some(DeepTag::Var) => matches!(list.elements.get(2), Some(Expr::Atom(Atom::Name(name), _)) if name == "matmul"),
            _ => false,
        })
    {
        return meta.ty().map(|ty| ty.expression().clone());
    }
    children.iter().find_map(find_matmul_type)
}

fn tagged_children(expr: &Expr) -> Option<(DeepTag, &[Expr])> {
    match expr {
        Expr::Node(node, _) => Some((node.tag(), node.children_slice())),
        Expr::List(list, _) => Some((list.tag()?, list.elements.get(2..)?)),
        _ => None,
    }
}

fn concrete_tensor_metadata(expr: &Expr) -> Option<(Vec<i64>, String)> {
    let (tag, children) = tagged_children(expr)?;
    if tag != DeepTag::TTensor || children.len() < 2 {
        return None;
    }
    let mut dims = Vec::new();
    for dim in &children[..children.len() - 1] {
        let (dim_tag, dim_children) = tagged_children(dim)?;
        if dim_tag != DeepTag::DLit {
            return None;
        }
        let Expr::Atom(Atom::Int(value), _) = dim_children.first()? else {
            return None;
        };
        dims.push(*value);
    }
    let (prim_tag, prim_children) = tagged_children(children.last()?)?;
    if prim_tag != DeepTag::TPrim {
        return None;
    }
    let Expr::Atom(Atom::Name(prim), _) = prim_children.first()? else {
        return None;
    };
    Some((dims, prim.clone()))
}

#[test]
fn resolved_owner_preserves_equivalent_concrete_annotation() {
    let source = r#"
(def {} f
  (fn {} (params {}
      (a {type: (t-tensor {} (d-lit {} 4) (d-lit {} 4) (t-prim {} f32))})
      (good {type: (t-tensor {} (d-lit {} 4) (d-lit {} 4) (t-prim {} f32))}))
    (app {type: (t-tensor {} (d-lit {} 4) (d-lit {} 4) (t-prim {} f32))}
      (var {} matmul) (var {} a) (var {} good))))
"#;
    let exprs = parse_and_stamp(source).expect("fixture must parse and stamp");
    let checked = check_ir_program(&exprs).expect("fully resolved fixture must check");
    let stamped = find_matmul_type(&checked.exprs()[0]).expect("checked matmul metadata");
    assert_eq!(
        concrete_tensor_metadata(&stamped),
        Some((vec![4, 4], "f32".to_string())),
        "writeback must preserve the concrete source shape and precision"
    );
}
