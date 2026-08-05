//! Checker annotation writeback may only preserve or increase information.

use chelis_deep::{Atom, DeepTag, Expr, parse_and_stamp};
use chelis_types::check_ir_program;

fn find_matmul_type(expr: &Expr) -> Option<Expr> {
    let (tag, meta, children): (DeepTag, &chelis_deep::MetaMap, &[Expr]) = match expr {
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
        return meta
            .entries
            .iter()
            .find(|(key, _)| key == "type")
            .map(|(_, value)| value.clone());
    }
    children.iter().find_map(find_matmul_type)
}

#[test]
fn unresolved_owner_never_clobbers_a_concrete_annotation() {
    let source = r#"
(def {} f
  (fn {} (params {} a (good {type: (t-tensor {} (d-lit {} 4) (d-lit {} 4) (t-prim {} f32))}))
    (app {type: (t-tensor {} (d-lit {} 4) (d-lit {} 4) (t-prim {} f32))}
      (var {} matmul) (var {} a) (var {} good))))
"#;
    let exprs = parse_and_stamp(source).expect("fixture must parse and stamp");
    let original = find_matmul_type(&exprs[0]).expect("fixture carries matmul metadata");
    match check_ir_program(&exprs) {
        Err(result) => assert!(
            result.errors.iter().any(|error| {
                error.message.contains("unresolved") || error.message.contains("annotation")
            }),
            "loud failure must name the unresolved annotation obligation: {:?}",
            result.errors
        ),
        Ok(checked) => {
            let stamped = find_matmul_type(&checked.exprs()[0]).expect("checked matmul metadata");
            assert_eq!(
                stamped, original,
                "an unresolved owner type may not replace concrete source metadata"
            );
        }
    }
}
