use std::collections::BTreeMap;

use chelis_deep::{DeepTag, Expr as DeepExpr};
use chelis_types::CheckedProgram;

use crate::{AllRootNames, IrName, RootMetadata, TensorRootNames};

pub(crate) fn root_metadata(
    program: &CheckedProgram,
    lowered_names: Option<&BTreeMap<String, bool>>,
) -> RootMetadata {
    let all_names = AllRootNames(root_names_from_checked_exprs(
        program.exprs(),
        program.type_env(),
        None,
    ));
    let owned_lowered_names;
    let lowered_names = match lowered_names {
        Some(map) => Some(map),
        None => {
            owned_lowered_names =
                chelis_ir::lower::top_level_lowering_map(program.exprs(), program.type_env());
            Some(&owned_lowered_names)
        }
    };
    let tensor_names = TensorRootNames(root_names_from_checked_exprs(
        program.exprs(),
        program.type_env(),
        lowered_names,
    ));
    RootMetadata {
        all_names,
        tensor_names,
    }
}

fn root_names_from_checked_exprs(
    exprs: &[DeepExpr],
    type_env: &BTreeMap<String, DeepExpr>,
    lowered_names: Option<&BTreeMap<String, bool>>,
) -> Vec<IrName> {
    let mut names = Vec::new();
    for expr in exprs {
        collect_checked_decl_names(expr, type_env, lowered_names, &mut names);
    }
    names
}

fn collect_checked_decl_names(
    expr: &DeepExpr,
    type_env: &BTreeMap<String, DeepExpr>,
    lowered_names: Option<&BTreeMap<String, bool>>,
    output: &mut Vec<IrName>,
) {
    let Some((tag, children)) = tagged_children(expr) else {
        return;
    };
    match tag {
        DeepTag::Module => {
            for child in children.iter().skip(1) {
                collect_checked_decl_names(child, type_env, lowered_names, output);
            }
        }
        DeepTag::Def => {
            if let Some(name) = children.first().and_then(symbol_name) {
                if lowered_names.is_some_and(|map| !map.get(name).copied().unwrap_or(false)) {
                    return;
                }
                let value = children.get(1);
                let ty = type_env
                    .get(name)
                    .or_else(|| value.and_then(expr_type_metadata));
                extend_root_names(name, ty, value, output);
            }
        }
        _ => {}
    }
}

fn tagged_children(expr: &DeepExpr) -> Option<(DeepTag, &[DeepExpr])> {
    match expr {
        DeepExpr::List(list, _) => Some((list.tag()?, list.elements.get(2..)?)),
        DeepExpr::Node(node, _) => Some((node.tag(), node.children_slice())),
        _ => None,
    }
}

fn extend_root_names(
    name: &str,
    ty: Option<&DeepExpr>,
    value: Option<&DeepExpr>,
    output: &mut Vec<IrName>,
) {
    if let Some((tag, children)) = ty.and_then(tagged_children) {
        if tag == DeepTag::TFn {
            extend_root_names(name, children.last(), None, output);
            return;
        }
        if tag == DeepTag::TTuple {
            for (index, child) in children.iter().enumerate() {
                extend_root_names(&format!("{name}.{index}"), Some(child), None, output);
            }
            return;
        }
    }
    if let Some((DeepTag::Tuple, children)) = value.and_then(tagged_children) {
        for (index, child) in children.iter().enumerate() {
            extend_root_names(
                &format!("{name}.{index}"),
                expr_type_metadata(child),
                Some(child),
                output,
            );
        }
        return;
    }
    output.push(IrName::new(name));
}

fn expr_type_metadata(expr: &DeepExpr) -> Option<&DeepExpr> {
    let metadata = match expr {
        DeepExpr::List(list, _) => match list.elements.get(1) {
            Some(DeepExpr::Map(metadata, _)) => metadata,
            _ => return None,
        },
        DeepExpr::Node(node, _) => node.meta(),
        _ => return None,
    };
    metadata.ty().map(|ty| ty.expression())
}

fn symbol_name(expr: &DeepExpr) -> Option<&str> {
    match expr {
        DeepExpr::Atom(chelis_deep::Atom::Name(name), _) => Some(name.as_str()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_name_reads_the_typed_deep_name_atom() {
        let expression = DeepExpr::Atom(
            chelis_deep::Atom::Name("root".to_string()),
            chelis_deep::Span::new(0, 4),
        );

        assert_eq!(symbol_name(&expression), Some("root"));
    }

    #[test]
    fn typed_node_root_collection_matches_list_root_collection() {
        let source = "(module {} m (def {} out (tuple {} \
            (lit {type: (t-prim {} f32)} 1.0) \
            (lit {type: (t-prim {} f32)} 2.0))))";
        let list_exprs = chelis_deep::parser::parse_str(source).expect("list Deep must parse");
        let typed_exprs =
            chelis_deep::parse_and_stamp_file(source).expect("typed Deep must parse and stamp");
        assert!(matches!(typed_exprs.first(), Some(DeepExpr::Node(_, _))));

        let expected = vec![IrName::new("out.0"), IrName::new("out.1")];
        assert_eq!(
            root_names_from_checked_exprs(&list_exprs, &BTreeMap::new(), None),
            expected
        );
        assert_eq!(
            root_names_from_checked_exprs(&typed_exprs, &BTreeMap::new(), None),
            expected
        );
    }

    #[test]
    fn empty_tuple_declaration_has_no_flattened_root_names() {
        let expressions = chelis_deep::parser::parse_str("(def {} empty (tuple {}))")
            .expect("empty tuple Deep must parse");

        assert!(root_names_from_checked_exprs(&expressions, &BTreeMap::new(), None).is_empty());
    }

    #[test]
    fn mixed_tuple_declaration_keeps_each_canonical_position() {
        let source = "(def {} mixed (tuple {} \
            (lit {type: (t-prim {} int32)} 1) \
            (lit {type: (t-tensor {} (d-name {} n) (t-prim {} f32))} 2.0)))";
        let expressions =
            chelis_deep::parser::parse_str(source).expect("mixed tuple Deep must parse");

        assert_eq!(
            root_names_from_checked_exprs(&expressions, &BTreeMap::new(), None),
            vec![IrName::new("mixed.0"), IrName::new("mixed.1")]
        );
    }

    #[test]
    fn typed_node_non_root_declaration_does_not_create_a_root_name() {
        let source = "(module {} m (defsig {} f (t-fn {} (t-prim {} f32))))";
        let typed_exprs =
            chelis_deep::parse_and_stamp_file(source).expect("typed Deep must parse and stamp");

        assert!(root_names_from_checked_exprs(&typed_exprs, &BTreeMap::new(), None).is_empty());
    }
}
