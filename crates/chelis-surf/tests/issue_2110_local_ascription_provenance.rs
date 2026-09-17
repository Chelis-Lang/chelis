use chelis_deep::annotations::BindingTypeOrigin;
use chelis_deep::{Atom, DeepTag, Expr, Metadata, Span};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;

fn node_parts(expr: &Expr) -> Option<(DeepTag, &Metadata, &[Expr])> {
    match expr {
        Expr::Node(node, _) => Some((node.tag(), node.meta(), node.children_slice())),
        Expr::List(list, _) => {
            let tag = list.tag()?;
            let Expr::Map(meta, _) = list.elements.get(1)? else {
                return None;
            };
            Some((tag, meta, &list.elements[2..]))
        }
        _ => None,
    }
}

fn collect_bindings<'a>(expr: &'a Expr, out: &mut Vec<(&'a Expr, &'a Expr)>) {
    let Some((tag, _, children)) = node_parts(expr) else {
        return;
    };
    if tag == DeepTag::Bind {
        for pair in children.as_chunks::<2>().0 {
            out.push((&pair[0], &pair[1]));
        }
    }
    for child in children {
        collect_bindings(child, out);
    }
}

fn binding_name(expr: &Expr) -> &str {
    match expr {
        Expr::Atom(Atom::Name(name), _) => name,
        other => panic!("expected binding name, got {other:?}"),
    }
}

fn metadata(expr: &Expr) -> &Metadata {
    node_parts(expr)
        .map(|(_, meta, _)| meta)
        .unwrap_or_else(|| panic!("expected metadata-bearing expression, got {expr:?}"))
}

fn source_at(source: &str, span: Span) -> &str {
    &source[span.offset..span.end()]
}

#[test]
fn explicit_local_tensor_ascription_is_distinct_from_inferred_type_metadata() {
    let source = r#"
def f(x: tensor[*, f32]) -> tensor[*, f32] = {
  explicit: tensor[9, f32] =
    pad(x, [[0i64, 0i64]], 0.0f32)
  inferred = 1.0f32
  explicit
}
"#;
    let decls = parse_str(source).expect("Surf parse");
    let program = desugar_program(&decls);
    let mut bindings = Vec::new();
    for expr in &program {
        collect_bindings(expr, &mut bindings);
    }

    let explicit = bindings
        .iter()
        .find(|(name, _)| binding_name(name) == "explicit")
        .expect("explicit binding");
    let inferred = bindings
        .iter()
        .find(|(name, _)| binding_name(name) == "inferred")
        .expect("inferred binding");

    let explicit_meta = metadata(explicit.1);
    assert_eq!(
        explicit_meta
            .surf_binding_type()
            .expect("explicit binding marker")
            .value(),
        &BindingTypeOrigin::Explicit
    );
    assert!(
        explicit_meta.ty().is_some(),
        "the exact authored type remains ordinary type metadata too"
    );
    assert_eq!(source_at(source, explicit.0.span()), "explicit");
    assert_eq!(
        source_at(
            source,
            explicit_meta
                .surf_binding_type()
                .expect("explicit binding marker")
                .span()
        ),
        "tensor[9, f32]"
    );

    let inferred_meta = metadata(inferred.1);
    assert_eq!(
        inferred_meta
            .surf_binding_type()
            .expect("inferred binding marker")
            .value(),
        &BindingTypeOrigin::Inferred
    );
    assert!(
        inferred_meta.ty().is_some(),
        "the typed literal supplies ordinary inferred type metadata"
    );
    assert_eq!(source_at(source, inferred.0.span()), "inferred");
}
