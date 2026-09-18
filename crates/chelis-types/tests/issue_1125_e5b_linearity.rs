//! chelis#1125 PP7/E5b: the linearity reader uses the shared total carrier
//! view without changing binding-generation or diagnostic semantics.

use chelis_deep::{Atom, Expr, List, Metadata};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::{CheckedProgram, check_ir_program, check_linearity, check_typed_program};
use syn::visit::{self, Visit};

fn path_ends_with(path: &syn::Path, expected: &str) -> bool {
    path.segments
        .last()
        .is_some_and(|segment| segment.ident == expected)
}

fn type_path_ends_with(ty: &syn::Type, expected: &str) -> bool {
    matches!(ty, syn::Type::Path(path) if path_ends_with(&path.path, expected))
}

fn optional_decoded_parts_return(ty: &syn::Type) -> bool {
    let syn::Type::Path(option) = ty else {
        return false;
    };
    let Some(segment) = option.path.segments.last() else {
        return false;
    };
    if segment.ident != "Option" {
        return false;
    }
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return false;
    };
    let Some(syn::GenericArgument::Type(syn::Type::Tuple(parts))) = arguments.args.first() else {
        return false;
    };
    let mut parts = parts.elems.iter();
    let (Some(tag), Some(metadata), Some(children), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return false;
    };
    let syn::Type::Reference(metadata) = metadata else {
        return false;
    };
    let syn::Type::Reference(children) = children else {
        return false;
    };
    let syn::Type::Slice(children) = children.elem.as_ref() else {
        return false;
    };
    type_path_ends_with(tag, "DeepTag")
        && type_path_ends_with(&metadata.elem, "Metadata")
        && type_path_ends_with(&children.elem, "Expr")
}

#[derive(Default)]
struct DecodedAdapterBody {
    calls_carrier: bool,
    matches_decoded_node: bool,
    returns_some: bool,
    returns_none: bool,
}

impl<'ast> Visit<'ast> for DecodedAdapterBody {
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if call.method == "carrier" {
            self.calls_carrier = true;
        }
        visit::visit_expr_method_call(self, call);
    }

    fn visit_path(&mut self, path: &'ast syn::Path) {
        self.matches_decoded_node |= path_ends_with(path, "DecodedNode");
        self.returns_some |= path_ends_with(path, "Some");
        self.returns_none |= path_ends_with(path, "None");
        visit::visit_path(self, path);
    }
}

fn private_optional_decoded_adapters(source: &str) -> Vec<String> {
    let file = syn::parse_file(source).expect("linearity source parses as Rust");

    #[derive(Default)]
    struct PrivateAdapterScan {
        names: Vec<String>,
    }

    impl PrivateAdapterScan {
        fn inspect(
            &mut self,
            visibility: &syn::Visibility,
            signature: &syn::Signature,
            block: &syn::Block,
        ) {
            if !matches!(visibility, syn::Visibility::Inherited) {
                return;
            }
            let syn::ReturnType::Type(_, ty) = &signature.output else {
                return;
            };
            if !optional_decoded_parts_return(ty) {
                return;
            }
            let mut body = DecodedAdapterBody::default();
            body.visit_block(block);
            if body.calls_carrier
                && body.matches_decoded_node
                && body.returns_some
                && body.returns_none
            {
                self.names.push(signature.ident.to_string());
            }
        }
    }

    impl<'ast> Visit<'ast> for PrivateAdapterScan {
        fn visit_item_fn(&mut self, function: &'ast syn::ItemFn) {
            self.inspect(&function.vis, &function.sig, &function.block);
            visit::visit_item_fn(self, function);
        }

        fn visit_impl_item_fn(&mut self, function: &'ast syn::ImplItemFn) {
            self.inspect(&function.vis, &function.sig, &function.block);
            visit::visit_impl_item_fn(self, function);
        }
    }

    let mut scan = PrivateAdapterScan::default();
    scan.visit_file(&file);
    scan.names
}

fn legacy_metadata(metadata: &Metadata) -> Metadata {
    metadata
        .map_expressions(&mut |value, _| legacy_expr(value))
        .expect("legacy fixture preserves metadata payloads")
}

fn legacy_expr(expr: &Expr) -> Expr {
    match expr {
        Expr::Atom(_, _) => expr.clone(),
        Expr::List(list, span) => Expr::List(
            List {
                elements: list.elements.iter().map(legacy_expr).collect(),
            },
            *span,
        ),
        Expr::Map(metadata, span) => Expr::Map(legacy_metadata(metadata), *span),
        Expr::MetaExpr(metadata_expr, span) => Expr::MetaExpr(
            chelis_deep::MetaExpr {
                metadata: legacy_metadata(&metadata_expr.metadata),
                expr: Box::new(legacy_expr(&metadata_expr.expr)),
            },
            *span,
        ),
        Expr::Node(node, span) => {
            let mut elements = Vec::with_capacity(node.child_count() + 2);
            elements.push(Expr::Atom(Atom::Tag(node.tag()), *span));
            elements.push(Expr::Map(legacy_metadata(node.meta()), *span));
            elements.extend(node.children_slice().iter().map(legacy_expr));
            Expr::List(List { elements }, *span)
        }
        Expr::BareList(elements, span) => {
            Expr::BareList(elements.iter().map(legacy_expr).collect(), *span)
        }
        Expr::UnknownForm(data) => Expr::UnknownForm(Box::new(chelis_deep::UnknownFormData {
            head: data.head.clone(),
            meta: legacy_metadata(&data.meta),
            children: data.children.iter().map(legacy_expr).collect(),
            span: data.span,
        })),
    }
}

fn checked_program(source: &str, legacy: bool) -> CheckedProgram {
    let declarations = parse_str(source).expect("Surf fixture parses");
    let successor = desugar_program(&declarations);
    if legacy {
        let legacy = successor.iter().map(legacy_expr).collect::<Vec<_>>();
        check_ir_program(&legacy).expect("legacy fixture type-checks")
    } else {
        check_typed_program(&successor).expect("successor fixture type-checks")
    }
}

fn ordered_linearity_diagnostics(program: &CheckedProgram) -> Vec<String> {
    match check_linearity(program) {
        Ok(_) => Vec::new(),
        Err(errors) => errors
            .into_iter()
            .map(|error| format!("[{:?}] {}", error.kind, error.message))
            .collect(),
    }
}

#[test]
fn nested_destructure_alias_single_consume_matches_successor_and_legacy() {
    let source = r#"
def ok(pair: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] =
  {
    (left, right) = pair
    alias: tensor[4, f32] = left
    consumed_left: tensor[4, f32] = realize(alias)
    consumed_right: tensor[4, f32] = realize(right)
    add(consumed_left, consumed_right)
  }
"#;

    let successor = ordered_linearity_diagnostics(&checked_program(source, false));
    let legacy = ordered_linearity_diagnostics(&checked_program(source, true));
    assert!(successor.is_empty(), "positive control: {successor:?}");
    assert_eq!(
        successor, legacy,
        "carrier choice must not change acceptance"
    );
}

#[test]
fn nested_destructure_alias_reuse_preserves_diagnostic_order_across_carriers() {
    let source = r#"
def bad(pair: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] =
  {
    (left, right) = pair
    alias: tensor[4, f32] = left
    consumed_left: tensor[4, f32] = realize(alias)
    consumed_right: tensor[4, f32] = realize(right)
    add(left, consumed_left)
  }
"#;

    let successor = ordered_linearity_diagnostics(&checked_program(source, false));
    let legacy = ordered_linearity_diagnostics(&checked_program(source, true));
    assert_eq!(
        successor, legacy,
        "binding identity, diagnostic ownership, and order must match"
    );
    assert!(
        successor.iter().any(|message| {
            message.contains("[UseAfterConsume]")
                && message.contains("variable `left`")
                && message.contains("consumed by realize")
        }),
        "negative control must retain the component/alias consume diagnostic: {successor:?}"
    );
}

#[test]
fn linearity_reader_has_no_private_optional_adapter_or_node_bridge() {
    let source = include_str!("../src/linearity.rs");
    let production = source
        .split("#[cfg(test)]")
        .next()
        .expect("linearity source has a production prefix");

    assert!(
        private_optional_decoded_adapters(production).is_empty(),
        "E5b forbids private Option adapters that can erase a non-decoded carrier"
    );
    assert!(
        !production.contains(".to_list("),
        "E5b forbids Node-to-List bridges in semantic linearity readers"
    );
    assert!(
        !production.contains("Expr::List("),
        "linearity readers must disposition legacy lists through Expr::carrier"
    );
    for disposition in [
        "ExprCarrier::DecodedNode",
        "ExprCarrier::StructuralList",
        "ExprCarrier::UndecodableHead",
        "ExprCarrier::Atom",
        "ExprCarrier::MetadataMap",
        "ExprCarrier::MetadataExpression",
        "ExprCarrier::MalformedLegacyList",
    ] {
        assert!(
            production.contains(disposition),
            "missing explicit carrier disposition `{disposition}`"
        );
    }
}

#[test]
fn optional_decoded_adapter_ratchet_is_name_independent() {
    let renamed_adapter = r#"
fn decoded_parts(expr: &Expr) -> Option<(DeepTag, &Metadata, &[Expr])> {
    match expr.carrier() {
        ExprCarrier::DecodedNode(tag, metadata, children) => Some((tag, metadata, children)),
        _ => None,
    }
}

fn pre_declare_one(expr: &Expr) {
    match decoded_parts(expr) {
        Some((DeepTag::Module, _, children)) => {
            for child in children {
                pre_declare_one(child);
            }
        }
        Some(_) | None => {}
    }
}
"#;
    assert_eq!(
        private_optional_decoded_adapters(renamed_adapter),
        vec!["decoded_parts"],
        "renaming the adapter and routing a reader through Some/None must not bypass the ratchet"
    );

    let unrelated_option = r#"
fn maybe_name(enabled: bool) -> Option<&'static str> {
    enabled.then_some("name")
}
"#;
    assert!(
        private_optional_decoded_adapters(unrelated_option).is_empty(),
        "ordinary Option helpers are outside the decoded-carrier adapter class"
    );
}
