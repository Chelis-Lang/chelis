//! chelis#1125 PP7/E5e slice 1: annotation-reader carrier parity.

use chelis_deep::{DeepTag, Expr, ExprCarrier, parse_and_stamp_file};
use chelis_types::errors::CheckError;
use chelis_types::{CheckedProgram, check_ir_program, check_typed_program};
use std::collections::BTreeSet;
use syn::visit::Visit;

fn legacy(source: &str) -> Vec<Expr> {
    chelis_deep::parser::parse_str(source).expect("fixture parses")
}

fn stamped(source: &str) -> Vec<Expr> {
    parse_and_stamp_file(source).expect("fixture stamps")
}

fn messages(errors: &[CheckError]) -> Vec<String> {
    errors
        .iter()
        .map(|error| format!("[{:?}] {}", error.kind, error.message))
        .collect()
}

fn checked_pair(source: &str) -> (CheckedProgram, CheckedProgram) {
    let ir = check_ir_program(&legacy(source)).expect("legacy ingress accepts fixture");
    let typed = check_typed_program(&stamped(source)).expect("stamped ingress accepts fixture");
    (ir, typed)
}

fn rejection_pair(source: &str) -> (Vec<String>, Vec<String>) {
    let ir = check_ir_program(&legacy(source))
        .expect_err("legacy ingress rejects fixture")
        .errors;
    let typed = check_typed_program(&stamped(source))
        .expect_err("stamped ingress rejects fixture")
        .errors;
    (messages(&ir), messages(&typed))
}

fn symbol(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(chelis_deep::Atom::Name(name), _) => Some(name),
        _ => None,
    }
}

fn named_def<'a>(program: &'a CheckedProgram, name: &str) -> &'a Expr {
    program
        .annotated_exprs()
        .iter()
        .find(|expr| {
            matches!(
                expr.carrier(),
                ExprCarrier::DecodedNode(DeepTag::Def, _, children)
                    if children.first().and_then(symbol) == Some(name)
            )
        })
        .expect("annotated definition exists")
}

fn function_params<'a>(program: &'a CheckedProgram, def_name: &str) -> &'a Expr {
    let ExprCarrier::DecodedNode(DeepTag::Def, _, def_children) =
        named_def(program, def_name).carrier()
    else {
        unreachable!("named_def returns a definition")
    };
    let ExprCarrier::DecodedNode(DeepTag::Fn, _, fn_children) = def_children[1].carrier() else {
        panic!("definition body remains a function carrier")
    };
    &fn_children[0]
}

fn parameter_type(program: &CheckedProgram, def_name: &str, param_name: &str) -> String {
    let ExprCarrier::DecodedNode(DeepTag::Params, _, params) =
        function_params(program, def_name).carrier()
    else {
        panic!("function parameters remain a decoded params carrier")
    };
    let (name, metadata) = match params[0].carrier() {
        ExprCarrier::UndecodableHead(name, metadata, children) => {
            assert!(children.is_empty(), "parameter metadata is not a child");
            (name, metadata)
        }
        ExprCarrier::StructuralList(elements) => match elements {
            [
                Expr::Atom(chelis_deep::Atom::Name(name), _),
                Expr::Map(metadata, _),
            ] => (name.as_str(), metadata),
            other => panic!("unexpected stamped parameter shape: {other:?}"),
        },
        other => panic!("annotated parameter has the wrong carrier: {other:?}"),
    };
    assert_eq!(name, param_name);
    chelis_deep::printer::print_expr_flat(
        metadata
            .ty()
            .expect("declared parameter type is written")
            .expression(),
    )
}

#[test]
fn declared_function_parameter_types_are_written_on_both_carriers() {
    let source = "
        (defsig {} identity
          (t-fn {} (t-prim {} i32) (t-prim {} i32)))
        (def {} identity
          (fn {} (params {} value) (var {} value)))";
    let (ir, typed) = checked_pair(source);

    assert_eq!(parameter_type(&ir, "identity", "value"), "(t-prim {} i32)");
    assert_eq!(
        parameter_type(&typed, "identity", "value"),
        "(t-prim {} i32)"
    );

    for (ingress, program) in [("lenient", &ir), ("stamped", &typed)] {
        assert!(
            matches!(function_params(program, "identity"), Expr::Node(_, _)),
            "{ingress} ingress must retain its Params node"
        );
        let ExprCarrier::DecodedNode(DeepTag::Def, _, def) =
            named_def(program, "identity").carrier()
        else {
            unreachable!()
        };
        assert!(
            matches!(def[1], Expr::Node(_, _)),
            "{ingress} annotation must preserve the stamped function carrier"
        );
    }
}

#[test]
fn declared_function_body_mismatch_has_ordered_ingress_parity() {
    let source = "
        (defsig {} identity
          (t-fn {} (t-prim {} i32) (t-prim {} i32)))
        (def {} identity
          (fn {} (params {} value)
            (lit {type: (t-prim {} bool)} true)))";
    let (ir, typed) = rejection_pair(source);

    assert_eq!(typed, ir, "diagnostic order is part of PP7 parity");
    assert!(
        typed
            .iter()
            .any(|message| message.contains("doesn't match declared signature")),
        "{typed:?}"
    );
}

const MATCH_PREFIX: &str = "
    (deftype {} Choice () (variant {} Left) (variant {} Right))
    (defsig {} choose
      (t-fn {} (t-adt {} Choice) (t-prim {} i32)))
    (def {} choose
      (fn {} (params {} choice)
        (match {} (var {} choice)";

#[test]
fn empty_match_guards_are_structural_on_both_ingresses() {
    let source = format!(
        "{MATCH_PREFIX}
          (arm {{}} (pat-ctor {{}} Left) () (lit {{type: (t-prim {{}} i32)}} 1))
          (arm {{}} (pat-ctor {{}} Right) () (lit {{type: (t-prim {{}} i32)}} 2)))))"
    );
    let _ = checked_pair(&source);
}

#[test]
fn non_boolean_match_guards_have_ordered_ingress_parity() {
    let source = format!(
        "{MATCH_PREFIX}
          (arm {{}} (pat-ctor {{}} Left)
            (lit {{type: (t-prim {{}} i32)}} 1)
            (lit {{type: (t-prim {{}} i32)}} 1))
          (arm {{}} (pat-ctor {{}} Right) () (lit {{type: (t-prim {{}} i32)}} 2)))))"
    );
    let (ir, typed) = rejection_pair(&source);

    assert_eq!(typed, ir, "diagnostic order is part of PP7 parity");
    assert!(
        typed
            .iter()
            .any(|message| message.contains("match arm guard must be bool")),
        "{typed:?}"
    );
}

#[test]
fn present_structural_match_guards_reject_on_both_ingresses() {
    use chelis_deep::{Atom, Metadata, Span};

    let span = Span::new(0, 0);
    let name = |value: &str| Expr::Atom(Atom::Name(value.to_string()), span);
    let int = |value| Expr::Atom(Atom::Int(value), span);
    let stamped_node = |tag, children| Expr::node(tag, Metadata::default(), children, span);

    let stamped_program = vec![stamped_node(
        DeepTag::Def,
        vec![
            name("guarded"),
            stamped_node(
                DeepTag::Match,
                vec![
                    stamped_node(DeepTag::Lit, vec![int(0)]),
                    stamped_node(
                        DeepTag::Arm,
                        vec![
                            stamped_node(DeepTag::PatWild, vec![]),
                            Expr::BareList(vec![int(1)], span),
                            stamped_node(DeepTag::Lit, vec![int(2)]),
                        ],
                    ),
                ],
            ),
        ],
    )];

    let ir_errors = check_ir_program(&stamped_program)
        .expect_err("a present structural guard must reject on the IR ingress")
        .errors;
    let stamped_errors = check_typed_program(&stamped_program)
        .expect_err("a present structural guard must reject")
        .errors;
    assert!(!ir_errors.is_empty());
    assert!(!stamped_errors.is_empty());
}

#[derive(Default)]
struct DispatchMatches<'ast> {
    node: Vec<&'ast syn::ExprMatch>,
}

fn path_name(expr: &syn::Expr) -> Option<String> {
    let syn::Expr::Path(path) = expr else {
        return None;
    };
    path.path
        .segments
        .last()
        .map(|segment| segment.ident.to_string())
}

impl<'ast> Visit<'ast> for DispatchMatches<'ast> {
    fn visit_expr_match(&mut self, node: &'ast syn::ExprMatch) {
        match node.expr.as_ref() {
            syn::Expr::MethodCall(call)
                if call.method == "tag"
                    && path_name(call.receiver.as_ref()).as_deref() == Some("node") =>
            {
                self.node.push(node);
            }
            _ => {}
        }
        syn::visit::visit_expr_match(self, node);
    }
}

#[derive(Default)]
struct DeepTags(BTreeSet<String>);

impl<'ast> Visit<'ast> for DeepTags {
    fn visit_path(&mut self, path: &'ast syn::Path) {
        let segments = path.segments.iter().collect::<Vec<_>>();
        if segments.len() >= 2 && segments[segments.len() - 2].ident == "DeepTag" {
            self.0
                .insert(segments[segments.len() - 1].ident.to_string());
        }
        syn::visit::visit_path(self, path);
    }
}

fn dispatched_tags(dispatch: &syn::ExprMatch) -> BTreeSet<String> {
    let mut result = BTreeSet::new();
    for arm in &dispatch.arms {
        let mut tags = DeepTags::default();
        tags.visit_pat(&arm.pat);
        for tag in tags.0 {
            assert!(
                result.insert(tag.clone()),
                "duplicate disposition for {tag}"
            );
        }
    }
    result
}

#[test]
fn root_node_dispatch_names_every_deep_tag() {
    let source = include_str!("../src/infer/expr.rs");
    let syntax = syn::parse_file(source).expect("inference source parses");
    let function = syntax
        .items
        .iter()
        .find_map(|item| match item {
            syn::Item::Fn(function)
                if function.sig.ident == "infer_expr_with_type_metadata_ownership" =>
            {
                Some(function)
            }
            _ => None,
        })
        .expect("root inference function remains present");
    let mut matches = DispatchMatches::default();
    matches.visit_block(&function.block);
    assert_eq!(matches.node.len(), 1, "expected one stamped Node dispatch");

    let vocabulary = DeepTag::ALL
        .iter()
        .map(|tag| format!("{tag:?}"))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        dispatched_tags(matches.node[0]),
        vocabulary,
        "the root dispatch must name every tag in an explicit arm, never absorb one in a catch-all"
    );
}
