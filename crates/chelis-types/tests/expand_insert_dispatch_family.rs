//! Narrow source-keyed parity control for the vmap movement analysis.
//!
//! The expand/insert migration left two production dispatch families whose
//! keys are operation-name literals: recognizing a movement operation and
//! selecting its extent argument. Both names must remain in both functions.
//! This test intentionally does not inventory unrelated compiler literals.

use std::collections::BTreeSet;

use syn::parse::Parser;
use syn::visit::{self, Visit};
use syn::{Expr, ExprMacro, ExprMatch, Item, ItemFn, Lit, Pat, Token};

const VMAP_EXTENT_SOURCE: &str = include_str!("../src/infer/vmap_extent.rs");
const REQUIRED_DISPATCH_FAMILIES: &[&str] = &["movement_operation", "movement_bound_dependencies"];
const REQUIRED_OPERATIONS: &[&str] = &["expand", "insert"];

fn function(source: &str, name: &str) -> Result<ItemFn, String> {
    let file = syn::parse_file(source).map_err(|error| format!("source did not parse: {error}"))?;
    file.items
        .into_iter()
        .find_map(|item| match item {
            Item::Fn(function) if function.sig.ident == name => Some(function),
            _ => None,
        })
        .ok_or_else(|| format!("missing required dispatch family `{name}`"))
}

fn expr_is_ident(expr: &Expr, expected: &str) -> bool {
    matches!(
        expr,
        Expr::Path(path)
            if path.qself.is_none()
                && path.path.segments.len() == 1
                && path.path.is_ident(expected)
    )
}

fn string_literals(pattern: &Pat, literals: &mut BTreeSet<String>) {
    match pattern {
        Pat::Lit(pattern) => {
            if let Lit::Str(literal) = &pattern.lit {
                literals.insert(literal.value());
            }
        }
        Pat::Or(pattern) => {
            for case in &pattern.cases {
                string_literals(case, literals);
            }
        }
        _ => {}
    }
}

fn parse_matches_pattern(node: &ExprMacro, subject: &str) -> Result<Option<Pat>, String> {
    if !node.mac.path.is_ident("matches") {
        return Ok(None);
    }
    let parser = |input: syn::parse::ParseStream<'_>| {
        let expression: Expr = input.parse()?;
        input.parse::<Token![,]>()?;
        let pattern = Pat::parse_multi_with_leading_vert(input)?;
        if !input.is_empty() {
            return Err(input.error("guarded or trailing matches! syntax is not supported here"));
        }
        Ok((expression, pattern))
    };
    let (expression, pattern) = parser
        .parse2(node.mac.tokens.clone())
        .map_err(|error| format!("could not parse matches! dispatch: {error}"))?;
    Ok(expr_is_ident(&expression, subject).then_some(pattern))
}

#[derive(Default)]
struct MovementOperationPatterns {
    patterns: Vec<Pat>,
    errors: Vec<String>,
}

impl<'ast> Visit<'ast> for MovementOperationPatterns {
    fn visit_expr_macro(&mut self, node: &'ast ExprMacro) {
        match parse_matches_pattern(node, "name") {
            Ok(Some(pattern)) => self.patterns.push(pattern),
            Ok(None) => {}
            Err(error) => self.errors.push(error),
        }
        visit::visit_expr_macro(self, node);
    }
}

fn required_movement_operation_literals(source: &str) -> Result<BTreeSet<String>, String> {
    let function = function(source, "movement_operation")?;
    let mut visitor = MovementOperationPatterns::default();
    visitor.visit_item_fn(&function);
    if let Some(error) = visitor.errors.into_iter().next() {
        return Err(error);
    }
    let [pattern] = visitor.patterns.as_slice() else {
        return Err(format!(
            "dispatch family `movement_operation` must contain exactly one matches!(name, ...) pattern; found {}",
            visitor.patterns.len()
        ));
    };
    let mut literals = BTreeSet::new();
    string_literals(pattern, &mut literals);
    Ok(literals)
}

fn is_args_get_two(expr: &Expr) -> bool {
    let Expr::MethodCall(into_iter) = expr else {
        return false;
    };
    if into_iter.method != "into_iter" || !into_iter.args.is_empty() {
        return false;
    }
    let Expr::MethodCall(get) = into_iter.receiver.as_ref() else {
        return false;
    };
    if get.method != "get" || !expr_is_ident(&get.receiver, "args") || get.args.len() != 1 {
        return false;
    }
    matches!(
        get.args.first(),
        Some(Expr::Lit(literal))
            if matches!(&literal.lit, Lit::Int(index) if index.base10_digits() == "2")
    )
}

fn selects_third_argument(expr: &Expr) -> bool {
    let Expr::Call(call) = expr else {
        return false;
    };
    let Expr::Path(function) = call.func.as_ref() else {
        return false;
    };
    function
        .path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "new")
        && call.args.len() == 1
        && call.args.first().is_some_and(is_args_get_two)
}

#[derive(Default)]
struct BoundDependencyPatterns {
    patterns: Vec<Pat>,
}

impl<'ast> Visit<'ast> for BoundDependencyPatterns {
    fn visit_expr_match(&mut self, node: &'ast ExprMatch) {
        if expr_is_ident(&node.expr, "operation") {
            for arm in &node.arms {
                if selects_third_argument(&arm.body) {
                    self.patterns.push(arm.pat.clone());
                }
            }
        }
        visit::visit_expr_match(self, node);
    }
}

fn required_bound_dependency_literals(source: &str) -> Result<BTreeSet<String>, String> {
    let function = function(source, "movement_bound_dependencies")?;
    let mut visitor = BoundDependencyPatterns::default();
    visitor.visit_item_fn(&function);
    let [pattern] = visitor.patterns.as_slice() else {
        return Err(format!(
            "dispatch family `movement_bound_dependencies` must contain exactly one operation arm selecting args.get(2); found {}",
            visitor.patterns.len()
        ));
    };
    let mut literals = BTreeSet::new();
    string_literals(pattern, &mut literals);
    Ok(literals)
}

fn validate_dispatch_family_parity(source: &str) -> Result<(), String> {
    let families = [
        (
            "movement_operation",
            required_movement_operation_literals(source)?,
        ),
        (
            "movement_bound_dependencies",
            required_bound_dependency_literals(source)?,
        ),
    ];
    for (family, literals) in families {
        for operation in REQUIRED_OPERATIONS {
            if !literals.contains(*operation) {
                return Err(format!(
                    "dispatch family `{family}` is missing operation `{operation}`"
                ));
            }
        }
    }
    Ok(())
}

fn replace_once(source: &str, original: &str, replacement: &str) -> String {
    assert_eq!(
        source.matches(original).count(),
        1,
        "fixture must contain exactly one `{original}`"
    );
    source.replacen(original, replacement, 1)
}

fn remove_dispatch_operation(source: &str, family: &str, operation: &str) -> String {
    match (family, operation) {
        ("movement_operation", "expand") => replace_once(
            source,
            r#""expand" | "insert" | "reshape""#,
            r#""insert" | "reshape""#,
        ),
        ("movement_operation", "insert") => replace_once(
            source,
            r#""expand" | "insert" | "reshape""#,
            r#""expand" | "reshape""#,
        ),
        ("movement_bound_dependencies", "expand") => replace_once(
            source,
            r#""expand" | "insert" => Box::new(args.get(2).into_iter())"#,
            r#""insert" => Box::new(args.get(2).into_iter())"#,
        ),
        ("movement_bound_dependencies", "insert") => replace_once(
            source,
            r#""expand" | "insert" => Box::new(args.get(2).into_iter())"#,
            r#""expand" => Box::new(args.get(2).into_iter())"#,
        ),
        _ => panic!("unsupported mutation `{family}` / `{operation}`"),
    }
}

fn plant_decoy_literal(source: &str, family: &str, operation: &str) -> String {
    let function_start = source
        .find(&format!("fn {family}"))
        .unwrap_or_else(|| panic!("fixture is missing function `{family}`"));
    let block_start = source[function_start..]
        .find('{')
        .map(|offset| function_start + offset)
        .unwrap_or_else(|| panic!("function `{family}` has no body"));
    let subject = match family {
        "movement_operation" => "name",
        "movement_bound_dependencies" => "operation",
        _ => panic!("unsupported dispatch family `{family}`"),
    };
    let mut mutated = source.to_string();
    mutated.insert_str(
        block_start + 1,
        &format!("\n    let _operation_label_for_diagnostics = {subject} == \"{operation}\";"),
    );
    mutated
}

#[test]
fn vmap_movement_dispatch_families_cover_expand_and_insert() {
    validate_dispatch_family_parity(VMAP_EXTENT_SOURCE)
        .expect("every required source-keyed family must cover both movement operations");
}

#[test]
fn each_dispatch_family_operation_is_mutation_live() {
    for family in REQUIRED_DISPATCH_FAMILIES {
        for operation in REQUIRED_OPERATIONS {
            let mutated = remove_dispatch_operation(VMAP_EXTENT_SOURCE, family, operation);
            let error = validate_dispatch_family_parity(&mutated)
                .expect_err("removing a required operation must fail the parity control");
            assert!(
                error.contains(family) && error.contains(operation),
                "mutation failure must identify `{family}` / `{operation}`, got {error}"
            );
        }
    }
}

#[test]
fn unrelated_literals_cannot_authorize_a_dispatch_family() {
    for family in REQUIRED_DISPATCH_FAMILIES {
        for operation in REQUIRED_OPERATIONS {
            let removed = remove_dispatch_operation(VMAP_EXTENT_SOURCE, family, operation);
            let with_decoy = plant_decoy_literal(&removed, family, operation);
            let error = validate_dispatch_family_parity(&with_decoy)
                .expect_err("a decoy literal outside the dispatch pattern must not satisfy parity");
            assert!(
                error.contains(family) && error.contains(operation),
                "decoy failure must identify `{family}` / `{operation}`, got {error}"
            );
        }
    }
}
