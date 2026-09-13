//! Narrow source-keyed parity control for the vmap movement analysis.
//!
//! The expand/insert migration left two production dispatch families whose
//! keys are operation-name literals: recognizing a movement operation and
//! selecting its extent argument. Both names must remain in both functions.
//! This test intentionally does not inventory unrelated compiler literals.

use quote::ToTokens;
use syn::{Item, ItemFn};

const VMAP_EXTENT_SOURCE: &str = include_str!("../src/infer/vmap_extent.rs");
const REQUIRED_DISPATCH_FAMILIES: &[&str] = &["movement_operation", "movement_bound_dependencies"];
const REQUIRED_OPERATIONS: &[&str] = &["expand", "insert"];

fn function_tokens(source: &str, name: &str) -> Result<String, String> {
    let file = syn::parse_file(source).map_err(|error| format!("source did not parse: {error}"))?;
    let function = file.items.iter().find_map(|item| match item {
        Item::Fn(function) if function.sig.ident == name => Some(function),
        _ => None,
    });
    function
        .map(ItemFn::to_token_stream)
        .map(|tokens| tokens.to_string())
        .ok_or_else(|| format!("missing required dispatch family `{name}`"))
}

fn validate_dispatch_family_parity(source: &str) -> Result<(), String> {
    for family in REQUIRED_DISPATCH_FAMILIES {
        let tokens = function_tokens(source, family)?;
        for operation in REQUIRED_OPERATIONS {
            let literal = format!("\"{operation}\"");
            if !tokens.contains(&literal) {
                return Err(format!(
                    "dispatch family `{family}` is missing operation `{operation}`"
                ));
            }
        }
    }
    Ok(())
}

fn replace_function_literal(source: &str, family: &str, operation: &str) -> String {
    let function_start = source
        .find(&format!("fn {family}("))
        .unwrap_or_else(|| panic!("fixture is missing function `{family}`"));
    let function_end = source[function_start + 1..]
        .find("\nfn ")
        .map(|offset| function_start + 1 + offset)
        .unwrap_or(source.len());
    let function = &source[function_start..function_end];
    let literal = format!("\"{operation}\"");
    let literal_start = function
        .find(&literal)
        .unwrap_or_else(|| panic!("function `{family}` is missing literal `{operation}`"));

    let mut mutated = source.to_string();
    let start = function_start + literal_start;
    mutated.replace_range(
        start..start + literal.len(),
        &format!("\"__removed_{operation}__\""),
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
            let mutated = replace_function_literal(VMAP_EXTENT_SOURCE, family, operation);
            let error = validate_dispatch_family_parity(&mutated)
                .expect_err("removing a required operation must fail the parity control");
            assert!(
                error.contains(family) && error.contains(operation),
                "mutation failure must identify `{family}` / `{operation}`, got {error}"
            );
        }
    }
}
