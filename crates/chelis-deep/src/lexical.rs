//! Shared readers for parameter, pattern, and literal carriers.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::role::{AritySpec, arity_contract};
use crate::{Atom, DeepTag, Expr, ExprCarrier};

/// Read the parameter names through the canonical carrier boundary.
pub fn lexical_param_names(expr: &Expr) -> Vec<String> {
    let ExprCarrier::DecodedNode(DeepTag::Params, _, params) = expr.carrier() else {
        return Vec::new();
    };
    let valid = match arity_contract(DeepTag::Params) {
        AritySpec::Fixed(expected) => params.len() == expected,
        AritySpec::AtLeast(minimum) => params.len() >= minimum,
        AritySpec::Range(minimum, maximum) => params.len() >= minimum && params.len() <= maximum,
    };
    if !valid {
        return Vec::new();
    }
    params
        .iter()
        .filter_map(|param| lexical_param_name_and_type(param).map(|(name, _)| name.to_string()))
        .collect()
}

/// Read the names that the lexical pattern traversal binds.
///
/// The caller retains syntax admission and checked-program authority.
pub fn lexical_pattern_names(expr: &Expr) -> Vec<String> {
    let mut names = Vec::new();
    collect_pattern_names(expr, &mut names);
    names
}

fn collect_pattern_names(expr: &Expr, names: &mut Vec<String>) {
    let ExprCarrier::DecodedNode(tag, _, children) = expr.carrier() else {
        return;
    };
    match tag {
        DeepTag::PatVar => {
            if let Some(name) = children.first().and_then(symbol_name) {
                names.push(name.to_string());
            }
        }
        DeepTag::PatAs => {
            if let Some(name) = children.first().and_then(symbol_name) {
                names.push(name.to_string());
            }
            if let Some(inner) = children.get(1) {
                collect_pattern_names(inner, names);
            }
        }
        _ => {
            for child in children {
                collect_pattern_names(child, names);
            }
        }
    }
}

/// Read a parameter name and its optional declared type.
///
/// Metadata-wrapped names preserve parameters that collide with Deep tags.
/// The result describes syntax and grants no checked-type authority.
pub fn lexical_param_name_and_type(param: &Expr) -> Option<(&str, Option<&Expr>)> {
    match param.carrier() {
        ExprCarrier::Atom(Atom::Name(name)) => Some((name.as_str(), None)),
        ExprCarrier::StructuralList(elements) => {
            let name = elements.first().and_then(symbol_name)?;
            let ty = elements.get(1).and_then(|meta| {
                let Expr::Map(meta, _) = meta else {
                    return None;
                };
                meta.ty().map(|value| value.expression())
            });
            Some((name, ty))
        }
        ExprCarrier::UndecodableHead(name, metadata, []) => {
            Some((name, metadata.ty().map(|value| value.expression())))
        }
        ExprCarrier::MetadataExpression(meta) => {
            let Expr::Atom(Atom::Name(name), _) = meta.expr.as_ref() else {
                return None;
            };
            Some((
                name.as_str(),
                meta.metadata.ty().map(|value| value.expression()),
            ))
        }
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_) => None,
    }
}

/// Read a string atom or the first child of a decoded literal.
pub fn string_literal(expr: &Expr) -> Option<&str> {
    match expr.carrier() {
        ExprCarrier::Atom(Atom::Str(value)) => Some(value.as_str()),
        ExprCarrier::DecodedNode(DeepTag::Lit, _, children) => {
            children.first().and_then(|child| match child.carrier() {
                ExprCarrier::Atom(Atom::Str(value)) => Some(value.as_str()),
                ExprCarrier::DecodedNode(_, _, _)
                | ExprCarrier::StructuralList(_)
                | ExprCarrier::UndecodableHead(_, _, _)
                | ExprCarrier::Atom(_)
                | ExprCarrier::MetadataMap(_)
                | ExprCarrier::MetadataExpression(_) => None,
            })
        }
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => None,
    }
}

fn symbol_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Name(name), _) => Some(name.as_str()),
        _ => None,
    }
}
