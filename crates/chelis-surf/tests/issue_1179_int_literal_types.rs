//! chelis#1179: integer literals are tensor dimensions, not types.
//!
//! `spec/02-surf-syntax.md` permits `IntLit` only inside a tensor type's
//! `DimList` (`DimExpr <- IntLit / Ident / '*' / '..' Ident`). Every other
//! type position takes a `TypeExpr`, which has no integer production, and
//! the Tree-sitter grammar rejects the spelling. The Rust parser previously
//! shared one integer arm between the two positions, so `def g(a: 732) ->
//! f32 = a` parsed silently and desugared to `(t-var {} 732)`.

use chelis_surf::ast::{Decl, TypeExpr};
use chelis_surf::parser::{parse_str, parse_str_legacy_v018};

fn assert_rejects_integer_type(source: &str) {
    let error = parse_str(source)
        .err()
        .unwrap_or_else(|| panic!("{source:?} must not parse"));
    let message = error.to_string();
    assert!(
        message.contains("integer literal"),
        "diagnostic should name the integer literal; got: {message}"
    );
    assert!(
        message.contains("tensor["),
        "diagnostic should point at the one position that takes integers; got: {message}"
    );
    assert!(
        message.contains("at byte"),
        "diagnostic should carry a source offset; got: {message}"
    );
}

#[test]
fn integer_literal_rejects_in_parameter_annotation() {
    assert_rejects_integer_type("def g(a: 732) -> f32 = a\n");
}

#[test]
fn integer_literal_rejects_in_return_type() {
    assert_rejects_integer_type("def g(a: f32) -> 732 = a\n");
}

#[test]
fn integer_literal_rejects_in_binding_annotation() {
    assert_rejects_integer_type("value: 732 = source\n");
}

#[test]
fn integer_literal_rejects_in_sig_type() {
    assert_rejects_integer_type("sig g: 732 -> f32\n");
}

#[test]
fn integer_literal_rejects_in_type_application_argument() {
    assert_rejects_integer_type("value: Option[732] = source\n");
}

#[test]
fn integer_literal_rejects_in_reference_type() {
    assert_rejects_integer_type("def g(a: &732) -> f32 = a\n");
}

#[test]
fn integer_literal_rejects_in_tuple_type() {
    assert_rejects_integer_type("value: (732, f32) = source\n");
}

#[test]
fn integer_literal_rejects_in_variant_field_type() {
    assert_rejects_integer_type("type Wrap = | Wrap(732)\n");
}

#[test]
fn integer_literal_rejects_in_legacy_mode() {
    // The restriction is grammar-level, not a canonicalization rule, so the
    // v0.18 migration parser rejects the same spelling.
    assert!(
        parse_str_legacy_v018("def g(a: 732) -> f32 = a\n").is_err(),
        "legacy parse mode must reject integer literals in type position"
    );
}

#[test]
fn tensor_dimension_integers_still_parse_in_parameter_annotation() {
    let decls = parse_str("def g(a: tensor[3, 4, f32]) -> tensor[3, 4, f32] = a\n")
        .expect("integer tensor dimensions parse");
    let Decl::FunDef { params, .. } = &decls[0] else {
        panic!("expected a function definition");
    };
    let Some(TypeExpr::Tensor(dims, precision, _)) = &params[0].ty else {
        panic!("expected a tensor parameter annotation");
    };
    assert_eq!(dims.len(), 2);
    assert!(matches!(&dims[0], TypeExpr::Named(n, _) if n == "3"));
    assert!(matches!(&dims[1], TypeExpr::Named(n, _) if n == "4"));
    assert_eq!(precision, "f32");
}

#[test]
fn tensor_dimension_integers_still_parse_in_binding_annotation() {
    let decls = parse_str("value: tensor[732, f32] = source\n")
        .expect("integer tensor dimension parses in a binding annotation");
    let Decl::LetDef { ty: Some(ty), .. } = &decls[0] else {
        panic!("expected an annotated binding");
    };
    let TypeExpr::Tensor(dims, precision, _) = ty else {
        panic!("expected a tensor annotation");
    };
    assert!(matches!(&dims[0], TypeExpr::Named(n, _) if n == "732"));
    assert_eq!(precision, "f32");
}
