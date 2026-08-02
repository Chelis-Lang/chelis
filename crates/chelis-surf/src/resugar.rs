//! Typed Deep-to-Surf AST resugaring.
//!
//! This is the structural boundary shared by Deep emitters and the canonical
//! Surf printer.  The foundation implements only the expression forms needed
//! to close the known `cast` and `par` emitter failures.  Every other Deep tag
//! remains an explicit error until its spec-derived tests land in the v0.19
//! cutover; this module never substitutes `()` or a comment for source code.

use chelis_deep::ast::{Atom, Expr as DeepExpr, MetaMap};
use chelis_deep::{DeepTag, LiteralSuffix, Span};
use thiserror::Error;

use crate::ast::{Expr, Literal, TypeExpr};

/// Failure to structurally resugar a Deep expression.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ResugarError {
    #[error("expected a canonical Deep node, found {found}")]
    ExpectedNode { found: String },

    #[error("Deep `{tag}` expects exactly {expected} children, found {actual}")]
    ExactArity {
        tag: &'static str,
        expected: usize,
        actual: usize,
    },

    #[error("Deep `{tag}` expects at least {minimum} child{plural}, found {actual}", plural = if *minimum == 1 { "" } else { "ren" })]
    MinimumArity {
        tag: &'static str,
        minimum: usize,
        actual: usize,
    },

    #[error("Deep `{tag}` child {index} must be {expected}")]
    InvalidChild {
        tag: &'static str,
        index: usize,
        expected: &'static str,
    },

    #[error("Deep `{tag}` is outside the tested resugaring foundation")]
    OutsideFoundation { tag: &'static str },

    #[error("Deep name `{name}` is not a valid Surf {role} identifier")]
    InvalidSurfaceIdentifier { name: String, role: &'static str },

    #[error("foundation resugaring unexpectedly constructed unsupported Surf {kind}")]
    InvalidSurfaceAst { kind: &'static str },
}

#[derive(Clone, Copy)]
struct NodeRef<'a> {
    tag: DeepTag,
    meta: &'a MetaMap,
    children: &'a [DeepExpr],
    span: Span,
}

/// Resugar one Deep expression to the shared Surf AST.
///
/// The caller prints the result with [`crate::format::format_expression`].
/// Keeping construction and rendering separate makes it impossible for this
/// path to invent a second spelling for an AST construct.
pub fn resugar_expression(expr: &DeepExpr) -> Result<Expr, ResugarError> {
    let node = node_ref(expr)?;
    let annotation = if node.tag == DeepTag::Lit {
        None
    } else {
        meta_value(node.meta, "type")
            .map(resugar_type)
            .transpose()?
    };
    let span = node.span;
    let expression = resugar_node(node)?;
    let expression = match annotation {
        Some(ty) => Expr::Annotate(Box::new(expression), ty, span),
        None => expression,
    };
    validate_surface_expression(&expression)?;
    Ok(expression)
}

fn node_ref(expr: &DeepExpr) -> Result<NodeRef<'_>, ResugarError> {
    match expr {
        DeepExpr::Node(node, span) => Ok(NodeRef {
            tag: node.tag(),
            meta: node.meta(),
            children: node.children_slice(),
            span: *span,
        }),
        DeepExpr::List(list, span) => {
            let Some(DeepExpr::Atom(Atom::Tag(tag), _)) = list.elements.first() else {
                return Err(ResugarError::ExpectedNode {
                    found: describe_deep(expr),
                });
            };
            let Some(DeepExpr::Map(meta, _)) = list.elements.get(1) else {
                return Err(ResugarError::InvalidChild {
                    tag: tag.as_str(),
                    index: 1,
                    expected: "a metadata map",
                });
            };
            Ok(NodeRef {
                tag: *tag,
                meta,
                children: &list.elements[2..],
                span: *span,
            })
        }
        _ => Err(ResugarError::ExpectedNode {
            found: describe_deep(expr),
        }),
    }
}

fn resugar_node(node: NodeRef<'_>) -> Result<Expr, ResugarError> {
    use DeepTag as T;

    match node.tag {
        T::Var => {
            exact(&node, 1)?;
            let name = name_child(&node, 0)?;
            if starts_uppercase(name) {
                Ok(Expr::Constructor(name.to_string(), node.span))
            } else {
                Ok(Expr::Var(name.to_string(), node.span))
            }
        }
        T::Lit => resugar_literal(node),
        T::App => {
            at_least(&node, 1)?;
            let function = resugar_expression(&node.children[0])?;
            let arguments = node.children[1..]
                .iter()
                .map(resugar_expression)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Expr::Apply(Box::new(function), arguments, node.span))
        }
        T::Cast => {
            exact(&node, 2)?;
            let precision =
                primitive_type_name(&node.children[1]).ok_or(ResugarError::InvalidChild {
                    tag: node.tag.as_str(),
                    index: 1,
                    expected: "a `(t-prim {} precision)` node",
                })?;
            Ok(Expr::Cast(
                Box::new(resugar_expression(&node.children[0])?),
                precision.to_string(),
                node.span,
            ))
        }
        T::Par => {
            at_least(&node, 1)?;
            Ok(Expr::Par(
                node.children
                    .iter()
                    .map(resugar_expression)
                    .collect::<Result<Vec<_>, _>>()?,
                node.span,
            ))
        }

        // Compile-time-total disposition over the closed 62-tag vocabulary.
        // Adding support to this boundary requires moving a tag out of this
        // arm together with its positive and negative tests.
        T::Module
        | T::Import
        | T::ImportAll
        | T::Export
        | T::Def
        | T::Defsig
        | T::Deftype
        | T::Typealias
        | T::Variant
        | T::Field
        | T::Defdim
        | T::Fn
        | T::Let
        | T::Match
        | T::Arm
        | T::If
        | T::Record
        | T::Access
        | T::Pipe
        | T::Block
        | T::Tuple
        | T::TupleGet
        | T::RecordUpdate
        | T::HandleEffect
        | T::Borrow
        | T::PatVar
        | T::PatLit
        | T::PatCtor
        | T::PatTuple
        | T::PatRecord
        | T::PatWild
        | T::PatAs
        | T::TPrim
        | T::TFn
        | T::TTensor
        | T::TRef
        | T::TAdt
        | T::TVar
        | T::TUnit
        | T::TTuple
        | T::DName
        | T::DVar
        | T::DLit
        | T::DRank
        | T::Grad
        | T::Vmap
        | T::Jit
        | T::Realize
        | T::Copy
        | T::Quote
        | T::Unquote
        | T::Splice
        | T::Params
        | T::Bind
        | T::Kv
        | T::Effects
        | T::Resource => Err(ResugarError::OutsideFoundation {
            tag: node.tag.as_str(),
        }),
    }
}

fn resugar_literal(node: NodeRef<'_>) -> Result<Expr, ResugarError> {
    exact(&node, 1)?;
    let literal_type = literal_type(node.meta)?;
    let literal = match (&node.children[0], literal_type) {
        (DeepExpr::Atom(Atom::Int(value), _), LiteralType::Unspecified) => Literal::Int(*value),
        (DeepExpr::Atom(Atom::Int(value), _), LiteralType::Numeric(suffix)) => {
            Literal::TypedInt(*value, suffix)
        }
        (DeepExpr::Atom(Atom::Float(value), _), LiteralType::Unspecified) => Literal::Float(*value),
        (DeepExpr::Atom(Atom::Float(value), _), LiteralType::Numeric(suffix))
            if suffix.is_float() =>
        {
            Literal::TypedFloat(*value, suffix)
        }
        (DeepExpr::Atom(Atom::Str(value), _), LiteralType::Unspecified | LiteralType::String) => {
            Literal::Str(value.clone())
        }
        (DeepExpr::Atom(Atom::Bool(value), _), LiteralType::Unspecified | LiteralType::Bool) => {
            Literal::Bool(*value)
        }
        (DeepExpr::BareList(items, _), LiteralType::Unspecified | LiteralType::Unit)
            if items.is_empty() =>
        {
            return Ok(Expr::Tuple(Vec::new(), node.span));
        }
        (DeepExpr::List(list, _), LiteralType::Unspecified | LiteralType::Unit)
            if list.elements.is_empty() =>
        {
            return Ok(Expr::Tuple(Vec::new(), node.span));
        }
        _ => {
            return Err(ResugarError::InvalidChild {
                tag: node.tag.as_str(),
                index: 0,
                expected: "a literal compatible with its primitive `type` metadata",
            });
        }
    };
    Ok(Expr::Lit(literal, node.span))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LiteralType {
    Unspecified,
    Unit,
    Bool,
    String,
    Numeric(LiteralSuffix),
}

fn literal_type(meta: &MetaMap) -> Result<LiteralType, ResugarError> {
    let Some(value) = meta
        .entries
        .iter()
        .find(|(key, _)| key == "type")
        .map(|(_, value)| value)
    else {
        return Ok(LiteralType::Unspecified);
    };
    if let Ok(unit) = node_ref(value)
        && unit.tag == DeepTag::TUnit
    {
        exact(&unit, 0)?;
        return Ok(LiteralType::Unit);
    }
    let Some(name) = primitive_type_name(value) else {
        return Err(ResugarError::InvalidChild {
            tag: DeepTag::Lit.as_str(),
            index: 1,
            expected: "primitive or unit `type` metadata",
        });
    };
    let suffix = match name {
        "bool" => return Ok(LiteralType::Bool),
        "string" => return Ok(LiteralType::String),
        "f32" => LiteralSuffix::F32,
        "f64" => LiteralSuffix::F64,
        "bf16" => LiteralSuffix::Bf16,
        "f16" => LiteralSuffix::F16,
        "int8" => LiteralSuffix::I8,
        "int16" => LiteralSuffix::I16,
        "int32" => LiteralSuffix::I32,
        "int64" => LiteralSuffix::I64,
        _ => {
            return Err(ResugarError::InvalidChild {
                tag: DeepTag::Lit.as_str(),
                index: 1,
                expected: "a numeric primitive `type` metadata value",
            });
        }
    };
    Ok(LiteralType::Numeric(suffix))
}

fn meta_value<'a>(meta: &'a MetaMap, key: &str) -> Option<&'a DeepExpr> {
    meta.entries
        .iter()
        .find(|(candidate, _)| candidate == key)
        .map(|(_, value)| value)
}

fn resugar_type(expr: &DeepExpr) -> Result<TypeExpr, ResugarError> {
    let node = node_ref(expr)?;
    match node.tag {
        DeepTag::TPrim => {
            exact(&node, 1)?;
            Ok(TypeExpr::Named(
                name_child(&node, 0)?.to_string(),
                node.span,
            ))
        }
        DeepTag::TUnit => {
            exact(&node, 0)?;
            Ok(TypeExpr::Tuple(Vec::new(), node.span))
        }
        _ => Err(ResugarError::OutsideFoundation {
            tag: node.tag.as_str(),
        }),
    }
}

fn validate_surface_expression(expression: &Expr) -> Result<(), ResugarError> {
    match expression {
        Expr::Lit(..) => Ok(()),
        Expr::Var(name, _) => require_name(name, "value", is_lower_identifier),
        Expr::Constructor(name, _) => require_name(name, "constructor", is_qualified_type_name),
        Expr::Apply(function, arguments, _) => {
            validate_surface_expression(function)?;
            for argument in arguments {
                validate_surface_expression(argument)?;
            }
            Ok(())
        }
        Expr::Cast(value, precision, _) => {
            validate_surface_expression(value)?;
            require_name(precision, "precision", is_lower_identifier)
        }
        Expr::Par(values, _) | Expr::Tuple(values, _) => {
            for value in values {
                validate_surface_expression(value)?;
            }
            Ok(())
        }
        Expr::Annotate(value, ty, _) => {
            validate_surface_expression(value)?;
            validate_surface_type(ty)
        }
        _ => Err(ResugarError::InvalidSurfaceAst { kind: "expression" }),
    }
}

fn validate_surface_type(ty: &TypeExpr) -> Result<(), ResugarError> {
    match ty {
        TypeExpr::Named(name, _) => require_name(name, "type", is_lower_identifier),
        TypeExpr::Tuple(types, _) if types.is_empty() => Ok(()),
        _ => Err(ResugarError::InvalidSurfaceAst { kind: "type" }),
    }
}

fn require_name(
    name: &str,
    role: &'static str,
    predicate: impl FnOnce(&str) -> bool,
) -> Result<(), ResugarError> {
    if predicate(name) {
        Ok(())
    } else {
        Err(ResugarError::InvalidSurfaceIdentifier {
            name: name.to_string(),
            role,
        })
    }
}

fn is_lower_identifier(name: &str) -> bool {
    let mut bytes = name.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    (first == b'_' || first.is_ascii_lowercase())
        && name != "_"
        && bytes.all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
        && !is_reserved_word(name)
}

fn is_type_identifier(name: &str) -> bool {
    let mut bytes = name.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    first.is_ascii_uppercase() && bytes.all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
}

fn is_qualified_type_name(name: &str) -> bool {
    name.split('.').all(is_type_identifier)
}

fn is_reserved_word(name: &str) -> bool {
    matches!(
        name,
        "def"
            | "sig"
            | "type"
            | "dim"
            | "macro"
            | "match"
            | "with"
            | "fn"
            | "module"
            | "import"
            | "if"
            | "then"
            | "else"
            | "grad"
            | "vmap"
            | "jit"
            | "realize"
            | "copy"
            | "tensor"
            | "cast"
            | "export"
            | "par"
            | "true"
            | "false"
    )
}

fn primitive_type_name(expr: &DeepExpr) -> Option<&str> {
    let node = node_ref(expr).ok()?;
    (node.tag == DeepTag::TPrim && node.children.len() == 1)
        .then(|| atom_name(&node.children[0]))
        .flatten()
}

fn name_child<'a>(node: &NodeRef<'a>, index: usize) -> Result<&'a str, ResugarError> {
    node.children
        .get(index)
        .and_then(atom_name)
        .ok_or(ResugarError::InvalidChild {
            tag: node.tag.as_str(),
            index,
            expected: "a name atom",
        })
}

fn atom_name(expr: &DeepExpr) -> Option<&str> {
    match expr {
        DeepExpr::Atom(Atom::Name(name), _) => Some(name),
        _ => None,
    }
}

fn exact(node: &NodeRef<'_>, expected: usize) -> Result<(), ResugarError> {
    if node.children.len() == expected {
        Ok(())
    } else {
        Err(ResugarError::ExactArity {
            tag: node.tag.as_str(),
            expected,
            actual: node.children.len(),
        })
    }
}

fn at_least(node: &NodeRef<'_>, minimum: usize) -> Result<(), ResugarError> {
    if node.children.len() >= minimum {
        Ok(())
    } else {
        Err(ResugarError::MinimumArity {
            tag: node.tag.as_str(),
            minimum,
            actual: node.children.len(),
        })
    }
}

fn starts_uppercase(name: &str) -> bool {
    name.chars().next().is_some_and(char::is_uppercase)
}

fn describe_deep(expr: &DeepExpr) -> String {
    match expr {
        DeepExpr::Atom(_, _) => "an atom".to_string(),
        DeepExpr::List(_, _) => "an untagged list".to_string(),
        DeepExpr::Map(_, _) => "a metadata map".to_string(),
        DeepExpr::MetaExpr(_, _) => "a legacy metadata wrapper".to_string(),
        DeepExpr::Node(_, _) => "a typed node".to_string(),
        DeepExpr::BareList(_, _) => "a structural bare list".to_string(),
        DeepExpr::UnknownForm(data) => format!("unknown form `{}`", data.head),
    }
}
