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

use crate::ast::{Expr, Literal};

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
    resugar_node(node)
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
    let suffix = literal_suffix(node.meta)?;
    let literal = match (&node.children[0], suffix) {
        (DeepExpr::Atom(Atom::Int(value), _), None | Some(LiteralSuffix::I32)) => {
            Literal::Int(*value)
        }
        (DeepExpr::Atom(Atom::Int(value), _), Some(suffix)) => Literal::TypedInt(*value, suffix),
        (DeepExpr::Atom(Atom::Float(value), _), None | Some(LiteralSuffix::F32)) => {
            Literal::Float(*value)
        }
        (DeepExpr::Atom(Atom::Float(value), _), Some(suffix)) if suffix.is_float() => {
            Literal::TypedFloat(*value, suffix)
        }
        (DeepExpr::Atom(Atom::Str(value), _), None) => Literal::Str(value.clone()),
        (DeepExpr::Atom(Atom::Bool(value), _), None) => Literal::Bool(*value),
        (DeepExpr::BareList(items, _), None) if items.is_empty() => {
            return Ok(Expr::Tuple(Vec::new(), node.span));
        }
        (DeepExpr::List(list, _), None) if list.elements.is_empty() => {
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

fn literal_suffix(meta: &MetaMap) -> Result<Option<LiteralSuffix>, ResugarError> {
    let Some(value) = meta
        .entries
        .iter()
        .find(|(key, _)| key == "type")
        .map(|(_, value)| value)
    else {
        return Ok(None);
    };
    let Some(name) = primitive_type_name(value) else {
        return Err(ResugarError::InvalidChild {
            tag: DeepTag::Lit.as_str(),
            index: 1,
            expected: "primitive `type` metadata",
        });
    };
    let suffix = match name {
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
    Ok(Some(suffix))
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
