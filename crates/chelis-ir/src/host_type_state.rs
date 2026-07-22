//! Typed host-type states for the chelis#730 Phase 2 migration.
//!
//! This module separates the three questions the legacy `HostType::Unknown`
//! sentinel answered at once:
//!
//! 1. what logical type term did the checker describe?
//! 2. has inference made that term fully concrete?
//! 3. can a particular backend represent the concrete type at its host ABI?
//!
//! Decode failures are not type terms. Polymorphic and inference variables are
//! not concrete types. Backend support is not a logical-type fact. Keeping
//! those boundaries distinct prevents a missing or unsupported type from being
//! converted into a plausible emitted value.

use std::fmt;

use chelis_deep::ast::{Atom, Expr, List};
use chelis_types::types::Prim;

use crate::dag::{DimInfo, TensorType};

/// A host type before polymorphism and inference have been resolved.
///
/// Unlike the legacy `HostType`, this vocabulary has no anonymous `Unknown`.
/// Every non-concrete state says why it is not concrete and preserves the
/// identity needed to resolve it.
#[derive(Debug, Clone, PartialEq)]
pub enum HostTypeTerm {
    Scalar(HostPrecisionTerm),
    Function(Vec<HostTypeTerm>, Box<HostTypeTerm>),
    Adt(String, Vec<HostTypeTerm>),
    List(Box<HostTypeTerm>),
    Dict(Box<HostTypeTerm>, Box<HostTypeTerm>),
    Tuple(Vec<HostTypeTerm>),
    Tensor(HostTensorTypeTerm),
    Option(Box<HostTypeTerm>),
    MappedFile,
    Unit,
    /// A named type-polymorphic variable from a signature.
    TypeVariable(String),
    /// An inference hole with stable identity. Empty collection literals use
    /// this state until context supplies their element type.
    InferenceVariable(HostInferenceVar),
    /// Bottom: an expression such as `fail` that does not produce a value.
    Never,
}

/// A scalar or tensor element precision before precision polymorphism has
/// been specialized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostPrecisionTerm {
    Concrete(Prim),
    Variable(String),
}

/// A tensor shape before rank polymorphism has been specialized.
#[derive(Debug, Clone, PartialEq)]
pub enum HostShapeTerm {
    Concrete(Vec<DimInfo>),
    /// A Tier-2/Tier-3 shape containing one or more named rank spreads.
    /// Concrete slots stay in order so `(d-rank pre) (d-name seq)
    /// (d-rank post)` does not collapse three distinct facts into one marker.
    Polymorphic(Vec<HostShapeSlot>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum HostShapeSlot {
    Dim(DimInfo),
    RankVariable(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct HostTensorTypeTerm {
    pub precision: HostPrecisionTerm,
    pub shape: HostShapeTerm,
}

/// Stable identity for an inference hole.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HostInferenceVar(pub u32);

/// A completely resolved logical host type.
///
/// Exact scalar precision survives here even when a backend does not yet have
/// a host representation for it. That target-specific decision belongs to the
/// backend's fallible `HostAbiType::try_from_concrete`, not parsing or
/// inference.
#[derive(Debug, Clone, PartialEq)]
pub enum ConcreteHostType {
    Scalar(Prim),
    Function(Vec<ConcreteHostType>, Box<ConcreteHostType>),
    Adt(String, Vec<ConcreteHostType>),
    List(Box<ConcreteHostType>),
    Dict(Box<ConcreteHostType>, Box<ConcreteHostType>),
    Tuple(Vec<ConcreteHostType>),
    Tensor(TensorType),
    Option(Box<ConcreteHostType>),
    MappedFile,
    Unit,
}

/// Why a logical type term could not be made concrete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostTypeResolutionError {
    UnresolvedTypeVariable {
        name: String,
    },
    UnresolvedPrecisionVariable {
        name: String,
    },
    UnresolvedRankVariable {
        name: String,
    },
    UnresolvedInferenceVariable {
        variable: HostInferenceVar,
    },
    /// A manually-constructed `HostShapeTerm::Polymorphic` violated its
    /// non-empty-rank-slot invariant. The Deep decoder cannot create this.
    InvalidPolymorphicShape,
    /// `Never` is valid during inference and joins, but it has no value
    /// representation of its own and must disappear before an ABI boundary.
    NeverReachedValueBoundary,
}

impl fmt::Display for HostTypeResolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnresolvedTypeVariable { name } => {
                write!(f, "unresolved host type variable `{name}`")
            }
            Self::UnresolvedPrecisionVariable { name } => {
                write!(f, "unresolved host precision variable `{name}`")
            }
            Self::UnresolvedRankVariable { name } => {
                write!(f, "unresolved host rank variable `{name}`")
            }
            Self::UnresolvedInferenceVariable { variable } => {
                write!(f, "unresolved host inference variable `{}`", variable.0)
            }
            Self::InvalidPolymorphicShape => {
                f.write_str("polymorphic host shape contains no rank-variable slot")
            }
            Self::NeverReachedValueBoundary => {
                f.write_str("bottom host type reached a value-representation boundary")
            }
        }
    }
}

impl std::error::Error for HostTypeResolutionError {}

/// Failure to decode checker metadata into a logical host type term.
///
/// Absence and malformed content are separate failures. Neither manufactures
/// an inference variable or a polymorphic term.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostTypeDecodeError {
    MissingTypeMetadata,
    MalformedTypeSyntax { detail: String },
    UnknownPrimitive { name: String },
}

impl fmt::Display for HostTypeDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingTypeMetadata => f.write_str("missing host type metadata"),
            Self::MalformedTypeSyntax { detail } => {
                write!(f, "malformed host type syntax: {detail}")
            }
            Self::UnknownPrimitive { name } => {
                write!(f, "unknown host primitive type `{name}`")
            }
        }
    }
}

impl std::error::Error for HostTypeDecodeError {}

impl HostTypeTerm {
    /// Resolve a term only when no polymorphic, inference, or bottom state
    /// remains. This function never chooses a default type.
    pub fn into_concrete(self) -> Result<ConcreteHostType, HostTypeResolutionError> {
        match self {
            Self::Scalar(HostPrecisionTerm::Concrete(precision)) => {
                Ok(ConcreteHostType::Scalar(precision))
            }
            Self::Scalar(HostPrecisionTerm::Variable(name)) => {
                Err(HostTypeResolutionError::UnresolvedPrecisionVariable { name })
            }
            Self::Function(params, ret) => Ok(ConcreteHostType::Function(
                params
                    .into_iter()
                    .map(Self::into_concrete)
                    .collect::<Result<Vec<_>, _>>()?,
                Box::new(ret.into_concrete()?),
            )),
            Self::Adt(name, args) => Ok(ConcreteHostType::Adt(
                name,
                args.into_iter()
                    .map(Self::into_concrete)
                    .collect::<Result<Vec<_>, _>>()?,
            )),
            Self::List(inner) => Ok(ConcreteHostType::List(Box::new(inner.into_concrete()?))),
            Self::Dict(key, value) => Ok(ConcreteHostType::Dict(
                Box::new(key.into_concrete()?),
                Box::new(value.into_concrete()?),
            )),
            Self::Tuple(items) => Ok(ConcreteHostType::Tuple(
                items
                    .into_iter()
                    .map(Self::into_concrete)
                    .collect::<Result<Vec<_>, _>>()?,
            )),
            Self::Tensor(HostTensorTypeTerm { precision, shape }) => {
                let precision = match precision {
                    HostPrecisionTerm::Concrete(precision) => precision,
                    HostPrecisionTerm::Variable(name) => {
                        return Err(HostTypeResolutionError::UnresolvedPrecisionVariable { name });
                    }
                };
                let dims = match shape {
                    HostShapeTerm::Concrete(dims) => dims,
                    HostShapeTerm::Polymorphic(slots) => {
                        let Some(name) = slots.into_iter().find_map(|slot| match slot {
                            HostShapeSlot::RankVariable(name) => Some(name),
                            HostShapeSlot::Dim(_) => None,
                        }) else {
                            return Err(HostTypeResolutionError::InvalidPolymorphicShape);
                        };
                        return Err(HostTypeResolutionError::UnresolvedRankVariable { name });
                    }
                };
                Ok(ConcreteHostType::Tensor(TensorType { dims, precision }))
            }
            Self::Option(inner) => Ok(ConcreteHostType::Option(Box::new(inner.into_concrete()?))),
            Self::MappedFile => Ok(ConcreteHostType::MappedFile),
            Self::Unit => Ok(ConcreteHostType::Unit),
            Self::TypeVariable(name) => {
                Err(HostTypeResolutionError::UnresolvedTypeVariable { name })
            }
            Self::InferenceVariable(variable) => {
                Err(HostTypeResolutionError::UnresolvedInferenceVariable { variable })
            }
            Self::Never => Err(HostTypeResolutionError::NeverReachedValueBoundary),
        }
    }
}

/// Decode one Deep type node into a logical host-type term.
///
/// This boundary preserves polymorphism and exact primitive identity. It does
/// not inspect backend capability and does not allocate inference variables:
/// the caller owns inference identity when a valid expression is
/// underconstrained.
pub fn decode_host_type(expr: &Expr) -> Result<HostTypeTerm, HostTypeDecodeError> {
    if let Expr::MetaExpr(meta, _) = expr {
        return decode_host_type(&meta.expr);
    }
    let list = match expr {
        Expr::List(list, _) => list,
        _ => return Err(malformed("expected a Deep type node")),
    };
    let tag = list_tag(list).ok_or_else(|| malformed("type node has no symbolic tag"))?;
    let children = list_children(list)?;
    match tag {
        "t-prim" => {
            let name = one_symbol_child(children, "t-prim")?;
            let precision =
                Prim::parse_name(name).ok_or_else(|| HostTypeDecodeError::UnknownPrimitive {
                    name: name.to_string(),
                })?;
            Ok(HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(precision)))
        }
        "t-var" => Ok(HostTypeTerm::TypeVariable(
            one_symbol_child(children, "t-var")?.to_string(),
        )),
        "t-ref" => {
            let inner = one_child(children, "t-ref")?;
            decode_host_type(inner)
        }
        "t-tensor" => decode_tensor_type(children),
        "t-adt" => decode_adt_type(children),
        "t-tuple" => Ok(HostTypeTerm::Tuple(
            children
                .iter()
                .map(decode_host_type)
                .collect::<Result<Vec<_>, _>>()?,
        )),
        "t-fn" => {
            let (ret, params) = children
                .split_last()
                .ok_or_else(|| malformed("t-fn has no return type"))?;
            Ok(HostTypeTerm::Function(
                params
                    .iter()
                    .map(decode_host_type)
                    .collect::<Result<Vec<_>, _>>()?,
                Box::new(decode_host_type(ret)?),
            ))
        }
        "t-unit" if children.is_empty() => Ok(HostTypeTerm::Unit),
        "t-unit" => Err(malformed("t-unit must not have children")),
        other => Err(malformed(format!("unknown type tag `{other}`"))),
    }
}

/// Decode the `type` entry on a Deep expression's metadata map.
///
/// Absence and invalid type syntax have separate errors. This is the boundary
/// that replaces `expr_type(...).unwrap_or(Unknown)` in the legacy lowerer.
pub fn decode_host_type_metadata(expr: &Expr) -> Result<HostTypeTerm, HostTypeDecodeError> {
    match expr {
        Expr::List(list, _) => {
            let metadata = match list.elements.get(1) {
                Some(Expr::Map(metadata, _)) => metadata,
                Some(_) => return Err(malformed("Deep node metadata slot is not a map")),
                None => return Err(HostTypeDecodeError::MissingTypeMetadata),
            };
            let type_expr = metadata
                .entries
                .iter()
                .find_map(|(key, value)| (key == "type").then_some(value))
                .ok_or(HostTypeDecodeError::MissingTypeMetadata)?;
            decode_host_type(type_expr)
        }
        Expr::MetaExpr(meta, _) => {
            if let Some(type_expr) = meta
                .entries
                .iter()
                .find_map(|(key, value)| (key == "type").then_some(value))
            {
                decode_host_type(type_expr)
            } else {
                decode_host_type_metadata(&meta.expr)
            }
        }
        _ => Err(HostTypeDecodeError::MissingTypeMetadata),
    }
}

fn decode_tensor_type(children: &[Expr]) -> Result<HostTypeTerm, HostTypeDecodeError> {
    let (precision_expr, dim_exprs) = children
        .split_last()
        .ok_or_else(|| malformed("t-tensor has no precision child"))?;
    let precision = decode_precision(precision_expr)?;
    let slots = dim_exprs
        .iter()
        .map(decode_shape_slot)
        .collect::<Result<Vec<_>, _>>()?;
    let shape = if slots
        .iter()
        .any(|slot| matches!(slot, HostShapeSlot::RankVariable(_)))
    {
        HostShapeTerm::Polymorphic(slots)
    } else {
        HostShapeTerm::Concrete(
            slots
                .into_iter()
                .map(|slot| match slot {
                    HostShapeSlot::Dim(dim) => dim,
                    HostShapeSlot::RankVariable(_) => {
                        unreachable!("rank-variable presence checked above")
                    }
                })
                .collect(),
        )
    };
    Ok(HostTypeTerm::Tensor(HostTensorTypeTerm {
        precision,
        shape,
    }))
}

fn decode_precision(expr: &Expr) -> Result<HostPrecisionTerm, HostTypeDecodeError> {
    let list = match expr {
        Expr::List(list, _) => list,
        _ => return Err(malformed("tensor precision is not a type node")),
    };
    let tag = list_tag(list).ok_or_else(|| malformed("precision node has no symbolic tag"))?;
    let children = list_children(list)?;
    match tag {
        "t-prim" => {
            let name = one_symbol_child(children, "tensor t-prim")?;
            Prim::parse_name(name)
                .map(HostPrecisionTerm::Concrete)
                .ok_or_else(|| HostTypeDecodeError::UnknownPrimitive {
                    name: name.to_string(),
                })
        }
        "t-var" => Ok(HostPrecisionTerm::Variable(
            one_symbol_child(children, "tensor t-var")?.to_string(),
        )),
        other => Err(malformed(format!(
            "tensor precision must be t-prim or t-var, got `{other}`"
        ))),
    }
}

fn decode_shape_slot(expr: &Expr) -> Result<HostShapeSlot, HostTypeDecodeError> {
    if let Expr::Atom(Atom::Symbol(name), _) = expr {
        return Ok(HostShapeSlot::Dim(DimInfo::Named(name.clone(), None)));
    }
    if let Expr::Atom(Atom::Int(value), _) = expr {
        return nonnegative_dim(*value);
    }
    let list = match expr {
        Expr::List(list, _) => list,
        _ => return Err(malformed("tensor dimension is not a dimension node")),
    };
    let tag = list_tag(list).ok_or_else(|| malformed("dimension node has no symbolic tag"))?;
    let children = list_children(list)?;
    match tag {
        "d-name" | "d-var" => Ok(HostShapeSlot::Dim(DimInfo::Named(
            one_symbol_child(children, tag)?.to_string(),
            None,
        ))),
        "d-rank" => Ok(HostShapeSlot::RankVariable(
            one_symbol_child(children, "d-rank")?.to_string(),
        )),
        "d-lit" => match one_child(children, "d-lit")? {
            Expr::Atom(Atom::Int(value), _) => nonnegative_dim(*value),
            _ => Err(malformed("d-lit child is not an integer")),
        },
        other => Err(malformed(format!("unknown dimension tag `{other}`"))),
    }
}

fn nonnegative_dim(value: i64) -> Result<HostShapeSlot, HostTypeDecodeError> {
    usize::try_from(value)
        .map(DimInfo::Lit)
        .map(HostShapeSlot::Dim)
        .map_err(|_| malformed(format!("negative tensor dimension `{value}`")))
}

fn decode_adt_type(children: &[Expr]) -> Result<HostTypeTerm, HostTypeDecodeError> {
    let (name_expr, args) = children
        .split_first()
        .ok_or_else(|| malformed("t-adt has no type name"))?;
    let name = symbol_name(name_expr).ok_or_else(|| malformed("t-adt name is not a symbol"))?;
    let args = args
        .iter()
        .map(decode_host_type)
        .collect::<Result<Vec<_>, _>>()?;
    match (name, args.as_slice()) {
        ("Option", [inner]) => Ok(HostTypeTerm::Option(Box::new(inner.clone()))),
        ("List", [inner]) => Ok(HostTypeTerm::List(Box::new(inner.clone()))),
        ("Dict", [key, value]) => Ok(HostTypeTerm::Dict(
            Box::new(key.clone()),
            Box::new(value.clone()),
        )),
        ("MappedFile", []) => Ok(HostTypeTerm::MappedFile),
        ("Option" | "List" | "Dict" | "MappedFile", _) => Err(malformed(format!(
            "reserved host ADT `{name}` has the wrong arity"
        ))),
        _ => Ok(HostTypeTerm::Adt(name.to_string(), args)),
    }
}

fn list_tag(list: &List) -> Option<&str> {
    list.elements.first().and_then(symbol_name)
}

fn list_children(list: &List) -> Result<&[Expr], HostTypeDecodeError> {
    match list.elements.get(1) {
        Some(Expr::Map(_, _)) => Ok(&list.elements[2..]),
        Some(_) => Err(malformed("type node metadata slot is not a map")),
        None => Err(malformed("type node has no metadata slot")),
    }
}

fn symbol_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Symbol(name), _) => Some(name),
        _ => None,
    }
}

fn one_child<'a>(children: &'a [Expr], context: &str) -> Result<&'a Expr, HostTypeDecodeError> {
    match children {
        [child] => Ok(child),
        _ => Err(malformed(format!(
            "{context} requires exactly one child, got {}",
            children.len()
        ))),
    }
}

fn one_symbol_child<'a>(
    children: &'a [Expr],
    context: &str,
) -> Result<&'a str, HostTypeDecodeError> {
    symbol_name(one_child(children, context)?)
        .ok_or_else(|| malformed(format!("{context} child is not a symbol")))
}

fn malformed(detail: impl Into<String>) -> HostTypeDecodeError {
    HostTypeDecodeError::MalformedTypeSyntax {
        detail: detail.into(),
    }
}
