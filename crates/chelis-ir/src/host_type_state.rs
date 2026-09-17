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

use chelis_deep::DeepTag;
use std::fmt;

use chelis_deep::ast::{Atom, Expr, Metadata};
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
    Fn(Vec<HostTypeTerm>, Box<HostTypeTerm>),
    Adt(String, Vec<HostTypeTerm>),
    List(Box<HostTypeTerm>),
    Dict(Box<HostTypeTerm>, Box<HostTypeTerm>),
    Tuple(Vec<HostTypeTerm>),
    /// A tensor whose precision and rank have already been resolved. Named
    /// runtime dimensions remain concrete logical dimensions.
    Tensor(TensorType),
    /// A tensor retaining a named precision or rank variable until call-site
    /// specialization supplies it.
    PolymorphicTensor(HostTensorTypeTerm),
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
    // Compatibility spellings for the scalar variants while the host lowerer
    // migrates from its former coarse enum.  They are exact logical
    // precisions, not defaults: i32 and i64 no longer collapse at the
    // syntax boundary, and the narrow widths remain representable terms.
    #[allow(non_upper_case_globals)]
    pub const Int64: Self = Self::Scalar(HostPrecisionTerm::Concrete(Prim::Int64));
    #[allow(non_upper_case_globals)]
    pub const Float32: Self = Self::Scalar(HostPrecisionTerm::Concrete(Prim::F32));
    #[allow(non_upper_case_globals)]
    pub const Float64: Self = Self::Scalar(HostPrecisionTerm::Concrete(Prim::F64));
    #[allow(non_upper_case_globals)]
    pub const Bool: Self = Self::Scalar(HostPrecisionTerm::Concrete(Prim::Bool));
    #[allow(non_upper_case_globals)]
    pub const String: Self = Self::Scalar(HostPrecisionTerm::Concrete(Prim::String));

    /// Whether inference, named polymorphism, rank polymorphism, or bottom
    /// still prevents this term from crossing a concrete value boundary.
    pub fn is_unresolved(&self) -> bool {
        match self {
            Self::TypeVariable(_)
            | Self::InferenceVariable(_)
            | Self::Never
            | Self::Scalar(HostPrecisionTerm::Variable(_)) => true,
            Self::PolymorphicTensor(_) => true,
            Self::Fn(params, ret) => params.iter().any(Self::is_unresolved) || ret.is_unresolved(),
            Self::Adt(_, args) | Self::Tuple(args) => args.iter().any(Self::is_unresolved),
            Self::List(inner) | Self::Option(inner) => inner.is_unresolved(),
            Self::Dict(key, value) => key.is_unresolved() || value.is_unresolved(),
            Self::Scalar(HostPrecisionTerm::Concrete(_))
            | Self::Tensor(_)
            | Self::MappedFile
            | Self::Unit => false,
        }
    }

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
            Self::Fn(params, ret) => Ok(ConcreteHostType::Function(
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
            Self::Tensor(tensor) => Ok(ConcreteHostType::Tensor(tensor)),
            Self::PolymorphicTensor(HostTensorTypeTerm { precision, shape }) => {
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

impl ConcreteHostType {
    // Readable spellings for exact resolved scalar types. These are aliases
    // for `Scalar(ConcretePrecision)`, not the legacy coarse host enum: each
    // constant preserves the full logical precision at the ABI boundary.
    #[allow(non_upper_case_globals)]
    pub const Int32: Self = Self::Scalar(Prim::Int32);
    #[allow(non_upper_case_globals)]
    pub const Int64: Self = Self::Scalar(Prim::Int64);
    #[allow(non_upper_case_globals)]
    pub const Float32: Self = Self::Scalar(Prim::F32);
    #[allow(non_upper_case_globals)]
    pub const Float64: Self = Self::Scalar(Prim::F64);
    #[allow(non_upper_case_globals)]
    pub const Bool: Self = Self::Scalar(Prim::Bool);
    #[allow(non_upper_case_globals)]
    pub const String: Self = Self::Scalar(Prim::String);
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
    let (tag, _, children) = stamped_parts(expr)?;
    match tag {
        DeepTag::TPrim => {
            let name = one_symbol_child(children, "t-prim")?;
            let precision =
                Prim::parse_name(name).ok_or_else(|| HostTypeDecodeError::UnknownPrimitive {
                    name: name.to_string(),
                })?;
            Ok(HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(precision)))
        }
        DeepTag::TVar => Ok(HostTypeTerm::TypeVariable(
            one_symbol_child(children, "t-var")?.to_string(),
        )),
        DeepTag::TRef => {
            let inner = one_child(children, "t-ref")?;
            decode_host_type(inner)
        }
        DeepTag::TTensor => decode_tensor_type(children),
        DeepTag::TAdt => decode_adt_type(children),
        DeepTag::TTuple => Ok(HostTypeTerm::Tuple(
            children
                .iter()
                .map(decode_host_type)
                .collect::<Result<Vec<_>, _>>()?,
        )),
        DeepTag::TFn => {
            let (ret, params) = children
                .split_last()
                .ok_or_else(|| malformed("t-fn has no return type"))?;
            Ok(HostTypeTerm::Fn(
                params
                    .iter()
                    .map(decode_host_type)
                    .collect::<Result<Vec<_>, _>>()?,
                Box::new(decode_host_type(ret)?),
            ))
        }
        DeepTag::TUnit if children.is_empty() => Ok(HostTypeTerm::Unit),
        DeepTag::TUnit => Err(malformed("t-unit must not have children")),
        other => Err(malformed(format!("unknown type tag `{}`", other.as_str()))),
    }
}

/// Decode the `type` entry on a Deep expression's metadata map.
///
/// Absence and invalid type syntax have separate errors. This is the boundary
/// that replaces `expr_type(...).unwrap_or(Unknown)` in the legacy lowerer.
pub fn decode_host_type_metadata(expr: &Expr) -> Result<HostTypeTerm, HostTypeDecodeError> {
    match expr {
        Expr::List(_, _) | Expr::Node(_, _) => {
            let (_, metadata, _) = stamped_parts(expr)?;
            let type_expr = metadata
                .ty()
                .map(|value| value.expression())
                .ok_or(HostTypeDecodeError::MissingTypeMetadata)?;
            decode_host_type(type_expr)
        }
        Expr::MetaExpr(meta, _) => {
            if let Some(type_expr) = meta.metadata.ty().map(|value| value.expression()) {
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
    match (&precision, &shape) {
        (HostPrecisionTerm::Concrete(precision), HostShapeTerm::Concrete(dims)) => {
            Ok(HostTypeTerm::Tensor(TensorType {
                dims: dims.clone(),
                precision: *precision,
            }))
        }
        _ => Ok(HostTypeTerm::PolymorphicTensor(HostTensorTypeTerm {
            precision,
            shape,
        })),
    }
}

fn decode_precision(expr: &Expr) -> Result<HostPrecisionTerm, HostTypeDecodeError> {
    let (tag, _, children) =
        stamped_parts(expr).map_err(|_| malformed("tensor precision is not a type node"))?;
    match tag {
        DeepTag::TPrim => {
            let name = one_symbol_child(children, "tensor t-prim")?;
            Prim::parse_name(name)
                .map(HostPrecisionTerm::Concrete)
                .ok_or_else(|| HostTypeDecodeError::UnknownPrimitive {
                    name: name.to_string(),
                })
        }
        DeepTag::TVar => Ok(HostPrecisionTerm::Variable(
            one_symbol_child(children, "tensor t-var")?.to_string(),
        )),
        other => Err(malformed(format!(
            "tensor precision must be t-prim or t-var, got `{}`",
            other.as_str()
        ))),
    }
}

fn decode_shape_slot(expr: &Expr) -> Result<HostShapeSlot, HostTypeDecodeError> {
    if let Expr::Atom(Atom::Name(name), _) = expr {
        return Ok(HostShapeSlot::Dim(DimInfo::Named(name.clone(), None)));
    }
    if let Expr::Atom(Atom::Int(value), _) = expr {
        return nonnegative_dim(*value);
    }
    let (tag, _, children) =
        stamped_parts(expr).map_err(|_| malformed("tensor dimension is not a dimension node"))?;
    match tag {
        DeepTag::DName | DeepTag::DVar => Ok(HostShapeSlot::Dim(DimInfo::Named(
            one_symbol_child(children, tag.as_str())?.to_string(),
            None,
        ))),
        DeepTag::DRank => Ok(HostShapeSlot::RankVariable(
            one_symbol_child(children, "d-rank")?.to_string(),
        )),
        DeepTag::DLit => match one_child(children, "d-lit")? {
            Expr::Atom(Atom::Int(value), _) => nonnegative_dim(*value),
            _ => Err(malformed("d-lit child is not an integer")),
        },
        other => Err(malformed(format!(
            "unknown dimension tag `{}`",
            other.as_str()
        ))),
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
        .map(|argument| {
            let (tag, _, _) = stamped_parts(argument)?;
            if matches!(
                tag,
                DeepTag::DName | DeepTag::DVar | DeepTag::DLit | DeepTag::DRank
            ) {
                // Nominal dimensions constrain checking but have no host value
                // representation. Preserve their nominal argument slot with
                // the existing private layout witness so constructor arity
                // remains aligned while the value itself is erased.
                Ok(HostTypeTerm::Unit)
            } else {
                decode_host_type(argument)
            }
        })
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

// Transitional E5b adapter; `Expr::carrier` owns physical-carrier decoding.
fn stamped_parts(expr: &Expr) -> Result<(DeepTag, &Metadata, &[Expr]), HostTypeDecodeError> {
    match expr.carrier() {
        chelis_deep::ExprCarrier::DecodedNode(tag, metadata, children) => {
            Ok((tag, metadata, children))
        }
        chelis_deep::ExprCarrier::MalformedLegacyList(list) => match list.elements.get(1) {
            Some(Expr::Map(meta, _)) => Ok((
                list.tag()
                    .ok_or_else(|| malformed("type node has no symbolic tag"))?,
                meta,
                &list.elements[2..],
            )),
            Some(_) => Err(malformed("type node metadata slot is not a map")),
            None => Err(malformed("type node has no metadata slot")),
        },
        chelis_deep::ExprCarrier::StructuralList(_)
        | chelis_deep::ExprCarrier::UndecodableHead(_, _, _)
        | chelis_deep::ExprCarrier::Atom(_)
        | chelis_deep::ExprCarrier::MetadataMap(_)
        | chelis_deep::ExprCarrier::MetadataExpression(_) => {
            Err(malformed("expected a Deep type node"))
        }
    }
}

fn symbol_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Name(name), _) => Some(name),
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
