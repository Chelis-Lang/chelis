//! Resolved C-host ABI vocabulary for chelis#730 Phase 2.
//!
//! Logical host types are resolved in `chelis-ir` without consulting a
//! backend.  This module is the target-specific boundary: it accepts only a
//! [`ConcreteHostType`] and either produces an exact C-host representation or
//! returns the shared structured [`Unsupported`] diagnostic.  No unresolved
//! type term can be represented here, and no negative decision selects an
//! alternate ABI type.

use chelis_ir::ConcreteHostType;
use chelis_types::types::Prim;
use chelis_types::unsupported::{Stage, Unsupported, UnsupportedKind};

/// A host value whose complete logical type has an implemented C ABI.
///
/// Kept crate-private so callers cannot manufacture a purportedly-resolved
/// ABI value.  Construction is exclusively through [`Self::try_from_concrete`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum HostAbiType {
    I32,
    I64,
    F32,
    F64,
    Bool,
    String,
    Function {
        params: Vec<HostAbiType>,
        ret: Box<HostAbiType>,
    },
    Adt {
        name: String,
        args: Vec<HostAbiType>,
    },
    List(Box<HostAbiType>),
    Dict(Box<HostAbiType>, Box<HostAbiType>),
    Tuple(Vec<HostAbiType>),
    Tensor,
    Option(Box<HostAbiType>),
    MappedFile,
    Unit,
}

impl HostAbiType {
    /// Resolve the private pre-Table-B C-host capability adapter.
    ///
    /// This match is intentionally exhaustive over the logical vocabulary.
    /// Table B (chelis#729 Phase 4) will replace the decisions without
    /// changing this fallible boundary.
    pub(crate) fn try_from_concrete(ty: &ConcreteHostType) -> Result<Self, Unsupported> {
        Ok(match ty {
            ConcreteHostType::Scalar(Prim::Int32) => Self::I32,
            ConcreteHostType::Scalar(Prim::Int64) => Self::I64,
            ConcreteHostType::Scalar(Prim::F32) => Self::F32,
            ConcreteHostType::Scalar(Prim::F64) => Self::F64,
            ConcreteHostType::Scalar(Prim::Bool) => Self::Bool,
            ConcreteHostType::Scalar(Prim::String) => Self::String,
            // These are known logical scalar types, but the C host lane does
            // not yet implement scalar storage/boxing for them.  This is an
            // implementation gap, not a semantic rejection (chelis#714;
            // [05-UNS-1]).  Keep the exact dtype in the diagnostic.
            ConcreteHostType::Scalar(
                precision @ (Prim::F16 | Prim::Bf16 | Prim::Int8 | Prim::Int16),
            ) => return Err(unimplemented_scalar(*precision)),
            // f8e4m3 is deferred and inadmissible in the active language per
            // spec/04-type-system.md section 1.1.1.  It still has a `Prim`
            // identity so this boundary can reject it precisely.
            ConcreteHostType::Scalar(Prim::F8e4m3) => {
                return Err(rejected_dtype(
                    Prim::F8e4m3,
                    "deferred by spec/04-type-system.md section 1.1.1 ([05-UNS-1])",
                ));
            }
            ConcreteHostType::Function(params, ret) => Self::Function {
                params: params
                    .iter()
                    .map(Self::try_from_concrete)
                    .collect::<Result<Vec<_>, _>>()?,
                ret: Box::new(Self::try_from_concrete(ret)?),
            },
            ConcreteHostType::Adt(name, args) => Self::Adt {
                name: name.clone(),
                args: args
                    .iter()
                    .map(Self::try_from_concrete)
                    .collect::<Result<Vec<_>, _>>()?,
            },
            ConcreteHostType::List(inner) => Self::List(Box::new(Self::try_from_concrete(inner)?)),
            ConcreteHostType::Dict(key, value) => Self::Dict(
                Box::new(Self::try_from_concrete(key)?),
                Box::new(Self::try_from_concrete(value)?),
            ),
            ConcreteHostType::Tuple(items) => Self::Tuple(
                items
                    .iter()
                    .map(Self::try_from_concrete)
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            ConcreteHostType::Tensor(tensor) => {
                if tensor.precision == Prim::F8e4m3 {
                    return Err(rejected_dtype(
                        Prim::F8e4m3,
                        "deferred by spec/04-type-system.md section 1.1.1 ([05-UNS-1])",
                    ));
                }
                Self::Tensor
            }
            ConcreteHostType::Option(inner) => {
                Self::Option(Box::new(Self::try_from_concrete(inner)?))
            }
            ConcreteHostType::MappedFile => Self::MappedFile,
            ConcreteHostType::Unit => Self::Unit,
        })
    }

    pub(crate) fn c_type_name(&self) -> &'static str {
        match self {
            Self::I32 => "int32_t",
            Self::I64 => "int64_t",
            Self::F32 => "float",
            Self::F64 => "double",
            Self::Bool => "bool",
            Self::String => "chelis_string",
            // The current callback runtime carries function values as opaque
            // pointers.  This is the implemented ABI for a *resolved*
            // function type, not the deleted unresolved-type fallback.
            Self::Function { .. } => "void*",
            Self::Adt { .. } => "chelis_adt*",
            Self::List(_) => "chelis_list*",
            Self::Dict(_, _) => "chelis_dict*",
            Self::Tuple(_) => "chelis_tuple*",
            Self::Tensor => "chelis_tensor*",
            Self::MappedFile => "chelis_mapped_file*",
            Self::Option(inner) => match inner.as_ref() {
                Self::I64 => "chelis_option_i64",
                Self::F64 => "chelis_option_f64",
                Self::I32
                | Self::F32
                | Self::Bool
                | Self::String
                | Self::Function { .. }
                | Self::Adt { .. }
                | Self::List(_)
                | Self::Dict(_, _)
                | Self::Tuple(_)
                | Self::Tensor
                | Self::Option(_)
                | Self::MappedFile
                | Self::Unit => "chelis_option_value",
            },
            Self::Unit => "int",
        }
    }
}

fn unimplemented_scalar(precision: Prim) -> Unsupported {
    rejected_dtype(
        precision,
        "C-host scalar ABI support is tracked by chelis#714; no alternate dtype is permitted by [05-UNS-1]",
    )
}

fn rejected_dtype(precision: Prim, hint: &'static str) -> Unsupported {
    Unsupported::new(
        UnsupportedKind::Dtype(precision.name().to_string()),
        "C host ABI selection",
        Stage::Codegen("c"),
        hint,
    )
}
