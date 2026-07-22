//! Resolved C-host ABI vocabulary for chelis#730 Phase 2.
//!
//! Logical host types are resolved in `chelis-ir` without consulting a
//! backend.  This module is the target-specific boundary: it accepts only a
//! [`ConcreteHostType`] and either produces an exact C-host representation or
//! returns the shared structured [`Unsupported`] diagnostic.  No unresolved
//! type term can be represented here, and no negative decision selects an
//! alternate ABI type.

use chelis_ir::ConcreteHostType;
use chelis_ir::host::{
    ConcreteHostBinding, ConcreteHostCallback, ConcreteHostCallbackKind, ConcreteHostExpr,
    ConcreteHostExprKind, ConcreteHostFunction, ConcreteHostParam, ConcreteHostProgram,
    HostBinding, HostCallback, HostCallbackKind, HostExpr, HostExprKind, HostFunction,
    HostMatchArm, HostParam, HostPatternBinding, HostProgram,
};
use chelis_types::types::Prim;
use chelis_types::unsupported::{Stage, Unsupported, UnsupportedKind};

/// A host value whose complete logical type has an implemented C ABI.
///
/// Kept crate-private so callers cannot manufacture a purportedly-resolved
/// ABI value.  Construction is exclusively through [`Self::try_from_concrete`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum HostAbiType {
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Float64,
    Bool,
    String,
    Fn(Vec<HostAbiType>, Box<HostAbiType>),
    Adt(String, Vec<HostAbiType>),
    List(Box<HostAbiType>),
    Dict(Box<HostAbiType>, Box<HostAbiType>),
    Tuple(Vec<HostAbiType>),
    Tensor(chelis_ir::TensorType),
    Option(Box<HostAbiType>),
    MappedFile,
    Unit,
}

pub(crate) type HostAbiProgram = HostProgram<HostAbiType>;
pub(crate) type HostAbiBinding = HostBinding<HostAbiType>;
pub(crate) type HostAbiFunction = HostFunction<HostAbiType>;
pub(crate) type HostAbiParam = HostParam<HostAbiType>;
pub(crate) type HostAbiCallback = HostCallback<HostAbiType>;
pub(crate) type HostAbiCallbackKind = HostCallbackKind<HostAbiType>;
pub(crate) type HostAbiExpr = HostExpr<HostAbiType>;
pub(crate) type HostAbiExprKind = HostExprKind<HostAbiType>;
pub(crate) type HostAbiMatchArm = HostMatchArm<HostAbiType>;

impl HostAbiType {
    /// Resolve the private pre-Table-B C-host capability adapter.
    ///
    /// This match is intentionally exhaustive over the logical vocabulary.
    /// Table B (chelis#729 Phase 4) will replace the decisions without
    /// changing this fallible boundary.
    pub(crate) fn try_from_concrete(ty: &ConcreteHostType) -> Result<Self, Unsupported> {
        Ok(match ty {
            ConcreteHostType::Scalar(Prim::Int8) => Self::Int8,
            ConcreteHostType::Scalar(Prim::Int16) => Self::Int16,
            ConcreteHostType::Scalar(Prim::Int32) => Self::Int32,
            ConcreteHostType::Scalar(Prim::Int64) => Self::Int64,
            ConcreteHostType::Scalar(Prim::F32) => Self::Float32,
            ConcreteHostType::Scalar(Prim::F64) => Self::Float64,
            ConcreteHostType::Scalar(Prim::Bool) => Self::Bool,
            ConcreteHostType::Scalar(Prim::String) => Self::String,
            // No exact C host-scalar storage/rounding contract exists for
            // reduced floats before chelis#729 Table A/B.  Reject them as
            // known logical dtypes; never recreate the former widened-double
            // path that happened to be green for exactly-representable values.
            ConcreteHostType::Scalar(precision @ (Prim::F16 | Prim::Bf16)) => {
                return Err(unimplemented_scalar(*precision));
            }
            // f8e4m3 is deferred and inadmissible in the active language per
            // spec/04-type-system.md section 1.1.1.  It still has a `Prim`
            // identity so this boundary can reject it precisely.
            ConcreteHostType::Scalar(Prim::F8e4m3) => {
                return Err(rejected_dtype(
                    Prim::F8e4m3,
                    "deferred by spec/04-type-system.md section 1.1.1 ([05-UNS-1])",
                ));
            }
            ConcreteHostType::Function(params, ret) => Self::Fn(
                params
                    .iter()
                    .map(Self::try_from_concrete)
                    .collect::<Result<Vec<_>, _>>()?,
                Box::new(Self::try_from_concrete(ret)?),
            ),
            ConcreteHostType::Adt(name, args) => Self::Adt(
                name.clone(),
                args.iter()
                    .map(Self::try_from_concrete)
                    .collect::<Result<Vec<_>, _>>()?,
            ),
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
                Self::Tensor(tensor.clone())
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
            Self::Int8 => "int8_t",
            Self::Int16 => "int16_t",
            Self::Int32 => "int32_t",
            Self::Int64 => "int64_t",
            Self::Float32 => "float",
            Self::Float64 => "double",
            Self::Bool => "bool",
            Self::String => "chelis_string",
            // The current callback runtime carries function values as opaque
            // pointers.  This is the implemented ABI for a *resolved*
            // function type, not the deleted unresolved-type fallback.
            Self::Fn(_, _) => "void*",
            Self::Adt(_, _) => "chelis_adt*",
            Self::List(_) => "chelis_list*",
            Self::Dict(_, _) => "chelis_dict*",
            Self::Tuple(_) => "chelis_tuple*",
            Self::Tensor(_) => "chelis_tensor*",
            Self::MappedFile => "chelis_mapped_file*",
            Self::Option(inner) => match inner.as_ref() {
                Self::Int64 => "chelis_option_i64",
                Self::Float64 => "chelis_option_f64",
                Self::Int8
                | Self::Int16
                | Self::Int32
                | Self::Float32
                | Self::Bool
                | Self::String
                | Self::Fn(_, _)
                | Self::Adt(_, _)
                | Self::List(_)
                | Self::Dict(_, _)
                | Self::Tuple(_)
                | Self::Tensor(_)
                | Self::Option(_)
                | Self::MappedFile
                | Self::Unit => "chelis_option_value",
            },
            Self::Unit => "int",
        }
    }
}

pub(crate) fn project_program(
    program: &ConcreteHostProgram,
) -> Result<HostAbiProgram, Unsupported> {
    Ok(HostAbiProgram {
        globals: program
            .globals
            .iter()
            .cloned()
            .map(project_binding)
            .collect::<Result<Vec<_>, _>>()?,
        global_tensor_helpers: program.global_tensor_helpers.clone(),
        functions: program
            .functions
            .iter()
            .cloned()
            .map(project_function)
            .collect::<Result<Vec<_>, _>>()?,
        summary_rejections: program.summary_rejections.clone(),
    })
}

fn project_binding(binding: ConcreteHostBinding) -> Result<HostAbiBinding, Unsupported> {
    Ok(HostAbiBinding {
        name: binding.name,
        display_name: binding.display_name,
        ty: HostAbiType::try_from_concrete(&binding.ty)?,
        value: project_expr(binding.value)?,
    })
}

fn project_function(function: ConcreteHostFunction) -> Result<HostAbiFunction, Unsupported> {
    Ok(HostAbiFunction {
        name: function.name,
        params: function
            .params
            .into_iter()
            .map(project_param)
            .collect::<Result<Vec<_>, _>>()?,
        ret_ty: HostAbiType::try_from_concrete(&function.ret_ty)?,
        body: project_expr(function.body)?,
        tensor_helpers: function.tensor_helpers,
        specialization: function.specialization,
        summary_rejections: function.summary_rejections,
    })
}

fn project_param(param: ConcreteHostParam) -> Result<HostAbiParam, Unsupported> {
    Ok(HostAbiParam {
        name: param.name,
        ty: HostAbiType::try_from_concrete(&param.ty)?,
    })
}

fn project_callback(callback: ConcreteHostCallback) -> Result<HostAbiCallback, Unsupported> {
    let kind = match callback.kind {
        ConcreteHostCallbackKind::Named { function, params } => HostAbiCallbackKind::Named {
            function,
            params: params
                .into_iter()
                .map(project_param)
                .collect::<Result<Vec<_>, _>>()?,
        },
        ConcreteHostCallbackKind::Inline { params, body } => HostAbiCallbackKind::Inline {
            params: params
                .into_iter()
                .map(project_param)
                .collect::<Result<Vec<_>, _>>()?,
            body: Box::new(project_expr(*body)?),
        },
    };
    Ok(HostAbiCallback {
        kind,
        ret_ty: HostAbiType::try_from_concrete(&callback.ret_ty)?,
    })
}

fn project_expr(expr: ConcreteHostExpr) -> Result<HostAbiExpr, Unsupported> {
    let kind = match expr.kind {
        ConcreteHostExprKind::Int(value) => HostAbiExprKind::Int(value),
        ConcreteHostExprKind::Float(value) => HostAbiExprKind::Float(value),
        ConcreteHostExprKind::Bool(value) => HostAbiExprKind::Bool(value),
        ConcreteHostExprKind::String(value) => HostAbiExprKind::String(value),
        ConcreteHostExprKind::List(items, ty) => HostAbiExprKind::List(
            items
                .into_iter()
                .map(project_expr)
                .collect::<Result<Vec<_>, _>>()?,
            HostAbiType::try_from_concrete(&ty)?,
        ),
        ConcreteHostExprKind::Tuple(items, ty) => HostAbiExprKind::Tuple(
            items
                .into_iter()
                .map(project_expr)
                .collect::<Result<Vec<_>, _>>()?,
            HostAbiType::try_from_concrete(&ty)?,
        ),
        ConcreteHostExprKind::Var(name, ty) => {
            HostAbiExprKind::Var(name, HostAbiType::try_from_concrete(&ty)?)
        }
        ConcreteHostExprKind::Call {
            function,
            args,
            arg_tys,
            ty,
        } => HostAbiExprKind::Call {
            function,
            args: args
                .into_iter()
                .map(project_expr)
                .collect::<Result<Vec<_>, _>>()?,
            arg_tys: arg_tys
                .iter()
                .map(HostAbiType::try_from_concrete)
                .collect::<Result<Vec<_>, _>>()?,
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::Builtin { name, args, ty } => HostAbiExprKind::Builtin {
            name,
            args: args
                .into_iter()
                .map(project_expr)
                .collect::<Result<Vec<_>, _>>()?,
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::AdtConstruct { ctor, fields, ty } => HostAbiExprKind::AdtConstruct {
            ctor,
            fields: fields
                .into_iter()
                .map(project_expr)
                .collect::<Result<Vec<_>, _>>()?,
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::AdtFieldAccess {
            base,
            field_index,
            ty,
        } => HostAbiExprKind::AdtFieldAccess {
            base: Box::new(project_expr(*base)?),
            field_index,
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ty,
        } => HostAbiExprKind::If {
            cond: Box::new(project_expr(*cond)?),
            then_expr: Box::new(project_expr(*then_expr)?),
            else_expr: Box::new(project_expr(*else_expr)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::MatchOption {
            scrutinee,
            bind_name,
            some_expr,
            none_expr,
            ty,
        } => HostAbiExprKind::MatchOption {
            scrutinee: Box::new(project_expr(*scrutinee)?),
            bind_name,
            some_expr: Box::new(project_expr(*some_expr)?),
            none_expr: Box::new(project_expr(*none_expr)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ty,
        } => HostAbiExprKind::MatchAdt {
            scrutinee: Box::new(project_expr(*scrutinee)?),
            arms: arms
                .into_iter()
                .map(|arm| {
                    Ok(HostMatchArm {
                        ctor: arm.ctor,
                        bindings: arm
                            .bindings
                            .into_iter()
                            .map(|binding| {
                                Ok(HostPatternBinding {
                                    name: binding.name,
                                    ty: HostAbiType::try_from_concrete(&binding.ty)?,
                                    field_index: binding.field_index,
                                })
                            })
                            .collect::<Result<Vec<_>, Unsupported>>()?,
                        expr: project_expr(arm.expr)?,
                    })
                })
                .collect::<Result<Vec<_>, Unsupported>>()?,
            default_expr: default_expr
                .map(|expr| project_expr(*expr).map(Box::new))
                .transpose()?,
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::Let { bindings, body, ty } => HostAbiExprKind::Let {
            bindings: bindings
                .into_iter()
                .map(project_binding)
                .collect::<Result<Vec<_>, _>>()?,
            body: Box::new(project_expr(*body)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::Map { callback, list, ty } => HostAbiExprKind::Map {
            callback: project_callback(callback)?,
            list: Box::new(project_expr(*list)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::Filter { callback, list, ty } => HostAbiExprKind::Filter {
            callback: project_callback(callback)?,
            list: Box::new(project_expr(*list)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::Fold {
            callback,
            init,
            list,
            ty,
        } => HostAbiExprKind::Fold {
            callback: project_callback(callback)?,
            init: Box::new(project_expr(*init)?),
            list: Box::new(project_expr(*list)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::Scan {
            callback,
            init,
            list,
            ty,
        } => HostAbiExprKind::Scan {
            callback: project_callback(callback)?,
            init: Box::new(project_expr(*init)?),
            list: Box::new(project_expr(*list)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::Partition { callback, list, ty } => HostAbiExprKind::Partition {
            callback: project_callback(callback)?,
            list: Box::new(project_expr(*list)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::FlatMap { callback, list, ty } => HostAbiExprKind::FlatMap {
            callback: project_callback(callback)?,
            list: Box::new(project_expr(*list)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::WithSeed { seed, body, ty } => HostAbiExprKind::WithSeed {
            seed: Box::new(project_expr(*seed)?),
            body: Box::new(project_expr(*body)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::TensorCall { helper, args, ty } => HostAbiExprKind::TensorCall {
            helper,
            args: args
                .into_iter()
                .map(project_expr)
                .collect::<Result<Vec<_>, _>>()?,
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::Unit => HostAbiExprKind::Unit,
    };
    Ok(HostAbiExpr {
        kind,
        span_id: expr.span_id,
        merged_spans: expr.merged_spans,
    })
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
