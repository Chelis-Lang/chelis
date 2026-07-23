//! Resolved C-host ABI vocabulary for chelis#730 Phase 2.
//!
//! Logical host types are resolved in `chelis-ir` without consulting a
//! backend.  This module is the target-specific boundary: it accepts only a
//! [`ConcreteHostType`] and either produces an exact C-host representation or
//! returns the shared structured [`Unsupported`] diagnostic.  No unresolved
//! type term can be represented here, and no negative decision selects an
//! alternate ABI type.
//!
//! The ABI vocabulary and its constructor stay private to this backend:
//!
//! ```compile_fail
//! use chelis_backend_c::host_abi::HostAbiType;
//! ```

use chelis_ir::ConcreteHostType;
use chelis_ir::host::{
    ConcreteHostBinding, ConcreteHostCallback, ConcreteHostCallbackKind, ConcreteHostExpr,
    ConcreteHostExprKind, ConcreteHostFunction, ConcreteHostParam, ConcreteHostProgram,
    HostBinding, HostCallback, HostCallbackKind, HostExpr, HostExprKind, HostFunction,
    HostMatchArm, HostParam, HostPatternBinding, HostProgram,
};
use chelis_types::types::Prim;
use chelis_types::unsupported::{Stage, Unsupported, UnsupportedKind};
use std::collections::HashSet;

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
    /// A typed C function-pointer parameter or direct callback argument.
    ///
    /// This is deliberately not a general value representation.  The only
    /// constructor is [`Self::try_callback_signature`], and projection uses
    /// it only for declared callback parameters and statically-known callback
    /// arguments.  Function results, bindings, fields, and container elements
    /// all cross [`Self::try_from_concrete`], which rejects function values.
    Callback(Vec<HostAbiType>, Box<HostAbiType>),
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
            ConcreteHostType::Function(_, _) => {
                return Err(unsupported_function_value(ty, "C host ABI value selection"));
            }
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

    /// Return the standalone C type spelling for value ABIs.
    ///
    /// A callback has no standalone spelling: C function-pointer syntax
    /// requires the identifier inside the declarator.  Keeping that state out
    /// of this method removes the former `Fn -> void *` erasure path.
    pub(crate) fn c_type_name(&self) -> Option<&'static str> {
        match self {
            Self::Int8 => Some("int8_t"),
            Self::Int16 => Some("int16_t"),
            Self::Int32 => Some("int32_t"),
            Self::Int64 => Some("int64_t"),
            Self::Float32 => Some("float"),
            Self::Float64 => Some("double"),
            Self::Bool => Some("bool"),
            Self::String => Some("chelis_string"),
            Self::Callback(_, _) => None,
            Self::Adt(_, _) => Some("chelis_adt*"),
            Self::List(_) => Some("chelis_list*"),
            Self::Dict(_, _) => Some("chelis_dict*"),
            Self::Tuple(_) => Some("chelis_tuple*"),
            Self::Tensor(_) => Some("chelis_tensor*"),
            Self::MappedFile => Some("chelis_mapped_file*"),
            Self::Option(inner) => match inner.as_ref() {
                Self::Int64 => Some("chelis_option_i64"),
                Self::Float64 => Some("chelis_option_f64"),
                Self::Int8
                | Self::Int16
                | Self::Int32
                | Self::Float32
                | Self::Bool
                | Self::String
                | Self::Callback(_, _)
                | Self::Adt(_, _)
                | Self::List(_)
                | Self::Dict(_, _)
                | Self::Tuple(_)
                | Self::Tensor(_)
                | Self::Option(_)
                | Self::MappedFile
                | Self::Unit => Some("chelis_option_value"),
            },
            Self::Unit => Some("int"),
        }
    }

    fn try_callback_signature(ty: &ConcreteHostType) -> Result<Self, Unsupported> {
        let ConcreteHostType::Function(params, ret) = ty else {
            return Err(invalid_callback_shape(format!(
                "callback position carries non-function type {ty:?}"
            )));
        };
        Ok(Self::Callback(
            params
                .iter()
                .map(Self::try_from_concrete)
                .collect::<Result<Vec<_>, _>>()?,
            Box::new(Self::try_from_concrete(ret)?),
        ))
    }
}

pub(crate) fn project_program(
    program: &ConcreteHostProgram,
) -> Result<HostAbiProgram, Unsupported> {
    let declared_callbacks = program
        .functions
        .iter()
        .map(|function| function.name.clone())
        .collect::<HashSet<_>>();
    Ok(HostAbiProgram {
        globals: program
            .globals
            .iter()
            .cloned()
            .map(|binding| project_binding(binding, &declared_callbacks))
            .collect::<Result<Vec<_>, _>>()?,
        global_tensor_helpers: program.global_tensor_helpers.clone(),
        functions: program
            .functions
            .iter()
            .cloned()
            .map(|function| project_function(function, &declared_callbacks))
            .collect::<Result<Vec<_>, _>>()?,
        summary_rejections: program.summary_rejections.clone(),
    })
}

fn project_binding(
    binding: ConcreteHostBinding,
    allowed_callbacks: &HashSet<String>,
) -> Result<HostAbiBinding, Unsupported> {
    Ok(HostAbiBinding {
        name: binding.name,
        display_name: binding.display_name,
        ty: HostAbiType::try_from_concrete(&binding.ty)?,
        value: project_expr(binding.value, allowed_callbacks)?,
    })
}

fn project_function(
    function: ConcreteHostFunction,
    declared_callbacks: &HashSet<String>,
) -> Result<HostAbiFunction, Unsupported> {
    let mut allowed_callbacks = declared_callbacks.clone();
    for param in &function.params {
        if matches!(param.ty, ConcreteHostType::Function(_, _)) {
            allowed_callbacks.insert(param.name.clone());
        }
    }
    Ok(HostAbiFunction {
        name: function.name,
        params: function
            .params
            .into_iter()
            .map(project_function_param)
            .collect::<Result<Vec<_>, _>>()?,
        ret_ty: HostAbiType::try_from_concrete(&function.ret_ty)?,
        body: project_expr(function.body, &allowed_callbacks)?,
        tensor_helpers: function.tensor_helpers,
        specialization: function.specialization,
        summary_rejections: function.summary_rejections,
    })
}

fn project_value_param(param: ConcreteHostParam) -> Result<HostAbiParam, Unsupported> {
    Ok(HostAbiParam {
        name: param.name,
        ty: HostAbiType::try_from_concrete(&param.ty)?,
    })
}

fn project_function_param(param: ConcreteHostParam) -> Result<HostAbiParam, Unsupported> {
    let ty = match &param.ty {
        ConcreteHostType::Function(_, _) => HostAbiType::try_callback_signature(&param.ty)?,
        _ => HostAbiType::try_from_concrete(&param.ty)?,
    };
    Ok(HostAbiParam {
        name: param.name,
        ty,
    })
}

fn project_callback(
    callback: ConcreteHostCallback,
    allowed_callbacks: &HashSet<String>,
) -> Result<HostAbiCallback, Unsupported> {
    let kind = match callback.kind {
        ConcreteHostCallbackKind::Named { function, params } => HostAbiCallbackKind::Named {
            function: if allowed_callbacks.contains(&function) {
                function
            } else {
                return Err(unsupported_function_symbol(&function));
            },
            params: params
                .into_iter()
                .map(project_value_param)
                .collect::<Result<Vec<_>, _>>()?,
        },
        ConcreteHostCallbackKind::Inline { params, body } => HostAbiCallbackKind::Inline {
            params: params
                .into_iter()
                .map(project_value_param)
                .collect::<Result<Vec<_>, _>>()?,
            body: Box::new(project_expr(*body, allowed_callbacks)?),
        },
    };
    Ok(HostAbiCallback {
        kind,
        ret_ty: HostAbiType::try_from_concrete(&callback.ret_ty)?,
    })
}

fn project_expr(
    expr: ConcreteHostExpr,
    allowed_callbacks: &HashSet<String>,
) -> Result<HostAbiExpr, Unsupported> {
    let kind = match expr.kind {
        ConcreteHostExprKind::Int(value) => HostAbiExprKind::Int(value),
        ConcreteHostExprKind::Float(value) => HostAbiExprKind::Float(value),
        ConcreteHostExprKind::Bool(value) => HostAbiExprKind::Bool(value),
        ConcreteHostExprKind::String(value) => HostAbiExprKind::String(value),
        ConcreteHostExprKind::List(items, ty) => HostAbiExprKind::List(
            items
                .into_iter()
                .map(|expr| project_expr(expr, allowed_callbacks))
                .collect::<Result<Vec<_>, _>>()?,
            HostAbiType::try_from_concrete(&ty)?,
        ),
        ConcreteHostExprKind::Tuple(items, ty) => HostAbiExprKind::Tuple(
            items
                .into_iter()
                .map(|expr| project_expr(expr, allowed_callbacks))
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
        } => {
            if chelis_ir::host::is_host_unresolved_marker(&function) {
                return Err(unsupported_callable_use(&function));
            }
            if args.len() != arg_tys.len() {
                return Err(invalid_callback_shape(format!(
                    "call `{function}` has {} arguments but {} checked argument types",
                    args.len(),
                    arg_tys.len()
                )));
            }
            let (args, arg_tys) = args
                .into_iter()
                .zip(arg_tys.iter())
                .map(|(arg, arg_ty)| {
                    if matches!(arg_ty, ConcreteHostType::Function(_, _)) {
                        Ok((
                            project_callback_argument(arg, arg_ty, allowed_callbacks)?,
                            HostAbiType::try_callback_signature(arg_ty)?,
                        ))
                    } else {
                        Ok((
                            project_expr(arg, allowed_callbacks)?,
                            HostAbiType::try_from_concrete(arg_ty)?,
                        ))
                    }
                })
                .collect::<Result<Vec<_>, Unsupported>>()?
                .into_iter()
                .unzip();
            HostAbiExprKind::Call {
                function,
                args,
                arg_tys,
                ty: HostAbiType::try_from_concrete(&ty)?,
            }
        }
        ConcreteHostExprKind::Builtin { name, args, ty } => {
            if chelis_ir::host::is_host_unresolved_marker(&name) {
                return Err(unsupported_callable_use(&name));
            }
            HostAbiExprKind::Builtin {
                name,
                args: args
                    .into_iter()
                    .map(|expr| project_expr(expr, allowed_callbacks))
                    .collect::<Result<Vec<_>, _>>()?,
                ty: HostAbiType::try_from_concrete(&ty)?,
            }
        }
        ConcreteHostExprKind::AdtConstruct { ctor, fields, ty } => HostAbiExprKind::AdtConstruct {
            ctor,
            fields: fields
                .into_iter()
                .map(|expr| project_expr(expr, allowed_callbacks))
                .collect::<Result<Vec<_>, _>>()?,
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::AdtFieldAccess {
            base,
            field_index,
            ty,
        } => HostAbiExprKind::AdtFieldAccess {
            base: Box::new(project_expr(*base, allowed_callbacks)?),
            field_index,
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ty,
        } => HostAbiExprKind::If {
            cond: Box::new(project_expr(*cond, allowed_callbacks)?),
            then_expr: Box::new(project_expr(*then_expr, allowed_callbacks)?),
            else_expr: Box::new(project_expr(*else_expr, allowed_callbacks)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::MatchOption {
            scrutinee,
            bind_name,
            some_expr,
            none_expr,
            ty,
        } => HostAbiExprKind::MatchOption {
            scrutinee: Box::new(project_expr(*scrutinee, allowed_callbacks)?),
            bind_name,
            some_expr: Box::new(project_expr(*some_expr, allowed_callbacks)?),
            none_expr: Box::new(project_expr(*none_expr, allowed_callbacks)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ty,
        } => HostAbiExprKind::MatchAdt {
            scrutinee: Box::new(project_expr(*scrutinee, allowed_callbacks)?),
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
                        expr: project_expr(arm.expr, allowed_callbacks)?,
                    })
                })
                .collect::<Result<Vec<_>, Unsupported>>()?,
            default_expr: default_expr
                .map(|expr| project_expr(*expr, allowed_callbacks).map(Box::new))
                .transpose()?,
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::Let { bindings, body, ty } => HostAbiExprKind::Let {
            bindings: bindings
                .into_iter()
                .map(|binding| project_binding(binding, allowed_callbacks))
                .collect::<Result<Vec<_>, _>>()?,
            body: Box::new(project_expr(*body, allowed_callbacks)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::Map { callback, list, ty } => HostAbiExprKind::Map {
            callback: project_callback(callback, allowed_callbacks)?,
            list: Box::new(project_expr(*list, allowed_callbacks)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::Filter { callback, list, ty } => HostAbiExprKind::Filter {
            callback: project_callback(callback, allowed_callbacks)?,
            list: Box::new(project_expr(*list, allowed_callbacks)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::Fold {
            callback,
            init,
            list,
            ty,
        } => HostAbiExprKind::Fold {
            callback: project_callback(callback, allowed_callbacks)?,
            init: Box::new(project_expr(*init, allowed_callbacks)?),
            list: Box::new(project_expr(*list, allowed_callbacks)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::Scan {
            callback,
            init,
            list,
            ty,
        } => HostAbiExprKind::Scan {
            callback: project_callback(callback, allowed_callbacks)?,
            init: Box::new(project_expr(*init, allowed_callbacks)?),
            list: Box::new(project_expr(*list, allowed_callbacks)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::Partition { callback, list, ty } => HostAbiExprKind::Partition {
            callback: project_callback(callback, allowed_callbacks)?,
            list: Box::new(project_expr(*list, allowed_callbacks)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::FlatMap { callback, list, ty } => HostAbiExprKind::FlatMap {
            callback: project_callback(callback, allowed_callbacks)?,
            list: Box::new(project_expr(*list, allowed_callbacks)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::WithSeed { seed, body, ty } => HostAbiExprKind::WithSeed {
            seed: Box::new(project_expr(*seed, allowed_callbacks)?),
            body: Box::new(project_expr(*body, allowed_callbacks)?),
            ty: HostAbiType::try_from_concrete(&ty)?,
        },
        ConcreteHostExprKind::TensorCall { helper, args, ty } => HostAbiExprKind::TensorCall {
            helper,
            args: args
                .into_iter()
                .map(|expr| project_expr(expr, allowed_callbacks))
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

fn project_callback_argument(
    expr: ConcreteHostExpr,
    expected: &ConcreteHostType,
    allowed_callbacks: &HashSet<String>,
) -> Result<HostAbiExpr, Unsupported> {
    let ConcreteHostExprKind::Var(name, actual) = expr.kind else {
        return Err(unsupported_function_value(
            expected,
            "C host callback argument selection",
        ));
    };
    if &actual != expected {
        return Err(invalid_callback_shape(format!(
            "callback `{name}` has type {actual:?}, expected {expected:?}"
        )));
    }
    if !allowed_callbacks.contains(&name) {
        return Err(unsupported_function_symbol(&name));
    }
    Ok(HostAbiExpr {
        kind: HostAbiExprKind::Var(name, HostAbiType::try_callback_signature(expected)?),
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

fn unsupported_function_value(ty: &ConcreteHostType, context: &'static str) -> Unsupported {
    Unsupported::new(
        UnsupportedKind::HostAbi(format!("function value `{ty:?}`")),
        context,
        Stage::Codegen("c"),
        "the C host backend supports typed callback parameters and direct statically-known \
         callback arguments, but no first-class function-value ABI; specialize the call or \
         reject the containing construct ([05-UNS-1]; chelis#730)",
    )
}

fn unsupported_function_symbol(name: &str) -> Unsupported {
    Unsupported::new(
        UnsupportedKind::HostAbi(format!("function value `{name}`")),
        "C host callback argument selection",
        Stage::Codegen("c"),
        "only a declared function symbol or an in-scope typed callback parameter can cross \
         this boundary; dynamic function values have no C host ABI ([05-UNS-1]; chelis#730)",
    )
}

/// The frozen rejection for an internal unresolved-callee marker. The
/// marker spelling never enters the diagnostic; the payload names the
/// semantic class instead, per marker kind.
fn unsupported_callable_use(marker: &str) -> Unsupported {
    if marker == chelis_ir::host::HOST_UNRESOLVED_TRANSFORM_MARKER {
        return Unsupported::new(
            UnsupportedKind::HostAbi("unresolved `grad`/`vmap` transform application".to_string()),
            "C host ABI callable-use projection",
            Stage::Codegen("c"),
            "the host lane recognized an AD transform it could not lower; rewrite the \
             differentiated body to pure tensor ops (sum, add, mul, einsum) or run under \
             `chelis eval` ([05-UNS-1]; chelis#730)",
        );
    }
    Unsupported::new(
        UnsupportedKind::HostAbi("unresolved function value".to_string()),
        "C host ABI callable-use projection",
        Stage::Codegen("c"),
        "the host lowerer did not resolve this application to a declared function symbol or \
         typed callback parameter; unresolved callables have no raw C call target \
         ([05-UNS-1]; chelis#730)",
    )
}

fn invalid_callback_shape(detail: String) -> Unsupported {
    Unsupported::new(
        UnsupportedKind::Construct(detail),
        "C host callback ABI projection",
        Stage::Codegen("c"),
        "checked callable metadata and the resolved host program disagree; no fallback \
         callable representation is permitted ([05-UNS-1]; chelis#730)",
    )
}
