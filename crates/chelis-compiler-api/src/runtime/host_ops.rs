use chelis_unord::UnordMap;

use chelis_deep::ast::{Atom, Expr, ExprCarrier};
use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue as IrTensorValue, eval_tensor_roots_with};
use chelis_ir::tier2;
use chelis_types::{
    ArgReduceOp, BUILTIN_NAMES, CompareOp, FloatBinOp, FloatUnOp, IntBinOp, IntUnOp, NumericTrap,
    ScalarValue, TensorReduceOp, arg_reduce_tensor_groups, cast_scalar, compare_scalars,
    compare_tensors, float_binop, float_scalar_tensor_binop, float_tensor_binop,
    float_tensor_scalar_binop, float_tensor_unop, float_unop, int_binop, int_scalar_tensor_binop,
    int_tensor_binop, int_tensor_scalar_binop, int_tensor_unop, int_unop, reduce_tensor_groups,
    scalar_from_f64, scalar_from_i64, tensor_from_scalars, types::Prim, uniform_sample,
};

use super::transforms::*;
use super::*;

pub(super) fn pattern_matches(
    value: &RuntimeValue,
    pattern: &Expr,
    bindings: &mut UnordMap<String, RuntimeValue>,
    adt_fields: &UnordMap<String, Vec<String>>,
) -> Result<bool, String> {
    let (tag, kids) = match pattern.carrier() {
        ExprCarrier::DecodedNode(tag, _, children) => (tag, children),
        ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_)
        | ExprCarrier::MalformedLegacyList(_) => return Ok(false),
    };
    match tag {
        DeepTag::PatVar => {
            if let Some(name) = kids.first().and_then(symbol_name) {
                bindings.insert(name.to_string(), value.clone());
                return Ok(true);
            }
            Ok(false)
        }
        DeepTag::PatWild => Ok(true),
        DeepTag::PatLit => {
            let lit = kids
                .first()
                .ok_or_else(|| "pat-lit missing value".to_string())?;
            Ok(match (value, lit) {
                (RuntimeValue::Scalar(payload), Expr::Atom(Atom::Int(rhs), _))
                    if payload.dtype().is_integer() =>
                {
                    payload.as_i64() == *rhs
                }
                (RuntimeValue::Scalar(payload), Expr::Atom(Atom::Float(rhs), _))
                    if payload.dtype().is_float() =>
                {
                    payload.as_f64_lossy() == *rhs
                }
                (RuntimeValue::Bool(lhs), Expr::Atom(Atom::Bool(rhs), _)) => lhs == rhs,
                (RuntimeValue::String(lhs), Expr::Atom(Atom::Str(rhs), _)) => lhs == rhs,
                _ => false,
            })
        }
        DeepTag::PatCtor => {
            let Some(ctor) = kids.first().and_then(symbol_name) else {
                return Ok(false);
            };
            let RuntimeValue::Adt {
                ctor: got, fields, ..
            } = value
            else {
                return Ok(false);
            };
            if ctor != got || kids.len().saturating_sub(1) != fields.len() {
                return Ok(false);
            }
            for (subpat, field) in kids.iter().skip(1).zip(fields) {
                if !pattern_matches(field, subpat, bindings, adt_fields)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        DeepTag::PatRecord => {
            let Some(ctor) = kids.first().and_then(symbol_name) else {
                return Ok(false);
            };
            let RuntimeValue::Adt {
                ctor: got,
                fields,
                field_names,
            } = value
            else {
                return Ok(false);
            };
            if ctor != got {
                return Ok(false);
            }
            let Some(declared_fields) = adt_fields.get(ctor).cloned().or(field_names.clone())
            else {
                return Ok(false);
            };
            for kv_expr in kids.iter().skip(1) {
                let kv_kids = match kv_expr.carrier() {
                    ExprCarrier::DecodedNode(DeepTag::Kv, _, children) => children,
                    ExprCarrier::DecodedNode(_, _, _)
                    | ExprCarrier::StructuralList(_)
                    | ExprCarrier::UndecodableHead(_, _, _)
                    | ExprCarrier::Atom(_)
                    | ExprCarrier::MetadataMap(_)
                    | ExprCarrier::MetadataExpression(_)
                    | ExprCarrier::MalformedLegacyList(_) => {
                        return Err("pat-record field must be a decoded `kv` node".to_string());
                    }
                };
                if kv_kids.len() != 2 {
                    return Err("pat-record `kv` field must contain a name and pattern".to_string());
                }
                let field_name = kv_kids
                    .first()
                    .and_then(symbol_name)
                    .ok_or_else(|| "pat-record field name must be a symbol".to_string())?;
                let pattern_expr = kv_kids
                    .get(1)
                    .ok_or_else(|| "pat-record field missing pattern".to_string())?;
                let Some(index) = declared_fields
                    .iter()
                    .position(|declared| declared == field_name)
                else {
                    return Ok(false);
                };
                let Some(field_value) = fields.get(index) else {
                    return Ok(false);
                };
                if !pattern_matches(field_value, pattern_expr, bindings, adt_fields)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        DeepTag::PatTuple => {
            let RuntimeValue::Tuple(items) = value else {
                return Ok(false);
            };
            if items.len() != kids.len() {
                return Ok(false);
            }
            for (subpat, item) in kids.iter().zip(items) {
                if !pattern_matches(item, subpat, bindings, adt_fields)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

pub(crate) fn collect_adt_ctor_fields(exprs: &[Expr]) -> UnordMap<String, Vec<String>> {
    let mut out = UnordMap::new();
    for expr in top_level_items(exprs) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) != Some(DeepTag::Deftype) {
            continue;
        }
        let kids = children(list);
        for variant in kids.iter().skip(2) {
            let Some(variant_list) = as_list(variant) else {
                continue;
            };
            if tag(variant_list) != Some(DeepTag::Variant) {
                continue;
            }
            let variant_kids = children(variant_list);
            let Some(ctor) = variant_kids.first().and_then(symbol_name) else {
                continue;
            };
            let mut fields = Vec::new();
            for field in variant_kids.iter().skip(1) {
                let Some(field_list) = as_list(field) else {
                    continue;
                };
                if tag(field_list) != Some(DeepTag::Field) {
                    continue;
                }
                if let Some(name) = children(field_list).first().and_then(symbol_name) {
                    fields.push(name.to_string());
                }
            }
            if !fields.is_empty() {
                out.insert(ctor.to_string(), fields);
            }
        }
    }
    out
}

pub(super) fn terminal_name_matches(full_name: &str, short_name: &str) -> bool {
    full_name == short_name || terminal_name(full_name) == terminal_name(short_name)
}

fn terminal_name(name: &str) -> &str {
    name.rsplit_once("__")
        .map(|(_, tail)| tail)
        .or_else(|| name.rsplit_once('.').map(|(_, tail)| tail))
        .unwrap_or(name)
}

fn tensor_result(
    template: &RuntimeTensorValue,
    storage: chelis_types::TensorStorage,
) -> RuntimeValue {
    RuntimeValue::Tensor(RuntimeTensorValue::new(IrTensorValue::from_storage(
        template.value.shape.clone(),
        storage,
    )))
}

/// The only host-runtime binary arithmetic dispatcher. Callable names map
/// to closed kernel enums before this boundary; arithmetic never enters as
/// an `f64`/`i64` closure.
pub(super) fn numeric_binop(
    args: &[RuntimeValue],
    int_op: Option<IntBinOp>,
    float_op: Option<FloatBinOp>,
) -> Result<RuntimeValue, String> {
    match (args.first(), args.get(1)) {
        (Some(RuntimeValue::Scalar(lhs)), Some(RuntimeValue::Scalar(rhs)))
            if lhs.dtype().is_integer() && rhs.dtype().is_integer() =>
        {
            let op = int_op.ok_or_else(|| "numeric op does not accept integer args".to_string())?;
            int_binop(op, lhs.value(), rhs.value())
                .map(RuntimeValue::from_scalar_value)
                .map_err(|error| error.to_string())
        }
        (Some(RuntimeValue::Scalar(lhs)), Some(RuntimeValue::Scalar(rhs)))
            if lhs.dtype().is_float() && rhs.dtype().is_float() =>
        {
            let op = float_op.ok_or_else(|| "numeric op does not accept float args".to_string())?;
            float_binop(op, lhs.value(), rhs.value())
                .map(RuntimeValue::from_scalar_value)
                .map_err(|error| error.to_string())
        }
        (Some(RuntimeValue::Tensor(lhs)), Some(RuntimeValue::Tensor(rhs))) => {
            if lhs.value.shape != rhs.value.shape {
                return Err(format!(
                    "tensor shapes must match for elementwise op, got {:?} vs {:?}",
                    lhs.value.shape, rhs.value.shape
                ));
            }
            let storage = if lhs.precision.is_integer() && rhs.precision.is_integer() {
                let op = int_op
                    .ok_or_else(|| "numeric op does not accept integer tensors".to_string())?;
                int_tensor_binop(op, lhs.value.storage(), rhs.value.storage())
            } else if lhs.precision.is_float() && rhs.precision.is_float() {
                let op = float_op
                    .ok_or_else(|| "numeric op does not accept float tensors".to_string())?;
                float_tensor_binop(op, lhs.value.storage(), rhs.value.storage())
            } else {
                return Err(format!(
                    "numeric op expects matching int or float tensors, got {} and {}",
                    lhs.precision.name(),
                    rhs.precision.name()
                ));
            }
            .map_err(|error| error.to_string())?;
            Ok(tensor_result(lhs, storage))
        }
        (Some(RuntimeValue::Tensor(tensor)), Some(RuntimeValue::Scalar(scalar))) => {
            let storage = if tensor.precision.is_integer() && scalar.dtype().is_integer() {
                let op = int_op
                    .ok_or_else(|| "numeric op does not accept integer tensors".to_string())?;
                int_tensor_scalar_binop(op, tensor.value.storage(), scalar.value())
            } else if tensor.precision.is_float() && scalar.dtype().is_float() {
                let op = float_op
                    .ok_or_else(|| "numeric op does not accept float tensors".to_string())?;
                float_tensor_scalar_binop(op, tensor.value.storage(), scalar.value())
            } else {
                return Err(format!(
                    "numeric op expects matching int or float args, got {} and {}",
                    tensor.precision.name(),
                    scalar.dtype().name()
                ));
            }
            .map_err(|error| error.to_string())?;
            Ok(tensor_result(tensor, storage))
        }
        (Some(RuntimeValue::Scalar(scalar)), Some(RuntimeValue::Tensor(tensor))) => {
            let storage = if scalar.dtype().is_integer() && tensor.precision.is_integer() {
                let op = int_op
                    .ok_or_else(|| "numeric op does not accept integer tensors".to_string())?;
                int_scalar_tensor_binop(op, scalar.value(), tensor.value.storage())
            } else if scalar.dtype().is_float() && tensor.precision.is_float() {
                let op = float_op
                    .ok_or_else(|| "numeric op does not accept float tensors".to_string())?;
                float_scalar_tensor_binop(op, scalar.value(), tensor.value.storage())
            } else {
                return Err(format!(
                    "numeric op expects matching int or float args, got {} and {}",
                    scalar.dtype().name(),
                    tensor.precision.name()
                ));
            }
            .map_err(|error| error.to_string())?;
            Ok(tensor_result(tensor, storage))
        }
        other => Err(format!(
            "numeric op expects matching int or float args, got {other:?}"
        )),
    }
}

pub(super) fn numeric_unop(
    args: &[RuntimeValue],
    int_op: Option<IntUnOp>,
    float_op: Option<FloatUnOp>,
) -> Result<RuntimeValue, String> {
    match args.first() {
        Some(RuntimeValue::Scalar(value)) if value.dtype().is_integer() => {
            let op =
                int_op.ok_or_else(|| "numeric op does not accept an integer arg".to_string())?;
            int_unop(op, value.value())
                .map(RuntimeValue::from_scalar_value)
                .map_err(|error| error.to_string())
        }
        Some(RuntimeValue::Scalar(value)) if value.dtype().is_float() => {
            let op =
                float_op.ok_or_else(|| "numeric op does not accept a float arg".to_string())?;
            float_unop(op, value.value())
                .map(RuntimeValue::from_scalar_value)
                .map_err(|error| error.to_string())
        }
        Some(RuntimeValue::Tensor(tensor)) if tensor.precision.is_integer() => {
            let op =
                int_op.ok_or_else(|| "numeric op does not accept an integer tensor".to_string())?;
            let storage =
                int_tensor_unop(op, tensor.value.storage()).map_err(|error| error.to_string())?;
            Ok(tensor_result(tensor, storage))
        }
        Some(RuntimeValue::Tensor(tensor)) if tensor.precision.is_float() => {
            let op =
                float_op.ok_or_else(|| "numeric op does not accept a float tensor".to_string())?;
            let storage =
                float_tensor_unop(op, tensor.value.storage()).map_err(|error| error.to_string())?;
            Ok(tensor_result(tensor, storage))
        }
        other => Err(format!(
            "numeric op expects int or float arg, got {other:?}"
        )),
    }
}

pub(super) fn eval_div(args: &[RuntimeValue]) -> Result<RuntimeValue, String> {
    numeric_binop(args, Some(IntBinOp::TruncDiv), Some(FloatBinOp::Div))
}

pub(super) fn eval_mod(args: &[RuntimeValue]) -> Result<RuntimeValue, String> {
    numeric_binop(args, Some(IntBinOp::Rem), None)
}

pub(super) fn eval_floor_div(args: &[RuntimeValue]) -> Result<RuntimeValue, String> {
    numeric_binop(args, Some(IntBinOp::FloorDiv), Some(FloatBinOp::FloorDiv))
}

pub(super) fn eval_trunc_div(args: &[RuntimeValue]) -> Result<RuntimeValue, String> {
    numeric_binop(args, Some(IntBinOp::TruncDiv), None)
}

pub(super) fn bit_int_binop(
    args: &[RuntimeValue],
    op: impl Fn(i64, i64) -> i64,
) -> Result<RuntimeValue, String> {
    match (args.first(), args.get(1)) {
        (Some(RuntimeValue::Scalar(lp)), Some(RuntimeValue::Scalar(rp)))
            if lp.dtype().is_integer() && rp.dtype().is_integer() =>
        {
            // Pre-WS-A0 stored every int as i64; preserve i64-precision
            // arithmetic but pin the result dtype to the operand dtype
            // when both sides agree, else widen to i64. Matches the
            // §5.1 "no implicit precision promotion" rule for matched
            // operands, and fails closed for mixed widths.
            let (ldt, rdt) = (lp.dtype(), rp.dtype());
            let result_dtype = if ldt == rdt { ldt } else { Prim::Int64 };
            RuntimeValue::scalar_like_int(result_dtype, op(lp.as_i64(), rp.as_i64()))
        }
        other => Err(format!("integer op expects int args, got {other:?}")),
    }
}

#[derive(Clone, Copy)]
pub(super) enum IntShiftOp {
    Left,
    Right,
}

pub(super) fn int_shift_binop(
    args: &[RuntimeValue],
    op: IntShiftOp,
) -> Result<RuntimeValue, String> {
    match (args.first(), args.get(1)) {
        (Some(RuntimeValue::Scalar(lp)), Some(RuntimeValue::Scalar(rp)))
            if lp.dtype().is_integer() && rp.dtype().is_integer() =>
        {
            let rhs = rp.as_i64();
            if rhs < 0 {
                return Err(format!("shift amount must be non-negative, got {rhs}"));
            }
            let width = match lp.dtype() {
                Prim::Int8 => 8_u32,
                Prim::Int16 => 16,
                Prim::Int32 => 32,
                Prim::Int64 => 64,
                _ => unreachable!("integer guard above excludes non-integer shift operands"),
            };
            // [04-NUM-13]: shifts are width-bounded, not host-language
            // shifts. A count at or above the declared width produces the
            // fully shifted-out value (zero for left shift/nonnegative right
            // shift, all ones for negative arithmetic right shift). Avoid
            // invoking Rust's debug-panic shift path for those counts.
            let lhs = lp.as_i64();
            let raw = if rhs >= i64::from(width) {
                match op {
                    IntShiftOp::Left => 0,
                    IntShiftOp::Right if lhs < 0 => -1,
                    IntShiftOp::Right => 0,
                }
            } else {
                match op {
                    IntShiftOp::Left => lhs.wrapping_shl(rhs as u32),
                    IntShiftOp::Right => lhs.wrapping_shr(rhs as u32),
                }
            };
            // [04-NUM-13] discards bits at the declared width. Narrow before
            // the trapping finalizer so an in-spec shift such as `1i8 << 7`
            // stores -128 instead of being misclassified as arithmetic
            // overflow.
            let wrapped = match lp.dtype() {
                Prim::Int8 => (raw as i8) as i64,
                Prim::Int16 => (raw as i16) as i64,
                Prim::Int32 => (raw as i32) as i64,
                other => {
                    debug_assert_eq!(other, Prim::Int64);
                    raw
                }
            };
            RuntimeValue::scalar_like_int(lp.dtype(), wrapped)
        }
        other => Err(format!("shift op expects int args, got {other:?}")),
    }
}

pub(super) fn float_unop_with_tensor(
    args: &[RuntimeValue],
    op: FloatUnOp,
) -> Result<RuntimeValue, String> {
    numeric_unop(args, None, Some(op))
}

fn comparison_scalar(value: &RuntimeValue) -> Option<ScalarValue> {
    match value {
        RuntimeValue::Scalar(payload) => Some(payload.value()),
        RuntimeValue::Bool(value) => Some(
            scalar_from_i64("compare", Prim::Bool, i64::from(*value))
                .expect("a bool is always in the bool dtype domain"),
        ),
        _ => None,
    }
}

pub(super) fn compare_eq(args: &[RuntimeValue]) -> Result<RuntimeValue, String> {
    compare_runtime(args, CompareOp::Eq)
}

pub(super) fn compare_runtime(
    args: &[RuntimeValue],
    op: CompareOp,
) -> Result<RuntimeValue, String> {
    match (args.first(), args.get(1)) {
        (Some(lhs), Some(rhs))
            if comparison_scalar(lhs).is_some() && comparison_scalar(rhs).is_some() =>
        {
            compare_scalars(
                op,
                comparison_scalar(lhs).expect("comparison scalar guard"),
                comparison_scalar(rhs).expect("comparison scalar guard"),
            )
            .map(RuntimeValue::Bool)
            .map_err(|error| error.to_string())
        }
        (Some(RuntimeValue::String(lhs)), Some(RuntimeValue::String(rhs))) => match op {
            CompareOp::Eq => Ok(RuntimeValue::Bool(lhs == rhs)),
            CompareOp::Ne => Ok(RuntimeValue::Bool(lhs != rhs)),
            _ => Err("ordered comparison does not accept string args".to_string()),
        },
        (Some(RuntimeValue::Tensor(lhs)), Some(RuntimeValue::Tensor(rhs))) => {
            tensor_compare_value(lhs, rhs, op).map(RuntimeValue::Tensor)
        }
        // A scalar beside a tensor. These two arms used to broadcast the
        // scalar and return a `tensor[D, bool]`. chelis#1506 makes the form a
        // type error under `[05-OP-36]`, so a CHECKED program can no longer
        // reach here; this lane is also driven by API callers with unchecked
        // input, which is why the arms become a named refusal rather than a
        // deletion. Deleting them would fall through to the generic arm below,
        // whose message names no rule and reads like an internal error, and
        // `[05-UNS-4]` says a pre-codegen gate "SHALL NOT be the sole defense
        // against an unsupported case reaching emission". Broadcasting here
        // was itself the value substitution `[05-UNS-1]` forbids.
        (Some(RuntimeValue::Tensor(_)), Some(scalar)) if comparison_scalar(scalar).is_some() => {
            Err(mixed_comparison_surface_error())
        }
        (Some(scalar), Some(RuntimeValue::Tensor(_))) if comparison_scalar(scalar).is_some() => {
            Err(mixed_comparison_surface_error())
        }
        other => Err(format!("comparison expects matching args, got {other:?}")),
    }
}

/// `[05-UNS-1]`: an unsupported case is refused, never substituted with a
/// value. The checker rejects a scalar beside a tensor under `[05-OP-36]`
/// (chelis#1506); this is the runtime backstop for input that did not come
/// through it.
fn mixed_comparison_surface_error() -> String {
    "[05-UNS-1] comparison does not admit a scalar beside a tensor: \
     spec/05-risc-primitives.md [05-OP-36] makes a mixed surface a type error. \
     Give the scalar the tensor's shape explicitly, as in \
     `gt(xs, expand(to_tensor([1.5f32]), 0i32, shape(xs, 0i32)))`."
        .to_string()
}

pub(super) fn ordered_compare(
    args: &[RuntimeValue],
    op: CompareOp,
) -> Result<RuntimeValue, String> {
    compare_runtime(args, op)
}

pub(super) fn tensor_compare_value(
    lhs: &RuntimeTensorValue,
    rhs: &RuntimeTensorValue,
    op: CompareOp,
) -> Result<RuntimeTensorValue, String> {
    if lhs.value.shape != rhs.value.shape {
        return Err("tensor comparison expects matching tensor shape".to_string());
    }
    let storage = compare_tensors(op, lhs.value.storage(), rhs.value.storage())
        .map_err(|error| error.to_string())?;
    Ok(RuntimeTensorValue::new(IrTensorValue::from_storage(
        lhs.value.shape.clone(),
        storage,
    )))
}

pub(super) fn bool_binop(
    args: &[RuntimeValue],
    op: impl Fn(bool, bool) -> bool,
) -> Result<RuntimeValue, String> {
    match (args.first(), args.get(1)) {
        (Some(RuntimeValue::Bool(lhs)), Some(RuntimeValue::Bool(rhs))) => {
            Ok(RuntimeValue::Bool(op(*lhs, *rhs)))
        }
        other => Err(format!("bool op expects bool args, got {other:?}")),
    }
}

pub(super) fn bool_unop(
    args: &[RuntimeValue],
    op: impl Fn(bool) -> bool,
) -> Result<RuntimeValue, String> {
    match args.first() {
        Some(RuntimeValue::Bool(value)) => Ok(RuntimeValue::Bool(op(*value))),
        other => Err(format!("bool op expects bool arg, got {other:?}")),
    }
}

/// Element-wise tensor-bool binary op. Per the pinned decision for
/// issue Chelis-Lang/chelis#185 the tensor-bool arms are SEPARATE from
/// the scalar `bool_binop` helper: tensor-bool semantics require
/// explicit precision + shape checking that scalar broadcasting would
/// hide. The IR evaluator does not have a dedicated bool path —
/// `tier2::lower_and`/`lower_or` lower to `Mul`/`MaxElem` over
/// 0.0/1.0-encoded bool tensors — but the host runtime stores
/// `tensor[D, bool]` as f64 data with `precision == Prim::Bool` (see
/// `tensor_compare_value` / `tensor_compare_scalar` which produce
/// 0.0/1.0 entries). Treat the truth value as `element != 0.0`,
/// re-encode the result the same way, and pin the output precision to
/// `Bool`.
pub(super) fn tensor_bool_binop(
    lhs: &RuntimeTensorValue,
    rhs: &RuntimeTensorValue,
    op: impl Fn(bool, bool) -> bool,
) -> Result<RuntimeTensorValue, String> {
    if lhs.precision != Prim::Bool || rhs.precision != Prim::Bool {
        return Err(format!(
            "tensor bool op expects tensor[D, bool] inputs, got lhs precision {} and rhs precision {}",
            lhs.precision.name(),
            rhs.precision.name()
        ));
    }
    if lhs.value.shape != rhs.value.shape {
        return Err(format!(
            "tensor bool op expects matching shapes, got {:?} vs {:?}",
            lhs.value.shape, rhs.value.shape
        ));
    }
    let data = lhs
        .value
        .to_f64_lossy_vec()
        .into_iter()
        .zip(rhs.value.to_f64_lossy_vec())
        .map(|(l, r)| if op(l != 0.0, r != 0.0) { 1 } else { 0 })
        .collect::<Vec<i64>>();
    RuntimeTensorValue::from_wide_int("bool", Prim::Bool, lhs.value.shape.clone(), data)
}

/// Element-wise tensor-bool unary op. See `tensor_bool_binop` for the
/// rationale on keeping this separate from the scalar `bool_unop` arm.
pub(super) fn tensor_bool_unop(
    tensor: &RuntimeTensorValue,
    op: impl Fn(bool) -> bool,
) -> Result<RuntimeTensorValue, String> {
    if tensor.precision != Prim::Bool {
        return Err(format!(
            "tensor bool op expects tensor[D, bool] input, got precision {}",
            tensor.precision.name()
        ));
    }
    let data = tensor
        .value
        .to_f64_lossy_vec()
        .into_iter()
        .map(|value| if op(value != 0.0) { 1 } else { 0 })
        .collect::<Vec<i64>>();
    RuntimeTensorValue::from_wide_int("bool", Prim::Bool, tensor.value.shape.clone(), data)
}

/// Declared-tensor element dtype of a Deep type expression: `Some(prim)`
/// exactly when the expression is a `t-tensor` whose element type is an
/// active tensor dtype.
pub(super) fn declared_tensor_prim(expr: &Expr) -> Option<Prim> {
    let list = as_list(expr)?;
    if tag(list) != Some(DeepTag::TTensor) {
        return None;
    }
    extract_prim_from_type_expr(expr)
}

/// Ingress-finalize a tensor value at a declared element dtype (the
/// host-lane mirror of the DAG evaluator's Load ingress; chelis#729
/// Phase 1). Identity when the dtypes already agree; float targets apply
/// the dtype's rounding, integer/bool targets domain-check loudly.
pub(super) fn ingress_tensor_to_declared(
    tensor: RuntimeTensorValue,
    prim: Prim,
) -> Result<RuntimeTensorValue, String> {
    if tensor.precision == prim {
        return Ok(tensor);
    }
    let storage = chelis_types::finalize_tensor("param", prim, tensor.value.storage().to_raw())
        .map_err(|trap| trap.to_string())?;
    Ok(RuntimeTensorValue::new(IrTensorValue::from_storage(
        tensor.value.shape.clone(),
        storage,
    )))
}

pub(super) fn expect_tensor_arg(
    args: &[RuntimeValue],
    index: usize,
) -> Result<RuntimeTensorValue, String> {
    match args.get(index) {
        Some(RuntimeValue::Tensor(value)) => Ok(value.clone()),
        other => Err(format!(
            "expected tensor arg at index {index}, got {other:?}"
        )),
    }
}

pub(super) fn expect_string_arg(args: &[RuntimeValue], index: usize) -> Result<String, String> {
    match args.get(index) {
        Some(RuntimeValue::String(value)) => Ok(value.clone()),
        other => Err(format!(
            "expected string arg at index {index}, got {other:?}"
        )),
    }
}

pub(super) fn expect_list_arg(
    args: &[RuntimeValue],
    index: usize,
) -> Result<Vec<RuntimeValue>, String> {
    match args.get(index) {
        Some(RuntimeValue::List(items)) => Ok(items.clone()),
        other => Err(format!("expected list arg at index {index}, got {other:?}")),
    }
}

pub(super) fn expect_dict_arg(
    args: &[RuntimeValue],
    index: usize,
) -> Result<Vec<(RuntimeValue, RuntimeValue)>, String> {
    match args.get(index) {
        Some(RuntimeValue::Dict(entries)) => Ok(entries.clone()),
        other => Err(format!("expected dict arg at index {index}, got {other:?}")),
    }
}

pub(super) fn expect_int_arg(args: &[RuntimeValue], index: usize) -> Result<i64, String> {
    match args.get(index) {
        Some(RuntimeValue::Scalar(payload)) if payload.dtype().is_integer() => Ok(payload.as_i64()),
        other => Err(format!("expected int arg at index {index}, got {other:?}")),
    }
}

pub(super) fn expect_bool_arg(args: &[RuntimeValue], index: usize) -> Result<bool, String> {
    match args.get(index) {
        Some(RuntimeValue::Bool(value)) => Ok(*value),
        other => Err(format!("expected bool arg at index {index}, got {other:?}")),
    }
}

pub(super) fn expect_float_arg(args: &[RuntimeValue], index: usize) -> Result<f64, String> {
    match args.get(index) {
        Some(RuntimeValue::Scalar(payload)) if payload.dtype().is_float() => {
            Ok(payload.as_f64_lossy())
        }
        Some(RuntimeValue::Scalar(payload)) if payload.dtype().is_integer() => {
            Ok(payload.as_f64_lossy())
        }
        other => Err(format!(
            "expected float arg at index {index}, got {other:?}"
        )),
    }
}

pub(super) fn ensure_dict_key_supported(value: &RuntimeValue) -> Result<(), String> {
    match value {
        RuntimeValue::Scalar(payload) if payload.dtype().is_integer() => Ok(()),
        RuntimeValue::String(_) => Ok(()),
        other => Err(format!(
            "dict keys must be i64 or string in 3d, got {other:?}"
        )),
    }
}

pub(super) fn runtime_value_eq(lhs: &RuntimeValue, rhs: &RuntimeValue) -> bool {
    match (lhs, rhs) {
        (RuntimeValue::Scalar(lp), RuntimeValue::Scalar(rp))
            if lp.dtype().is_integer() && rp.dtype().is_integer() =>
        {
            lp.as_i64() == rp.as_i64()
        }
        (RuntimeValue::Scalar(lp), RuntimeValue::Scalar(rp))
            if lp.dtype().is_float() && rp.dtype().is_float() =>
        {
            lp.as_f64_lossy() == rp.as_f64_lossy()
        }
        (RuntimeValue::Bool(lhs), RuntimeValue::Bool(rhs)) => lhs == rhs,
        (RuntimeValue::String(lhs), RuntimeValue::String(rhs)) => lhs == rhs,
        (RuntimeValue::Tuple(lhs), RuntimeValue::Tuple(rhs)) => {
            lhs.len() == rhs.len()
                && lhs
                    .iter()
                    .zip(rhs)
                    .all(|(lhs, rhs)| runtime_value_eq(lhs, rhs))
        }
        _ => false,
    }
}

pub(super) fn upsert_dict_entry(
    dict: &mut Vec<(RuntimeValue, RuntimeValue)>,
    key: RuntimeValue,
    value: RuntimeValue,
) {
    if let Some((_, existing)) = dict
        .iter_mut()
        .find(|(existing_key, _)| runtime_value_eq(existing_key, &key))
    {
        *existing = value;
    } else {
        dict.push((key, value));
    }
}

pub(super) fn dict_lookup<'a>(
    dict: &'a [(RuntimeValue, RuntimeValue)],
    key: &RuntimeValue,
) -> Option<&'a RuntimeValue> {
    dict.iter()
        .find(|(existing_key, _)| runtime_value_eq(existing_key, key))
        .map(|(_, value)| value)
}

/// V2-F2: element-wise precision conversion for `cast(tensor[..], q)` in
/// the host runtime. Spec §2.7 lists `cast` as a first-class precision
/// transform; the IR DAG and C backend already handle the tensor form
/// (`crates/chelis-ir/src/lower.rs::lower_cast`,
/// `crates/chelis-backend-c/src/emit.rs::emit_cast`). This helper closes
/// the runtime/host-lane gap that PR #58's red-team v2 surfaced.
///
/// The output shape is preserved element-for-element (C8 in
/// `crates/chelis-ir/src/verify.rs`: cast dims must not change).
pub(super) fn cast_tensor_value(
    tensor: RuntimeTensorValue,
    target_prim: Prim,
) -> Result<RuntimeValue, String> {
    let RuntimeTensorValue {
        value: ir_value,
        precision: src_prim,
    } = tensor;
    // chelis#729: one CHECKED cast ladder for both eval surfaces
    // (`chelis_types::cast_raw` via the DAG evaluator's `cast_tensor`).
    // Integer storage reads exactly, float targets finalize at width
    // (f16/bf16 casts genuinely round, chelis#717), and out-of-range or
    // out-of-domain elements TRAP per spec/04 section 5.2; the named
    // lossy forms remain chelis#759's future surface.
    chelis_ir::eval::cast_tensor(&ir_value, src_prim, target_prim)
        .map(|value| RuntimeValue::Tensor(RuntimeTensorValue::new(value)))
}

/// The [05-OP-6] tensor rung, routed through the same sealed kernel the
/// DAG evaluator uses so the two eval surfaces cannot diverge.
pub(super) fn cast_trunc_tensor_value(
    tensor: RuntimeTensorValue,
    target_prim: Prim,
) -> Result<RuntimeValue, String> {
    let RuntimeTensorValue {
        value: ir_value,
        precision: _,
    } = tensor;
    chelis_ir::eval::cast_trunc_tensor(&ir_value, target_prim)
        .map(|value| RuntimeValue::Tensor(RuntimeTensorValue::new(value)))
}

/// Bucket 4b: recursively flatten a nested numeric/bool list into a
/// rank-N tensor. Every nesting level contributes one outer dimension;
/// the innermost level must be uniformly numeric or bool. All sibling
/// sub-lists at the same level must have matching length and matching
/// precision.
///
/// The checked leaf dtype and rank are required inputs. Empty Lists carry
/// no payload evidence, and hidden trailing extents require checked witnesses
/// rather than a guessed dtype or shape ([05-OP-57]).
pub(super) fn nested_list_to_tensor_data(
    outer: &[RuntimeValue],
    precision: Prim,
    expected_shape: &[Option<usize>],
) -> Result<(Vec<usize>, ListTensorData), String> {
    let Some((expected_extent, inner_extents)) = expected_shape.split_first() else {
        return Err("to_tensor expected a positive-rank checked result".into());
    };
    if expected_extent.is_some_and(|extent| extent != outer.len()) {
        return Err("to_tensor List length disagrees with its checked tensor extent".into());
    }
    if outer.is_empty() {
        let mut shape = vec![0];
        for extent in inner_extents {
            shape.push(extent.ok_or(
                "to_tensor cannot resolve an inner extent hidden by an empty List [05-OP-57]",
            )?);
        }
        return Ok((shape, list_to_tensor_data(outer, precision)?));
    }
    if inner_extents.is_empty() {
        let data = list_to_tensor_data(outer, precision)?;
        return Ok((vec![outer.len()], data));
    }

    let mut rows = outer.iter().enumerate().map(|(index, value)| {
        let RuntimeValue::List(inner) = value else {
            return Err(format!(
                "to_tensor expects homogeneous nested lists; element {index} is not a List"
            ));
        };
        nested_list_to_tensor_data(inner, precision, inner_extents)
    });
    let (inner_shape, mut data) = rows.next().expect("nonempty List checked above")?;
    for row in rows {
        let (shape, row_data) = row?;
        if shape != inner_shape {
            return Err("to_tensor requires uniform inner shape".into());
        }
        data.extend(row_data)?;
    }
    let mut shape = vec![outer.len()];
    shape.extend(inner_shape);
    Ok((shape, data))
}

/// Wide ingress buffer for `to_tensor`: exact i64 for the integer/bool
/// families, exact f64 images for floats (chelis#729 Phase 1; ends the
/// f64-collapse of exact i64 elements, chelis#684).
pub(super) enum ListTensorData {
    Int(Vec<i64>),
    Float(Vec<f64>),
}

impl ListTensorData {
    pub(super) fn into_raw(self) -> chelis_types::RawTensor {
        match self {
            ListTensorData::Int(v) => chelis_types::RawTensor::Int(v),
            ListTensorData::Float(v) => chelis_types::RawTensor::Float(v),
        }
    }

    fn len(&self) -> usize {
        match self {
            ListTensorData::Int(v) => v.len(),
            ListTensorData::Float(v) => v.len(),
        }
    }

    fn extend(&mut self, other: ListTensorData) -> Result<(), String> {
        match (self, other) {
            (ListTensorData::Int(a), ListTensorData::Int(b)) => {
                a.extend(b);
                Ok(())
            }
            (ListTensorData::Float(a), ListTensorData::Float(b)) => {
                a.extend(b);
                Ok(())
            }
            (ListTensorData::Int(_), ListTensorData::Float(_))
            | (ListTensorData::Float(_), ListTensorData::Int(_)) => {
                Err("to_tensor requires homogeneous numeric or bool list elements".to_string())
            }
        }
    }
}

fn list_to_tensor_data(values: &[RuntimeValue], precision: Prim) -> Result<ListTensorData, String> {
    // The declared dtype is authoritative even for zero elements. Nonempty
    // payloads validate against it; they never choose or change that dtype.
    let mut ints = Vec::new();
    let mut floats = Vec::new();
    for value in values {
        match value {
            RuntimeValue::Scalar(payload)
                if payload.dtype() == precision && precision.is_integer() =>
            {
                ints.push(payload.as_i64());
            }
            RuntimeValue::Scalar(payload)
                if payload.dtype() == precision && precision.is_float() =>
            {
                floats.push(payload.as_f64_lossy());
            }
            RuntimeValue::Bool(value) if precision == Prim::Bool => {
                ints.push(i64::from(*value));
            }
            _ => return Err(
                "to_tensor requires homogeneous numeric or bool list elements at the checked dtype"
                    .into(),
            ),
        }
    }
    if precision.is_float() {
        Ok(ListTensorData::Float(floats))
    } else if precision.is_integer() || precision == Prim::Bool {
        Ok(ListTensorData::Int(ints))
    } else {
        Err("to_tensor requires an active numeric or bool element dtype".into())
    }
}

pub(super) fn tensor_to_list_values(
    tensor: &RuntimeTensorValue,
) -> Result<Vec<RuntimeValue>, String> {
    if tensor.value.shape.len() != 1 {
        return Err(format!(
            "to_list expects a rank-1 tensor, got rank {} tensor",
            tensor.value.shape.len()
        ));
    }
    // chelis#729 Phase 1: elements read the sealed per-dtype storage
    // directly, so i64 lists stay exact above 2^53 and float elements
    // carry their own width (the probe-2 to_list narrowing is gone).
    let mut values = Vec::with_capacity(tensor.value.len());
    for index in 0..tensor.value.len() {
        let element = tensor.value.storage().scalar_at(index);
        let element = match element.as_bool_exact() {
            Some(flag) => RuntimeValue::Bool(flag),
            None => RuntimeValue::from_scalar_value(element),
        };
        values.push(element);
    }
    Ok(values)
}

/// Shared row collector for the `pad_sequences*` family: the pad scalar fixes
/// the exact output dtype, every non-empty row must carry that same dtype,
/// and the padded row-major data stays in a wide ingress buffer until final
/// storage construction (chelis#729 Phase 1, section C3).
fn pad_sequences_rows(
    sequences: &[RuntimeValue],
    pad: &RuntimeValue,
    op: &str,
) -> Result<(Prim, ListTensorData, Vec<usize>), String> {
    let pad_precision = match pad {
        RuntimeValue::Scalar(payload)
            if payload.dtype().is_integer() || payload.dtype().is_float() =>
        {
            payload.dtype()
        }
        other => {
            return Err(format!("{op} expects numeric pad value, got {other:?}"));
        }
    };
    let pad_is_int = pad_precision.is_integer();
    let mut rows = Vec::with_capacity(sequences.len());
    let mut lens = Vec::with_capacity(sequences.len());
    for sequence in sequences {
        let RuntimeValue::List(items) = sequence else {
            return Err(format!("{op} expects nested lists, got {sequence:?}"));
        };
        let row = list_to_tensor_data(items, pad_precision)?;
        lens.push(row.len());
        rows.push(row);
    }
    let mut data = if pad_is_int {
        ListTensorData::Int(Vec::new())
    } else {
        ListTensorData::Float(Vec::new())
    };
    for row in rows {
        // An empty row carries the family of its (empty) collector;
        // coerce it to the pad family so extend type-checks.
        let row = match (&data, row) {
            (ListTensorData::Int(_), ListTensorData::Float(v)) if v.is_empty() => {
                ListTensorData::Int(Vec::new())
            }
            (ListTensorData::Float(_), ListTensorData::Int(v)) if v.is_empty() => {
                ListTensorData::Float(Vec::new())
            }
            (_, row) => row,
        };
        data.extend(row)?;
    }
    Ok((pad_precision, data, lens))
}

pub(super) fn pad_sequences_value(
    sequences: &[RuntimeValue],
    pad: &RuntimeValue,
) -> Result<(Prim, ListTensorData, usize, usize), String> {
    let (pad_precision, rows, lens) = pad_sequences_rows(sequences, pad, "pad_sequences")?;
    let width = lens.iter().copied().fold(0usize, usize::max);
    let batch = lens.len();
    let data = pad_rows(rows, &lens, width, pad, batch)?;
    Ok((pad_precision, data, batch, width))
}

pub(super) fn pad_sequences_to_value(
    sequences: &[RuntimeValue],
    width: i64,
    pad: &RuntimeValue,
) -> Result<(Prim, ListTensorData, usize), String> {
    if width < 0 {
        return Err(format!(
            "pad_sequences_to requires non-negative width, got {width}"
        ));
    }
    let (pad_precision, rows, lens) = pad_sequences_rows(sequences, pad, "pad_sequences_to")?;
    let width = width as usize;
    let batch = lens.len();
    let data = pad_rows(rows, &lens, width, pad, batch)?;
    Ok((pad_precision, data, batch))
}

/// Lay the concatenated rows out row-major at `width`, truncating long
/// rows and filling short ones with the pad scalar (exact per family).
fn pad_rows(
    rows: ListTensorData,
    lens: &[usize],
    width: usize,
    pad: &RuntimeValue,
    batch: usize,
) -> Result<ListTensorData, String> {
    let RuntimeValue::Scalar(payload) = pad else {
        return Err(format!(
            "pad_sequences expects numeric pad value, got {pad:?}"
        ));
    };
    match rows {
        ListTensorData::Int(flat) => {
            let pad_value = payload.as_i64();
            let mut out = Vec::with_capacity(batch * width);
            let mut offset = 0usize;
            for &len in lens {
                let used = len.min(width);
                out.extend_from_slice(&flat[offset..offset + used]);
                out.extend(std::iter::repeat_n(pad_value, width.saturating_sub(used)));
                offset += len;
            }
            Ok(ListTensorData::Int(out))
        }
        ListTensorData::Float(flat) => {
            let pad_value = payload.as_f64_lossy();
            let mut out = Vec::with_capacity(batch * width);
            let mut offset = 0usize;
            for &len in lens {
                let used = len.min(width);
                out.extend_from_slice(&flat[offset..offset + used]);
                out.extend(std::iter::repeat_n(pad_value, width.saturating_sub(used)));
                offset += len;
            }
            Ok(ListTensorData::Float(out))
        }
    }
}

/// Element count for a host shape.
///
/// A zero extent means zero elements ([05-OP-33]), and the answer does not
/// depend on where the zero sits, so it short-circuits rather than folding
/// past it: `[2^32, 2^32, 0]` reaches `2^64` before it reaches the zero.
/// `chelis-ir`'s `numel` holds the same contract.
///
/// The `.max(1)` this replaces read as the rank-zero convention, but
/// `[].iter().product()` is already one, so the clamp only ever fired on a
/// zero-containing shape, where it fabricated an element that does not exist.
/// Callers use the count as a `0..n` bound over an output buffer, so that
/// phantom element drove `linear_to_indices` into `linear % 0`, or produced a
/// `picks` vector one longer than the storage its shape declares.
fn tensor_numel(shape: &[usize]) -> usize {
    if shape.contains(&0) {
        return 0;
    }
    shape.iter().product()
}

fn linear_to_indices(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut indices = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        indices[axis] = linear % shape[axis];
        linear /= shape[axis];
    }
    indices
}

fn indices_to_linear(indices: &[usize], shape: &[usize]) -> usize {
    let mut linear = 0usize;
    for (axis, value) in indices.iter().enumerate() {
        linear *= shape[axis];
        linear += value;
    }
    linear
}

pub(super) fn normalize_axis(rank: usize, axis: i64, op: &str) -> Result<usize, String> {
    // chelis#522: apply the from-the-end negative-axis convention uniformly
    // (`-1` == last axis, `-rank` == axis 0), matching the type checker
    // (`chelis_types::normalize_static_axis`), the IR lowerer
    // (`chelis_ir::lower::normalize_axis`), and spec/05-risc-primitives.md.
    // Before this the host evaluator rejected every negative axis with
    // "requires non-negative axis", so `sum(x, -1)` passed `chelis check` /
    // `chelis build` (both normalize) but failed `chelis eval` — a check↔eval
    // soundness gap (same class as #364). An axis still out of `0..rank` after
    // the offset (e.g. `rank` or `-rank-1`) is rejected loud, not wrapped.
    let normalized = if axis < 0 { axis + rank as i64 } else { axis };
    if normalized < 0 || normalized as usize >= rank {
        return Err(format!("{op} axis {axis} out of bounds for rank {rank}"));
    }
    Ok(normalized as usize)
}

pub(super) fn expect_int_list(values: &[RuntimeValue], op: &str) -> Result<Vec<usize>, String> {
    values
        .iter()
        .map(|value| match value {
            RuntimeValue::Scalar(payload) if payload.dtype().is_integer() => {
                let v = payload.as_i64();
                if v >= 0 {
                    Ok(v as usize)
                } else {
                    Err(format!("{op} expects non-negative sizes, got {v}"))
                }
            }
            other => Err(format!("{op} expects i64 sizes, got {other:?}")),
        })
        .collect()
}

#[derive(Clone, Copy)]
pub(super) enum ReduceOp {
    Sum,
    Min,
    Max,
    Prod,
    Argmax,
    Argmin,
}

/// Reducer selector for the host-runtime `reduce_window_*` family.
/// Mirrors `chelis_ir::dag::ReduceWindowKind` so host-runtime eval and
/// IR-evaluator paths agree on the operational meaning of each builtin
/// name. See `spec/05-risc-primitives.md` §2.3.1.
#[derive(Clone, Copy)]
pub(super) enum ReduceWindowOp {
    Max,
    Min,
    Sum,
    Mean,
}

pub(super) fn tensor_reduce_host(
    tensor: &RuntimeTensorValue,
    axis: i64,
    op: ReduceOp,
) -> Result<RuntimeTensorValue, String> {
    let rank = tensor.value.shape.len();
    let axis = normalize_axis(rank, axis, "reduction")?;
    let mut out_shape: Vec<usize> = tensor.value.shape.clone();
    let axis_len = out_shape.remove(axis);
    if axis_len == 0 {
        return Err("reduction over empty axis is undefined".to_string());
    }
    let out_numel = tensor_numel(&out_shape);
    let mut groups = Vec::with_capacity(out_numel);
    for out_linear in 0..out_numel {
        let out_indices = linear_to_indices(out_linear, &out_shape);
        let mut group = Vec::with_capacity(axis_len);
        for k in 0..axis_len {
            let mut in_indices = Vec::with_capacity(rank);
            let mut oi = 0;
            for dim in 0..rank {
                if dim == axis {
                    in_indices.push(k);
                } else {
                    in_indices.push(out_indices[oi]);
                    oi += 1;
                }
            }
            group.push(indices_to_linear(&in_indices, &tensor.value.shape));
        }
        groups.push(group);
    }

    let storage = match op {
        ReduceOp::Argmax => {
            arg_reduce_tensor_groups(ArgReduceOp::Argmax, tensor.value.storage(), &groups)
        }
        ReduceOp::Argmin => {
            arg_reduce_tensor_groups(ArgReduceOp::Argmin, tensor.value.storage(), &groups)
        }
        ReduceOp::Sum => {
            let accumulator = tensor.precision.default_reduce_sum_accumulator()?;
            let result = tensor.precision.default_reduce_sum_result_precision()?;
            reduce_tensor_groups(
                TensorReduceOp::Sum {
                    accumulator,
                    result,
                },
                tensor.value.storage(),
                &groups,
            )
        }
        ReduceOp::Min => {
            reduce_tensor_groups(TensorReduceOp::MinReduce, tensor.value.storage(), &groups)
        }
        ReduceOp::Max => {
            reduce_tensor_groups(TensorReduceOp::MaxReduce, tensor.value.storage(), &groups)
        }
        ReduceOp::Prod => {
            reduce_tensor_groups(TensorReduceOp::ProdReduce, tensor.value.storage(), &groups)
        }
    };
    let storage = storage.map_err(|err| err.to_string())?;
    Ok(RuntimeTensorValue::new(IrTensorValue::from_storage(
        out_shape, storage,
    )))
}

pub(super) fn tensor_count_host(
    tensor: &RuntimeTensorValue,
    axes: &[usize],
) -> Result<RuntimeTensorValue, String> {
    chelis_ir::eval::count_tensor(&tensor.value, axes).map(RuntimeTensorValue::new)
}

/// Permute axes of a tensor, given an `axes` permutation. `axes[i]` is the
/// source axis for output axis `i`.
pub(super) fn tensor_permute_host(
    tensor: &RuntimeTensorValue,
    axes: &[usize],
) -> Result<RuntimeTensorValue, String> {
    let rank = tensor.value.shape.len();
    if axes.len() != rank {
        return Err(format!(
            "permute expects {rank} axis arguments for rank-{rank} tensor, got {}",
            axes.len()
        ));
    }
    let mut seen = vec![false; rank];
    for &axis in axes {
        if axis >= rank {
            return Err(format!("permute axis {axis} out of bounds for rank {rank}"));
        }
        if seen[axis] {
            return Err(format!("permute axes contain duplicate axis {axis}"));
        }
        seen[axis] = true;
    }
    let in_shape = tensor.value.shape.clone();
    let out_shape: Vec<usize> = axes.iter().map(|&a| in_shape[a]).collect();
    let out_numel = tensor_numel(&out_shape);
    let mut picks = vec![0usize; out_numel];
    for in_linear in 0..tensor.value.len() {
        let in_indices = linear_to_indices(in_linear, &in_shape);
        let out_indices: Vec<usize> = axes.iter().map(|&a| in_indices[a]).collect();
        picks[indices_to_linear(&out_indices, &out_shape)] = in_linear;
    }
    // reuse_* contract: permute is element-preserving (section C3).
    Ok(RuntimeTensorValue::new(IrTensorValue::from_storage(
        out_shape,
        tensor.value.storage().reuse_gather(&picks),
    )))
}

/// Execute matrix multiplication through its typed, rank-generic lowering.
pub(super) fn tensor_matmul_host(
    lhs: &RuntimeTensorValue,
    rhs: &RuntimeTensorValue,
) -> Result<RuntimeTensorValue, String> {
    let a = &lhs.value.shape;
    let b = &rhs.value.shape;
    if a.len() < 2 || b.len() < 2 {
        return Err("matmul requires both operand ranks to be at least two".to_string());
    }
    if lhs.precision != rhs.precision || !lhs.precision.is_float() {
        return Err("matmul requires one matching active float dtype".to_string());
    }
    if a[a.len() - 1] != b[b.len() - 2] {
        return Err("matmul shared-axis mismatch".to_string());
    }
    for (&a_extent, &b_extent) in a[..a.len() - 2]
        .iter()
        .rev()
        .zip(b[..b.len() - 2].iter().rev())
    {
        if a_extent != b_extent && a_extent != 1 && b_extent != 1 {
            return Err("matmul batch-axis mismatch".to_string());
        }
    }
    let mut dag = Dag::new();
    let lhs_ty = tensor_type_for(lhs);
    let rhs_ty = tensor_type_for(rhs);
    let lhs_name = format!("{COMPOSED_PLACEHOLDER_PREFIX}0");
    let rhs_name = format!("{COMPOSED_PLACEHOLDER_PREFIX}1");
    let lhs_id = add_load(&mut dag, lhs_name.clone(), lhs_ty.clone());
    let rhs_id = add_load(&mut dag, rhs_name.clone(), rhs_ty.clone());
    let root = tier2::lower_matmul(&mut dag, lhs_id, rhs_id, &lhs_ty, &rhs_ty, None);
    let mut inputs = UnordMap::new();
    inputs.insert(lhs_name, lhs.value.clone());
    inputs.insert(rhs_name, rhs.value.clone());
    extract_root(&dag, &inputs, root, "matmul")
}

/// `insert`: replicate a tensor along a NEW axis.
///
/// `insert(b, axis, count)` produces a tensor of shape `[..., count, ...]`
/// with `count` inserted at position `axis`, where every slice along the new
/// axis is a copy of `b`. The output rank is always `input_rank + 1`
/// (`spec/04-type-system.md` section 4.7.2).
///
/// This function used to serve both names and always inserted, because the
/// old positional `expand` had two candidate result shapes and the host
/// runtime cannot see the user annotation that chose between them. Picking
/// the same-rank branch whenever `in_shape[axis] == 1` is what made
/// `chelis eval` disagree with `chelis check` on `examples/linreg.ch`
/// (Bucket 4a). With one shape per operation there is no choice left to guess
/// at: `insert` lands here and `expand` lands in `tensor_expand_host`.
pub(super) fn tensor_insert_host(
    builtin: &str,
    tensor: &RuntimeTensorValue,
    axis: usize,
    count: usize,
) -> Result<RuntimeTensorValue, String> {
    let in_shape = tensor.value.shape.clone();
    let in_rank = in_shape.len();
    if axis > in_rank {
        return Err(format!(
            "{builtin} axis {axis} out of bounds for rank-{in_rank} tensor (insert position must be <= rank)"
        ));
    }

    // INSERT: create a new axis of size `count` at position `axis`.
    let mut out_shape = Vec::with_capacity(in_rank + 1);
    out_shape.extend_from_slice(&in_shape[..axis]);
    out_shape.push(count);
    out_shape.extend_from_slice(&in_shape[axis..]);

    let out_numel = tensor_numel(&out_shape);
    let mut picks = Vec::with_capacity(out_numel);
    for out_linear in 0..out_numel {
        let out_indices = linear_to_indices(out_linear, &out_shape);
        // Drop the inserted axis to recover the input index.
        let mut in_indices = out_indices;
        in_indices.remove(axis);
        picks.push(indices_to_linear(&in_indices, &in_shape));
    }
    // reuse_* contract: expand is element-preserving (section C3).
    Ok(RuntimeTensorValue::new(IrTensorValue::from_storage(
        out_shape,
        tensor.value.storage().reuse_gather(&picks),
    )))
}

/// `expand`: broadcast an existing size-1 axis.
///
/// `spec/05-risc-primitives.md` section 2.4.1: "`expand` sets the extent at
/// `axis` and is well formed only when the operand's extent at `axis` is 1
/// (the size-1 broadcast of section 2.4's table); the operation is a claim
/// that the operand's extent at `axis` is 1. [...] A symbolic or runtime
/// operand extent at `axis` other than 1 fails that claim's runtime extent
/// guard and traps `Domain`."
///
/// The claim is checked here rather than assumed. The host runtime is the
/// only evaluator a `chelis eval` of a user program reaches, so a claim this
/// function did not check would execute unguarded on the lane a user is most
/// likely to run. The guard the compiled and DAG lanes place from the derived
/// equality classes compares the same values at the same trap kind; this one
/// fires at the operation because the host interpreter has no entry at which
/// to hoist it.
///
/// The trap renders through [`NumericTrap`], so the line is
/// `numeric trap: domain in expand at i64` verbatim: the guarded result is
/// an extent under [05-DIM-1] and not a tensor element, which is why the
/// dtype slot is `i64` rather than the tensor's precision
/// (`spec/04-type-system.md` section 4.7).
pub(super) fn tensor_expand_host(
    builtin: &str,
    tensor: &RuntimeTensorValue,
    axis: usize,
    count: usize,
) -> Result<RuntimeTensorValue, String> {
    let in_shape = tensor.value.shape.clone();
    let in_rank = in_shape.len();
    if axis >= in_rank {
        return Err(format!(
            "{builtin} axis {axis} out of bounds for rank-{in_rank} tensor (the broadcast \
             axis must be within the operand's rank)"
        ));
    }
    if in_shape[axis] != 1 {
        let trap = NumericTrap::Domain {
            op: "expand",
            prim: Prim::Int64,
        };
        let observed = in_shape[axis];
        return Err(format!(
            "{trap}\n  {builtin} claims the operand's extent at axis {axis} is 1, observed \
             {observed}"
        ));
    }

    // BROADCAST: set the size-1 axis to `count`, reading index 0 there.
    let mut out_shape = in_shape.clone();
    out_shape[axis] = count;

    let out_numel = tensor_numel(&out_shape);
    let mut picks = Vec::with_capacity(out_numel);
    for out_linear in 0..out_numel {
        let mut in_indices = linear_to_indices(out_linear, &out_shape);
        in_indices[axis] = 0;
        picks.push(indices_to_linear(&in_indices, &in_shape));
    }
    // reuse_* contract: expand is element-preserving (section C3).
    Ok(RuntimeTensorValue::new(IrTensorValue::from_storage(
        out_shape,
        tensor.value.storage().reuse_gather(&picks),
    )))
}

/// Strided windowed reduction host evaluator. Mirrors
/// `chelis_ir::eval::reduce_window` so host-runtime evaluation and
/// IR-evaluator runs produce byte-identical output for the four
/// `reduce_window_*` builtins. See `spec/05-risc-primitives.md`
/// §2.3.1 for the surface semantics.
pub(super) fn tensor_reduce_window_host(
    tensor: &RuntimeTensorValue,
    window_shape: &[usize],
    strides: &[usize],
    reducer: ReduceWindowOp,
    op_name: &str,
) -> Result<RuntimeTensorValue, String> {
    if window_shape.len() != strides.len() {
        return Err(format!(
            "{op_name} window_shape (len {}) and strides (len {}) must agree",
            window_shape.len(),
            strides.len()
        ));
    }
    let in_shape = &tensor.value.shape;
    let n = window_shape.len();
    if n == 0 {
        return Err(format!(
            "{op_name} requires a non-empty window_shape and strides"
        ));
    }
    if in_shape.len() < n {
        return Err(format!(
            "{op_name} window arity {n} exceeds tensor rank {}",
            in_shape.len()
        ));
    }
    let leading = in_shape.len() - n;
    let mut out_shape = in_shape[..leading].to_vec();
    for i in 0..n {
        let w = window_shape[i];
        let s = strides[i];
        if w == 0 {
            return Err(format!("{op_name} window_shape[{i}] must be >= 1"));
        }
        if s == 0 {
            return Err(format!("{op_name} strides[{i}] must be >= 1"));
        }
        let in_dim = in_shape[leading + i];
        if in_dim < w {
            return Err(format!(
                "{op_name} axis {} input dim {in_dim} < window_shape[{i}] = {w}",
                leading + i
            ));
        }
        out_shape.push((in_dim - w) / s + 1);
    }

    let out_numel = tensor_numel(&out_shape);
    let mut groups = Vec::with_capacity(out_numel);
    for out_flat in 0..out_numel {
        let out_indices = linear_to_indices(out_flat, &out_shape);
        let mut group = Vec::with_capacity(window_shape.iter().product());
        let mut window_pos = vec![0usize; n];
        loop {
            let mut src_indices = vec![0usize; in_shape.len()];
            src_indices[..leading].copy_from_slice(&out_indices[..leading]);
            for i in 0..n {
                src_indices[leading + i] = out_indices[leading + i] * strides[i] + window_pos[i];
            }
            let src_linear = indices_to_linear(&src_indices, in_shape);
            group.push(src_linear);
            let mut carry = n;
            for i in (0..n).rev() {
                window_pos[i] += 1;
                if window_pos[i] < window_shape[i] {
                    carry = i;
                    break;
                }
                window_pos[i] = 0;
            }
            if carry == n {
                break;
            }
        }
        groups.push(group);
    }
    let op = match reducer {
        ReduceWindowOp::Max => TensorReduceOp::ReduceWindowMax,
        ReduceWindowOp::Min => TensorReduceOp::ReduceWindowMin,
        ReduceWindowOp::Sum => TensorReduceOp::ReduceWindowSum,
        ReduceWindowOp::Mean => TensorReduceOp::ReduceWindowMean,
    };
    debug_assert_eq!(op.name(), op_name);
    let storage =
        reduce_tensor_groups(op, tensor.value.storage(), &groups).map_err(|err| err.to_string())?;
    Ok(RuntimeTensorValue::new(IrTensorValue::from_storage(
        out_shape, storage,
    )))
}

/// Pad each axis by `padding[i] = (lo_i, hi_i)`, filling the inserted
/// region with `fill`. Output dim i is `input_dim[i] + lo_i + hi_i`.
/// Mirrors the IR evaluator at `crates/chelis-ir/src/eval.rs::pad` so
/// eval-in-context output is byte-identical to a freshly-lowered DAG run.
///
/// Sibling sweep of issue Chelis-Lang/chelis#187 (pad had the same
/// 1-arg-`tensor_unop`-vs-parameterized-RISC-op antipattern as shrink
/// and stride).
pub(super) fn tensor_pad_host(
    tensor: &RuntimeTensorValue,
    padding: &[(usize, usize)],
    fill: f64,
) -> Result<RuntimeTensorValue, String> {
    let in_shape = &tensor.value.shape;
    if padding.len() != in_shape.len() {
        return Err(format!(
            "pad expects {} padding pairs for rank-{} tensor, got {}",
            in_shape.len(),
            in_shape.len(),
            padding.len()
        ));
    }
    let out_shape: Vec<usize> = padding
        .iter()
        .zip(in_shape.iter())
        .map(|((lo, hi), in_dim)| in_dim + lo + hi)
        .collect();
    let out_numel = tensor_numel(&out_shape);
    let mut map: Vec<Option<usize>> = vec![None; out_numel];
    let in_numel = tensor_numel(in_shape);
    for in_linear in 0..in_numel {
        let in_indices = linear_to_indices(in_linear, in_shape);
        let out_indices: Vec<usize> = in_indices
            .iter()
            .zip(padding.iter())
            .map(|(idx, (lo, _))| idx + lo)
            .collect();
        map[indices_to_linear(&out_indices, &out_shape)] = Some(in_linear);
    }
    // The fill ingress-finalizes at the buffer's dtype (loud on a fill
    // outside an integer dtype's domain).
    let fill = chelis_types::scalar_from_f64("pad", tensor.precision, fill)
        .map_err(|trap| trap.to_string())?;
    // reuse_* contract: pad moves existing elements and places a
    // finalized fill (section C3, element-preserving).
    Ok(RuntimeTensorValue::new(IrTensorValue::from_storage(
        out_shape,
        tensor.value.storage().reuse_fill_gather(&fill, &map),
    )))
}

/// Sub-tensor slice along every axis. For each axis the `bounds[i] =
/// (start_i, end_i)` carve out the half-open range `[start_i, end_i)`,
/// producing an output of dim `end_i - start_i`. The arithmetic mirrors
/// the IR-level evaluator at `crates/chelis-ir/src/eval.rs::shrink` so
/// eval-in-context output is byte-identical to a freshly-lowered DAG run.
///
/// Validates bounds at host-runtime so a malformed `shrink` call surfaces
/// as a loud `eval` error rather than silently returning garbage data --
/// the surface-level counterpart to `c10_shrink_invalid_bounds_is_error`
/// in `crates/chelis-ir/src/verify.rs`.
pub(super) fn tensor_shrink_host(
    tensor: &RuntimeTensorValue,
    bounds: &[(usize, usize)],
) -> Result<RuntimeTensorValue, String> {
    let in_shape = &tensor.value.shape;
    if bounds.len() != in_shape.len() {
        return Err(format!(
            "shrink expects {} bounds pairs for rank-{} tensor, got {}",
            in_shape.len(),
            in_shape.len(),
            bounds.len()
        ));
    }
    let mut out_shape = Vec::with_capacity(in_shape.len());
    for (axis, ((start, end), in_dim)) in bounds.iter().zip(in_shape.iter()).enumerate() {
        if start >= end {
            return Err(format!(
                "shrink axis {axis} bound [{start}, {end}] is empty or inverted (start >= end)"
            ));
        }
        if *end > *in_dim {
            return Err(format!(
                "shrink axis {axis} bound [{start}, {end}] is out of range for input dim {in_dim}"
            ));
        }
        out_shape.push(end - start);
    }
    let out_numel = tensor_numel(&out_shape);
    let mut picks = Vec::with_capacity(out_numel);
    for out_linear in 0..out_numel {
        let out_indices = linear_to_indices(out_linear, &out_shape);
        let in_indices: Vec<usize> = out_indices
            .iter()
            .zip(bounds.iter())
            .map(|(idx, (start, _))| idx + start)
            .collect();
        picks.push(indices_to_linear(&in_indices, in_shape));
    }
    // reuse_* contract: shrink is element-preserving (section C3).
    Ok(RuntimeTensorValue::new(IrTensorValue::from_storage(
        out_shape,
        tensor.value.storage().reuse_gather(&picks),
    )))
}

/// Strided view -- take every `strides[i]`-th element along axis i. Output
/// dim i is `ceil(input_dim[i] / strides[i])`. Zero strides are rejected
/// upstream (the eval_builtin arm validates positivity) but checked again
/// here to keep the function self-contained and to match the IR-level
/// `c10_stride_zero_step_is_error` invariant.
pub(super) fn tensor_stride_host(
    tensor: &RuntimeTensorValue,
    strides: &[usize],
) -> Result<RuntimeTensorValue, String> {
    let in_shape = &tensor.value.shape;
    if strides.len() != in_shape.len() {
        return Err(format!(
            "stride expects {} strides for rank-{} tensor, got {}",
            in_shape.len(),
            in_shape.len(),
            strides.len()
        ));
    }
    let mut out_shape = Vec::with_capacity(in_shape.len());
    for (axis, (step, in_dim)) in strides.iter().zip(in_shape.iter()).enumerate() {
        if *step == 0 {
            return Err(format!(
                "stride axis {axis} step 0 is not allowed (must be positive)"
            ));
        }
        out_shape.push(in_dim.div_ceil(*step));
    }
    let out_numel = tensor_numel(&out_shape);
    let mut picks = Vec::with_capacity(out_numel);
    for out_linear in 0..out_numel {
        let out_indices = linear_to_indices(out_linear, &out_shape);
        let in_indices: Vec<usize> = out_indices
            .iter()
            .zip(strides.iter())
            .map(|(idx, step)| idx * step.max(&1))
            .collect();
        picks.push(indices_to_linear(&in_indices, in_shape));
    }
    // reuse_* contract: stride is element-preserving (section C3).
    Ok(RuntimeTensorValue::new(IrTensorValue::from_storage(
        out_shape,
        tensor.value.storage().reuse_gather(&picks),
    )))
}

/// Convert a `RuntimeValue::List` of inner `List`s into a flat
/// `Vec<(usize, usize)>` of `[start, end]` bounds pairs. Each inner list
/// must have exactly two non-negative int entries (matching the
/// type-checker's `List[List[Int64]]` contract). Any other shape -- wrong
/// inner-list length, non-int entries, negative endpoints -- surfaces as
/// a loud host-runtime error.
pub(super) fn extract_bounds_pair_list(
    raw: &[RuntimeValue],
    op: &str,
) -> Result<Vec<(usize, usize)>, String> {
    raw.iter()
        .enumerate()
        .map(|(axis, item)| match item {
            RuntimeValue::List(pair) => {
                if pair.len() != 2 {
                    return Err(format!(
                        "{op} axis {axis} expects a [start, end] pair, got {} entries",
                        pair.len()
                    ));
                }
                let start = match &pair[0] {
                    RuntimeValue::Scalar(payload) if payload.dtype().is_integer() => {
                        payload.as_i64()
                    }
                    other => {
                        return Err(format!("{op} axis {axis} expects int start, got {other:?}"));
                    }
                };
                let end = match &pair[1] {
                    RuntimeValue::Scalar(payload) if payload.dtype().is_integer() => {
                        payload.as_i64()
                    }
                    other => {
                        return Err(format!("{op} axis {axis} expects int end, got {other:?}"));
                    }
                };
                if start < 0 || end < 0 {
                    return Err(format!(
                        "{op} axis {axis} bound [{start}, {end}] has negative endpoint"
                    ));
                }
                Ok((start as usize, end as usize))
            }
            other => Err(format!(
                "{op} axis {axis} expects a [start, end] pair list, got {other:?}"
            )),
        })
        .collect()
}

/// Numerically stable softmax along a single axis:
/// `softmax(x, axis)[i] = exp(x[i] - max(x, axis)) / sum_j exp(x[j] - max(x, axis))`.
/// Matches the spec §4.2 lowering used by `tier2::lower_softmax`.
///
/// Negative axes are normalized to `rank + axis` (e.g. `-1` is the last axis).
pub(super) fn tensor_softmax_host(
    tensor: &RuntimeTensorValue,
    axis: i64,
) -> Result<RuntimeTensorValue, String> {
    let rank = tensor.value.shape.len();
    if rank == 0 {
        return Err("softmax requires a tensor of rank >= 1".to_string());
    }
    let axis_usize = if axis < 0 {
        let neg = (-axis) as usize;
        if neg > rank {
            return Err(format!("softmax axis {axis} out of bounds for rank {rank}"));
        }
        rank - neg
    } else {
        let a = axis as usize;
        if a >= rank {
            return Err(format!("softmax axis {axis} out of bounds for rank {rank}"));
        }
        a
    };

    let in_shape = tensor.value.shape.clone();
    let axis_size = in_shape[axis_usize];
    if axis_size == 0 {
        return Err("softmax axis has size 0".to_string());
    }
    let numel = tensor_numel(&in_shape);
    let mut out = vec![0.0_f64; numel];
    let wide_in = tensor.value.to_f64_lossy_vec();

    // Iterate over each "slice" along the reduced axis: for every combination
    // of the other axes, compute max -> exp(x - max) -> sum -> divide.
    let mut reduced_shape = in_shape.clone();
    reduced_shape[axis_usize] = 1;
    let reduced_numel = tensor_numel(&reduced_shape);

    for slice_linear in 0..reduced_numel {
        let mut base_indices = linear_to_indices(slice_linear, &reduced_shape);
        // First pass: max over the axis.
        //
        // #173: a slice that contains `+Inf` (then `exp(+Inf - +Inf) =
        // exp(NaN) = NaN`) or that is all `-Inf` (then `exp(-Inf - -Inf) =
        // exp(NaN) = NaN`) must yield NaN, exactly as torch's
        // `torch.softmax` does (verified against torch 2.x CPU: every
        // `+Inf`-containing or all-`-Inf` slice returns NaN). The C
        // backend, the IR evaluator, and the spec'd lowering
        // (`tier2::lower_softmax`: max/sub/exp/sum/div) already produce
        // NaN via the standard formula; the host runtime previously
        // special-cased these to "natural limits" (uniform `1/N` for
        // all-`-Inf`, `1/K` one-hot for `+Inf`), silently diverging from
        // torch and from every other Chelis lane. The special-cases are
        // removed so the standard formula runs and NaN propagates. The
        // ONLY non-finite case the standard formula handles cleanly is a
        // mixed slice with `-Inf` but no `+Inf` (the finite max makes
        // `exp(-Inf - max) = 0`); that path is preserved below.
        let mut max_val = f64::NEG_INFINITY;
        for k in 0..axis_size {
            base_indices[axis_usize] = k;
            let in_linear = indices_to_linear(&base_indices, &in_shape);
            let v = wide_in[in_linear];
            if v.is_nan() {
                // NaN propagates: write NaN across the whole slice and
                // continue. This matches IEEE behavior of every other
                // numerical library (PyTorch / NumPy / JAX).
                for kk in 0..axis_size {
                    base_indices[axis_usize] = kk;
                    let l = indices_to_linear(&base_indices, &in_shape);
                    out[l] = f64::NAN;
                }
                // Restart the outer slice loop's bookkeeping cleanly.
                max_val = f64::NAN;
                break;
            }
            if v > max_val {
                max_val = v;
            }
        }
        if max_val.is_nan() {
            // NaN propagation handled above; nothing else to do for this slice.
            continue;
        }
        // Second pass: sum of exp(x - max). When the slice contains a
        // `+Inf` the max is `+Inf` and `exp(+Inf - +Inf) = exp(NaN) =
        // NaN`; when the slice is all `-Inf` the max is `-Inf` and
        // `exp(-Inf - -Inf) = exp(NaN) = NaN`. The NaN flows through the
        // sum and the normalize below, so every output element of that
        // slice is NaN — matching torch (#173).
        //
        // #170 (DO NOT "fix" this sum into the stride-4 cascade): the f64
        // accumulator here is intentional and is NOT a torch-parity gap.
        // (a) This host-eval softmax computes exp/sum/div in f64, whereas the
        //     lowered path (`tier2::lower_softmax` -> `RiscOp::Sum`) uses f32
        //     `expf` + the #163 f32 cascade. The two lanes are NOT guaranteed
        //     bit-identical: the f64 `exp` is more accurate than f32 `expf`
        //     (cf. #172), so per-element exponentials can differ before the
        //     sum even runs. What the repo actually proves is agreement within
        //     the 1e-6 relative parity tolerance the corpus oracle enforces
        //     (`chelis-cli/tests/parity.rs`) — not bit-identity. Swapping the
        //     f64 fold for the f32 cascade would not buy bit-identity (the
        //     exp mismatch remains) and would only lower the host lane's
        //     precision.
        // (b) torch's softmax is a FUSED kernel; neither the cascade nor an
        //     f64 fold reliably bit-matches it. So softmax is DOCUMENTED, not
        //     cascaded; only `sum`/`trace` take the cascade.
        let mut sum_exp = 0.0_f64;
        for k in 0..axis_size {
            base_indices[axis_usize] = k;
            let in_linear = indices_to_linear(&base_indices, &in_shape);
            sum_exp += (wide_in[in_linear] - max_val).exp();
        }
        if sum_exp == 0.0 {
            return Err("softmax sum-of-exp is zero (numerical underflow)".to_string());
        }
        // Third pass: write exp(x - max) / sum.
        for k in 0..axis_size {
            base_indices[axis_usize] = k;
            let in_linear = indices_to_linear(&base_indices, &in_shape);
            let numer = (wide_in[in_linear] - max_val).exp();
            out[in_linear] = numer / sum_exp;
        }
    }

    RuntimeTensorValue::from_wide("softmax", tensor.precision, in_shape, out)
}

// ---------------------------------------------------------------------------
// Composed Tier-2 host-runtime delegation
//
// `mean` / `layer_norm` / `conv` are not single RISC ops; they
// decompose into combinations of `RiscOp::Sum`, `RiscOp::Div`,
// `RiscOp::Sqrt`, `RiscOp::Mul`, `RiscOp::Pad`, etc. The canonical
// decomposition lives in `crates/chelis-ir/src/tier2.rs::lower_*`. To
// avoid drift between the host runtime and the C-backend / IR
// evaluator, we build a small ad-hoc DAG using the same tier2 lowering
// helper and forward-eval it through `chelis_ir::eval::eval_tensor_*`.
// This makes the host runtime byte-identical (to documented float
// tolerance) with what the IR evaluator would produce, per
// `feedback_evaluator_byte_identical_gate`.
// ---------------------------------------------------------------------------

const COMPOSED_PLACEHOLDER_PREFIX: &str = "__issue185_composed_arg_";

fn tensor_type_for(tensor: &RuntimeTensorValue) -> TensorType {
    TensorType {
        dims: tensor
            .value
            .shape
            .iter()
            .map(|&size| DimInfo::Lit(size))
            .collect(),
        precision: tensor.precision,
    }
}

fn add_load(dag: &mut Dag, name: String, ty: TensorType) -> NodeId {
    dag.add_node(RiscOp::Load { name: name.into() }, Vec::new(), ty, None)
}

fn extract_root(
    dag: &Dag,
    inputs: &UnordMap<String, IrTensorValue>,
    root: NodeId,
    op_label: &str,
) -> Result<RuntimeTensorValue, String> {
    let values =
        eval_tensor_roots_with(dag, &[root], |name| inputs.get(name).cloned()).map_err(|err| {
            if err.starts_with(chelis_types::NUMERIC_TRAP_PREFIX) {
                err
            } else {
                format!("{op_label}: IR eval failed: {err}")
            }
        })?;
    let tensor_value = values
        .get(&root)
        .cloned()
        .ok_or_else(|| format!("{op_label}: IR eval produced no value for root node"))?;
    let precision = dag
        .get(root)
        .map(|node| node.output_type.precision)
        .ok_or_else(|| format!("{op_label}: root node missing from DAG"))?;
    debug_assert_eq!(
        tensor_value.prim(),
        precision,
        "{op_label}: the DAG evaluator finalizes at the root's declared dtype"
    );
    Ok(RuntimeTensorValue::new(tensor_value))
}

/// Build a small DAG whose only input is `x`, attach the supplied
/// tier2 lowering helper, and forward-eval the resulting root through
/// the IR evaluator. Used by host-runtime arms that delegate a Tier-2
/// composed op to its canonical RISC decomposition.
pub(super) fn eval_composed_unary<F>(
    x: &RuntimeTensorValue,
    build: F,
) -> Result<RuntimeTensorValue, String>
where
    F: FnOnce(&mut Dag, NodeId, &TensorType) -> NodeId,
{
    let mut dag = Dag::new();
    let ty = tensor_type_for(x);
    let x_name = format!("{COMPOSED_PLACEHOLDER_PREFIX}0");
    let x_id = add_load(&mut dag, x_name.clone(), ty.clone());
    let root = build(&mut dag, x_id, &ty);
    let mut inputs = UnordMap::new();
    inputs.insert(x_name, x.value.clone());
    extract_root(&dag, &inputs, root, "composed unary tier2")
}

/// Build a small DAG with three tensor inputs `(x, gamma, beta)`,
/// attach the supplied tier2 lowering helper, and forward-eval the
/// resulting root. Used for `layer_norm`.
pub(super) fn eval_composed_triop<F>(
    x: &RuntimeTensorValue,
    gamma: &RuntimeTensorValue,
    beta: &RuntimeTensorValue,
    build: F,
) -> Result<RuntimeTensorValue, String>
where
    F: FnOnce(&mut Dag, NodeId, NodeId, NodeId, (&TensorType, &TensorType, &TensorType)) -> NodeId,
{
    let mut dag = Dag::new();
    let x_ty = tensor_type_for(x);
    let g_ty = tensor_type_for(gamma);
    let b_ty = tensor_type_for(beta);
    let x_name = format!("{COMPOSED_PLACEHOLDER_PREFIX}0");
    let g_name = format!("{COMPOSED_PLACEHOLDER_PREFIX}1");
    let b_name = format!("{COMPOSED_PLACEHOLDER_PREFIX}2");
    let x_id = add_load(&mut dag, x_name.clone(), x_ty.clone());
    let g_id = add_load(&mut dag, g_name.clone(), g_ty.clone());
    let b_id = add_load(&mut dag, b_name.clone(), b_ty.clone());
    let root = build(&mut dag, x_id, g_id, b_id, (&x_ty, &g_ty, &b_ty));
    let mut inputs = UnordMap::new();
    inputs.insert(x_name, x.value.clone());
    inputs.insert(g_name, gamma.value.clone());
    inputs.insert(b_name, beta.value.clone());
    extract_root(&dag, &inputs, root, "composed triop tier2")
}

/// Execute the same N-dimensional contraction graph as the compiled lane.
pub(super) fn conv_host(
    input: &RuntimeTensorValue,
    kernel: &RuntimeTensorValue,
    strides: &[usize],
    padding: &[(usize, usize)],
) -> Result<RuntimeTensorValue, String> {
    let shape = &input.value.shape;
    let kernel_shape = &kernel.value.shape;
    if shape.len() < 3 || kernel_shape.len() != shape.len() {
        return Err("conv requires equal input/kernel ranks of at least 3".to_string());
    }
    let rank = shape.len() - 2;
    if strides.len() != rank || padding.len() != rank {
        return Err(format!(
            "conv requires exactly {rank} stride and padding entries"
        ));
    }
    if shape[1] != kernel_shape[1]
        || input.precision != kernel.precision
        || !input.precision.is_float()
    {
        return Err("conv requires matching channels and one active float dtype".to_string());
    }
    let mut output_shape = vec![shape[0], kernel_shape[0]];
    for axis in 0..rank {
        let padded = shape[axis + 2]
            .checked_add(padding[axis].0)
            .and_then(|n| n.checked_add(padding[axis].1))
            .filter(|&n| i64::try_from(n).is_ok())
            .ok_or("conv padded extent overflows i64")?;
        let k = kernel_shape[axis + 2];
        if strides[axis] == 0 || k == 0 || k > padded {
            return Err(format!(
                "conv invalid kernel/stride/padding at spatial axis {axis}"
            ));
        }
        output_shape.push(
            ((padded - k) / strides[axis])
                .checked_add(1)
                .filter(|&n| i64::try_from(n).is_ok())
                .ok_or("conv output extent overflows i64")?,
        );
    }
    let output_ty = TensorType {
        dims: output_shape.into_iter().map(DimInfo::Lit).collect(),
        precision: input.precision,
    };
    let input_ty = tensor_type_for(input);
    let kernel_ty = tensor_type_for(kernel);
    let mut dag = Dag::new();
    let x_name = format!("{COMPOSED_PLACEHOLDER_PREFIX}0");
    let k_name = format!("{COMPOSED_PLACEHOLDER_PREFIX}1");
    let x_id = add_load(&mut dag, x_name.clone(), input_ty.clone());
    let k_id = add_load(&mut dag, k_name.clone(), kernel_ty.clone());
    let root = tier2::lower_conv(
        &mut dag, x_id, k_id, &input_ty, &kernel_ty, &output_ty, strides, padding, None,
    );
    let mut inputs = UnordMap::new();
    inputs.insert(x_name, input.value.clone());
    inputs.insert(k_name, kernel.value.clone());
    extract_root(&dag, &inputs, root, "conv")
}

pub(super) fn tensor_concat_value(
    parts: &[RuntimeValue],
    axis: i64,
) -> Result<RuntimeValue, String> {
    let tensors = parts
        .iter()
        .map(|value| match value {
            RuntimeValue::Tensor(tensor) => Ok(tensor.clone()),
            other => Err(format!("concat expects tensor parts, got {other:?}")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let first = tensors
        .first()
        .ok_or_else(|| "concat expects at least one tensor part".to_string())?;
    // chelis#368/#522: accept a negative concat axis (`-1` = last axis),
    // matching the negative-axis convention every other axis-taking op follows
    // (reductions, softmax) AND the IR `concat` lowering (`lower_tensor_concat`
    // normalizes `raw_axis < 0`). Before #368, the host forward path rejected
    // `concat(..., -1)` while the grad and C-build lanes — which lower through
    // `lower_tensor_concat` — accepted it, an eval-forward-vs-IR divergence.
    // `normalize_axis` now applies the from-the-end offset itself (#522), so
    // the prior inline `axis + rank` pre-pass is removed: doing it twice would
    // wrongly accept a doubly-out-of-range negative (e.g. `-4` on rank 3 ->
    // `-1` -> `2`). One offset, then the shared bounds check.
    let rank = first.value.shape.len();
    let axis = normalize_axis(rank, axis, "concat")?;
    for tensor in &tensors[1..] {
        if tensor.precision != first.precision {
            return Err("concat expects matching tensor precision".to_string());
        }
        if tensor.value.shape.len() != first.value.shape.len() {
            return Err("concat expects matching tensor rank".to_string());
        }
        for dim in 0..tensor.value.shape.len() {
            if dim != axis && tensor.value.shape[dim] != first.value.shape[dim] {
                return Err(format!(
                    "concat expects matching non-concatenated axes; axis {dim} differed"
                ));
            }
        }
    }
    let mut out_shape = first.value.shape.clone();
    out_shape[axis] = tensors.iter().map(|tensor| tensor.value.shape[axis]).sum();
    // reuse_* contract: concat moves existing elements only (section C3,
    // element-preserving). Seed a zero-filled buffer at the shared dtype,
    // then overwrite every slot from its owning part.
    let out_numel = tensor_numel(&out_shape);
    let zero = chelis_types::scalar_from_i64("concat", first.precision, 0)
        .map_err(|trap| trap.to_string())?;
    let seed = first
        .value
        .storage()
        .reuse_fill_gather(&zero, &vec![None; out_numel]);
    let mut out = IrTensorValue::from_storage(out_shape.clone(), seed);
    let mut axis_offset = 0usize;
    for tensor in &tensors {
        let mut writes = Vec::with_capacity(tensor.value.len());
        for linear in 0..tensor.value.len() {
            let mut index = linear_to_indices(linear, &tensor.value.shape);
            index[axis] += axis_offset;
            writes.push((indices_to_linear(&index, &out_shape), linear));
        }
        out = IrTensorValue::from_storage(
            out_shape.clone(),
            out.storage()
                .reuse_overwrite(tensor.value.storage(), writes),
        );
        axis_offset += tensor.value.shape[axis];
    }
    Ok(RuntimeValue::Tensor(RuntimeTensorValue::new(out)))
}

pub(super) fn tensor_reshape_value(
    tensor: &RuntimeTensorValue,
    shape: &[RuntimeValue],
) -> Result<RuntimeTensorValue, String> {
    let new_shape = expect_int_list(shape, "reshape")?;
    let expected = new_shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| "reshape target shape overflows usize".to_string())?;
    if expected != tensor.value.len() {
        return Err(format!(
            "reshape expects {} elements but tensor has {}",
            expected,
            tensor.value.len()
        ));
    }
    // reuse_* contract: reshape is element-preserving (section C3); the
    // buffer moves unchanged under a new shape.
    Ok(RuntimeTensorValue::new(IrTensorValue::from_storage(
        new_shape,
        tensor.value.storage().clone(),
    )))
}

pub(super) fn tensor_split_value(
    tensor: &RuntimeTensorValue,
    axis: i64,
    sizes: &[RuntimeValue],
) -> Result<RuntimeValue, String> {
    let axis = normalize_axis(tensor.value.shape.len(), axis, "split")?;
    let sizes = expect_int_list(sizes, "split")?;
    let total: usize = sizes.iter().sum();
    if total != tensor.value.shape[axis] {
        return Err(format!(
            "split sizes sum to {total}, expected {}",
            tensor.value.shape[axis]
        ));
    }
    let mut parts = Vec::with_capacity(sizes.len());
    let mut offset = 0usize;
    for size in sizes {
        let mut shape = tensor.value.shape.clone();
        shape[axis] = size;
        let numel = tensor_numel(&shape);
        let mut picks = Vec::with_capacity(numel);
        for linear in 0..numel {
            let mut index = linear_to_indices(linear, &shape);
            index[axis] += offset;
            picks.push(indices_to_linear(&index, &tensor.value.shape));
        }
        offset += size;
        // reuse_* contract: split is element-preserving (section C3).
        parts.push(RuntimeValue::Tensor(RuntimeTensorValue::new(
            IrTensorValue::from_storage(shape, tensor.value.storage().reuse_gather(&picks)),
        )));
    }
    Ok(RuntimeValue::List(parts))
}

pub(super) fn tensor_gather_value(
    tensor: &RuntimeTensorValue,
    indices: &RuntimeTensorValue,
    axis: i64,
) -> Result<RuntimeTensorValue, String> {
    let axis = normalize_axis(tensor.value.shape.len(), axis, "gather")?;
    if !indices.precision.is_integer() {
        return Err("gather expects integer tensor indices".to_string());
    }
    let mut out_shape = tensor.value.shape[..axis].to_vec();
    out_shape.extend_from_slice(&indices.value.shape);
    out_shape.extend_from_slice(&tensor.value.shape[axis + 1..]);
    let index_values = indices
        .value
        .storage()
        .to_i64_exact_vec()
        .expect("integer tensor storage reads exactly");
    let out_numel = tensor_numel(&out_shape);
    let mut picks = Vec::with_capacity(out_numel);
    for linear in 0..out_numel {
        let out_index = linear_to_indices(linear, &out_shape);
        let mut src_index = Vec::with_capacity(tensor.value.shape.len());
        src_index.extend_from_slice(&out_index[..axis]);
        let gathered_idx = &out_index[axis..axis + indices.value.shape.len()];
        let index_linear = indices_to_linear(gathered_idx, &indices.value.shape);
        let value = index_values[index_linear];
        if value < 0 || value as usize >= tensor.value.shape[axis] {
            return Err(format!("gather index {value} out of bounds at axis {axis}"));
        }
        src_index.push(value as usize);
        src_index.extend_from_slice(&out_index[axis + indices.value.shape.len()..]);
        picks.push(indices_to_linear(&src_index, &tensor.value.shape));
    }
    // reuse_* contract: gather is element-preserving (section C3).
    Ok(RuntimeTensorValue::new(IrTensorValue::from_storage(
        out_shape,
        tensor.value.storage().reuse_gather(&picks),
    )))
}

pub(super) fn tensor_scatter_value(
    base: &RuntimeTensorValue,
    indices: &RuntimeTensorValue,
    updates: &RuntimeTensorValue,
    axis: i64,
    mode: &str,
) -> Result<RuntimeTensorValue, String> {
    let axis = normalize_axis(base.value.shape.len(), axis, "scatter")?;
    if !indices.precision.is_integer() {
        return Err("scatter expects integer tensor indices".to_string());
    }
    let expected = tensor_gather_value(base, indices, axis as i64)?;
    if expected.value.shape != updates.value.shape || expected.precision != updates.precision {
        return Err("scatter updates must match gathered tensor shape and precision".to_string());
    }
    let index_values = indices
        .value
        .storage()
        .to_i64_exact_vec()
        .expect("integer tensor storage reads exactly");
    let mut writes = Vec::with_capacity(updates.value.len());
    for linear in 0..updates.value.len() {
        let update_index = linear_to_indices(linear, &updates.value.shape);
        let mut out_index = Vec::with_capacity(base.value.shape.len());
        out_index.extend_from_slice(&update_index[..axis]);
        let gathered_idx = &update_index[axis..axis + indices.value.shape.len()];
        let index_linear = indices_to_linear(gathered_idx, &indices.value.shape);
        let value = index_values[index_linear];
        if value < 0 || value as usize >= base.value.shape[axis] {
            return Err(format!(
                "scatter index {value} out of bounds at axis {axis}"
            ));
        }
        out_index.push(value as usize);
        out_index.extend_from_slice(&update_index[axis + indices.value.shape.len()..]);
        let out_linear = indices_to_linear(&out_index, &base.value.shape);
        writes.push((out_linear, linear));
    }
    match mode {
        // reuse_* contract: replace-scatter moves existing elements only
        // (section C3, element-preserving).
        "replace" => Ok(RuntimeTensorValue::new(IrTensorValue::from_storage(
            base.value.shape.clone(),
            base.value
                .storage()
                .reuse_overwrite(updates.value.storage(), writes),
        ))),
        "add" => {
            let mut out = base.value.to_f64_lossy_vec();
            let upd = updates.value.to_f64_lossy_vec();
            for (out_linear, linear) in writes {
                out[out_linear] += upd[linear];
            }
            RuntimeTensorValue::from_wide(
                "scatter_add",
                base.precision,
                base.value.shape.clone(),
                out,
            )
        }
        other => Err(format!("scatter mode must be replace or add, got {other}")),
    }
}

/// Element-wise replace-scatter with ONNX `ScatterElements` semantics
/// (`spec/05-risc-primitives.md` §3.5.1). `data`, `indices`, and
/// `updates` share a rank; `indices.shape == updates.shape`;
/// `output.shape == data.shape`. Each flat update coordinate `c` writes
/// `updates[c]` to `output[c with c[axis] := indices[c]]`. Duplicate
/// writes resolve last-write-wins in updates row-major flat order,
/// matching the DAG evaluator and the C backend.
pub(super) fn tensor_scatter_elements_value(
    data: &RuntimeTensorValue,
    indices: &RuntimeTensorValue,
    updates: &RuntimeTensorValue,
    axis: i64,
) -> Result<RuntimeTensorValue, String> {
    let axis = normalize_axis(data.value.shape.len(), axis, "scatter_elements")?;
    if !indices.precision.is_integer() {
        return Err("scatter_elements expects integer tensor indices".to_string());
    }
    if indices.value.shape != updates.value.shape {
        return Err("scatter_elements requires indices.shape == updates.shape".to_string());
    }
    if indices.value.shape.len() != data.value.shape.len() {
        return Err(
            "scatter_elements requires data, indices, and updates to share a rank".to_string(),
        );
    }
    let index_values = indices
        .value
        .storage()
        .to_i64_exact_vec()
        .expect("integer tensor storage reads exactly");
    let mut writes = Vec::with_capacity(updates.value.len());
    for (linear, &value) in index_values.iter().enumerate() {
        let coord = linear_to_indices(linear, &updates.value.shape);
        if value < 0 || value as usize >= data.value.shape[axis] {
            return Err(format!(
                "scatter_elements index {value} out of bounds at axis {axis}"
            ));
        }
        let mut out_index = coord.clone();
        out_index[axis] = value as usize;
        // Last-write-wins: deterministic-order overwrite.
        writes.push((indices_to_linear(&out_index, &data.value.shape), linear));
    }
    // reuse_* contract: element-preserving overwrite (section C3).
    Ok(RuntimeTensorValue::new(IrTensorValue::from_storage(
        data.value.shape.clone(),
        data.value
            .storage()
            .reuse_overwrite(updates.value.storage(), writes),
    )))
}

pub(super) fn tensor_where_value(
    cond: &RuntimeTensorValue,
    then_tensor: &RuntimeTensorValue,
    else_tensor: &RuntimeTensorValue,
) -> Result<RuntimeTensorValue, String> {
    if cond.precision != Prim::Bool {
        return Err("where expects bool tensor condition".to_string());
    }
    if cond.value.shape != then_tensor.value.shape
        || then_tensor.value.shape != else_tensor.value.shape
    {
        return Err(
            "where expects condition and both branches to have identical shape".to_string(),
        );
    }
    if then_tensor.precision != else_tensor.precision {
        return Err("where expects matching branch precision".to_string());
    }
    // reuse_* contract: `where` selects existing elements from the two
    // branches (section C3, element-preserving). Start from the then
    // branch and overwrite else-selected slots.
    let cond_mask = cond
        .value
        .storage()
        .to_i64_exact_vec()
        .expect("bool tensor storage reads exactly");
    let writes = cond_mask
        .iter()
        .enumerate()
        .filter(|(_, flag)| **flag == 0)
        .map(|(linear, _)| (linear, linear));
    Ok(RuntimeTensorValue::new(IrTensorValue::from_storage(
        then_tensor.value.shape.clone(),
        then_tensor
            .value
            .storage()
            .reuse_overwrite(else_tensor.value.storage(), writes),
    )))
}

pub(super) fn tensor_cumsum_value(
    tensor: &RuntimeTensorValue,
    axis: i64,
) -> Result<RuntimeTensorValue, String> {
    let axis = normalize_axis(tensor.value.shape.len(), axis, "cumsum")?;
    let mut data = tensor.value.to_f64_lossy_vec();
    // An empty operand has nothing to scan, and its axis decomposition is
    // never read. `outer` is the product of the extents BEFORE the axis, which
    // for an empty tensor are unconstrained: the zero elsewhere is what makes
    // the element count representable. Computing it anyway overflows `usize`
    // or spins an empty loop, matching the C runtime's guard in
    // `chelis_tensor_cumsum` / `chelis_tensor_sort`.
    if tensor.value.is_empty() {
        return RuntimeTensorValue::from_wide(
            "cumsum",
            tensor.precision,
            tensor.value.shape.clone(),
            data,
        );
    }
    let axis_size = tensor.value.shape[axis];
    let inner: usize = tensor.value.shape[axis + 1..]
        .iter()
        .product::<usize>()
        .max(1);
    let outer: usize = tensor.value.shape[..axis].iter().product::<usize>().max(1);
    // #170: cumsum is an inherently sequential prefix scan, NOT a reducible
    // tree — the stride-4 cascade does not apply. The f64 running accumulator
    // already matches `torch.cumsum` (verified: torch's cumsum is a
    // step-by-step prefix whose f32 result equals this f64 prefix rounded to
    // f32). No change needed; left as-is.
    for outer_idx in 0..outer {
        for inner_idx in 0..inner {
            let mut running = 0.0;
            for axis_idx in 0..axis_size {
                let linear = (outer_idx * axis_size + axis_idx) * inner + inner_idx;
                running += data[linear];
                data[linear] = running;
            }
        }
    }
    RuntimeTensorValue::from_wide("cumsum", tensor.precision, tensor.value.shape.clone(), data)
}

pub(super) fn tensor_sort_value(
    tensor: &RuntimeTensorValue,
    axis: i64,
) -> Result<RuntimeValue, String> {
    let axis = normalize_axis(tensor.value.shape.len(), axis, "sort")?;
    // An empty operand has nothing to scan, and its axis decomposition is
    // never read. `outer` is the product of the extents BEFORE the axis, which
    // for an empty tensor are unconstrained: the zero elsewhere is what makes
    // the element count representable. Computing it anyway overflows `usize`
    // or spins an empty loop, matching the C runtime's guard in
    // `chelis_tensor_cumsum` / `chelis_tensor_sort`.
    if tensor.value.is_empty() {
        return Ok(RuntimeValue::Tuple(vec![
            RuntimeValue::Tensor(RuntimeTensorValue::new(IrTensorValue::from_storage(
                tensor.value.shape.clone(),
                tensor.value.storage().reuse_gather(&[]),
            ))),
            RuntimeValue::Tensor(RuntimeTensorValue::from_wide_int(
                "sort",
                Prim::Int64,
                tensor.value.shape.clone(),
                Vec::new(),
            )?),
        ]));
    }
    let axis_size = tensor.value.shape[axis];
    let inner: usize = tensor.value.shape[axis + 1..]
        .iter()
        .product::<usize>()
        .max(1);
    let outer: usize = tensor.value.shape[..axis].iter().product::<usize>().max(1);
    let wide = tensor.value.to_f64_lossy_vec();
    let mut picks = vec![0usize; tensor.value.len()];
    let mut indices = vec![0i64; tensor.value.len()];
    for outer_idx in 0..outer {
        for inner_idx in 0..inner {
            let mut items = (0..axis_size)
                .map(|axis_idx| {
                    let linear = (outer_idx * axis_size + axis_idx) * inner + inner_idx;
                    (axis_idx, linear, wide[linear])
                })
                .collect::<Vec<_>>();
            items.sort_by(|(lhs_idx, _, lhs_val), (rhs_idx, _, rhs_val)| {
                lhs_val
                    .partial_cmp(rhs_val)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(lhs_idx.cmp(rhs_idx))
            });
            for (sorted_idx, (original_idx, original_linear, _)) in items.into_iter().enumerate() {
                let linear = (outer_idx * axis_size + sorted_idx) * inner + inner_idx;
                picks[linear] = original_linear;
                indices[linear] = original_idx as i64;
            }
        }
    }
    // reuse_* contract: the sorted values are a permutation of the input
    // (section C3, element-preserving); the index tensor is exact i64.
    Ok(RuntimeValue::Tuple(vec![
        RuntimeValue::Tensor(RuntimeTensorValue::new(IrTensorValue::from_storage(
            tensor.value.shape.clone(),
            tensor.value.storage().reuse_gather(&picks),
        ))),
        RuntimeValue::Tensor(RuntimeTensorValue::from_wide_int(
            "sort",
            Prim::Int64,
            tensor.value.shape.clone(),
            indices,
        )?),
    ]))
}

pub(super) fn tensor_diagonal_value(
    tensor: &RuntimeTensorValue,
    axis1: i64,
    axis2: i64,
) -> Result<RuntimeTensorValue, String> {
    let axis1 = normalize_axis(tensor.value.shape.len(), axis1, "diagonal")?;
    let axis2 = normalize_axis(tensor.value.shape.len(), axis2, "diagonal")?;
    if axis1 == axis2 {
        return Err("diagonal expects distinct axes".to_string());
    }
    let diag = tensor.value.shape[axis1].min(tensor.value.shape[axis2]);
    let mut out_shape = Vec::with_capacity(tensor.value.shape.len() - 1);
    for (index, size) in tensor.value.shape.iter().enumerate() {
        if index == axis1 {
            out_shape.push(diag);
        } else if index != axis2 {
            out_shape.push(*size);
        }
    }
    // chelis#1349: `out_index` holds one coordinate per retained OUTPUT
    // axis, so the diagonal's own coordinate lives at `axis1`'s position
    // after `axis2` is removed, which is one slot earlier whenever
    // `axis2 < axis1`. Indexing by the source axis number read past the
    // end for `axis1 == rank - 1` and, together with a reconstruction
    // walk that never consumed the diagonal's slot, handed later source
    // axes a coordinate belonging to a different axis (an out-of-bounds
    // storage pick whenever that coordinate's range exceeds the diagonal
    // extent).
    let diag_out_axis = if axis2 < axis1 { axis1 - 1 } else { axis1 };
    let out_numel = tensor_numel(&out_shape);
    let mut picks = Vec::with_capacity(out_numel);
    for linear in 0..out_numel {
        let out_index = linear_to_indices(linear, &out_shape);
        let mut src_index = Vec::with_capacity(tensor.value.shape.len());
        let mut out_pos = 0usize;
        let diag_idx = out_index[diag_out_axis];
        for index in 0..tensor.value.shape.len() {
            if index == axis1 {
                // `axis1` keeps an output slot (the diagonal's own), so
                // the walk must consume it.
                src_index.push(diag_idx);
                out_pos += 1;
            } else if index == axis2 {
                // `axis2` was removed from the output; nothing to consume.
                src_index.push(diag_idx);
            } else {
                src_index.push(out_index[out_pos]);
                out_pos += 1;
            }
        }
        picks.push(indices_to_linear(&src_index, &tensor.value.shape));
    }
    // reuse_* contract: diagonal is element-preserving (section C3).
    Ok(RuntimeTensorValue::new(IrTensorValue::from_storage(
        out_shape,
        tensor.value.storage().reuse_gather(&picks),
    )))
}

fn trace_balanced_sum(
    tensor: &RuntimeTensorValue,
    group: &[usize],
    accumulator: Prim,
    result: Prim,
) -> Result<ScalarValue, String> {
    let mut level = group
        .iter()
        .map(|&index| {
            cast_scalar(
                "trace",
                tensor.value.storage().scalar_at(index),
                accumulator,
            )
            .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if level.is_empty() {
        let zero = if accumulator.is_integer() {
            scalar_from_i64("trace", accumulator, 0)
        } else {
            scalar_from_f64("trace", accumulator, 0.0)
        }
        .map_err(|error| error.to_string())?;
        return cast_scalar("trace", zero, result).map_err(|error| error.to_string());
    }
    while level.len() > 1 {
        let mut source = level.into_iter();
        let mut next = Vec::with_capacity(source.len().div_ceil(2));
        while let Some(left) = source.next() {
            let combined = match source.next() {
                Some(right) if accumulator.is_integer() => {
                    int_binop(IntBinOp::Add, left, right).map_err(|error| error.to_string())?
                }
                Some(right) => {
                    float_binop(FloatBinOp::Add, left, right).map_err(|error| error.to_string())?
                }
                None => left,
            };
            next.push(combined);
        }
        level = next;
    }
    cast_scalar("trace", level[0], result).map_err(|error| error.to_string())
}

pub(super) fn tensor_trace_value(
    tensor: &RuntimeTensorValue,
    axis1: i64,
    axis2: i64,
) -> Result<RuntimeTensorValue, String> {
    // chelis#1349: the diagonal occupies output slot `axis1 - 1` when
    // `axis2 < axis1`, else `axis1` (see `tensor_diagonal_value`), and that
    // slot is the axis trace must reduce so both source axes are removed,
    // matching `infer_trace_result_type`. `min(axis1, axis2)` named it only
    // for `axis1 < axis2` and for adjacent reversed pairs; elsewhere it
    // reduced a retained axis and produced a shape the checker never
    // declared. Normalizing against the SOURCE rank with `?` also replaces
    // the prior `unwrap_or` silent fallback to the last axis.
    let source_rank = tensor.value.shape.len();
    let axis1 = normalize_axis(source_rank, axis1, "trace")?;
    let axis2 = normalize_axis(source_rank, axis2, "trace")?;
    let diagonal = tensor_diagonal_value(tensor, axis1 as i64, axis2 as i64)?;
    let rank = diagonal.value.shape.len();
    let axis = if axis2 < axis1 { axis1 - 1 } else { axis1 };
    let mut out_shape = diagonal.value.shape.clone();
    let axis_len = out_shape.remove(axis);
    let out_numel = tensor_numel(&out_shape);
    let mut groups = Vec::with_capacity(out_numel);
    for out_linear in 0..out_numel {
        let out_indices = linear_to_indices(out_linear, &out_shape);
        let mut group = Vec::with_capacity(axis_len);
        for axis_index in 0..axis_len {
            let mut input_indices = Vec::with_capacity(rank);
            let mut output_position = 0;
            for dimension in 0..rank {
                if dimension == axis {
                    input_indices.push(axis_index);
                } else {
                    input_indices.push(out_indices[output_position]);
                    output_position += 1;
                }
            }
            group.push(indices_to_linear(&input_indices, &diagonal.value.shape));
        }
        groups.push(group);
    }
    let accumulator = diagonal.precision.default_reduce_sum_accumulator()?;
    let result = diagonal.precision.default_reduce_sum_result_precision()?;
    let values = groups
        .iter()
        .map(|group| trace_balanced_sum(&diagonal, group, accumulator, result))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RuntimeTensorValue::new(IrTensorValue::from_storage(
        out_shape,
        tensor_from_scalars(result, &values),
    )))
}

pub(super) fn tensor_clamp_value(
    tensor: &RuntimeTensorValue,
    lo: &RuntimeTensorValue,
    hi: &RuntimeTensorValue,
) -> Result<RuntimeTensorValue, String> {
    let scalar_or_match = |bound: &RuntimeTensorValue| {
        bound.value.shape.is_empty() || bound.value.shape == tensor.value.shape
    };
    if tensor.precision != lo.precision || tensor.precision != hi.precision {
        return Err("clamp expects matching tensor precision".to_string());
    }
    if !scalar_or_match(lo) || !scalar_or_match(hi) {
        return Err(
            "clamp expects scalar tensor bounds or matching-shape tensor bounds".to_string(),
        );
    }
    let wide = tensor.value.to_f64_lossy_vec();
    let lo_wide = lo.value.to_f64_lossy_vec();
    let hi_wide = hi.value.to_f64_lossy_vec();
    let mut out = Vec::with_capacity(wide.len());
    for (linear, value) in wide.into_iter().enumerate() {
        let lo_value = if lo.value.shape.is_empty() {
            lo_wide[0]
        } else {
            lo_wide[linear]
        };
        let hi_value = if hi.value.shape.is_empty() {
            hi_wide[0]
        } else {
            hi_wide[linear]
        };
        out.push(value.clamp(lo_value, hi_value));
    }
    RuntimeTensorValue::from_wide("clamp", tensor.precision, tensor.value.shape.clone(), out)
}

pub(super) fn tensor_einsum_value(
    equation: &str,
    lhs: &RuntimeTensorValue,
    rhs: &RuntimeTensorValue,
) -> Result<RuntimeTensorValue, String> {
    let (inputs, output) = equation
        .split_once("->")
        .ok_or_else(|| "einsum equation must match [a-z]*,[a-z]*->[a-z]*".to_string())?;
    if output.contains("->") {
        return Err("einsum equation must contain exactly one `->`".to_string());
    }
    let (lhs_input, rhs_input) = inputs
        .split_once(',')
        .ok_or_else(|| "einsum equation must contain exactly two operands".to_string())?;
    if rhs_input.contains(',') {
        return Err("einsum equation must contain exactly two operands".to_string());
    }
    if !lhs_input
        .bytes()
        .chain(rhs_input.bytes())
        .chain(output.bytes())
        .all(|label| label.is_ascii_lowercase())
    {
        return Err("einsum labels must be lowercase ASCII `a` through `z`".to_string());
    }
    let lhs_labels = lhs_input.chars().collect::<Vec<_>>();
    let rhs_labels = rhs_input.chars().collect::<Vec<_>>();
    let out_labels = output.chars().collect::<Vec<_>>();
    if lhs_labels.len() != lhs.value.shape.len() || rhs_labels.len() != rhs.value.shape.len() {
        return Err("einsum label count must match operand rank".to_string());
    }
    let mut output_seen = [false; 26];
    for &label in &out_labels {
        let index = (label as u8 - b'a') as usize;
        if output_seen[index] {
            return Err(format!(
                "einsum output label `{label}` must occur exactly once"
            ));
        }
        output_seen[index] = true;
    }
    let mut dims = std::collections::BTreeMap::<char, usize>::new();
    for (label, size) in lhs_labels.iter().zip(&lhs.value.shape) {
        if let Some(prev) = dims.insert(*label, *size)
            && prev != *size
        {
            return Err(format!("einsum label `{label}` has inconsistent extents"));
        }
    }
    for (label, size) in rhs_labels.iter().zip(&rhs.value.shape) {
        if let Some(prev) = dims.insert(*label, *size)
            && prev != *size
        {
            return Err(format!("einsum label `{label}` has inconsistent extents"));
        }
    }
    let out_shape = out_labels
        .iter()
        .map(|label| {
            dims.get(label)
                .copied()
                .ok_or_else(|| format!("einsum output label `{label}` missing from inputs"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut reduction_labels = Vec::<char>::new();
    for label in lhs_labels.iter().chain(rhs_labels.iter()) {
        if !out_labels.contains(label) && !reduction_labels.contains(label) {
            reduction_labels.push(*label);
        }
    }
    let reduction_shape = reduction_labels
        .iter()
        .map(|label| dims.get(label).copied().unwrap_or(1))
        .collect::<Vec<_>>();
    // The host lane carries shapes as `usize`, but the language's extent
    // domain is i64 ([05-DIM-2]) and [05-OP-33] wants an unrepresentable
    // count to trap `Overflow`. Fold in i64 so this lane agrees with the C
    // runtime about where the ceiling is instead of inheriting the host's, and
    // short-circuit a zero extent so the answer does not depend on axis order:
    // a zero anywhere means zero elements, whatever the other extents are.
    let checked_product = |shape: &[usize], context: &str| {
        if shape.contains(&0) {
            return Ok(0_usize);
        }
        shape
            .iter()
            .try_fold(1_i64, |product, &extent| {
                let extent = i64::try_from(extent).map_err(|_| {
                    format!("Overflow: einsum {context} extent {extent} exceeds i64")
                })?;
                product
                    .checked_mul(extent)
                    .ok_or_else(|| format!("Overflow: einsum {context} extent product exceeds i64"))
            })
            .and_then(|product| {
                usize::try_from(product).map_err(|_| {
                    format!(
                        "Overflow: einsum {context} extent product {product} is not representable on this host"
                    )
                })
            })
    };
    let output_total = checked_product(&out_shape, "output")?;
    let reduction_total = checked_product(&reduction_shape, "reduction")?;
    // This legacy host einsum still accumulates in f64. Unlike matmul,
    // it does not yet delegate to the typed contraction implementation;
    // #1290 owns alignment with [05-OP-33]'s exact tree and widths.
    let lhs_wide = lhs.value.to_f64_lossy_vec();
    let rhs_wide = rhs.value.to_f64_lossy_vec();
    let mut out = vec![0.0; output_total];
    for (out_linear, slot) in out.iter_mut().enumerate() {
        let out_index = linear_to_indices(out_linear, &out_shape);
        let mut label_values = chelis_unord::UnordMap::<char, usize>::new();
        for (label, value) in out_labels.iter().zip(out_index.iter()) {
            label_values.insert(*label, *value);
        }
        let mut acc = 0.0_f64;
        for reduction_linear in 0..reduction_total {
            let reduction_index = linear_to_indices(reduction_linear, &reduction_shape);
            for (label, value) in reduction_labels.iter().zip(reduction_index.iter()) {
                label_values.insert(*label, *value);
            }
            let lhs_index = lhs_labels
                .iter()
                .map(|label| label_values[label])
                .collect::<Vec<_>>();
            let rhs_index = rhs_labels
                .iter()
                .map(|label| label_values[label])
                .collect::<Vec<_>>();
            acc += lhs_wide[indices_to_linear(&lhs_index, &lhs.value.shape)]
                * rhs_wide[indices_to_linear(&rhs_index, &rhs.value.shape)];
        }
        *slot = acc;
    }
    RuntimeTensorValue::from_wide("einsum", lhs.precision, out_shape, out)
}

/// Every exit in the eval lane truncates tensor element rendering after
/// this many elements, marking the cut with `, ...` - one rule at every
/// exit in both lanes ([05-OBS-5]; faithful_observation.md open question 4,
/// decided 2026-07-17, matching the compiled lane's existing form).
/// Full-element fidelity is `to_list`'s and the wire's job, never print's.
pub(super) const TENSOR_RENDER_LIMIT: usize = 32;

/// The [05-OBS-1] renderer for one eval-lane element of a TENSOR payload.
///
/// Width policy (chelis#732 Phase 1, deliberate): the eval tensor store is
/// f64-backed and its runtime precision tag is unreliable for float WIDTH
/// (chelis#717 pins checker-f64 tensors at an F32 tag), so float elements
/// format at the STORED width (f64) - narrowing at render time would
/// launder stored bits, which [05-OBS-1] forbids. The tag is trusted for
/// dtype CLASS only (bool/integer/float), which is structurally sound: the
/// class comes from what `to_tensor`/`cast`/the DAG actually stored.
/// Own-width float tensor digits arrive when chelis#729 repairs the value
/// metadata; scalar exits already render at their own width below.
/// Render a tensor payload: `tensor(shape=[..], data=[..])`, elements via
/// [`render_tensor_element`], truncated per [`TENSOR_RENDER_LIMIT`]. A
/// rank-0 tensor renders as its single element, bare: the
/// `tensor(shape=[], data=[..])` wrapper is not an exit form ([05-OBS-4],
/// the chelis#775 scalar-root decision - eval's internal rank-0
/// realization of scalar bindings must not leak into the observation
/// channel, and `print` of the same scalar already renders bare).
fn render_tensor(tensor: &RuntimeTensorValue) -> String {
    // chelis#729 Phase 1: elements render straight from the sealed
    // per-dtype storage (`element_ref` carries the element at its own
    // width), so the former f64-image bridge `render_tensor_element` and
    // its tag-vs-bits disagreement arm are structurally unreachable: the
    // storage variant IS the tag.
    if tensor.value.shape.is_empty() {
        assert!(
            !tensor.value.is_empty(),
            "render_tensor: rank-0 tensor with no element (IrTensorValue \
             guarantees numel(shape=[]) == 1 at construction)"
        );
        return chelis_types::format_element(
            tensor.precision,
            tensor.value.storage().element_ref(0),
        );
    }
    let visible = tensor.value.len().min(TENSOR_RENDER_LIMIT);
    let mut elements: Vec<String> = (0..visible)
        .map(|index| {
            chelis_types::format_element(
                tensor.precision,
                tensor.value.storage().element_ref(index),
            )
        })
        .collect();
    if tensor.value.len() > visible {
        elements.push("...".to_string());
    }
    format!(
        "tensor(shape={:?}, data=[{}])",
        tensor.value.shape,
        elements.join(", ")
    )
}

// pub(crate): compiler.rs pre-renders each evaluated root's display text
// through this exact function (the [05-OBS-1] single renderer) while the
// dtype tags still exist; see `EvaluatedRoot::display`.
pub(crate) fn render_value(value: &RuntimeValue) -> String {
    use chelis_types::{ElementRef, format_element};
    match value {
        RuntimeValue::Tensor(tensor) => render_tensor(tensor),
        RuntimeValue::Scalar(payload) => {
            // Scalars carry their dtype in the sealed storage variant
            // (the dtype/bits invariant holds by construction), so every
            // scalar exit renders at its OWN width per [05-OBS-2].
            format_element(payload.dtype(), payload.value().element_ref())
        }
        RuntimeValue::Bool(value) => format_element(Prim::Bool, ElementRef::Bool(*value)),
        RuntimeValue::String(value) => value.clone(),
        RuntimeValue::List(items) => format!(
            "[{}]",
            items
                .iter()
                .map(render_value)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        RuntimeValue::Dict(entries) => format!(
            "dict({})",
            entries
                .iter()
                .map(|(key, value)| format!("{}: {}", render_value(key), render_value(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        RuntimeValue::Tuple(items) => format!(
            "({})",
            items
                .iter()
                .map(render_value)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        // Show the user-facing (de-mangled) constructor name; a reef-linked
        // ADT carries the internal `Pkg__..__Ctor` form, which must not leak
        // to eval output (chelis#399). `demangle_ident` is a no-op on bare /
        // builtin constructors.
        RuntimeValue::Adt { ctor, fields, .. } if fields.is_empty() => {
            chelis_types::demangle_ident(ctor)
        }
        RuntimeValue::Adt { ctor, fields, .. } => format!(
            "{}({})",
            chelis_types::demangle_ident(ctor),
            fields
                .iter()
                .map(render_value)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        RuntimeValue::MappedFile(bytes) => format!("<mapped-file:{}>", bytes.len()),
        RuntimeValue::Closure { .. } => "<closure>".to_string(),
        RuntimeValue::Transform { kind, .. } => match kind {
            TransformKind::Grad => "<grad>".to_string(),
            TransformKind::Vmap => "<vmap>".to_string(),
        },
        RuntimeValue::Unit => "()".to_string(),
    }
}

// ---------------------------------------------------------------------------
// The diagnostic-rendering boundary (chelis#997 / faithful_observation.md
// FO-DIAG). It sits here, beside `render_value`, because the two are one
// policy: `render_value` is the [05-OBS-1] exit renderer, and everything
// below frames its output for an error message. §C1.6 forbids a diagnostic
// laundering what it reports, so no caller picks its own numeric grammar,
// container shape, or truncation - a call site names WHAT it is reporting
// (a value, an argument slot, an ADT field list) and this boundary decides
// how it renders.
// ---------------------------------------------------------------------------

/// Byte cap on one diagnostic rendering. A malformed payload can be a whole
/// parsed document; past this many bytes the text is cut on a char boundary
/// and the elision is stated rather than silently dropped (chelis#903
/// review). Renderings that fit are byte-identical to the untruncated form.
const DIAGNOSTIC_RENDER_LIMIT: usize = 160;

/// Truncate one already-rendered diagnostic string at
/// [`DIAGNOSTIC_RENDER_LIMIT`]. Private to this boundary: truncation is a
/// property of the diagnostic channel, not a knob a call site turns.
fn truncate_for_diagnostic(full: String) -> String {
    if full.len() <= DIAGNOSTIC_RENDER_LIMIT {
        return full;
    }
    let mut cut = DIAGNOSTIC_RENDER_LIMIT;
    while !full.is_char_boundary(cut) {
        cut -= 1;
    }
    format!(
        "{}... ({} more bytes elided)",
        &full[..cut],
        full.len() - cut
    )
}

/// The untruncated body of [`describe_value`].
///
/// Every numeric payload reaches text through [`render_value`], hence
/// through `format_element` - the §C1.6 requirement. The only thing added
/// on top is a KIND TAG on the outermost value, because a mismatch
/// diagnostic ("expected an f64 value, got ...") has to name what arrived,
/// and the canonical exit form deliberately does not: `5` alone cannot
/// distinguish an i32 from an f64 whose shortest form has no fraction.
/// Nested structure is *not* re-tagged - it is `render_value`'s output
/// verbatim, so the diagnostic and the exit channel agree byte-for-byte on
/// every payload they both render.
fn describe_value_untruncated(value: &RuntimeValue) -> String {
    match value {
        // The dtype tag comes from the sealed storage variant, so it is the
        // width the digits were rendered at, not a guess.
        RuntimeValue::Scalar(payload) => {
            format!("{} {}", payload.dtype().name(), render_value(value))
        }
        RuntimeValue::Bool(_) => format!("bool {}", render_value(value)),
        // Quoted so an empty, blank, or control-carrying string is visible;
        // `escape_debug` is string escaping, not a numeric grammar, and a
        // string payload carries no digits for it to launder.
        RuntimeValue::String(text) => format!("string \"{}\"", text.escape_debug()),
        RuntimeValue::List(_) => format!("list {}", render_value(value)),
        RuntimeValue::Tuple(_) => format!("tuple {}", render_value(value)),
        // Tensors, dicts, ADTs, mapped files, closures, transforms, and unit
        // already name their own shape in the canonical form
        // (`tensor(shape=..)`, `dict(..)`, `JInt(..)`, `<closure>`, `()`).
        _ => render_value(value),
    }
}

/// Render one runtime value for a diagnostic message.
pub(crate) fn describe_value(value: &RuntimeValue) -> String {
    truncate_for_diagnostic(describe_value_untruncated(value))
}

/// Render an argument slot that may be absent (`args.first()` and friends),
/// so a missing argument reads as a missing argument instead of as a Rust
/// `Option` spelling.
pub(crate) fn describe_argument(slot: Option<&RuntimeValue>) -> String {
    match slot {
        Some(value) => describe_value(value),
        None => "nothing".to_string(),
    }
}

/// Render an ADT constructor's field list for a malformed-shape diagnostic.
/// Fields ARE tagged individually: `malformed JNum fields [i32 5]` names
/// the reason the shape was rejected, which an untagged `[5]` does not.
#[cfg(test)]
pub(crate) fn describe_fields(fields: &[RuntimeValue]) -> String {
    truncate_for_diagnostic(format!(
        "[{}]",
        fields
            .iter()
            .map(describe_value_untruncated)
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

pub(super) fn builtin_name(expr: &Expr) -> Option<&str> {
    let name = var_name(expr)?;
    BUILTIN_NAMES.contains(&name).then_some(name)
}

#[cfg(test)]
fn dropout_sample(seed: u64, index: u64) -> f64 {
    let mut x = seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    ((x >> 11) as f64) / ((1u64 << 53) as f64)
}

pub(super) fn uniform_like_value(
    template: &RuntimeTensorValue,
    low: f64,
    high: f64,
    seed: u64,
) -> RuntimeTensorValue {
    let low = low as f32;
    let high = high as f32;
    let values = (0..template.value.len())
        .map(|index| {
            uniform_sample(template.precision, low, high, seed, index as u64)
                .expect("uniform_like checker admits only active float dtypes")
        })
        .collect::<Vec<_>>();
    RuntimeTensorValue::new(IrTensorValue::from_storage(
        template.value.shape.clone(),
        tensor_from_scalars(template.precision, &values),
    ))
}

#[cfg(test)]
mod uniform_like_affine_tests {
    //! chelis#770/#937: `uniform_like_value` routes through the shared
    //! per-dtype sampler used by the IR evaluator. These pin the exact
    //! widened-f32 output at seed=42 / shape=[8] and the 1-ULP gap the old
    //! f64 affine left at elem[4] of [2,5), plus a negative range (unit-level
    //! only: the C cross-lane path can't be driven with a bare negative
    //! literal, a separate lowering gap).
    use super::*;

    fn template_f32(n: usize) -> RuntimeTensorValue {
        RuntimeTensorValue::from_wide("test", Prim::F32, vec![n], vec![0.0; n])
            .expect("zero template finalizes at f32")
    }

    fn template_f64(n: usize) -> RuntimeTensorValue {
        RuntimeTensorValue::from_wide("test", Prim::F64, vec![n], vec![0.0; n])
            .expect("zero template finalizes at f64")
    }

    #[test]
    fn affine_mirrors_c_f32_sampler_positive_range() {
        let out = uniform_like_value(&template_f32(8), 2.0, 5.0, 42);
        // Single correctly-rounded FMA, conforming to the compiled C sampler.
        // elem[4]: where the pre-#770 f64 affine rounded to the adjacent f32
        // (0x404215a9) instead of the sampler's 0x404215aa.
        assert_eq!(
            out.value.to_f64_lossy_vec()[4].to_bits(),
            (f32::from_bits(0x404215aa) as f64).to_bits(),
            "elem[4] must be the C f32 sampler value (0x404215aa), got {} (f32 bits {:#010x})",
            out.value.to_f64_lossy_vec()[4],
            (out.value.to_f64_lossy_vec()[4] as f32).to_bits(),
        );
        let old_f64_affine = 2.0 + (5.0 - 2.0) * dropout_sample(42, 4);
        assert_eq!((old_f64_affine as f32).to_bits(), 0x404215a9);
        assert_ne!(
            (out.value.to_f64_lossy_vec()[4] as f32).to_bits(),
            (old_f64_affine as f32).to_bits(),
            "the fix must not reproduce the old f64-affine rounding",
        );
        // elem[6]/[7]: where a single-rounding FMA and a plain two-rounding
        // affine disagree by 1 ULP — the exact bit the compiled C lane flips
        // between `-ffp-contract=fast` (FMA, 0x408f5273) and `=off` (two
        // roundings, 0x408f5274). Pin the FMA values; show two-rounding differs.
        assert_eq!(
            out.value.to_f64_lossy_vec()[6].to_bits(),
            (f32::from_bits(0x408f5273) as f64).to_bits(),
            "elem[6] must be the single-rounding FMA value (0x408f5273)",
        );
        assert_eq!(
            out.value.to_f64_lossy_vec()[7].to_bits(),
            (f32::from_bits(0x403ec1e7) as f64).to_bits(),
            "elem[7] must be the single-rounding FMA value (0x403ec1e7)",
        );
        let unit6 = dropout_sample(42, 6) as f32;
        let two_rounding_6 = 2.0f32 + (5.0f32 - 2.0f32) * unit6;
        assert_eq!(two_rounding_6.to_bits(), 0x408f5274);
        assert_ne!(
            (out.value.to_f64_lossy_vec()[6] as f32).to_bits(),
            two_rounding_6.to_bits(),
        );
    }

    #[test]
    fn affine_is_f32_for_negative_range() {
        let out = uniform_like_value(&template_f32(8), -3.0, -1.0, 42);
        assert_eq!(
            out.value.to_f64_lossy_vec()[3].to_bits(),
            (f32::from_bits(0xc010167a) as f64).to_bits(),
            "elem[3] must be the C f32 sampler value for [-3,-1) (0xc010167a)",
        );
    }

    #[test]
    fn affine_uses_f64_storage_and_f64_arithmetic_for_f64_template() {
        let out = uniform_like_value(&template_f64(8), 2.0, 5.0, 42);
        assert_eq!(out.precision, Prim::F64);
        let expected = uniform_sample(Prim::F64, 2.0, 5.0, 42, 4)
            .expect("f64 sample")
            .as_f64_lossy();
        assert_eq!(out.value.element_f64_lossy(4).to_bits(), expected.to_bits());
        assert_ne!(
            out.value.element_f64_lossy(4).to_bits(),
            (expected as f32 as f64).to_bits(),
            "f64 samples must not be widened f32 values"
        );
    }
}

#[cfg(test)]
mod normalize_axis_tests {
    //! chelis#522: `normalize_axis` is the single host-evaluator axis
    //! normalizer for every axis-taking primitive. These pin the
    //! from-the-end convention and the loud out-of-range rejection at the
    //! function boundary, independent of any one caller.
    use super::normalize_axis;

    #[test]
    fn accepts_last_axis_via_minus_one() {
        assert_eq!(normalize_axis(3, -1, "op").unwrap(), 2);
        assert_eq!(normalize_axis(2, -1, "op").unwrap(), 1);
    }

    #[test]
    fn accepts_first_axis_via_minus_rank() {
        assert_eq!(normalize_axis(3, -3, "op").unwrap(), 0);
        assert_eq!(normalize_axis(1, -1, "op").unwrap(), 0);
    }

    #[test]
    fn passes_through_non_negative_axes() {
        assert_eq!(normalize_axis(3, 0, "op").unwrap(), 0);
        assert_eq!(normalize_axis(3, 2, "op").unwrap(), 2);
    }

    #[test]
    fn rejects_axis_equal_to_rank() {
        let err = normalize_axis(3, 3, "reduction").expect_err("axis == rank is out of range");
        assert!(
            err.contains("reduction") && err.contains("out of bounds"),
            "expected an out-of-bounds reduction diagnostic, got {err:?}"
        );
    }

    #[test]
    fn rejects_axis_one_past_negative_rank() {
        // `-rank-1` normalizes to `-1`, still out of `0..rank`: reject loud,
        // do NOT silently wrap to a valid axis.
        let err = normalize_axis(3, -4, "gather").expect_err("-rank-1 is out of range");
        assert!(
            err.contains("gather") && err.contains("out of bounds"),
            "expected an out-of-bounds gather diagnostic, got {err:?}"
        );
    }
}

#[cfg(test)]
mod numeric_trap_forwarding_tests {
    use super::*;

    #[test]
    fn composed_evaluation_forwards_the_raising_primitive_trap_without_plumbing() {
        // i8 sum has an i32 default accumulator and therefore cannot
        // overflow on this two-element input. Use an i32 accumulator-edge
        // row so the test continues to exercise trap forwarding without
        // contradicting the §5.7.1 accumulator contract.
        let input = RuntimeTensorValue::from_wide_int(
            "test",
            Prim::Int32,
            vec![2],
            vec![i64::from(i32::MAX), 1],
        )
        .expect("input is representable at i32");
        let err = eval_composed_unary(&input, |dag, x, ty| {
            let output_ty = TensorType {
                dims: Vec::new(),
                precision: ty.precision,
            };
            dag.add_node(
                RiscOp::sum_default(0, ty.precision).expect("i8 sum is admitted"),
                vec![x],
                output_ty,
                None,
            )
        })
        .expect_err("the composed i32 sum must overflow");

        assert_eq!(err, "numeric trap: overflow in sum at i32");
        assert!(!err.contains("IR eval failed"));
        assert!(!err.contains("composed unary"));
    }
}
