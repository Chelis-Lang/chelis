use std::collections::HashMap;

use chelis_deep::ast::{Atom, Expr};
use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue as IrTensorValue, eval_tensor_roots_with};
use chelis_ir::tier2;
use chelis_types::{BUILTIN_NAMES, types::Prim};

use super::transforms::*;
use super::*;

pub(super) fn pattern_matches(
    value: &RuntimeValue,
    pattern: &Expr,
    bindings: &mut HashMap<String, RuntimeValue>,
    adt_fields: &HashMap<String, Vec<String>>,
) -> Result<bool, String> {
    let Some(list) = as_list(pattern) else {
        return Ok(false);
    };
    match tag(list) {
        Some(DeepTag::PatVar) => {
            if let Some(name) = children(list).first().and_then(symbol_name) {
                bindings.insert(name.to_string(), value.clone());
                return Ok(true);
            }
            Ok(false)
        }
        Some(DeepTag::PatWild) => Ok(true),
        Some(DeepTag::PatLit) => {
            let lit = children(list)
                .first()
                .ok_or_else(|| "pat-lit missing value".to_string())?;
            Ok(match (value, lit) {
                (RuntimeValue::Scalar(payload), Expr::Atom(Atom::Int(rhs), _))
                    if payload.dtype().is_integer() =>
                {
                    payload.bits().as_i64() == *rhs
                }
                (RuntimeValue::Scalar(payload), Expr::Atom(Atom::Float(rhs), _))
                    if payload.dtype().is_float() =>
                {
                    payload.bits().as_f64() == *rhs
                }
                (RuntimeValue::Bool(lhs), Expr::Atom(Atom::Bool(rhs), _)) => lhs == rhs,
                (RuntimeValue::String(lhs), Expr::Atom(Atom::Str(rhs), _)) => lhs == rhs,
                _ => false,
            })
        }
        Some(DeepTag::PatCtor) => {
            let kids = children(list);
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
        Some(DeepTag::PatRecord) => {
            let kids = children(list);
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
                let Some(kv_list) = as_list(kv_expr) else {
                    continue;
                };
                if tag(kv_list) != Some(DeepTag::Kv) {
                    continue;
                }
                let kv_kids = children(kv_list);
                let Some(field_name) = kv_kids.first().and_then(symbol_name) else {
                    continue;
                };
                let Some(pattern_expr) = kv_kids.get(1) else {
                    continue;
                };
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
        Some(DeepTag::PatTuple) => {
            let RuntimeValue::Tuple(items) = value else {
                return Ok(false);
            };
            if items.len() != children(list).len() {
                return Ok(false);
            }
            for (subpat, item) in children(list).iter().zip(items) {
                if !pattern_matches(item, subpat, bindings, adt_fields)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// chelis#520 D2: is a declared field type inside the field-wise
/// gradient slice? Float tensors and float scalars are; everything
/// else (ints, bools, strings, nested ADTs, generic type vars) is not.
/// Zero-argument `typealias` references resolve through `aliases`
/// (matching the checker's alias resolution; `seen` guards cycles);
/// a `t-adt` name that is not a known alias is a real nested ADT and
/// stays non-float.
fn field_type_is_float(ty: &Expr, aliases: &HashMap<String, Expr>, seen: &mut Vec<String>) -> bool {
    let Some(list) = as_list(ty) else {
        return false;
    };
    match tag(list) {
        Some(DeepTag::TPrim) => matches!(
            children(list).first().and_then(symbol_name),
            Some("f16" | "bf16" | "f32" | "f64")
        ),
        Some(DeepTag::TTensor) => children(list)
            .last()
            .is_some_and(|elem| field_type_is_float(elem, aliases, seen)),
        Some(DeepTag::TAdt) => {
            let kids = children(list);
            let Some(name) = kids.first().and_then(symbol_name) else {
                return false;
            };
            // Parameterized references would need argument substitution;
            // stay conservative (the checker skips those as well).
            if kids.len() > 1 || seen.iter().any(|s| s == name) {
                return false;
            }
            let Some(target) = aliases.get(name) else {
                return false;
            };
            seen.push(name.to_string());
            let result = field_type_is_float(target, aliases, seen);
            seen.pop();
            result
        }
        _ => false,
    }
}

/// Collect zero-parameter `typealias` targets so field types declared
/// through an alias resolve the way the checker resolves them.
fn collect_type_aliases(expr_sets: &[&[Expr]]) -> HashMap<String, Expr> {
    let mut aliases = HashMap::new();
    for exprs in expr_sets {
        for expr in top_level_items(exprs) {
            let Expr::List(list, _) = expr else {
                continue;
            };
            if tag(list) != Some(DeepTag::Typealias) {
                continue;
            }
            let kids = children(list);
            let (Some(name), Some(params), Some(target)) =
                (kids.first().and_then(symbol_name), kids.get(1), kids.get(2))
            else {
                continue;
            };
            let no_params = as_list(params).is_some_and(|p| p.elements.is_empty());
            if no_params {
                aliases.insert(name.to_string(), target.clone());
            }
        }
    }
    aliases
}

/// chelis#520 D2: map every constructor of a deftype to a rejection
/// reason when the TYPE is outside the field-wise gradient slice. The
/// checker types `grad` over such an ADT argument as non-differentiable
/// (`unit` payload), so the eval lane must reject loudly instead of
/// fabricating a gradient value the static type does not admit. Two
/// shapes are rejected: a type with any non-float-tensor field in ANY
/// variant (the constructed variant may be float-clean, but a mixed
/// sibling variant already poisons the static gradient type), and a
/// pure enum whose variants carry no fields at all (no continuous
/// payload to differentiate). Keyed by constructor name, matching the
/// global-ctor-namespace convention of [`collect_adt_ctor_fields`].
/// Takes every expr set at once (library plus program) so an alias
/// declared in one set resolves inside a deftype from another.
pub(crate) fn collect_adt_grad_rejections(expr_sets: &[&[Expr]]) -> HashMap<String, String> {
    let aliases = collect_type_aliases(expr_sets);
    let mut out = HashMap::new();
    for expr in expr_sets.iter().flat_map(|exprs| top_level_items(exprs)) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) != Some(DeepTag::Deftype) {
            continue;
        }
        let kids = children(list);
        let Some(type_name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let mut ctors: Vec<&str> = Vec::new();
        let mut offending: Option<(String, String)> = None;
        let mut has_any_field = false;
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
            ctors.push(ctor);
            for field in variant_kids.iter().skip(1) {
                let Some(field_list) = as_list(field) else {
                    continue;
                };
                if tag(field_list) != Some(DeepTag::Field) {
                    continue;
                }
                let field_kids = children(field_list);
                let Some(field_name) = field_kids.first().and_then(symbol_name) else {
                    continue;
                };
                has_any_field = true;
                let is_float = field_kids
                    .get(1)
                    .is_some_and(|ty| field_type_is_float(ty, &aliases, &mut Vec::new()));
                if !is_float && offending.is_none() {
                    offending = Some((ctor.to_string(), field_name.to_string()));
                }
            }
        }
        let reason = match (&offending, has_any_field) {
            (Some((bad_ctor, bad_field)), _) => Some(format!(
                "type `{type_name}` is not field-wise differentiable: field \
                 `{bad_field}` of constructor `{bad_ctor}` is not a float tensor, \
                 so the checker types this gradient as unit; mixed ADT gradients \
                 are not supported yet"
            )),
            (None, false) => Some(format!(
                "type `{type_name}` carries no fields in any constructor, so \
                 there is no float tensor payload to differentiate"
            )),
            (None, true) => None,
        };
        if let Some(reason) = reason {
            for ctor in ctors {
                out.insert(ctor.to_string(), reason.clone());
            }
        }
    }
    out
}

pub(crate) fn collect_adt_ctor_fields(exprs: &[Expr]) -> HashMap<String, Vec<String>> {
    let mut out = HashMap::new();
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

/// Per-dtype dispatch helper for two-argument numeric ops on host
/// scalars. WS-A0: each dtype gets the same code path but the result
/// dtype matches the operand dtype (per spec §5.1 "no implicit precision
/// promotion"). The closure is invoked at f64 precision and re-packed
/// at the operand dtype on the way out.
pub(super) fn dispatch_scalar_binop(
    lhs: &RuntimeValue,
    rhs: &RuntimeValue,
    op: &impl Fn(f64, f64) -> f64,
) -> Result<RuntimeValue, String> {
    match (lhs, rhs) {
        (RuntimeValue::Scalar(lp), RuntimeValue::Scalar(rp))
            if lp.dtype().is_integer() && rp.dtype().is_integer() =>
        {
            // Mirror pre-WS-A0 behavior: integer scalar ops compute the
            // result in f64 and truncate. Result dtype is the operand
            // dtype; if dtypes differ, widen to int64.
            let (ldt, rdt) = (lp.dtype(), rp.dtype());
            let (lb, rb) = (lp.bits(), rp.bits());
            let result_dtype = if ldt == rdt { ldt } else { Prim::Int64 };
            let value = op(lb.as_f64(), rb.as_f64()) as i64;
            RuntimeValue::scalar_like_int(result_dtype, value)
        }
        (RuntimeValue::Scalar(lp), RuntimeValue::Scalar(rp))
            if lp.dtype().is_float() && rp.dtype().is_float() =>
        {
            // Float-float: pick the wider of the two operand dtypes (no
            // implicit promotion when they match — but keep f64 if either
            // side is f64 so we don't downgrade an f64-typed value).
            //
            // E1 (WS-A0 RT-1 fixup): per spec/04-type-system.md §5.1
            // there is no implicit precision promotion. The mixed
            // narrow-float case (`(Bf16, F16)` and `(F16, Bf16)`) must
            // be rejected by the type checker before reaching this
            // dispatch; if the runtime ever observes it the type
            // checker has a hole. Replace the silent `_ => Prim::F32`
            // re-precisioning fallback with an `unreachable!` that
            // names the spec invariant.
            let (ldt, rdt) = (lp.dtype(), rp.dtype());
            let (lb, rb) = (lp.bits(), rp.bits());
            let result_dtype = match (ldt, rdt) {
                (Prim::F64, _) | (_, Prim::F64) => Prim::F64,
                (Prim::F32, _) | (_, Prim::F32) => Prim::F32,
                (Prim::Bf16, Prim::Bf16) => Prim::Bf16,
                (Prim::F16, Prim::F16) => Prim::F16,
                _ => unreachable!(
                    "type checker must reject mismatched float precisions per \
                     spec/04-type-system.md §5.1 (no implicit precision promotion); \
                     reached float-binop fallback with ({:?}, {:?})",
                    ldt, rdt
                ),
            };
            let value = op(lb.as_f64(), rb.as_f64());
            RuntimeValue::scalar_like_float(result_dtype, value)
        }
        _ => Err(format!(
            "numeric op expects matching int or float args, got ({lhs:?}, {rhs:?})"
        )),
    }
}

pub(super) fn numeric_binop(
    args: &[RuntimeValue],
    op: impl Fn(f64, f64) -> f64,
) -> Result<RuntimeValue, String> {
    match (args.first(), args.get(1)) {
        (Some(RuntimeValue::Tensor(lhs)), Some(RuntimeValue::Tensor(rhs))) => {
            tensor_numeric_binop(lhs, rhs, &op)
        }
        (Some(RuntimeValue::Tensor(lhs)), Some(rhs)) => tensor_scalar_binop(lhs, rhs, &op),
        (Some(lhs), Some(RuntimeValue::Tensor(rhs))) => scalar_tensor_binop(lhs, rhs, &op),
        (Some(lhs), Some(rhs)) => dispatch_scalar_binop(lhs, rhs, &op),
        other => Err(format!(
            "numeric op expects matching int or float args, got {other:?}"
        )),
    }
}

pub(super) fn numeric_unop(
    args: &[RuntimeValue],
    op: impl Fn(f64) -> f64,
) -> Result<RuntimeValue, String> {
    match args.first() {
        Some(RuntimeValue::Tensor(tensor)) => tensor_numeric_unop(tensor, &op),
        Some(RuntimeValue::Scalar(payload)) if payload.dtype().is_integer() => {
            RuntimeValue::scalar_like_int(payload.dtype(), op(payload.bits().as_f64()) as i64)
        }
        Some(RuntimeValue::Scalar(payload)) if payload.dtype().is_float() => {
            RuntimeValue::scalar_like_float(payload.dtype(), op(payload.bits().as_f64()))
        }
        other => Err(format!(
            "numeric op expects int or float arg, got {other:?}"
        )),
    }
}

fn tensor_numeric_binop(
    lhs: &RuntimeTensorValue,
    rhs: &RuntimeTensorValue,
    op: &impl Fn(f64, f64) -> f64,
) -> Result<RuntimeValue, String> {
    if lhs.value.shape != rhs.value.shape {
        return Err(format!(
            "tensor shapes must match for elementwise op, got {:?} vs {:?}",
            lhs.value.shape, rhs.value.shape
        ));
    }
    Ok(RuntimeValue::Tensor(RuntimeTensorValue {
        value: IrTensorValue::from_vec(
            lhs.value.shape.clone(),
            lhs.value
                .data
                .iter()
                .zip(&rhs.value.data)
                .map(|(l, r)| op(*l, *r))
                .collect(),
        ),
        precision: lhs.precision,
    }))
}

fn tensor_scalar_binop(
    tensor: &RuntimeTensorValue,
    scalar: &RuntimeValue,
    op: &impl Fn(f64, f64) -> f64,
) -> Result<RuntimeValue, String> {
    let scalar = runtime_scalar_as_f64(scalar)
        .ok_or_else(|| format!("numeric op expects scalar rhs, got {scalar:?}"))?;
    Ok(RuntimeValue::Tensor(RuntimeTensorValue {
        value: IrTensorValue::from_vec(
            tensor.value.shape.clone(),
            tensor
                .value
                .data
                .iter()
                .map(|value| op(*value, scalar))
                .collect(),
        ),
        precision: tensor.precision,
    }))
}

fn scalar_tensor_binop(
    scalar: &RuntimeValue,
    tensor: &RuntimeTensorValue,
    op: &impl Fn(f64, f64) -> f64,
) -> Result<RuntimeValue, String> {
    let scalar = runtime_scalar_as_f64(scalar)
        .ok_or_else(|| format!("numeric op expects scalar lhs, got {scalar:?}"))?;
    Ok(RuntimeValue::Tensor(RuntimeTensorValue {
        value: IrTensorValue::from_vec(
            tensor.value.shape.clone(),
            tensor
                .value
                .data
                .iter()
                .map(|value| op(scalar, *value))
                .collect(),
        ),
        precision: tensor.precision,
    }))
}

fn tensor_numeric_unop(
    tensor: &RuntimeTensorValue,
    op: &impl Fn(f64) -> f64,
) -> Result<RuntimeValue, String> {
    Ok(RuntimeValue::Tensor(RuntimeTensorValue {
        value: IrTensorValue::from_vec(
            tensor.value.shape.clone(),
            tensor.value.data.iter().map(|value| op(*value)).collect(),
        ),
        precision: tensor.precision,
    }))
}

/// Tensor-elementwise unary that runs through `f32` precision so the
/// host-runtime activation primitives stay byte-identical (to f32 ulp
/// tolerance) with the C backend's `chelis_host_*_f32` helpers, which
/// always go through `float` in `crates/chelis-backend-c/src/host_emit.rs`.
///
/// The closure receives an `f64` (cast down from `f32`) and returns an
/// `f64` (cast down from the float result of its body). The wrapper
/// itself takes care of the cast-down-cast-back at the boundary; the
/// caller need only ensure every internal transcendental is invoked
/// against an `f32` value (via `as f32` followed by libm `f32::*`).
pub(super) fn tensor_float_unop_f32(
    tensor: &RuntimeTensorValue,
    op: impl Fn(f32) -> f32,
) -> RuntimeTensorValue {
    RuntimeTensorValue {
        value: IrTensorValue::from_vec(
            tensor.value.shape.clone(),
            tensor
                .value
                .data
                .iter()
                .map(|value| op(*value as f32) as f64)
                .collect(),
        ),
        precision: tensor.precision,
    }
}

/// `relu(x) = max(0, x)`. Exact in any precision; we still take `f32`
/// here so the host-lane and C-lane storage shapes line up.
pub(super) fn activation_relu_f32(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

/// `sigmoid(x) = 1 / (1 + exp(-x))`. Mirrors `chelis_host_sigmoid_f32`
/// in `crates/chelis-backend-c/src/host_emit.rs:137` exactly — single
/// `expf` of `-x`, no f64 widening.
pub(super) fn activation_sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `tanh(x)` via `f32::tanh`. Matches the C backend's `tanhf` helper.
pub(super) fn activation_tanh_f32(x: f32) -> f32 {
    x.tanh()
}

/// `silu(x) = x * sigmoid(x)` (a.k.a. swish). Composed from
/// `activation_sigmoid_f32` so the f32-rounding profile is identical
/// to the C-backend helper — i.e., the C side computes
/// `x * chelis_host_sigmoid_f32(x)` and we mirror it 1:1.
pub(super) fn activation_silu_f32(x: f32) -> f32 {
    x * activation_sigmoid_f32(x)
}

/// `gelu(x)` via the tanh approximation, matching `School.Nn.Gelu.gelu_scalar`:
///
///   gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
///
/// We use the tanh-approx (not the erf-exact form) because the
/// C-backend host helper composes the same way and the Std layer is
/// the canonical reference. If/when a `Erf` RISC op is added the
/// exact form can replace this and both lanes must move together.
pub(super) fn activation_gelu_f32(x: f32) -> f32 {
    // The literal is the f64 value that `School.Nn.Gelu` and the C-backend
    // helper (`0.7978845608028654f` in host_emit.rs) both encode; the
    // explicit cast keeps the f32 round-trip identical to those lanes.
    // `clippy::excessive_precision` complains about the trailing digits
    // being beyond f32 representability — that's intentional (we want
    // the same source-level constant the other lanes use).
    #[allow(clippy::excessive_precision)]
    const C: f32 = 0.7978845608028654_f32; // sqrt(2/pi)
    const K: f32 = 0.044715_f32;
    let inner = C * (x + K * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

fn runtime_scalar_as_f64(value: &RuntimeValue) -> Option<f64> {
    match value {
        RuntimeValue::Scalar(payload) => Some(payload.bits().as_f64()),
        RuntimeValue::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        _ => None,
    }
}

pub(super) fn int_binop(
    args: &[RuntimeValue],
    op: impl Fn(i64, i64) -> i64,
) -> Result<RuntimeValue, String> {
    match (args.first(), args.get(1)) {
        (Some(RuntimeValue::Scalar(lp)), Some(RuntimeValue::Scalar(rp)))
            if lp.dtype().is_integer() && rp.dtype().is_integer() =>
        {
            // Pre-WS-A0 stored every int as i64; preserve i64-precision
            // arithmetic but pin the result dtype to the operand dtype
            // when both sides agree, else widen to int64. Matches the
            // §5.1 "no implicit precision promotion" rule for matched
            // operands, and fails closed for mixed widths.
            let (ldt, rdt) = (lp.dtype(), rp.dtype());
            let result_dtype = if ldt == rdt { ldt } else { Prim::Int64 };
            RuntimeValue::scalar_like_int(result_dtype, op(lp.bits().as_i64(), rp.bits().as_i64()))
        }
        other => Err(format!("integer op expects int args, got {other:?}")),
    }
}

/// Canonical evaluator diagnostic for integer division/remainder by zero
/// (#387). `div` and `mod` share one message so the two primitives trap
/// consistently. Per `spec/05-risc-primitives.md` (integer division) a
/// `1 / 0` (or `% 0`) on integer operands must trap rather than yield a
/// silently-wrong finite value; the C backend follows the platform's
/// SIGFPE for the same operands. Returning a clean `Err` halts evaluation
/// with `error: <message>` (exit 1) instead of an unhandled Rust panic.
const INT_DIV_ZERO_MSG: &str = "integer division or remainder by zero";

/// True when every operand resolves to an integer precision (scalar dtype
/// or integer-precision tensor). Integer `div`/`mod` follow C truncating
/// semantics and trap on a zero divisor; float operands keep IEEE-754
/// division (`1.0 / 0.0 == inf`), so the integer trap must not fire there.
fn operand_is_integer(value: &RuntimeValue) -> bool {
    match value {
        RuntimeValue::Scalar(payload) => payload.dtype().is_integer(),
        RuntimeValue::Tensor(tensor) => tensor.precision.is_integer(),
        _ => false,
    }
}

/// `div` evaluator entry: integer operands trap on a zero divisor and use
/// true integer (truncating, round-toward-zero) division; float operands
/// fall through to IEEE-754 `numeric_binop` (`1.0 / 0.0 == inf`). Routing
/// integers through real `i64` division (rather than the f64 round-trip
/// `lhs / rhs as i64`) also preserves the full `int64` range that the f64
/// mantissa would otherwise truncate. See #387.
pub(super) fn eval_div(args: &[RuntimeValue]) -> Result<RuntimeValue, String> {
    let both_integer = args.len() == 2
        && args.iter().all(operand_is_integer)
        && args
            .iter()
            .any(|v| matches!(v, RuntimeValue::Scalar(_)) || matches!(v, RuntimeValue::Tensor(_)));
    if both_integer {
        return checked_int_binop(args, |lhs, rhs| {
            if rhs == 0 {
                Err(INT_DIV_ZERO_MSG.to_string())
            } else {
                Ok(lhs.wrapping_div(rhs))
            }
        });
    }
    numeric_binop(args, |lhs, rhs| lhs / rhs)
}

/// `mod` evaluator entry: integer-only (the surface `mod` primitive),
/// trapping on a zero divisor with the same diagnostic as `eval_div` so
/// the two stay consistent (#387). Uses true integer remainder.
pub(super) fn eval_mod(args: &[RuntimeValue]) -> Result<RuntimeValue, String> {
    checked_int_binop(args, |lhs, rhs| {
        if rhs == 0 {
            Err(INT_DIV_ZERO_MSG.to_string())
        } else {
            Ok(lhs.wrapping_rem(rhs))
        }
    })
}

/// Integer floor division (round toward −∞). The quotient is the
/// truncating `/` corrected down by one when the remainder is nonzero and
/// the operands have opposite signs. Matches Python `//` / the C-backend
/// remainder-sign correction. Traps on a zero divisor. See chelis#178.
fn floor_div_i64(lhs: i64, rhs: i64) -> Result<i64, String> {
    if rhs == 0 {
        return Err(INT_DIV_ZERO_MSG.to_string());
    }
    let q = lhs.wrapping_div(rhs);
    let r = lhs.wrapping_rem(rhs);
    if r != 0 && ((r < 0) != (rhs < 0)) {
        Ok(q - 1)
    } else {
        Ok(q)
    }
}

/// `floor_div` evaluator entry (chelis#178): integer operands round the
/// quotient toward −∞ (and trap on a zero divisor); float operands compute
/// `floor(a / b)` under IEEE division (a zero divisor follows IEEE,
/// `floor(+inf) == +inf`, never traps).
pub(super) fn eval_floor_div(args: &[RuntimeValue]) -> Result<RuntimeValue, String> {
    let both_integer = args.len() == 2 && args.iter().all(operand_is_integer);
    if both_integer {
        return checked_int_binop(args, floor_div_i64);
    }
    numeric_binop(args, |lhs, rhs| (lhs / rhs).floor())
}

/// `trunc_div` evaluator entry (chelis#178): integer-only truncating
/// (round-toward-zero) division — the C/Rust integer `/` quotient. Traps
/// on a zero divisor with the shared diagnostic. The type checker rejects
/// float operands; this entry handles the integer (scalar/tensor) lanes.
pub(super) fn eval_trunc_div(args: &[RuntimeValue]) -> Result<RuntimeValue, String> {
    checked_int_binop(args, |lhs, rhs| {
        if rhs == 0 {
            Err(INT_DIV_ZERO_MSG.to_string())
        } else {
            Ok(lhs.wrapping_div(rhs))
        }
    })
}

/// Integer binop helper whose closure may fail (the failing path is the
/// zero-divisor trap). Handles integer scalars and integer-precision
/// tensors element-wise; the closure runs at `i64` precision. A tensor
/// result keeps the operand precision. Mixed scalar/tensor integer forms
/// broadcast the scalar across the tensor, mirroring `numeric_binop`.
fn checked_int_binop(
    args: &[RuntimeValue],
    op: impl Fn(i64, i64) -> Result<i64, String>,
) -> Result<RuntimeValue, String> {
    match (args.first(), args.get(1)) {
        (Some(RuntimeValue::Scalar(lp)), Some(RuntimeValue::Scalar(rp)))
            if lp.dtype().is_integer() && rp.dtype().is_integer() =>
        {
            let (ldt, rdt) = (lp.dtype(), rp.dtype());
            let result_dtype = if ldt == rdt { ldt } else { Prim::Int64 };
            let value = op(lp.bits().as_i64(), rp.bits().as_i64())?;
            RuntimeValue::scalar_like_int(result_dtype, value)
        }
        (Some(RuntimeValue::Tensor(lhs)), Some(RuntimeValue::Tensor(rhs)))
            if lhs.precision.is_integer() && rhs.precision.is_integer() =>
        {
            if lhs.value.shape != rhs.value.shape {
                return Err(format!(
                    "tensor shapes must match for elementwise op, got {:?} vs {:?}",
                    lhs.value.shape, rhs.value.shape
                ));
            }
            let mut data = Vec::with_capacity(lhs.value.data.len());
            for (l, r) in lhs.value.data.iter().zip(&rhs.value.data) {
                data.push(op(*l as i64, *r as i64)? as f64);
            }
            Ok(RuntimeValue::Tensor(RuntimeTensorValue {
                value: IrTensorValue::from_vec(lhs.value.shape.clone(), data),
                precision: lhs.precision,
            }))
        }
        (Some(RuntimeValue::Tensor(lhs)), Some(RuntimeValue::Scalar(rp)))
            if lhs.precision.is_integer() && rp.dtype().is_integer() =>
        {
            let rhs = rp.bits().as_i64();
            let mut data = Vec::with_capacity(lhs.value.data.len());
            for l in &lhs.value.data {
                data.push(op(*l as i64, rhs)? as f64);
            }
            Ok(RuntimeValue::Tensor(RuntimeTensorValue {
                value: IrTensorValue::from_vec(lhs.value.shape.clone(), data),
                precision: lhs.precision,
            }))
        }
        (Some(RuntimeValue::Scalar(lp)), Some(RuntimeValue::Tensor(rhs)))
            if lp.dtype().is_integer() && rhs.precision.is_integer() =>
        {
            let lhs = lp.bits().as_i64();
            let mut data = Vec::with_capacity(rhs.value.data.len());
            for r in &rhs.value.data {
                data.push(op(lhs, *r as i64)? as f64);
            }
            Ok(RuntimeValue::Tensor(RuntimeTensorValue {
                value: IrTensorValue::from_vec(rhs.value.shape.clone(), data),
                precision: rhs.precision,
            }))
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
            let rhs = rp.bits().as_i64();
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
            let lhs = lp.bits().as_i64();
            let value = if rhs >= i64::from(width) {
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
            RuntimeValue::scalar_like_int(lp.dtype(), value)
        }
        other => Err(format!("shift op expects int args, got {other:?}")),
    }
}

fn float_unop(args: &[RuntimeValue], op: impl Fn(f64) -> f64) -> Result<RuntimeValue, String> {
    match args.first() {
        Some(RuntimeValue::Scalar(payload)) if payload.dtype().is_float() => {
            RuntimeValue::scalar_like_float(payload.dtype(), op(payload.bits().as_f64()))
        }
        other => Err(format!("float op expects float arg, got {other:?}")),
    }
}

/// Float unary that accepts both scalar floats and tensors. Scalar args
/// run in `f64` via `scalar_op` (matching the C backend's libm `sqrt`/
/// `exp`/`log`/`sin` scalar emit in `host_emit.rs`). Tensor args run
/// elementwise in `f32` via `tensor_op` (matching the C backend's
/// `expf`/`logf`/`sinf`/`sqrtf` tensor emit and the activation-block
/// `f32` parity rule above).
pub(super) fn float_unop_with_tensor(
    args: &[RuntimeValue],
    scalar_op: impl Fn(f64) -> f64,
    tensor_op: impl Fn(f32) -> f32,
) -> Result<RuntimeValue, String> {
    match args.first() {
        Some(RuntimeValue::Tensor(tensor)) => Ok(RuntimeValue::Tensor(tensor_float_unop_f32(
            tensor, tensor_op,
        ))),
        _ => float_unop(args, scalar_op),
    }
}

/// Precision-aware fallible float unary for the EVAL-ONLY special
/// functions (chelis#902: `erf`/`erfc`/`norm_cdf`/`norm_ppf`).
///
/// Scalar floats compute in `f64` and re-pack at the operand's dtype
/// (the `round_to` precision-preserving rule). Tensor args run
/// elementwise: an `f64`-tagged tensor keeps the full `f64` result,
/// while narrower float tags round the `f64` result through `f32`
/// (their stored elements are already `f32`-quantized, so this is the
/// correctly-rounded value at the tensor's precision). This deliberately
/// differs from `float_unop_with_tensor`'s all-`f32` tensor rule: that
/// rule exists to stay byte-identical with the C backend's `expf`-style
/// tensor emit, and these builtins have no C emit to mirror — the build
/// lane rejects them (`EVAL_ONLY_HOST_BUILTINS`) and full `f64` tail
/// accuracy is their entire purpose.
///
/// The op is fallible so a per-element domain error (e.g. `norm_ppf`
/// outside [0, 1]) surfaces loudly with the failing element's index
/// instead of a silent NaN.
pub(super) fn special_float_unop(
    name: &str,
    args: &[RuntimeValue],
    op: impl Fn(f64) -> Result<f64, String>,
) -> Result<RuntimeValue, String> {
    if args.len() != 1 {
        return Err(format!("{name} expects 1 argument, got {}", args.len()));
    }
    match args.first() {
        Some(RuntimeValue::Tensor(tensor)) => {
            if !tensor.precision.is_float() {
                return Err(format!(
                    "{name} expects a float tensor, got precision {}",
                    tensor.precision.name()
                ));
            }
            let keep_f64 = tensor.precision == Prim::F64;
            let mut data = Vec::with_capacity(tensor.value.data.len());
            for (index, value) in tensor.value.data.iter().enumerate() {
                let out =
                    op(*value).map_err(|err| format!("{name}: tensor element {index}: {err}"))?;
                data.push(if keep_f64 { out } else { out as f32 as f64 });
            }
            Ok(RuntimeValue::Tensor(RuntimeTensorValue {
                value: IrTensorValue::from_vec(tensor.value.shape.clone(), data),
                precision: tensor.precision,
            }))
        }
        Some(RuntimeValue::Scalar(payload)) if payload.dtype().is_float() => {
            let out = op(payload.bits().as_f64())?;
            RuntimeValue::scalar_like_float(payload.dtype(), out)
        }
        other => Err(format!(
            "{name} expects a float scalar or float tensor argument, got {other:?}"
        )),
    }
}

/// Coerce a scalar `RuntimeValue` to its `f64` representation for
/// comparison with a tensor element. Returns `None` for non-scalar values.
fn scalar_as_f64(value: &RuntimeValue) -> Option<f64> {
    runtime_scalar_as_f64(value)
}

pub(super) fn compare_eq(args: &[RuntimeValue]) -> Result<RuntimeValue, String> {
    match (args.first(), args.get(1)) {
        (Some(RuntimeValue::Scalar(lp)), Some(RuntimeValue::Scalar(rp)))
            if lp.dtype().is_integer() && rp.dtype().is_integer() =>
        {
            Ok(RuntimeValue::Bool(lp.bits().as_i64() == rp.bits().as_i64()))
        }
        (Some(RuntimeValue::Scalar(lp)), Some(RuntimeValue::Scalar(rp)))
            if lp.dtype().is_float() && rp.dtype().is_float() =>
        {
            Ok(RuntimeValue::Bool(lp.bits().as_f64() == rp.bits().as_f64()))
        }
        (Some(RuntimeValue::Bool(lhs)), Some(RuntimeValue::Bool(rhs))) => {
            Ok(RuntimeValue::Bool(lhs == rhs))
        }
        (Some(RuntimeValue::String(lhs)), Some(RuntimeValue::String(rhs))) => {
            Ok(RuntimeValue::Bool(lhs == rhs))
        }
        // Element-wise tensor-tensor equality. The build-target lane already
        // supports this; the host evaluator was returning an error, blocking
        // IntCol/BoolCol construction and tensor-level is_nan in chelis test.
        (Some(RuntimeValue::Tensor(lhs)), Some(RuntimeValue::Tensor(rhs))) => {
            tensor_compare_value(lhs, rhs, |a, b| a == b).map(RuntimeValue::Tensor)
        }
        // Element-wise tensor-scalar equality: broadcast the scalar across
        // every element. Mirrors the build-target lane and unblocks
        // `is_nan_local`-style scalar comparisons against a tensor.
        (Some(RuntimeValue::Tensor(tensor)), Some(scalar)) if scalar_as_f64(scalar).is_some() => {
            let scalar_f = scalar_as_f64(scalar).expect("scalar guard");
            tensor_compare_scalar(tensor, scalar_f, |a, b| a == b).map(RuntimeValue::Tensor)
        }
        (Some(scalar), Some(RuntimeValue::Tensor(tensor))) if scalar_as_f64(scalar).is_some() => {
            let scalar_f = scalar_as_f64(scalar).expect("scalar guard");
            tensor_compare_scalar(tensor, scalar_f, |a, b| a == b).map(RuntimeValue::Tensor)
        }
        other => Err(format!("eq/neq expect matching scalar args, got {other:?}")),
    }
}

pub(super) fn ordered_compare(
    args: &[RuntimeValue],
    cmp: impl Fn(f64, f64) -> bool,
) -> Result<RuntimeValue, String> {
    match (args.first(), args.get(1)) {
        (Some(RuntimeValue::Scalar(lp)), Some(RuntimeValue::Scalar(rp)))
            if lp.dtype().is_integer() && rp.dtype().is_integer() =>
        {
            Ok(RuntimeValue::Bool(cmp(
                lp.bits().as_f64(),
                rp.bits().as_f64(),
            )))
        }
        (Some(RuntimeValue::Scalar(lp)), Some(RuntimeValue::Scalar(rp)))
            if lp.dtype().is_float() && rp.dtype().is_float() =>
        {
            Ok(RuntimeValue::Bool(cmp(
                lp.bits().as_f64(),
                rp.bits().as_f64(),
            )))
        }
        // Element-wise tensor-tensor ordering. Mirrors the build-target lane
        // and unblocks the same downstream tensor-level boolean ops.
        (Some(RuntimeValue::Tensor(lhs)), Some(RuntimeValue::Tensor(rhs))) => {
            tensor_compare_value(lhs, rhs, cmp).map(RuntimeValue::Tensor)
        }
        // Element-wise tensor-scalar ordering: broadcast the scalar across
        // every element. The result is a `tensor[D, bool]` mask.
        (Some(RuntimeValue::Tensor(tensor)), Some(scalar)) if scalar_as_f64(scalar).is_some() => {
            let scalar_f = scalar_as_f64(scalar).expect("scalar guard");
            tensor_compare_scalar(tensor, scalar_f, cmp).map(RuntimeValue::Tensor)
        }
        (Some(scalar), Some(RuntimeValue::Tensor(tensor))) if scalar_as_f64(scalar).is_some() => {
            // `cmp(scalar, tensor[i])` — flip the comparator so the helper
            // can keep using `cmp(tensor[i], scalar)` internally.
            let scalar_f = scalar_as_f64(scalar).expect("scalar guard");
            tensor_compare_scalar(tensor, scalar_f, |t, s| cmp(s, t)).map(RuntimeValue::Tensor)
        }
        other => Err(format!(
            "ordered comparison expects matching numeric args, got {other:?}"
        )),
    }
}

pub(super) fn tensor_compare_value(
    lhs: &RuntimeTensorValue,
    rhs: &RuntimeTensorValue,
    cmp: impl Fn(f64, f64) -> bool,
) -> Result<RuntimeTensorValue, String> {
    if lhs.precision != rhs.precision {
        return Err("tensor comparison expects matching tensor precision".to_string());
    }
    if lhs.value.shape != rhs.value.shape {
        return Err("tensor comparison expects matching tensor shape".to_string());
    }
    let data = lhs
        .value
        .data
        .iter()
        .zip(&rhs.value.data)
        .map(|(lhs, rhs)| if cmp(*lhs, *rhs) { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(lhs.value.shape.clone(), data),
        precision: Prim::Bool,
    })
}

/// Element-wise tensor-vs-scalar comparison. The scalar is broadcast across
/// every element of the tensor and the result is a `tensor[D, bool]` mask
/// with the same shape as the input tensor. The comparator is invoked as
/// `cmp(tensor_element, scalar)`; callers passing the scalar as the lhs
/// should pre-flip the comparator.
fn tensor_compare_scalar(
    tensor: &RuntimeTensorValue,
    scalar: f64,
    cmp: impl Fn(f64, f64) -> bool,
) -> Result<RuntimeTensorValue, String> {
    let data = tensor
        .value
        .data
        .iter()
        .map(|element| if cmp(*element, scalar) { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(tensor.value.shape.clone(), data),
        precision: Prim::Bool,
    })
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
        .data
        .iter()
        .zip(&rhs.value.data)
        .map(|(l, r)| if op(*l != 0.0, *r != 0.0) { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(lhs.value.shape.clone(), data),
        precision: Prim::Bool,
    })
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
        .data
        .iter()
        .map(|value| if op(*value != 0.0) { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(tensor.value.shape.clone(), data),
        precision: Prim::Bool,
    })
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
        Some(RuntimeValue::Scalar(payload)) if payload.dtype().is_integer() => {
            Ok(payload.bits().as_i64())
        }
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
            Ok(payload.bits().as_f64())
        }
        Some(RuntimeValue::Scalar(payload)) if payload.dtype().is_integer() => {
            Ok(payload.bits().as_f64())
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
            "dict keys must be int64 or string in 3d, got {other:?}"
        )),
    }
}

pub(super) fn runtime_value_eq(lhs: &RuntimeValue, rhs: &RuntimeValue) -> bool {
    match (lhs, rhs) {
        (RuntimeValue::Scalar(lp), RuntimeValue::Scalar(rp))
            if lp.dtype().is_integer() && rp.dtype().is_integer() =>
        {
            lp.bits().as_i64() == rp.bits().as_i64()
        }
        (RuntimeValue::Scalar(lp), RuntimeValue::Scalar(rp))
            if lp.dtype().is_float() && rp.dtype().is_float() =>
        {
            lp.bits().as_f64() == rp.bits().as_f64()
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
/// Storage for `IrTensorValue::data` is always `Vec<f64>` regardless of
/// the logical tensor precision. Float<->float casts where the storage
/// already covers both ranges (any `Prim::F64` source, or any `Prim::F32`
/// source widening to `Prim::F64`) are identity at the data level. The
/// `Prim::F64 -> Prim::F32` narrowing case rounds through `(x as f32) as
/// f64` so the runtime honors the precision loss honestly. Integer
/// targets truncate toward zero, matching the scalar arms above and
/// `(int32_t)f` in the C backend. The output shape is preserved
/// element-for-element (C8 in `crates/chelis-ir/src/verify.rs`: cast dims
/// must not change).
pub(super) fn cast_tensor_value(
    tensor: RuntimeTensorValue,
    target: &str,
) -> Result<RuntimeValue, String> {
    let target_prim = Prim::parse_name(target)
        .ok_or_else(|| format!("unsupported cast target `{target}` for tensor input"))?;
    let RuntimeTensorValue {
        value: ir_value,
        precision: src_prim,
    } = tensor;
    let IrTensorValue { data, shape } = ir_value;
    let converted: Vec<f64> = data
        .into_iter()
        .map(|x| convert_scalar_data(x, src_prim, target_prim))
        .collect();
    Ok(RuntimeValue::Tensor(RuntimeTensorValue {
        value: IrTensorValue::from_vec(shape, converted),
        precision: target_prim,
    }))
}

/// Element-wise scalar conversion for `cast_tensor_value`. Returns the
/// converted value in `f64` storage. Float-to-float narrowing rounds
/// through f32 to drop precision; float-to-int truncates; int-to-float
/// preserves value; bool encodes as 0.0 / 1.0 and decodes via `!= 0.0`.
fn convert_scalar_data(x: f64, src: Prim, dst: Prim) -> f64 {
    if src == dst {
        return x;
    }
    // Step 1: project the source storage into a normalized representation.
    // Floats stay floats; ints route through i64; bools route through 0/1.
    let as_int: Option<i64> = match src {
        Prim::Int8 | Prim::Int32 | Prim::Int64 => Some(x as i64),
        Prim::Bool => Some(if x != 0.0 { 1 } else { 0 }),
        _ => None,
    };
    // Step 2: emit the value in the target precision's storage convention.
    match dst {
        Prim::F64 => match as_int {
            Some(i) => i as f64,
            None => x,
        },
        Prim::F32 => match as_int {
            Some(i) => (i as f32) as f64,
            None => (x as f32) as f64,
        },
        Prim::Int8 => match as_int {
            Some(i) => (i as i8) as f64,
            None => (x as i8) as f64,
        },
        Prim::Int32 => match as_int {
            Some(i) => (i as i32) as f64,
            None => (x as i32) as f64,
        },
        Prim::Int64 => match as_int {
            Some(i) => i as f64,
            None => (x as i64) as f64,
        },
        Prim::Bool => match as_int {
            Some(i) => {
                if i != 0 {
                    1.0
                } else {
                    0.0
                }
            }
            None => {
                if x != 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        },
        // Reduced floats and `string` are not valid tensor element types
        // (see `Prim::is_valid_tensor_precision`); the checker rejects
        // them before this point. Fall back to identity rather than
        // silently corrupting data.
        _ => x,
    }
}

/// Bucket 4b: recursively flatten a nested numeric/bool list into a
/// rank-N tensor. Every nesting level contributes one outer dimension;
/// the innermost level must be uniformly numeric or bool. All sibling
/// sub-lists at the same level must have matching length and matching
/// precision.
///
/// Returns `(precision, shape, flat_data)`. Empty outer lists fall
/// back to an `[0]` shape with `Prim::F32` (matching the rank-1 path
/// behaviour for compatibility).
pub(super) fn nested_list_to_tensor_data(
    outer: &[RuntimeValue],
) -> Result<(Prim, Vec<usize>, Vec<f64>), String> {
    if outer.is_empty() {
        return Ok((Prim::F32, vec![0], Vec::new()));
    }

    // Decide whether this is a leaf level (numeric/bool elements) or a
    // recursive level (List elements) based on the first element. The
    // homogeneity check below catches the mixed case.
    let first_is_list = matches!(&outer[0], RuntimeValue::List(_));

    if !first_is_list {
        // Leaf level — same code path as the original list_to_tensor.
        let (precision, data) = list_to_tensor_data(outer)?;
        return Ok((precision, vec![data.len()], data));
    }

    let mut precision: Option<Prim> = None;
    let mut inner_shape: Option<Vec<usize>> = None;
    let mut data = Vec::new();
    for (idx, value) in outer.iter().enumerate() {
        let RuntimeValue::List(inner) = value else {
            return Err(format!(
                "to_tensor expects homogeneous nested lists; element {idx} is not a List"
            ));
        };
        let (sub_precision, sub_shape, sub_data) = nested_list_to_tensor_data(inner)?;
        match &precision {
            None => precision = Some(sub_precision),
            Some(p) if *p == sub_precision => {}
            Some(p) => {
                return Err(format!(
                    "to_tensor requires homogeneous numeric or bool elements; expected {p:?}, got {sub_precision:?} at element {idx}"
                ));
            }
        }
        match &inner_shape {
            None => inner_shape = Some(sub_shape),
            Some(s) if *s == sub_shape => {}
            Some(s) => {
                return Err(format!(
                    "to_tensor requires uniform inner shape; expected {s:?}, got {sub_shape:?} at element {idx}"
                ));
            }
        }
        data.extend(sub_data);
    }

    let mut shape = vec![outer.len()];
    shape.extend(inner_shape.unwrap_or_default());
    Ok((precision.unwrap_or(Prim::F32), shape, data))
}

fn list_to_tensor_data(values: &[RuntimeValue]) -> Result<(Prim, Vec<f64>), String> {
    // Element classification: integer scalars → Int64-precision tensor;
    // float scalars → F32-precision tensor; bools → Bool tensor. The
    // homogeneity check below pins the precision to whatever the first
    // typed element advertised.
    let mut precision = None;
    let mut data = Vec::with_capacity(values.len());
    for value in values {
        match value {
            RuntimeValue::Scalar(payload) if payload.dtype().is_integer() => {
                precision.get_or_insert(Prim::Int64);
                if precision != Some(Prim::Int64) {
                    return Err(
                        "to_tensor requires homogeneous numeric or bool list elements".to_string(),
                    );
                }
                data.push(payload.bits().as_f64());
            }
            RuntimeValue::Scalar(payload) if payload.dtype().is_float() => {
                precision.get_or_insert(Prim::F32);
                if precision != Some(Prim::F32) {
                    return Err(
                        "to_tensor requires homogeneous numeric or bool list elements".to_string(),
                    );
                }
                data.push(payload.bits().as_f64());
            }
            RuntimeValue::Bool(value) => {
                precision.get_or_insert(Prim::Bool);
                if precision != Some(Prim::Bool) {
                    return Err(
                        "to_tensor requires homogeneous numeric or bool list elements".to_string(),
                    );
                }
                data.push(if *value { 1.0 } else { 0.0 });
            }
            other => {
                return Err(format!(
                    "to_tensor expects numeric or bool list elements, got {other:?}"
                ));
            }
        }
    }
    Ok((precision.unwrap_or(Prim::F32), data))
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
    let mut values = Vec::with_capacity(tensor.value.data.len());
    for value in &tensor.value.data {
        let element = match tensor.precision {
            Prim::Bool => RuntimeValue::Bool(*value != 0.0),
            p if p.is_integer() => RuntimeValue::scalar_like_int(p, *value as i64)?,
            p if p.is_float() => RuntimeValue::scalar_like_float(p, *value)?,
            other => {
                return Err(format!(
                    "to_list expects numeric or bool tensor input, got {other:?}"
                ));
            }
        };
        values.push(element);
    }
    Ok(values)
}

pub(super) fn pad_sequences_value(
    sequences: &[RuntimeValue],
    pad: &RuntimeValue,
) -> Result<(Prim, Vec<f64>, usize, usize), String> {
    let (pad_precision, pad_value) = match pad {
        RuntimeValue::Scalar(payload) if payload.dtype().is_integer() => {
            (Prim::Int64, payload.bits().as_f64())
        }
        RuntimeValue::Scalar(payload) if payload.dtype().is_float() => {
            (Prim::F32, payload.bits().as_f64())
        }
        other => {
            return Err(format!(
                "pad_sequences expects numeric pad value, got {other:?}"
            ));
        }
    };
    let mut rows = Vec::<Vec<f64>>::with_capacity(sequences.len());
    let mut width = 0usize;
    for sequence in sequences {
        let RuntimeValue::List(items) = sequence else {
            return Err(format!(
                "pad_sequences expects nested lists, got {sequence:?}"
            ));
        };
        let (row_precision, row) = list_to_tensor_data(items)?;
        if row_precision != pad_precision {
            return Err("pad_sequences requires homogeneous numeric nested lists".to_string());
        }
        width = width.max(row.len());
        rows.push(row);
    }
    let batch = rows.len();
    let mut data = Vec::with_capacity(batch * width);
    for row in rows {
        data.extend(row.iter().copied());
        data.extend(std::iter::repeat_n(
            pad_value,
            width.saturating_sub(row.len()),
        ));
    }
    Ok((pad_precision, data, batch, width))
}

pub(super) fn pad_sequences_to_value(
    sequences: &[RuntimeValue],
    width: i64,
    pad: &RuntimeValue,
) -> Result<(Prim, Vec<f64>, usize), String> {
    if width < 0 {
        return Err(format!(
            "pad_sequences_to requires non-negative width, got {width}"
        ));
    }
    let (pad_precision, pad_value) = match pad {
        RuntimeValue::Scalar(payload) if payload.dtype().is_integer() => {
            (Prim::Int64, payload.bits().as_f64())
        }
        RuntimeValue::Scalar(payload) if payload.dtype().is_float() => {
            (Prim::F32, payload.bits().as_f64())
        }
        other => {
            return Err(format!(
                "pad_sequences_to expects numeric pad value, got {other:?}"
            ));
        }
    };
    let width = width as usize;
    let mut rows = Vec::<Vec<f64>>::with_capacity(sequences.len());
    for sequence in sequences {
        let RuntimeValue::List(items) = sequence else {
            return Err(format!(
                "pad_sequences_to expects nested lists, got {sequence:?}"
            ));
        };
        let (row_precision, row) = list_to_tensor_data(items)?;
        if row_precision != pad_precision {
            return Err("pad_sequences_to requires homogeneous numeric nested lists".to_string());
        }
        rows.push(row);
    }
    let batch = rows.len();
    let mut data = Vec::with_capacity(batch * width);
    for row in rows {
        let used = row.len().min(width);
        data.extend(row.into_iter().take(used));
        data.extend(std::iter::repeat_n(pad_value, width.saturating_sub(used)));
    }
    Ok((pad_precision, data, batch))
}

fn tensor_numel(shape: &[usize]) -> usize {
    shape.iter().product::<usize>().max(1)
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
                let v = payload.bits().as_i64();
                if v >= 0 {
                    Ok(v as usize)
                } else {
                    Err(format!("{op} expects non-negative sizes, got {v}"))
                }
            }
            other => Err(format!("{op} expects int64 sizes, got {other:?}")),
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
    let mut out = vec![0.0_f64; out_numel];
    let sum_in_f32 = matches!(op, ReduceOp::Sum) && tensor.precision == Prim::F32;
    #[allow(clippy::needless_range_loop)]
    for out_linear in 0..out_numel {
        let out_indices = linear_to_indices(out_linear, &out_shape);
        // Stride-4 ILP cascade lanes for Sum (issue #163, parity with
        // torch's CPU `row_sum` at n <= 16). Other reductions keep a
        // single accumulator since they're either associative
        // (Min/Prod) or position-tracking (Argmax/Argmin).
        let mut sum_lanes = [0.0_f64; 4];
        let mut sum_lanes_f32 = [0.0_f32; 4];
        let mut best_value = match op {
            ReduceOp::Sum => 0.0,
            ReduceOp::Min => f64::INFINITY,
            ReduceOp::Max => f64::NEG_INFINITY,
            ReduceOp::Prod => 1.0,
            ReduceOp::Argmax => f64::NEG_INFINITY,
            ReduceOp::Argmin => f64::INFINITY,
        };
        let mut best_index: usize = 0;
        // #172: Min/Max must PROPAGATE NaN to match torch (`torch.max`/
        // `torch.min` of any slice containing NaN return NaN, at every
        // position). A naive `value > best` / `value < best` silently
        // DROPS NaN (all NaN comparisons are false), which made the C
        // backend's SIMD `chelis_max_f32` position-dependent and diverged
        // from torch on both lanes. Track whether any NaN was seen and
        // force the Min/Max result to NaN if so. (Prod already propagates
        // via `*=`; Sum has its own IEEE accumulation; Argmax/Argmin index
        // semantics are unchanged.)
        let mut saw_nan = false;
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
            let in_linear = indices_to_linear(&in_indices, &tensor.value.shape);
            let value = tensor.value.data[in_linear];
            if value.is_nan() {
                saw_nan = true;
            }
            match op {
                ReduceOp::Sum => {
                    if sum_in_f32 {
                        sum_lanes_f32[k & 3] += value as f32;
                    } else {
                        sum_lanes[k & 3] += value;
                    }
                }
                ReduceOp::Min => {
                    if value < best_value {
                        best_value = value;
                    }
                }
                ReduceOp::Max => {
                    if value > best_value {
                        best_value = value;
                    }
                }
                ReduceOp::Prod => {
                    best_value *= value;
                }
                ReduceOp::Argmax => {
                    if value > best_value {
                        best_value = value;
                        best_index = k;
                    }
                }
                ReduceOp::Argmin => {
                    if value < best_value {
                        best_value = value;
                        best_index = k;
                    }
                }
            }
        }
        out[out_linear] = match op {
            ReduceOp::Sum => {
                if sum_in_f32 {
                    ((sum_lanes_f32[0] + sum_lanes_f32[1]) + (sum_lanes_f32[2] + sum_lanes_f32[3]))
                        as f64
                } else {
                    (sum_lanes[0] + sum_lanes[1]) + (sum_lanes[2] + sum_lanes[3])
                }
            }
            // #172: Min/Max propagate NaN (torch parity). Prod already
            // propagates through `best_value *= NaN`.
            ReduceOp::Min | ReduceOp::Max if saw_nan => f64::NAN,
            ReduceOp::Min | ReduceOp::Max | ReduceOp::Prod => best_value,
            // Argmax/Argmin: write the integer index into the f64 storage
            // slot. The surrounding `precision` tag is `Prim::Int64`
            // (set below per chelis#233), so downstream consumers read
            // these slots back as int64 scalars.
            ReduceOp::Argmax | ReduceOp::Argmin => best_index as f64,
        };
    }
    // chelis#233: argmax_reduce / argmin_reduce return integer indices,
    // not reduced operand values, so the storage precision must be
    // `Prim::Int64` regardless of the input precision. This matches the
    // type-system label widened in #230 and prevents downstream
    // primitives that branch on `RuntimeTensorValue::precision` (`eq`,
    // `to_list`, `tensor_to_scalar`) from misclassifying the result as
    // the input's float dtype. The other reductions return values at
    // the input dtype and preserve `tensor.precision`.
    let out_precision = match op {
        ReduceOp::Argmax | ReduceOp::Argmin => Prim::Int64,
        ReduceOp::Sum | ReduceOp::Min | ReduceOp::Max | ReduceOp::Prod => tensor.precision,
    };
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(out_shape, out),
        precision: out_precision,
    })
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
    let mut out = vec![0.0_f64; out_numel];
    for in_linear in 0..tensor.value.data.len() {
        let in_indices = linear_to_indices(in_linear, &in_shape);
        let out_indices: Vec<usize> = axes.iter().map(|&a| in_indices[a]).collect();
        let out_linear = indices_to_linear(&out_indices, &out_shape);
        out[out_linear] = tensor.value.data[in_linear];
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(out_shape, out),
        precision: tensor.precision,
    })
}

/// 2D matmul: lhs is [m, k], rhs is [k, n], output is [m, n].
pub(super) fn tensor_matmul_host(
    lhs: &RuntimeTensorValue,
    rhs: &RuntimeTensorValue,
) -> Result<RuntimeTensorValue, String> {
    if lhs.value.shape.len() != 2 || rhs.value.shape.len() != 2 {
        return Err(format!(
            "matmul host runtime currently supports only rank-2 × rank-2; got ranks {} and {}",
            lhs.value.shape.len(),
            rhs.value.shape.len()
        ));
    }
    let m = lhs.value.shape[0];
    let k_lhs = lhs.value.shape[1];
    let k_rhs = rhs.value.shape[0];
    let n = rhs.value.shape[1];
    if k_lhs != k_rhs {
        return Err(format!(
            "matmul shared-axis mismatch: lhs has {k_lhs}, rhs has {k_rhs}"
        ));
    }
    // #170 (DO NOT "fix" this into the stride-4 cascade): matmul does NOT
    // take the #163 `sum` cascade, and its f64 accumulator is intentional.
    // torch's CPU f32 matmul is a BLAS GEMM whose rounding is bit-exact
    // with a strict-f32 left-fold (verified k=20..257), NOT the cascade
    // (which is `sum`'s order — applying it here would CREATE a k>=128
    // divergence). The eval reference deliberately keeps a HIGHER-precision
    // f64 accumulator: it is the reference, the shipped C backend trades
    // precision for speed via `cblas_sgemm`, and the matmul eval-vs-C
    // parity tests use a TOLERANCE (not bit-identity) for exactly this
    // expected eval(f64)-vs-backend(BLAS) gap. Matching torch's f32-GEMM
    // bit pattern by downcasting eval to strict-f32 would lower precision,
    // couple the reference to torch's specific BLAS version, and still not
    // buy eval-vs-C bit-identity — net worse, no soundness win. So this is
    // a documented, expected precision characteristic, not a divergence.
    let mut out = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for kk in 0..k_lhs {
                acc += lhs.value.data[i * k_lhs + kk] * rhs.value.data[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![m, n], out),
        precision: lhs.precision,
    })
}

/// Replicate a tensor along a new axis.
///
/// Per the typer (`chelis-types::infer::check_expand_signature`),
/// `expand(b, axis, count)` is canonically an INSERT operation: it
/// produces a tensor of shape `[..., count, ...]` with `count` inserted
/// at position `axis`, where every "slice" along the new axis is a copy
/// of `b`. The output rank is always `input_rank + 1`.
///
/// The typer also accepts a same-rank "replace-singleton" interpretation
/// when the user explicitly annotates the result as same-rank, but the
/// host runtime has no access to user annotations, so it always picks
/// the canonical INSERT branch — which is the typer's first-preference
/// branch at infer.rs:7188 and the only branch synthesized by IR
/// lowering in `tier2::lower_softmax`/`lower_layer_norm`/`lower_matmul`.
/// Closes Bucket 4a: previously this function silently picked the
/// same-rank REPLICATE-singleton branch whenever `in_shape[axis] == 1`,
/// producing shape `[count]` for `expand([1], 0, count)` while the typer
/// accepted the `[count, 1]` annotation, leaving `chelis test`/`chelis
/// eval` disagreeing with `chelis check` on `examples/linreg.ch`.
pub(super) fn tensor_expand_host(
    tensor: &RuntimeTensorValue,
    axis: usize,
    count: usize,
) -> Result<RuntimeTensorValue, String> {
    let in_shape = tensor.value.shape.clone();
    let in_rank = in_shape.len();
    if axis > in_rank {
        return Err(format!(
            "expand axis {axis} out of bounds for rank-{in_rank} tensor (insert position must be <= rank)"
        ));
    }

    // INSERT: create a new axis of size `count` at position `axis`.
    let mut out_shape = Vec::with_capacity(in_rank + 1);
    out_shape.extend_from_slice(&in_shape[..axis]);
    out_shape.push(count);
    out_shape.extend_from_slice(&in_shape[axis..]);

    let out_numel = tensor_numel(&out_shape);
    let mut out = vec![0.0_f64; out_numel];
    for (out_linear, slot) in out.iter_mut().enumerate() {
        let out_indices = linear_to_indices(out_linear, &out_shape);
        // Drop the inserted axis to recover the input index.
        let mut in_indices = out_indices;
        in_indices.remove(axis);
        let in_linear = indices_to_linear(&in_indices, &in_shape);
        *slot = tensor.value.data[in_linear];
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(out_shape, out),
        precision: tensor.precision,
    })
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
    let mut out = vec![0.0_f64; out_numel];
    let window_volume: usize = window_shape.iter().product();
    let init_acc = match reducer {
        ReduceWindowOp::Max => f64::NEG_INFINITY,
        ReduceWindowOp::Min => f64::INFINITY,
        ReduceWindowOp::Sum | ReduceWindowOp::Mean => 0.0,
    };

    for (out_flat, slot) in out.iter_mut().enumerate() {
        let out_indices = linear_to_indices(out_flat, &out_shape);
        let mut acc = init_acc;
        let mut window_pos = vec![0usize; n];
        loop {
            let mut src_indices = vec![0usize; in_shape.len()];
            src_indices[..leading].copy_from_slice(&out_indices[..leading]);
            for i in 0..n {
                src_indices[leading + i] = out_indices[leading + i] * strides[i] + window_pos[i];
            }
            let src_linear = indices_to_linear(&src_indices, in_shape);
            let value = tensor.value.data[src_linear];
            // #172 sibling (intentionally NOT NaN-propagating here, mirrors the
            // C-emit note in `chelis-backend-c/src/emit.rs` ~3963): windowed
            // Max/Min use Rust `f64::max`/`f64::min`, which DROP NaN (return
            // the non-NaN operand) — the same NaN-dropping semantics as the C
            // backend's `fmaxf`/`fminf`, so eval and the backend stay
            // CONSISTENT here. The #172 NaN-propagation fix scoped itself to
            // `max_reduce` / `min_reduce`; flipping reduce_window forward
            // without also defining the NaN gradient-routing in the windowed
            // backward would create a fwd/bwd inconsistency. Tracked as a
            // follow-up; reduce_window has its own parity gate (spec §2.3).
            acc = match reducer {
                ReduceWindowOp::Max => acc.max(value),
                ReduceWindowOp::Min => acc.min(value),
                ReduceWindowOp::Sum | ReduceWindowOp::Mean => acc + value,
            };
            // Unreachable: `n == 0` already returned `Err` above (a windowed
            // reduction needs >= 1 windowed axis). Kept to mirror
            // `chelis_ir::eval::reduce_window`, whose internal walk has no
            // such early return and so relies on this guard.
            if n == 0 {
                break;
            }
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
        if matches!(reducer, ReduceWindowOp::Mean) {
            acc /= window_volume as f64;
        }
        *slot = acc;
    }

    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(out_shape, out),
        precision: tensor.precision,
    })
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
    let mut out = vec![fill; out_numel];
    let in_numel = tensor_numel(in_shape);
    for in_linear in 0..in_numel {
        let in_indices = linear_to_indices(in_linear, in_shape);
        let out_indices: Vec<usize> = in_indices
            .iter()
            .zip(padding.iter())
            .map(|(idx, (lo, _))| idx + lo)
            .collect();
        let out_linear = indices_to_linear(&out_indices, &out_shape);
        out[out_linear] = tensor.value.data[in_linear];
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(out_shape, out),
        precision: tensor.precision,
    })
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
    let mut out = vec![0.0_f64; out_numel];
    for (out_linear, slot) in out.iter_mut().enumerate() {
        let out_indices = linear_to_indices(out_linear, &out_shape);
        let in_indices: Vec<usize> = out_indices
            .iter()
            .zip(bounds.iter())
            .map(|(idx, (start, _))| idx + start)
            .collect();
        let in_linear = indices_to_linear(&in_indices, in_shape);
        *slot = tensor.value.data[in_linear];
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(out_shape, out),
        precision: tensor.precision,
    })
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
    let mut out = vec![0.0_f64; out_numel];
    for (out_linear, slot) in out.iter_mut().enumerate() {
        let out_indices = linear_to_indices(out_linear, &out_shape);
        let in_indices: Vec<usize> = out_indices
            .iter()
            .zip(strides.iter())
            .map(|(idx, step)| idx * step.max(&1))
            .collect();
        let in_linear = indices_to_linear(&in_indices, in_shape);
        *slot = tensor.value.data[in_linear];
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(out_shape, out),
        precision: tensor.precision,
    })
}

/// Convert a `RuntimeValue::List` of inner `List`s into a flat
/// `Vec<(usize, usize)>` of `[start, end]` bounds pairs. Each inner list
/// must have exactly two non-negative int entries (matching the
/// type-checker's `List[List[Int32]]` contract). Any other shape -- wrong
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
                        payload.bits().as_i64()
                    }
                    other => {
                        return Err(format!("{op} axis {axis} expects int start, got {other:?}"));
                    }
                };
                let end = match &pair[1] {
                    RuntimeValue::Scalar(payload) if payload.dtype().is_integer() => {
                        payload.bits().as_i64()
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
            let v = tensor.value.data[in_linear];
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
        //     f64 fold reliably bit-matches it (same situation as matmul —
        //     see `tensor_matmul_host`). So softmax is DOCUMENTED, not
        //     cascaded; only `sum`/`trace` take the cascade.
        let mut sum_exp = 0.0_f64;
        for k in 0..axis_size {
            base_indices[axis_usize] = k;
            let in_linear = indices_to_linear(&base_indices, &in_shape);
            sum_exp += (tensor.value.data[in_linear] - max_val).exp();
        }
        if sum_exp == 0.0 {
            return Err("softmax sum-of-exp is zero (numerical underflow)".to_string());
        }
        // Third pass: write exp(x - max) / sum.
        for k in 0..axis_size {
            base_indices[axis_usize] = k;
            let in_linear = indices_to_linear(&base_indices, &in_shape);
            let numer = (tensor.value.data[in_linear] - max_val).exp();
            out[in_linear] = numer / sum_exp;
        }
    }

    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(in_shape, out),
        precision: tensor.precision,
    })
}

// ---------------------------------------------------------------------------
// Composed Tier-2 host-runtime delegation
//
// `mean` / `layer_norm` / `conv2d` are not single RISC ops; they
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
    inputs: &HashMap<String, IrTensorValue>,
    root: NodeId,
    op_label: &str,
) -> Result<RuntimeTensorValue, String> {
    let values = eval_tensor_roots_with(dag, &[root], |name| inputs.get(name).cloned())
        .map_err(|err| format!("{op_label}: IR eval failed: {err}"))?;
    let tensor_value = values
        .get(&root)
        .cloned()
        .ok_or_else(|| format!("{op_label}: IR eval produced no value for root node"))?;
    let precision = dag
        .get(root)
        .map(|node| node.output_type.precision)
        .ok_or_else(|| format!("{op_label}: root node missing from DAG"))?;
    Ok(RuntimeTensorValue {
        value: tensor_value,
        precision,
    })
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
    let mut inputs = HashMap::new();
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
    let mut inputs = HashMap::new();
    inputs.insert(x_name, x.value.clone());
    inputs.insert(g_name, gamma.value.clone());
    inputs.insert(b_name, beta.value.clone());
    extract_root(&dag, &inputs, root, "composed triop tier2")
}

/// conv2d forward in the host runtime. `tier2::lower_conv2d` is shape-
/// polymorphic via its `output_ty` parameter and panics if the spatial
/// dims of `output_ty` disagree with the arithmetic derived from
/// `(input_dims, kernel_dims, stride, padding)`. The host runtime does
/// not have a downstream type-annotation source for the output type,
/// so we compute it inline from the four spatial parameters: that's
/// the same formula `lower_conv2d` reaches for via the
/// `raw_h_out`/`raw_w_out` fallback at
/// `crates/chelis-ir/src/tier2.rs:973-984`.
pub(super) fn conv2d_host(
    input: &RuntimeTensorValue,
    kernel: &RuntimeTensorValue,
    stride: usize,
    padding: usize,
) -> Result<RuntimeTensorValue, String> {
    if input.value.shape.len() != 4 {
        return Err(format!(
            "conv2d input must be rank-4 (batch, channels, h, w), got shape {:?}",
            input.value.shape
        ));
    }
    if kernel.value.shape.len() != 4 {
        return Err(format!(
            "conv2d kernel must be rank-4 (out_c, in_c, kh, kw), got shape {:?}",
            kernel.value.shape
        ));
    }
    let stride = stride.max(1);
    let batch = input.value.shape[0];
    let in_c = input.value.shape[1];
    let h_in = input.value.shape[2];
    let w_in = input.value.shape[3];
    let out_c = kernel.value.shape[0];
    let kernel_in_c = kernel.value.shape[1];
    let kh = kernel.value.shape[2];
    let kw = kernel.value.shape[3];
    if kernel_in_c != in_c {
        return Err(format!(
            "conv2d kernel input channels ({kernel_in_c}) must match input channels ({in_c})"
        ));
    }
    let padded_h = h_in + (2 * padding);
    let padded_w = w_in + (2 * padding);
    if padded_h < kh || padded_w < kw {
        return Err(format!(
            "conv2d kernel dims ({kh}, {kw}) exceed padded input dims ({padded_h}, {padded_w})"
        ));
    }
    let h_out = ((padded_h - kh) / stride) + 1;
    let w_out = ((padded_w - kw) / stride) + 1;
    let output_ty = TensorType {
        dims: vec![
            DimInfo::Lit(batch),
            DimInfo::Lit(out_c),
            DimInfo::Lit(h_out),
            DimInfo::Lit(w_out),
        ],
        precision: input.precision,
    };
    let input_ty = tensor_type_for(input);
    let kernel_ty = tensor_type_for(kernel);
    let mut dag = Dag::new();
    let x_name = format!("{COMPOSED_PLACEHOLDER_PREFIX}0");
    let k_name = format!("{COMPOSED_PLACEHOLDER_PREFIX}1");
    let x_id = add_load(&mut dag, x_name.clone(), input_ty.clone());
    let k_id = add_load(&mut dag, k_name.clone(), kernel_ty.clone());
    let root = tier2::lower_conv2d(
        &mut dag, x_id, k_id, &input_ty, &kernel_ty, &output_ty, stride, padding, None,
    );
    let mut inputs = HashMap::new();
    inputs.insert(x_name, input.value.clone());
    inputs.insert(k_name, kernel.value.clone());
    extract_root(&dag, &inputs, root, "conv2d")
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
    let mut out = vec![0.0; tensor_numel(&out_shape)];
    let mut axis_offset = 0usize;
    for tensor in &tensors {
        for linear in 0..tensor.value.data.len() {
            let mut index = linear_to_indices(linear, &tensor.value.shape);
            index[axis] += axis_offset;
            let out_linear = indices_to_linear(&index, &out_shape);
            out[out_linear] = tensor.value.data[linear];
        }
        axis_offset += tensor.value.shape[axis];
    }
    Ok(RuntimeValue::Tensor(RuntimeTensorValue {
        value: IrTensorValue::from_vec(out_shape, out),
        precision: first.precision,
    }))
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
    if expected != tensor.value.data.len() {
        return Err(format!(
            "reshape expects {} elements but tensor has {}",
            expected,
            tensor.value.data.len()
        ));
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(new_shape, tensor.value.data.clone()),
        precision: tensor.precision,
    })
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
        let mut data = vec![0.0; tensor_numel(&shape)];
        for (linear, slot) in data.iter_mut().enumerate() {
            let mut index = linear_to_indices(linear, &shape);
            index[axis] += offset;
            let src = indices_to_linear(&index, &tensor.value.shape);
            *slot = tensor.value.data[src];
        }
        offset += size;
        parts.push(RuntimeValue::Tensor(RuntimeTensorValue {
            value: IrTensorValue::from_vec(shape, data),
            precision: tensor.precision,
        }));
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
    let mut out = vec![0.0; tensor_numel(&out_shape)];
    for (linear, slot) in out.iter_mut().enumerate() {
        let out_index = linear_to_indices(linear, &out_shape);
        let mut src_index = Vec::with_capacity(tensor.value.shape.len());
        src_index.extend_from_slice(&out_index[..axis]);
        let gathered_idx = &out_index[axis..axis + indices.value.shape.len()];
        let index_linear = indices_to_linear(gathered_idx, &indices.value.shape);
        let value = indices.value.data[index_linear] as i64;
        if value < 0 || value as usize >= tensor.value.shape[axis] {
            return Err(format!("gather index {value} out of bounds at axis {axis}"));
        }
        src_index.push(value as usize);
        src_index.extend_from_slice(&out_index[axis + indices.value.shape.len()..]);
        let src_linear = indices_to_linear(&src_index, &tensor.value.shape);
        *slot = tensor.value.data[src_linear];
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(out_shape, out),
        precision: tensor.precision,
    })
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
    let mut out = base.value.data.clone();
    let mut seen = std::collections::HashSet::new();
    for linear in 0..updates.value.data.len() {
        let update_index = linear_to_indices(linear, &updates.value.shape);
        let mut out_index = Vec::with_capacity(base.value.shape.len());
        out_index.extend_from_slice(&update_index[..axis]);
        let gathered_idx = &update_index[axis..axis + indices.value.shape.len()];
        let index_linear = indices_to_linear(gathered_idx, &indices.value.shape);
        let value = indices.value.data[index_linear] as i64;
        if value < 0 || value as usize >= base.value.shape[axis] {
            return Err(format!(
                "scatter index {value} out of bounds at axis {axis}"
            ));
        }
        out_index.push(value as usize);
        out_index.extend_from_slice(&update_index[axis + indices.value.shape.len()..]);
        let out_linear = indices_to_linear(&out_index, &base.value.shape);
        match mode {
            "replace" => {
                if !seen.insert(out_linear) {
                    return Err(format!(
                        "scatter replace mode rejects duplicate target index {}",
                        out_linear
                    ));
                }
                out[out_linear] = updates.value.data[linear];
            }
            "add" => out[out_linear] += updates.value.data[linear],
            other => return Err(format!("scatter mode must be replace or add, got {other}")),
        }
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(base.value.shape.clone(), out),
        precision: base.precision,
    })
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
    let mut out = data.value.data.clone();
    for linear in 0..updates.value.data.len() {
        let coord = linear_to_indices(linear, &updates.value.shape);
        let value = indices.value.data[linear] as i64;
        if value < 0 || value as usize >= data.value.shape[axis] {
            return Err(format!(
                "scatter_elements index {value} out of bounds at axis {axis}"
            ));
        }
        let mut out_index = coord.clone();
        out_index[axis] = value as usize;
        let out_linear = indices_to_linear(&out_index, &data.value.shape);
        // Last-write-wins: deterministic-order overwrite.
        out[out_linear] = updates.value.data[linear];
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(data.value.shape.clone(), out),
        precision: data.precision,
    })
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
    let data = cond
        .value
        .data
        .iter()
        .zip(&then_tensor.value.data)
        .zip(&else_tensor.value.data)
        .map(|((cond, then_value), else_value)| {
            if *cond != 0.0 {
                *then_value
            } else {
                *else_value
            }
        })
        .collect();
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(then_tensor.value.shape.clone(), data),
        precision: then_tensor.precision,
    })
}

pub(super) fn tensor_cumsum_value(
    tensor: &RuntimeTensorValue,
    axis: i64,
) -> Result<RuntimeTensorValue, String> {
    let axis = normalize_axis(tensor.value.shape.len(), axis, "cumsum")?;
    let mut data = tensor.value.data.clone();
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
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(tensor.value.shape.clone(), data),
        precision: tensor.precision,
    })
}

pub(super) fn tensor_sort_value(
    tensor: &RuntimeTensorValue,
    axis: i64,
) -> Result<RuntimeValue, String> {
    let axis = normalize_axis(tensor.value.shape.len(), axis, "sort")?;
    let axis_size = tensor.value.shape[axis];
    let inner: usize = tensor.value.shape[axis + 1..]
        .iter()
        .product::<usize>()
        .max(1);
    let outer: usize = tensor.value.shape[..axis].iter().product::<usize>().max(1);
    let mut values = tensor.value.data.clone();
    let mut indices = vec![0.0; tensor.value.data.len()];
    for outer_idx in 0..outer {
        for inner_idx in 0..inner {
            let mut items = (0..axis_size)
                .map(|axis_idx| {
                    let linear = (outer_idx * axis_size + axis_idx) * inner + inner_idx;
                    (axis_idx, values[linear])
                })
                .collect::<Vec<_>>();
            items.sort_by(|(lhs_idx, lhs_val), (rhs_idx, rhs_val)| {
                lhs_val
                    .partial_cmp(rhs_val)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(lhs_idx.cmp(rhs_idx))
            });
            for (sorted_idx, (original_idx, value)) in items.into_iter().enumerate() {
                let linear = (outer_idx * axis_size + sorted_idx) * inner + inner_idx;
                values[linear] = value;
                indices[linear] = original_idx as f64;
            }
        }
    }
    Ok(RuntimeValue::Tuple(vec![
        RuntimeValue::Tensor(RuntimeTensorValue {
            value: IrTensorValue::from_vec(tensor.value.shape.clone(), values),
            precision: tensor.precision,
        }),
        RuntimeValue::Tensor(RuntimeTensorValue {
            value: IrTensorValue::from_vec(tensor.value.shape.clone(), indices),
            precision: Prim::Int64,
        }),
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
    let mut data = vec![0.0; tensor_numel(&out_shape)];
    for (linear, slot) in data.iter_mut().enumerate() {
        let out_index = linear_to_indices(linear, &out_shape);
        let mut src_index = Vec::with_capacity(tensor.value.shape.len());
        let mut out_pos = 0usize;
        let diag_idx = out_index[axis1];
        for index in 0..tensor.value.shape.len() {
            if index == axis1 || index == axis2 {
                src_index.push(diag_idx);
            } else {
                src_index.push(out_index[out_pos]);
                out_pos += 1;
            }
        }
        let src_linear = indices_to_linear(&src_index, &tensor.value.shape);
        *slot = tensor.value.data[src_linear];
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(out_shape, data),
        precision: tensor.precision,
    })
}

pub(super) fn tensor_trace_value(
    tensor: &RuntimeTensorValue,
    axis1: i64,
    axis2: i64,
) -> Result<RuntimeTensorValue, String> {
    let diagonal = tensor_diagonal_value(tensor, axis1, axis2)?;
    let rank = diagonal.value.shape.len();
    let axis = normalize_axis(rank, axis1.min(axis2), "trace").unwrap_or(rank.saturating_sub(1));
    // #170: trace = sum over the diagonal. Route the diagonal reduction
    // through `tensor_reduce_host`'s `Sum` path so it uses the SAME
    // stride-4 ILP f32 cascade as `RiscOp::Sum` (issue #163, torch
    // `row_sum` parity). The prior hand-rolled `sum += ...` left-fold in
    // f64 diverged from `torch.trace` (== `torch.sum(diagonal)`) by ~1 ULP
    // for diagonals longer than 16 f32 elements. Reusing the one verified
    // cascade also prevents the two summation orders from drifting apart.
    tensor_reduce_host(&diagonal, axis as i64, ReduceOp::Sum)
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
    let mut out = Vec::with_capacity(tensor.value.data.len());
    for linear in 0..tensor.value.data.len() {
        let lo_value = if lo.value.shape.is_empty() {
            lo.value.data[0]
        } else {
            lo.value.data[linear]
        };
        let hi_value = if hi.value.shape.is_empty() {
            hi.value.data[0]
        } else {
            hi.value.data[linear]
        };
        out.push(tensor.value.data[linear].clamp(lo_value, hi_value));
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(tensor.value.shape.clone(), out),
        precision: tensor.precision,
    })
}

pub(super) fn tensor_einsum_value(
    equation: &str,
    lhs: &RuntimeTensorValue,
    rhs: &RuntimeTensorValue,
) -> Result<RuntimeTensorValue, String> {
    if equation.contains("...") {
        return Err("einsum ellipsis support is deferred in 3h".to_string());
    }
    let (inputs, output) = equation
        .split_once("->")
        .ok_or_else(|| "einsum equation must contain explicit output".to_string())?;
    let operands = inputs.split(',').collect::<Vec<_>>();
    if operands.len() != 2 {
        return Err("einsum 3h currently supports exactly two operands".to_string());
    }
    let lhs_labels = operands[0].chars().collect::<Vec<_>>();
    let rhs_labels = operands[1].chars().collect::<Vec<_>>();
    let out_labels = output.chars().collect::<Vec<_>>();
    if lhs_labels.len() != lhs.value.shape.len() || rhs_labels.len() != rhs.value.shape.len() {
        return Err("einsum label count must match operand rank".to_string());
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
    // #170 (DO NOT "fix" into the cascade): einsum is a contraction sum,
    // same shape as matmul, and shares matmul's disposition. torch's f32
    // einsum follows its GEMM order (strict-f32 left-fold), NOT the #163
    // `sum` cascade. The eval reference keeps the higher-precision f64
    // accumulator deliberately — same rationale as `tensor_matmul_host`:
    // the eval(f64)-vs-C(BLAS) gap at large k is an expected, tolerance-
    // covered precision characteristic, not a divergence. See the comment
    // in `tensor_matmul_host`.
    let mut out = vec![0.0; tensor_numel(&out_shape)];
    for (out_linear, slot) in out.iter_mut().enumerate() {
        let out_index = linear_to_indices(out_linear, &out_shape);
        let mut label_values = std::collections::HashMap::<char, usize>::new();
        for (label, value) in out_labels.iter().zip(out_index.iter()) {
            label_values.insert(*label, *value);
        }
        let reduction_total = tensor_numel(&reduction_shape);
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
            acc += lhs.value.data[indices_to_linear(&lhs_index, &lhs.value.shape)]
                * rhs.value.data[indices_to_linear(&rhs_index, &rhs.value.shape)];
        }
        *slot = acc;
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(out_shape, out),
        precision: lhs.precision,
    })
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
fn render_tensor_element(precision: Prim, stored: f64) -> String {
    use chelis_types::{ElementRef, format_element};
    // Exhaustive over Prim, no `_` arm (loud_unsupported.md section C4.1).
    //
    // Tag-vs-bits disagreements print the BITS: when an integer- or
    // bool-tagged slot holds a value outside the tag's value set (the live
    // example: `mean` of an int64 tensor stores 187.5 - chelis#724/[#729]
    // domain territory), the element renders as the stored f64 so the
    // value bug stays visible instead of laundered through truncation
    // (faithful_observation.md non-goals: printing wrong stored bits
    // faithfully is a feature). Panicking here would turn a runnable
    // program's print into a crash ([05-UNS-3] forbids source-reachable
    // panics), and truncating would manufacture a well-formed lie.
    let faithful_f64 = || format_element(Prim::F64, ElementRef::F64(stored));
    let int_or_bits = |width: Prim| -> String {
        if stored.fract() != 0.0 || !stored.is_finite() {
            return faithful_f64();
        }
        // Exact i64 range at f64 precision: [-2^63, 2^63). A saturating
        // `as` cast outside it would print a near-miss integer for bits
        // that are not that integer.
        if !(-9223372036854775808.0..9223372036854775808.0).contains(&stored) {
            return faithful_f64();
        }
        let as_int = stored as i64;
        let element = match width {
            Prim::Int8 => i8::try_from(as_int).map(ElementRef::I8).ok(),
            Prim::Int16 => i16::try_from(as_int).map(ElementRef::I16).ok(),
            Prim::Int32 => i32::try_from(as_int).map(ElementRef::I32).ok(),
            Prim::Int64 => Some(ElementRef::I64(as_int)),
            Prim::F16
            | Prim::Bf16
            | Prim::F32
            | Prim::F64
            | Prim::F8e4m3
            | Prim::Bool
            | Prim::String => None,
        };
        match element {
            Some(element) => format_element(width, element),
            // Out of the width's range: same tag-vs-bits story.
            None => faithful_f64(),
        }
    };
    match precision {
        Prim::Bool => {
            if stored == 1.0 {
                format_element(Prim::Bool, ElementRef::Bool(true))
            } else if stored == 0.0 && !stored.is_sign_negative() {
                format_element(Prim::Bool, ElementRef::Bool(false))
            } else {
                // A bool-tagged slot holding neither +0 nor 1 prints the
                // bits - including -0.0, whose sign bit is stored state a
                // `false` rendering would launder (PR #792 red-team F6;
                // reachable from source via `print(neg(bool_tensor))`).
                faithful_f64()
            }
        }
        Prim::Int8 => int_or_bits(Prim::Int8),
        Prim::Int16 => int_or_bits(Prim::Int16),
        Prim::Int32 => int_or_bits(Prim::Int32),
        Prim::Int64 => int_or_bits(Prim::Int64),
        Prim::F16 | Prim::Bf16 | Prim::F32 | Prim::F64 => faithful_f64(),
        Prim::F8e4m3 => panic!(
            "render_tensor_element: f8e4m3 is not in the active dtype set \
             (spec/04-type-system.md section 1.1.1); the checker rejects it, \
             so no tensor tag can carry it"
        ),
        Prim::String => panic!(
            "render_tensor_element: no string tensors exist \
             (to_tensor/cast/DAG typing only produce numeric/bool tensors)"
        ),
    }
}

/// Render a tensor payload: `tensor(shape=[..], data=[..])`, elements via
/// [`render_tensor_element`], truncated per [`TENSOR_RENDER_LIMIT`]. A
/// rank-0 tensor renders as its single element, bare: the
/// `tensor(shape=[], data=[..])` wrapper is not an exit form ([05-OBS-4],
/// the chelis#775 scalar-root decision - eval's internal rank-0
/// realization of scalar bindings must not leak into the observation
/// channel, and `print` of the same scalar already renders bare).
fn render_tensor(tensor: &RuntimeTensorValue) -> String {
    if tensor.value.shape.is_empty() {
        let stored = *tensor.value.data.first().unwrap_or_else(|| {
            panic!(
                "render_tensor: rank-0 tensor with no element (IrTensorValue \
                 guarantees numel(shape=[]) == 1 at construction)"
            )
        });
        return render_tensor_element(tensor.precision, stored);
    }
    let visible = tensor.value.data.len().min(TENSOR_RENDER_LIMIT);
    let mut elements: Vec<String> = tensor.value.data[..visible]
        .iter()
        .map(|stored| render_tensor_element(tensor.precision, *stored))
        .collect();
    if tensor.value.data.len() > visible {
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
            // Scalars carry their dtype in ScalarBits (the dtype/bits
            // invariant is enforced at construction), so every scalar exit
            // renders at its OWN width per [05-OBS-2].
            let element = match payload.bits() {
                ScalarBits::I8(v) => ElementRef::I8(v),
                ScalarBits::I16(v) => ElementRef::I16(v),
                ScalarBits::I32(v) => ElementRef::I32(v),
                ScalarBits::I64(v) => ElementRef::I64(v),
                ScalarBits::F16(v) => ElementRef::F16(v),
                ScalarBits::Bf16(v) => ElementRef::Bf16(v),
                ScalarBits::F32(v) => ElementRef::F32(v),
                ScalarBits::F64(v) => ElementRef::F64(v),
            };
            format_element(payload.dtype(), element)
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

pub(super) fn builtin_name(expr: &Expr) -> Option<&str> {
    let name = var_name(expr)?;
    BUILTIN_NAMES.contains(&name).then_some(name)
}

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
    // chelis#770: evaluate the affine `low + (high - low) * unit` in f32,
    // conforming to the C f32 sampler `chelis_uniform_sample_f32`
    // (chelis-backend-c/src/emit.rs and host_emit.rs:
    // `return low + (high - low) * (float)unit;` with f32 `low`/`high`).
    // Under the default toolchain (`-march=native`, `-ffp-contract=fast`) the
    // C compiler contracts that `low + span * unit` into a single-rounding
    // fused multiply-add, so `span_f.mul_add(unit_f, low_f)` (also one
    // rounding) matches it bit-for-bit; a plain `low_f + span_f * unit_f`
    // (two roundings) drifts 1 ULP on some elements. `dropout_sample` stays
    // f64 (no C oracle; drives dropout thresholds); the f32 casts are local.
    let low_f = low as f32;
    let high_f = high as f32;
    let span_f = high_f - low_f;
    let data = template
        .value
        .data
        .iter()
        .enumerate()
        .map(|(index, _)| span_f.mul_add(dropout_sample(seed, index as u64) as f32, low_f) as f64)
        .collect::<Vec<_>>();
    RuntimeTensorValue {
        value: IrTensorValue::from_vec(template.value.shape.clone(), data),
        precision: template.precision,
    }
}

#[cfg(test)]
mod uniform_like_affine_tests {
    //! chelis#770: `uniform_like_value` evaluates the affine in f32,
    //! op-for-op with the C `chelis_uniform_sample_f32` sampler, so the host
    //! evaluator and the compiled C lane agree at f32. These pin the exact
    //! widened-f32 output at seed=42 / shape=[8] and the 1-ULP gap the old
    //! f64 affine left at elem[4] of [2,5), plus a negative range (unit-level
    //! only: the C cross-lane path can't be driven with a bare negative
    //! literal, a separate lowering gap).
    use super::*;

    fn template_f32(n: usize) -> RuntimeTensorValue {
        RuntimeTensorValue {
            value: IrTensorValue::from_vec(vec![n], vec![0.0; n]),
            precision: Prim::F32,
        }
    }

    #[test]
    fn affine_mirrors_c_f32_sampler_positive_range() {
        let out = uniform_like_value(&template_f32(8), 2.0, 5.0, 42);
        // Single correctly-rounded FMA, conforming to the compiled C sampler.
        // elem[4]: where the pre-#770 f64 affine rounded to the adjacent f32
        // (0x404215a9) instead of the sampler's 0x404215aa.
        assert_eq!(
            out.value.data[4].to_bits(),
            (f32::from_bits(0x404215aa) as f64).to_bits(),
            "elem[4] must be the C f32 sampler value (0x404215aa), got {} (f32 bits {:#010x})",
            out.value.data[4],
            (out.value.data[4] as f32).to_bits(),
        );
        let old_f64_affine = 2.0 + (5.0 - 2.0) * dropout_sample(42, 4);
        assert_eq!((old_f64_affine as f32).to_bits(), 0x404215a9);
        assert_ne!(
            (out.value.data[4] as f32).to_bits(),
            (old_f64_affine as f32).to_bits(),
            "the fix must not reproduce the old f64-affine rounding",
        );
        // elem[6]/[7]: where a single-rounding FMA and a plain two-rounding
        // affine disagree by 1 ULP — the exact bit the compiled C lane flips
        // between `-ffp-contract=fast` (FMA, 0x408f5273) and `=off` (two
        // roundings, 0x408f5274). Pin the FMA values; show two-rounding differs.
        assert_eq!(
            out.value.data[6].to_bits(),
            (f32::from_bits(0x408f5273) as f64).to_bits(),
            "elem[6] must be the single-rounding FMA value (0x408f5273)",
        );
        assert_eq!(
            out.value.data[7].to_bits(),
            (f32::from_bits(0x403ec1e7) as f64).to_bits(),
            "elem[7] must be the single-rounding FMA value (0x403ec1e7)",
        );
        let unit6 = dropout_sample(42, 6) as f32;
        let two_rounding_6 = 2.0f32 + (5.0f32 - 2.0f32) * unit6;
        assert_eq!(two_rounding_6.to_bits(), 0x408f5274);
        assert_ne!(
            (out.value.data[6] as f32).to_bits(),
            two_rounding_6.to_bits(),
        );
    }

    #[test]
    fn affine_is_f32_for_negative_range() {
        let out = uniform_like_value(&template_f32(8), -3.0, -1.0, 42);
        assert_eq!(
            out.value.data[3].to_bits(),
            (f32::from_bits(0xc010167a) as f64).to_bits(),
            "elem[3] must be the C f32 sampler value for [-3,-1) (0xc010167a)",
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
