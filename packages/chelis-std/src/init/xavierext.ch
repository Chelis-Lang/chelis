module Std.Init.XavierExt
import Std.Init.Random (normal_like)
export (xavier_uniform, xavier_normal, trunc_normal)
sig xavier_uniform[r, p: Float]: key -> &tensor[..r, p] -> p -> p -> tensor[..r, p]
-- [05-OP-35]: validate, draw one uniform from the key, then the pure scaling graph.
def xavier_uniform(k, template, fan_in, fan_out) = {
  _ = validate_xavier_params(fan_in, fan_out)
  xavier_uniform_given(uniform_like(k, template, 0.0, 1.0), template, fan_in, fan_out)
}
-- The pure layer: the Xavier uniform values of a unit uniform draw, shaped like the template.
sig xavier_uniform_given[r, p: Float]: tensor[..r, p] -> &tensor[..r, p] -> p -> p -> tensor[..r, p]
def xavier_uniform_given(raw, template, fan_in, fan_out) = {
  bound = sqrt(div(6.0, add(fan_in, fan_out)))
  flat = reshape(copy(raw), [numel(raw)])
  values = map(fn (x) -> mul(sub(mul(2.0, x), 1.0), bound), to_list(flat))
  _ = drop(raw)
  reshape(to_tensor(values), tensor_shape(template, cast(0, i32), cast(rank(template), i32), skip([cast(0, i64)], cast(1, i64))))
}
sig xavier_normal[r, p: Float]: key -> &tensor[..r, p] -> p -> p -> tensor[..r, p]
-- [05-OP-35]: the key passes unchanged to the one normal_like call.
def xavier_normal(k, template, fan_in, fan_out) = {
  _ = validate_xavier_params(fan_in, fan_out)
  std = sqrt(div(2.0, add(fan_in, fan_out)))
  normal_like(k, template, 0.0, std)
}
sig trunc_normal[r, p: Float]: key -> &tensor[..r, p] -> p -> p -> p -> p -> tensor[..r, p]
-- [05-OP-35]: validate, draw the normal values from the key with normal_like, then clip them to [a, b].
def trunc_normal(k, template, mean, std, a, b) = {
  _ = validate_trunc_params(mean, std, a, b)
  raw = normal_like(k, template, mean, std)
  flat = reshape(copy(raw), [numel(raw)])
  values = map(fn (x) -> if lt(x, a) then a else if gt(x, b) then b else x, to_list(flat))
  _ = drop(raw)
  reshape(to_tensor(values), tensor_shape(template, cast(0, i32), cast(rank(template), i32), skip([cast(0, i64)], cast(1, i64))))
}
def validate_xavier_params[p: Float](fan_in: p, fan_out: p) -> bool = {
  fan_sum = add(fan_in, fan_out)
  validate_domain(and(finite_float(fan_in), and(finite_float(fan_out), and(finite_float(fan_sum), gt(fan_sum, cast(0.0, p))))))
}
def validate_trunc_params[p: Float](mean: p, std: p, a: p, b: p) -> bool = validate_domain(and(finite_float(mean), and(finite_float(std), and(finite_float(a), and(finite_float(b), and(gte(std, cast(0.0, p)), lte(a, b)))))))
-- Invalid maps to integer 2, so the checked bool cast traps Domain before any draw; valid maps to 0. Avoiding a source `if` keeps validation in pathwise AD's forward graph.
def validate_domain(valid: bool) -> bool = cast(mul(sub(cast(1, i64), cast(valid, i64)), cast(2, i64)), bool)
def finite_float[p: Float](value: p) -> bool = not(or(neq(value, value), or(eq(value, div(cast(1.0, p), cast(0.0, p))), eq(value, div(cast(-1.0, p), cast(0.0, p))))))
def tensor_shape[r, p](template: &tensor[..r, p], axis: i32, limit: i32, out: List[i64]) -> List[i64] = if gte(axis, limit) then out else tensor_shape(template, add(axis, cast(1, i32)), limit, append(out, shape(template, axis)))
