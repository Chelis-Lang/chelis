module Std.Init.XavierExt
import Std.Init.Random (normal_like)
export (xavier_uniform, xavier_normal, trunc_normal)
sig xavier_uniform: &tensor[..r, p_float] -> p_float -> p_float -> tensor[..r, p_float] ! { Random }
def xavier_uniform(template, fan_in, fan_out) = {
  _ = validate_xavier_params(fan_in, fan_out)
  bound = sqrt(div(6.0, add(fan_in, fan_out)))
  raw = uniform_like(template, 0.0, 1.0)
  flat = reshape(copy(raw), [numel(raw)])
  values = map(fn (x) -> mul(sub(mul(2.0, x), 1.0), bound), to_list(flat))
  _ = drop(raw)
  reshape(to_tensor(values), tensor_shape(template, cast(0, int32), cast(rank(template), int32), drop([cast(0, int64)], cast(1, int64))))
}
sig xavier_normal: &tensor[..r, p_float] -> p_float -> p_float -> tensor[..r, p_float] ! { Random }
def xavier_normal(template, fan_in, fan_out) = {
  _ = validate_xavier_params(fan_in, fan_out)
  std = sqrt(div(2.0, add(fan_in, fan_out)))
  normal_like(template, 0.0, std)
}
sig trunc_normal: &tensor[..r, p_float] -> p_float -> p_float -> p_float -> p_float -> tensor[..r, p_float] ! { Random }
def trunc_normal(template, mean, std, a, b) = {
  _ = validate_trunc_params(mean, std, a, b)
  raw = normal_like(template, mean, std)
  flat = reshape(copy(raw), [numel(raw)])
  values = map(fn (x) -> if lt(x, a) then a else if gt(x, b) then b else x, to_list(flat))
  _ = drop(raw)
  reshape(to_tensor(values), tensor_shape(template, cast(0, int32), cast(rank(template), int32), drop([cast(0, int64)], cast(1, int64))))
}
def validate_xavier_params[p_float](fan_in: p_float, fan_out: p_float) -> bool = {
  fan_sum = add(fan_in, fan_out)
  validate_domain(and(finite_float(fan_in), and(finite_float(fan_out), and(finite_float(fan_sum), gt(fan_sum, 0.0)))))
}
def validate_trunc_params[p_float](mean: p_float, std: p_float, a: p_float, b: p_float) -> bool = validate_domain(and(finite_float(mean), and(finite_float(std), and(finite_float(a), and(finite_float(b), and(gte(std, 0.0), lte(a, b)))))))
-- Invalid maps to integer 2, so the checked bool cast traps Domain before Random; valid maps to 0. Avoiding a source `if` keeps validation in pathwise AD's forward graph.
def validate_domain(valid: bool) -> bool = cast(mul(sub(cast(1, int64), cast(valid, int64)), cast(2, int64)), bool)
def finite_float[p_float](value: p_float) -> bool = not(or(neq(value, value), or(eq(value, div(1.0, 0.0)), eq(value, div(-1.0, 0.0)))))
def tensor_shape[p](template: &tensor[..r, p], axis: int32, limit: int32, out: List[int64]) -> List[int64] = if gte(axis, limit) then out else tensor_shape(template, add(axis, cast(1, int32)), limit, append(out, shape(template, axis)))
