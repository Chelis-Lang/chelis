module Std.Init.Kaiming
import Std.Init.Random (normal_like)
export (kaiming_uniform, kaiming_normal)
sig kaiming_uniform[r, p: Float]: key -> &tensor[..r, p] -> p -> tensor[..r, p]
-- [05-OP-35]: validate, draw one uniform from the key, then the pure scaling graph.
def kaiming_uniform(k, template, fan_in) = {
  _ = validate_fan_in(fan_in)
  kaiming_uniform_given(uniform_like(k, template, 0.0, 1.0), template, fan_in)
}
-- The pure layer: the Kaiming uniform values of a unit uniform draw, shaped like the template.
sig kaiming_uniform_given[r, p: Float]: tensor[..r, p] -> &tensor[..r, p] -> p -> tensor[..r, p]
def kaiming_uniform_given(raw, template, fan_in) = {
  bound = sqrt(div(6.0, fan_in))
  flat = reshape(copy(raw), [numel(raw)])
  values = map(fn (x) -> mul(sub(mul(2.0, x), 1.0), bound), to_list(flat))
  _ = drop(raw)
  reshape(to_tensor(values), tensor_shape(template, cast(0, i32), cast(rank(template), i32), skip([cast(0, i64)], cast(1, i64))))
}
def tensor_shape[r, p](template: &tensor[..r, p], axis: i32, limit: i32, out: List[i64]) -> List[i64] = if gte(axis, limit) then out else tensor_shape(template, add(axis, cast(1, i32)), limit, append(out, shape(template, axis)))
sig kaiming_normal[r, p: Float]: key -> &tensor[..r, p] -> p -> tensor[..r, p]
-- [05-OP-35]: the key passes unchanged to the one normal_like call.
def kaiming_normal(k, template, fan_in) = {
  _ = validate_fan_in(fan_in)
  std = sqrt(div(2.0, fan_in))
  normal_like(k, template, 0.0, std)
}
def validate_fan_in[p: Float](fan_in: p) -> bool = validate_domain(and(finite_float(fan_in), gt(fan_in, cast(0.0, p))))
-- Invalid maps to integer 2, so the checked bool cast traps Domain before any draw; valid maps to 0. Avoiding a source `if` keeps validation in pathwise AD's forward graph.
def validate_domain(valid: bool) -> bool = cast(mul(sub(cast(1, i64), cast(valid, i64)), cast(2, i64)), bool)
def finite_float[p: Float](value: p) -> bool = not(or(neq(value, value), or(eq(value, div(cast(1.0, p), cast(0.0, p))), eq(value, div(cast(-1.0, p), cast(0.0, p))))))
