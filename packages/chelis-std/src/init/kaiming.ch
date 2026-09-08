module Std.Init.Kaiming
import Std.Init.Random (normal_like)
export (kaiming_uniform, kaiming_normal)
sig kaiming_uniform[p: Float]: &tensor[..r, p] -> p -> tensor[..r, p] ! { Random }
def kaiming_uniform(template, fan_in) = {
  _ = validate_fan_in(fan_in)
  bound = sqrt(div(6.0, fan_in))
  raw = uniform_like(template, 0.0, 1.0)
  flat = reshape(copy(raw), [numel(raw)])
  values = map(fn (x) -> mul(sub(mul(2.0, x), 1.0), bound), to_list(flat))
  _ = drop(raw)
  reshape(to_tensor(values), tensor_shape(template, cast(0, int32), cast(rank(template), int32), drop([cast(0, int64)], cast(1, int64))))
}
def tensor_shape[p](template: &tensor[..r, p], axis: int32, limit: int32, out: List[int64]) -> List[int64] = if gte(axis, limit) then out else tensor_shape(template, add(axis, cast(1, int32)), limit, append(out, shape(template, axis)))
sig kaiming_normal[p: Float]: &tensor[..r, p] -> p -> tensor[..r, p] ! { Random }
def kaiming_normal(template, fan_in) = {
  _ = validate_fan_in(fan_in)
  std = sqrt(div(2.0, fan_in))
  normal_like(template, 0.0, std)
}
def validate_fan_in[p: Float](fan_in: p) -> bool = validate_domain(and(finite_float(fan_in), gt(fan_in, cast(0.0, p))))
-- Invalid maps to integer 2, so the checked bool cast traps Domain before Random; valid maps to 0. Avoiding a source `if` keeps validation in pathwise AD's forward graph.
def validate_domain(valid: bool) -> bool = cast(mul(sub(cast(1, int64), cast(valid, int64)), cast(2, int64)), bool)
def finite_float[p: Float](value: p) -> bool = not(or(neq(value, value), or(eq(value, div(cast(1.0, p), cast(0.0, p))), eq(value, div(cast(-1.0, p), cast(0.0, p))))))
