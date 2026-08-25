module Std.Init.Random
export (normal_like)
sig normal_like: &tensor[..r, p_float] -> p_float -> p_float -> tensor[..r, p_float] ! { Random }
def normal_like(template, mean, std) = {
  _ = validate_normal_params(mean, std)
  u1 = uniform_like(template, 1e-7, 1.0)
  u2 = uniform_like(template, 0.0, 1.0)
  u1_flat = reshape(copy(u1), [numel(u1)])
  u2_flat = reshape(copy(u2), [numel(u2)])
  radii = map(fn (x) -> sqrt(mul(-2.0, log(x))), to_list(u1_flat))
  cos_terms = map(fn (x) -> cos(mul(6.283185307179586, x)), to_list(u2_flat))
  z = mul(to_tensor(radii), to_tensor(cos_terms))
  values = map(fn (x) -> add(mean, mul(std, x)), to_list(z))
  _ = drop(u1)
  _ = drop(u2)
  reshape(to_tensor(values), tensor_shape(template, cast(0, int32), cast(rank(template), int32), drop([cast(0, int64)], cast(1, int64))))
}
def validate_normal_params[p_float](mean: p_float, std: p_float) -> bool = validate_domain(and(finite_float(mean), and(finite_float(std), gte(std, 0.0))))
-- Invalid maps to integer 2, so the checked bool cast traps Domain before Random; valid maps to 0. Avoiding a source `if` keeps validation in pathwise AD's forward graph.
def validate_domain(valid: bool) -> bool = cast(mul(sub(cast(1, int64), cast(valid, int64)), cast(2, int64)), bool)
def finite_float[p_float](value: p_float) -> bool = not(or(neq(value, value), or(eq(value, div(1.0, 0.0)), eq(value, div(-1.0, 0.0)))))
def tensor_shape[p](template: &tensor[..r, p], axis: int32, limit: int32, out: List[int64]) -> List[int64] = if gte(axis, limit) then out else tensor_shape(template, add(axis, cast(1, int32)), limit, append(out, shape(template, axis)))
