module Std.Init.Random
export (normal_like)
sig normal_like[r, p: Float]: key -> &tensor[..r, p] -> p -> p -> tensor[..r, p]
-- [05-OP-35]: validate, draw the two unit uniforms from the key, then the pure Box-Muller graph.
def normal_like(k, template, mean, std) = {
  _ = validate_normal_params(mean, std)
  normal_like_given(normal_like_sample(k, template), template, mean, std)
}
-- The sampling layer: split the key, then draw u1 from the left half and u2 from the right.
sig normal_like_sample[r, p: Float]: key -> &tensor[..r, p] -> (tensor[..r, p], tensor[..r, p])
def normal_like_sample(k, template) = {
  (k1, k2) = split_key(k)
  (uniform_like(k1, template, 1e-7, 1.0), uniform_like(k2, template, 0.0, 1.0))
}
-- The pure layer: the Box-Muller normal values of two unit uniform draws, shaped like the template.
sig normal_like_given[r, p: Float]: (tensor[..r, p], tensor[..r, p]) -> &tensor[..r, p] -> p -> p -> tensor[..r, p]
def normal_like_given(units, template, mean, std) = {
  (u1, u2) = units
  u1_flat = reshape(copy(u1), [numel(u1)])
  u2_flat = reshape(copy(u2), [numel(u2)])
  radii = map(fn (x) -> sqrt(mul(-2.0, log(x))), to_list(u1_flat))
  cos_terms = map(fn (x) -> cos(mul(6.283185307179586, x)), to_list(u2_flat))
  z = mul(to_tensor(radii), to_tensor(cos_terms))
  values = map(fn (x) -> add(mean, mul(std, x)), to_list(z))
  _ = drop(u1)
  _ = drop(u2)
  reshape(to_tensor(values), tensor_shape(template, cast(0, i32), cast(rank(template), i32), skip([cast(0, i64)], cast(1, i64))))
}
def validate_normal_params[p: Float](mean: p, std: p) -> bool = validate_domain(and(finite_float(mean), and(finite_float(std), gte(std, cast(0.0, p)))))
-- Invalid maps to integer 2, so the checked bool cast traps Domain before any draw; valid maps to 0. Avoiding a source `if` keeps validation in pathwise AD's forward graph.
def validate_domain(valid: bool) -> bool = cast(mul(sub(cast(1, i64), cast(valid, i64)), cast(2, i64)), bool)
def finite_float[p: Float](value: p) -> bool = not(or(neq(value, value), or(eq(value, div(cast(1.0, p), cast(0.0, p))), eq(value, div(cast(-1.0, p), cast(0.0, p))))))
def tensor_shape[r, p](template: &tensor[..r, p], axis: i32, limit: i32, out: List[i64]) -> List[i64] = if gte(axis, limit) then out else tensor_shape(template, add(axis, cast(1, i32)), limit, append(out, shape(template, axis)))
