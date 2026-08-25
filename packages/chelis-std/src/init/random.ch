module Std.Init.Random
export (normal_like)
sig normal_like: &tensor[..r, p_float] -> p_float -> p_float -> tensor[..r, p_float] ! { Random }
def normal_like(template, mean, std) = {
  u1 = uniform_like(template, 1e-7, 1.0)
  u2 = uniform_like(template, 0.0, 1.0)
  values = map(fn (pair) -> {
    cos_term = cos(mul(6.283185307179586, pair.1))
    radius = sqrt(mul(-2.0, log(pair.0)))
    add(mean, mul(std, mul(radius, cos_term)))
  }, zip(to_list(u1), to_list(u2)))
  _ = drop(u1)
  _ = drop(u2)
  reshape(to_tensor(values), tensor_shape(template, cast(0, int32), cast(rank(template), int32), drop([cast(0, int64)], cast(1, int64))))
}
def tensor_shape[p](template: &tensor[..r, p], axis: int32, limit: int32, out: List[int64]) -> List[int64] = if gte(axis, limit) then out else tensor_shape(template, add(axis, cast(1, int32)), limit, append(out, shape(template, axis)))
