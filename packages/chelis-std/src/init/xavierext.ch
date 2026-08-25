module Std.Init.XavierExt
import Std.Init.Random (normal_like)
export (xavier_uniform, xavier_normal, trunc_normal)
sig xavier_uniform: &tensor[..r, p_float] -> p_float -> p_float -> tensor[..r, p_float] ! { Random }
def xavier_uniform(template, fan_in, fan_out) = {
  bound = sqrt(div(6.0, add(fan_in, fan_out)))
  raw = uniform_like(template, 0.0, 1.0)
  values = map(fn (x) -> mul(sub(mul(2.0, x), 1.0), bound), to_list(raw))
  _ = drop(raw)
  reshape(to_tensor(values), tensor_shape(template, cast(0, int32), cast(rank(template), int32), drop([cast(0, int64)], cast(1, int64))))
}
sig xavier_normal: &tensor[..r, p_float] -> p_float -> p_float -> tensor[..r, p_float] ! { Random }
def xavier_normal(template, fan_in, fan_out) = {
  std = sqrt(div(2.0, add(fan_in, fan_out)))
  normal_like(template, 0.0, std)
}
sig trunc_normal: &tensor[..r, p_float] -> p_float -> p_float -> p_float -> p_float -> tensor[..r, p_float] ! { Random }
def trunc_normal(template, mean, std, a, b) = {
  raw = normal_like(template, mean, std)
  values = map(fn (x) -> if lt(x, a) then a else if gt(x, b) then b else x, to_list(raw))
  _ = drop(raw)
  reshape(to_tensor(values), tensor_shape(template, cast(0, int32), cast(rank(template), int32), drop([cast(0, int64)], cast(1, int64))))
}
def tensor_shape[p](template: &tensor[..r, p], axis: int32, limit: int32, out: List[int64]) -> List[int64] = if gte(axis, limit) then out else tensor_shape(template, add(axis, cast(1, int32)), limit, append(out, shape(template, axis)))
