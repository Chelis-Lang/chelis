module Std.Init.XavierExt
import Std.Init.Random (normal_like)
export (xavier_uniform, xavier_normal, trunc_normal)
sig xavier_uniform: &tensor[n, f32] -> f32 -> f32 -> tensor[n, f32] ! { Random }
def xavier_uniform(template, fan_in, fan_out) = {
  bound = sqrt(div(cast(6.0, f32), add(fan_in, fan_out)))
  raw = uniform_like(template, 0.0, 1.0)
  two = cast(2.0, f32)
  one = cast(1.0, f32)
  values = to_list(raw)
  _ = drop(raw)
  to_tensor(map(fn (x: f32) -> mul(sub(mul(two, x), one), bound), values))
}
sig xavier_normal: &tensor[n, f32] -> f32 -> f32 -> tensor[n, f32] ! { Random }
def xavier_normal(template, fan_in, fan_out) = {
  std = sqrt(div(cast(2.0, f32), add(fan_in, fan_out)))
  normal_like(template, cast(0.0, f32), std)
}
sig trunc_normal: &tensor[n, f32] -> f32 -> f32 -> f32 -> f32 -> tensor[n, f32] ! { Random }
def trunc_normal(template, mean, std, a, b) = {
  raw = normal_like(template, mean, std)
  values = to_list(raw)
  _ = drop(raw)
  to_tensor(map(fn (x: f32) -> if lt(x, a) then a else if gt(x, b) then b else x, values))
}
