module Std.Init.XavierExt
import Std.Init.Random (normal_like)
export (xavier_uniform, xavier_normal, trunc_normal)
def xavier_uniform[n](template: tensor[n, f32], fan_in: f32, fan_out: f32) -> tensor[n, f32] ! { Random } = {
  bound = sqrt(div(cast(6.0, f32), add(fan_in, fan_out)))
  raw = uniform_like(template, 0.0, 1.0)
  two = cast(2.0, f32)
  one = cast(1.0, f32)
  to_tensor(map(fn (x: f32) -> mul(sub(mul(two, x), one), bound), to_list(raw)))
}
def xavier_normal[n](template: tensor[n, f32], fan_in: f32, fan_out: f32) -> tensor[n, f32] ! { Random } = {
  std = sqrt(div(cast(2.0, f32), add(fan_in, fan_out)))
  normal_like(template, cast(0.0, f32), std)
}
def trunc_normal[n](template: tensor[n, f32], mean: f32, std: f32, a: f32, b: f32) -> tensor[n, f32] ! { Random } = {
  raw = normal_like(template, mean, std)
  to_tensor(map(fn (x: f32) -> if lt(x, a) then a else if gt(x, b) then b else x, to_list(raw)))
}
