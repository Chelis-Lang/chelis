module Std.Init.Kaiming
import Std.Init.Random (normal_like)
export (kaiming_uniform, kaiming_normal)
sig kaiming_uniform: &tensor[n, f32] -> f32 -> tensor[n, f32] ! { Random }
def kaiming_uniform(template, fan_in) = {
  bound = sqrt(div(cast(6.0, f32), fan_in))
  raw = uniform_like(template, 0.0, 1.0)
  two = cast(2.0, f32)
  one = cast(1.0, f32)
  values = to_list(raw)
  _ = drop(raw)
  to_tensor(map(fn (x: f32) -> mul(sub(mul(two, x), one), bound), values))
}
sig kaiming_normal: &tensor[n, f32] -> f32 -> tensor[n, f32] ! { Random }
def kaiming_normal(template, fan_in) = {
  std = sqrt(div(cast(2.0, f32), fan_in))
  normal_like(template, cast(0.0, f32), std)
}
