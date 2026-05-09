module Std.Init.Kaiming
import Std.Init.Random (normal_like)
export (kaiming_uniform, kaiming_normal)
def kaiming_uniform[n](template: &tensor[n, f32], fan_in: f32) -> tensor[n, f32] ! { Random } = {
  bound = sqrt(div(cast(6.0, f32), fan_in))
  raw = uniform_like(template, 0.0, 1.0)
  two = cast(2.0, f32)
  one = cast(1.0, f32)
  values = to_list(raw)
  _ = drop(raw)
  to_tensor(map(fn (x: f32) -> mul(sub(mul(two, x), one), bound), values))
}
def kaiming_normal[n](template: &tensor[n, f32], fan_in: f32) -> tensor[n, f32] ! { Random } = {
  std = sqrt(div(cast(2.0, f32), fan_in))
  normal_like(template, cast(0.0, f32), std)
}
