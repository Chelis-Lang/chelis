module Std.Init.Random
export (normal_like)
def normal_like[n](template: tensor[n, f32], mean: f32, std: f32) -> tensor[n, f32] ! { Random } = {
  u1 = uniform_like(copy(template), 0.0000001, 1.0)
  u2 = uniform_like(template, 0.0, 1.0)
  to_tensor(map(fn (pair: (f32, f32)) -> {
    a = pair.0
    b = pair.1
    two_pi_b = mul(cast(6.2831855, f32), b)
    cos_term = sin(sub(cast(1.5707964, f32), two_pi_b))
    radius = sqrt(mul(cast(-2.0, f32), log(a)))
    z = mul(radius, cos_term)
    add(mean, mul(std, z))
  }, zip(to_list(u1), to_list(u2))))
}
