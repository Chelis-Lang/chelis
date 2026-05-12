module Std.Loss.KlDiv
export (kl_divergence)
sig kl_divergence: &tensor[n, f32] -> &tensor[n, f32] -> f32
def kl_divergence(p, q) = {
  p_list = to_list(copy(p))
  q_list = to_list(copy(q))
  if neq(len(p_list), len(q_list)) then fail("kl_divergence: p and q have different lengths") else fold(fn (acc: f32, pair: (f32, f32)) -> add(acc, if eq(pair.0, cast(0.0, f32)) then cast(0.0, f32) else mul(pair.0, sub(log(pair.0), log(pair.1)))), cast(0.0, f32), zip(p_list, q_list))
}
