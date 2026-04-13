module Std.Nn.Gelu
export (forward, tanh_scalar, gelu_scalar)
-- GELU activation using the tanh approximation, because `erf` is not a
-- Chelis primitive. Ships the OpenAI/BERT tanh-approx form:
--   gelu_tanh(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
-- where tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z)).
-- This is NOT bit-identical to the exact erf-based GELU; it is the
-- standard approximation used by GPT-2/BERT. Documented in
-- spec/design/chelis_phase3_plan.md as an acknowledged limitation.
-- Rank-1 element-wise via to_list/map/to_tensor; callers with
-- multi-dim tensors should flatten first.
def forward[n](x: tensor[n, f32]) -> tensor[n, f32] = to_tensor(map(fn (v: f32) -> gelu_scalar(v), to_list(x)))
def tanh_scalar(z: f32) -> f32 = {
  e_pos = exp(z)
  e_neg = exp(neg(z))
  div(sub(e_pos, e_neg), add(e_pos, e_neg))
}
def gelu_scalar(v: f32) -> f32 = {
  c = cast(0.7978845608028654, f32)
  k = cast(0.044715, f32)
  half = cast(0.5, f32)
  one = cast(1.0, f32)
  inner = mul(c, add(v, mul(k, mul(v, mul(v, v)))))
  mul(half, mul(v, add(one, tanh_scalar(inner))))
}
