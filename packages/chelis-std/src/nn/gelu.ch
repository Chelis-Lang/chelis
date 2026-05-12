module Std.Nn.Gelu
export (forward, tanh_scalar, gelu_scalar)
sig forward: &tensor[n, f32] -> tensor[n, f32]
def forward(x) = to_tensor(map(fn (v: f32) -> gelu_scalar(v), to_list(x)))
sig tanh_scalar: f32 -> f32
def tanh_scalar(z) = {
  e_pos = exp(z)
  e_neg = exp(neg(z))
  div(sub(e_pos, e_neg), add(e_pos, e_neg))
}
sig gelu_scalar: f32 -> f32
def gelu_scalar(v) = {
  c = cast(0.7978845608028654, f32)
  k = cast(0.044715, f32)
  half = cast(0.5, f32)
  one = cast(1.0, f32)
  inner = mul(c, add(v, mul(k, mul(v, mul(v, v)))))
  mul(half, mul(v, add(one, tanh_scalar(inner))))
}
