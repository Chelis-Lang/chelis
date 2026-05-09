module Std.Nn.Gelu
export (forward, tanh_scalar, gelu_scalar)
def forward[n](x: &tensor[n, f32]) -> tensor[n, f32] = to_tensor(map(fn (v: f32) -> gelu_scalar(v), to_list(x)))
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
