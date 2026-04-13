module Std.Nn.RmsNorm
export (forward, rms_scale)
-- RMSNorm: y = x * rsqrt(mean(x^2) + eps) * gain.
-- Reduction is over the last (and only) axis; rank-1 variant. For
-- transformer hidden states with shape [batch, seq, dim], split to
-- rank-1 rows before shipping to RmsNorm.forward.
-- rsqrt(y) = 1 / sqrt(y) computed inline because there is no `rsqrt`
-- primitive.
def forward[n](x: tensor[n, f32], gain: tensor[n, f32], eps: f32) -> tensor[n, f32] = {
  scale = rms_scale(copy(x), eps)
  scaled = to_tensor(map(fn (v: f32) -> mul(v, scale), to_list(x)))
  mul(scaled, gain)
}
def rms_scale[n](x: tensor[n, f32], eps: f32) -> f32 = {
  squared = map(fn (v: f32) -> mul(v, v), to_list(x))
  count = cast(len(squared), f32)
  total = fold(fn (acc: f32, v: f32) -> add(acc, v), cast(0.0, f32), squared)
  ms = div(total, count)
  div(cast(1.0, f32), sqrt(add(ms, eps)))
}
