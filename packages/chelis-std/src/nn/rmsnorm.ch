module Std.Nn.RmsNorm
export (forward, rms_scale)
sig forward: &tensor[n, f32] -> &tensor[n, f32] -> f32 -> tensor[n, f32]
def forward(x, gain, eps) = {
  scale = rms_scale(x, eps)
  scaled = to_tensor(map(fn (v: f32) -> mul(v, scale), to_list(x)))
  out = mul(scaled, gain)
  _ = drop(scaled)
  out
}
sig rms_scale: &tensor[n, f32] -> f32 -> f32
def rms_scale(x, eps) = {
  squared = map(fn (v: f32) -> mul(v, v), to_list(x))
  count = cast(len(squared), f32)
  total = fold(fn (acc: f32, v: f32) -> add(acc, v), cast(0.0, f32), squared)
  ms = div(total, count)
  div(cast(1.0, f32), sqrt(add(ms, eps)))
}
