def aligned[d](x: tensor[d, f32], gain: tensor[fixed, f32]) -> tensor[d, f32] = mul(x, gain)
out = {
  f = aligned
  g = f
  sum(g(to_tensor([1.0f32, 2.0f32]), to_tensor([1.0f32, 2.0f32])), fixed)
}
