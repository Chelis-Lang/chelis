module Sink_Panic_Localization
sig causal_sdpa_with_sink: &tensor[s, d, f32] -> &tensor[s, d, f32] -> &tensor[s, d, f32] -> &tensor[s, s, f32] -> &tensor[s, s, f32] -> f32 -> tensor[s, d, f32]
def causal_sdpa_with_sink(q, k, v, scale, mask, sink) = {
  kt = permute(k, 1, 0)
  scores = matmul(q, kt)
  scaled = mul(scores, scale)
  masked = add(scaled, mask)
  seq = shape(&masked, cast(0, i32))
  sink_scalar = scalar_to_tensor(sink)
  sink_rows: tensor[s, f32] = insert(sink_scalar, cast(0, i32), seq)
  sink_col = insert(sink_rows, cast(1, i32), cast(1, i64))
  extended = join_columns(masked, sink_col)
  weights_ext = softmax(extended, -1)
  weights = shrink(weights_ext, [[cast(0, i64), seq], [cast(0, i64), seq]])
  out = matmul(weights, v)
  _ = kt
  _ = scores
  _ = scaled
  _ = masked
  _ = sink_scalar
  _ = sink_rows
  _ = sink_col
  _ = extended
  _ = weights_ext
  _ = weights
  out
}
def join_columns[s](x: tensor[s, *, f32], y: tensor[s, *, f32]) = concat([x, y], 1i32)
def zero_qk() -> tensor[2, 3, f32] = to_tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
def sink_output(sink: f32) -> tensor[2, 3, f32] = {
  v = to_tensor([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])
  scale = to_tensor([[1.0, 1.0], [1.0, 1.0]])
  mask = to_tensor([[0.0, -1000000000.0], [0.0, 0.0]])
  causal_sdpa_with_sink(zero_qk(), zero_qk(), v, scale, mask, sink)
}
output = sink_output(0.0)
