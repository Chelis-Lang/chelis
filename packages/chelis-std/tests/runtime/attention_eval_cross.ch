module Std.Tests.Runtime.AttentionEvalCross
import Std.Test (assert_close_tensor)
import Std.Nn.Attention (scaled_dot_product_attention)
def test_attention_eval_matches_build() -> unit ! { Test } = {
  zero_row = [cast(0.0, f32), cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)]
  ones_row = [cast(1.0, f32), cast(1.0, f32), cast(1.0, f32), cast(1.0, f32)]
  q = (pad_sequences_to([zero_row, zero_row, zero_row, zero_row], cast(4, int64), cast(0.0, f32)) : tensor[4, 4, f32])
  k = (pad_sequences_to([zero_row, zero_row, zero_row, zero_row], cast(4, int64), cast(0.0, f32)) : tensor[4, 4, f32])
  v = (pad_sequences_to([
    [cast(1.0, f32), cast(1.0, f32), cast(1.0, f32), cast(1.0, f32)],
    [cast(2.0, f32), cast(2.0, f32), cast(2.0, f32), cast(2.0, f32)],
    [cast(3.0, f32), cast(3.0, f32), cast(3.0, f32), cast(3.0, f32)],
    [cast(4.0, f32), cast(4.0, f32), cast(4.0, f32), cast(4.0, f32)]
  ], cast(4, int64), cast(0.0, f32)) : tensor[4, 4, f32])
  scale = (pad_sequences_to([ones_row, ones_row, ones_row, ones_row], cast(4, int64), cast(0.0, f32)) : tensor[4, 4, f32])
  out = scaled_dot_product_attention(q, k, v, scale)
  flat = reshape(out, [cast(16, int64)])
  assert_close_tensor(flat, to_tensor([cast(2.5, f32), cast(2.5, f32), cast(2.5, f32), cast(2.5, f32), cast(2.5, f32), cast(2.5, f32), cast(2.5, f32), cast(2.5, f32), cast(2.5, f32), cast(2.5, f32), cast(2.5, f32), cast(2.5, f32), cast(2.5, f32), cast(2.5, f32), cast(2.5, f32), cast(2.5, f32)]), cast(0.001, f32), "uniform attention is mean of v")
}
