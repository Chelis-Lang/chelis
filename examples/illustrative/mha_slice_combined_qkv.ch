-- Illustrative: slice a fused QKV projection into Q / K / V along the
-- feature axis using shrink, then run a tiny attention block.
--
-- shrink(&qkv, [[axis0_start, axis0_end], [axis1_start, axis1_end]])
-- extracts the sub-tensor qkv[axis0_start..axis0_end, axis1_start..axis1_end].
-- Here the fused qkv has feature dim 192 = 3 * 64; Q takes columns 0..64,
-- K takes 64..128, V takes 128..192. See spec/05-risc-primitives.md section 2.4.
--
-- The dims are concrete (seq = 8) to keep the slice bounds literal;
-- shrink also accepts computed bounds, as examples/checked_reshape.ch shows.
def block(x: tensor[8, 256, f32], wqkv: tensor[256, 192, f32], wo: tensor[64, 256, f32]) -> tensor[8, 256, f32] = {
  qkv = matmul(x, wqkv)
  q = shrink(&qkv, [[0i64, 8i64], [0i64, 64i64]])
  k = shrink(&qkv, [[0i64, 8i64], [64i64, 128i64]])
  v = shrink(&qkv, [[0i64, 8i64], [128i64, 192i64]])
  scores = matmul(q, permute(k, 1, 0))
  probs = softmax(scores, 1)
  attn_out = probs |> matmul(v) |> matmul(wo)
  add(x, attn_out)
}
