-- Illustrative: slice a fused QKV projection into Q / K / V along the
-- feature axis using shrink, then run a tiny attention block.
--
-- shrink(&qkv, [[axis0_start, axis0_end], [axis1_start, axis1_end]])
-- extracts the sub-tensor qkv[axis0_start..axis0_end, axis1_start..axis1_end].
-- Here the fused qkv has feature dim 192 = 3 * 64; Q takes columns 0..64,
-- K takes 64..128, V takes 128..192. See spec/05-risc-primitives.md section 2.4.
--
-- This file uses concrete dims (seq = 8) because the current type-checker
-- requires literal int bounds in shrink -- symbolic-axis windowing is
-- out of scope for issue Chelis-Lang/chelis 187.
def block(x: tensor[8, 256, f32], wqkv: tensor[256, 192, f32], wo: tensor[64, 256, f32]) -> tensor[8, 256, f32] = {
  qkv = matmul(copy(x), wqkv)
  q = shrink(&qkv, [[0, 8], [0, 64]])
  k = shrink(&qkv, [[0, 8], [64, 128]])
  v = shrink(&qkv, [[0, 8], [128, 192]])
  scores = matmul(q, permute(k, 1, 0))
  probs = softmax(scores, 1)
  attn_out = matmul(matmul(probs, v), wo)
  add(x, attn_out)
}
