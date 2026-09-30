def block(x: tensor[seq, 256, f32], wq: tensor[256, 64, f32], wk: tensor[256, 64, f32], wv: tensor[256, 64, f32], wo: tensor[64, 256, f32]) -> tensor[seq, 256, f32] = {
  q = matmul(x, wq)
  k = matmul(x, wk)
  v = matmul(x, wv)
  scores = matmul(q, permute(k, 1, 0))
  probs = softmax(scores, 1)
  attn_out = probs |> matmul(v) |> matmul(wo)
  add(x, attn_out)
}
