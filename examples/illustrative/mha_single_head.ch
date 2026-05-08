def block(x: tensor[seq, 256, f32], wq: tensor[256, 64, f32], wk: tensor[256, 64, f32], wv: tensor[256, 64, f32], wo: tensor[64, 256, f32]) -> tensor[seq, 256, f32] = {
  q = matmul(copy(x), wq)
  k = matmul(copy(x), wk)
  v = matmul(copy(x), wv)
  scores = matmul(q, permute(k, 1, 0))
  probs = softmax(scores, 1)
  attn_out = matmul(matmul(probs, v), wo)
  add(x, attn_out)
}
