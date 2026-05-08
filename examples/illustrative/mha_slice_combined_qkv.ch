def block(x: tensor[seq, 256, f32], wqkv: tensor[256, 192, f32], wo: tensor[192, 256, f32]) -> tensor[seq, 256, f32] = {
  qkv = matmul(copy(x), wqkv)
  q = shrink(&qkv)
  k = shrink(&qkv)
  v = shrink(&qkv)
  scores = matmul(q, permute(k, 1, 0))
  probs = softmax(scores, 1)
  attn_out = matmul(matmul(probs, v), wo)
  add(x, attn_out)
}
