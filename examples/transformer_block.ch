def forward(x: tensor[seq, 256, f32], wq: tensor[256, 64, f32], wk: tensor[256, 64, f32], wv: tensor[256, 64, f32], wo: tensor[64, 256, f32], ff1: tensor[256, 1024, f32], ff2: tensor[1024, 256, f32], gamma1: tensor[256, f32], beta1: tensor[256, f32], gamma2: tensor[256, f32], beta2: tensor[256, f32]) -> tensor[seq, 256, f32] = {
  q = matmul(x, wq)
  k = matmul(x, wk)
  v = matmul(x, wv)
  scores = matmul(q, permute(k, 1, 0))
  probs = softmax(scores, 1)
  attn = matmul(matmul(probs, v), wo)
  norm1 = layer_norm(add(x, attn), gamma1, beta1)
  hidden = matmul(norm1, ff1) |> relu |> matmul(ff2)
  residual = add(norm1, hidden)
  out = layer_norm(residual, gamma2, beta2)
  out
}
