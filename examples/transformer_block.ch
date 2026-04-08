-- Phase 1e executable benchmark: sequence-only transformer-block-style forward pass.
-- Shapes: d_model=256, n_heads=4, head_dim=64, d_ff=1024.

def forward(
  x: tensor[seq, 256, f32],
  wq0: tensor[256, 64, f32],
  wk0: tensor[256, 64, f32],
  wv0: tensor[256, 64, f32],
  wo0: tensor[64, 256, f32],
  wq1: tensor[256, 64, f32],
  wk1: tensor[256, 64, f32],
  wv1: tensor[256, 64, f32],
  wo1: tensor[64, 256, f32],
  wq2: tensor[256, 64, f32],
  wk2: tensor[256, 64, f32],
  wv2: tensor[256, 64, f32],
  wo2: tensor[64, 256, f32],
  wq3: tensor[256, 64, f32],
  wk3: tensor[256, 64, f32],
  wv3: tensor[256, 64, f32],
  wo3: tensor[64, 256, f32],
  ff1: tensor[256, 1024, f32],
  ff2: tensor[1024, 256, f32],
  gamma1: tensor[256, f32],
  beta1: tensor[256, f32],
  gamma2: tensor[256, f32],
  beta2: tensor[256, f32]
) -> tensor[seq, 256, f32] =
  let q0 = matmul(x, wq0)
  let k0 = matmul(x, wk0)
  let v0 = matmul(x, wv0)
  let scores0 = matmul(q0, permute(k0, 1, 0))
  let probs0 = softmax(scores0, 1)
  let head0 = matmul(matmul(probs0, v0), wo0)

  let q1 = matmul(x, wq1)
  let k1 = matmul(x, wk1)
  let v1 = matmul(x, wv1)
  let scores1 = matmul(q1, permute(k1, 1, 0))
  let probs1 = softmax(scores1, 1)
  let head1 = matmul(matmul(probs1, v1), wo1)

  let q2 = matmul(x, wq2)
  let k2 = matmul(x, wk2)
  let v2 = matmul(x, wv2)
  let scores2 = matmul(q2, permute(k2, 1, 0))
  let probs2 = softmax(scores2, 1)
  let head2 = matmul(matmul(probs2, v2), wo2)

  let q3 = matmul(x, wq3)
  let k3 = matmul(x, wk3)
  let v3 = matmul(x, wv3)
  let scores3 = matmul(q3, permute(k3, 1, 0))
  let probs3 = softmax(scores3, 1)
  let head3 = matmul(matmul(probs3, v3), wo3)

  let attn_out = add(add(head0, head1), add(head2, head3))
  let norm1 = layer_norm(add(x, attn_out), gamma1, beta1)
  let ff_out = matmul(relu(matmul(norm1, ff1)), ff2)
  in layer_norm(add(norm1, ff_out), gamma2, beta2)
