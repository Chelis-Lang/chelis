def attention_scores(q: tensor[batch, heads, seq, head_dim, f32], k_t: tensor[batch, heads, head_dim, seq, f32]) -> tensor[batch, heads, seq, seq, f32] = matmul(q, k_t)
