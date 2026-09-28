def sample(x: tensor[4, f32]) -> tensor[4, f32] = dropout(key_from_seed(1i64), x, 0.5f32)
