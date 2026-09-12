def sample(x: tensor[4, f32]) -> tensor[4, f32] = with seed(42i64) { dropout(x, 0.5f32) }
