def sample(x: tensor[4, f32]) -> tensor[4, f32] = 1i64 |> key_from_seed |> dropout(x, 0.5f32)
