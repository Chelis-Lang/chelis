template = to_tensor([0.0f32, 0.0f32])
sampled = -1i64 |> key_from_seed |> uniform_like(template, 0.0f32, 1.0f32)
