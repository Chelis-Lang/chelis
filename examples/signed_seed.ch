template = to_tensor([0.0f32, 0.0f32])
sampled = with seed(-1i64) { uniform_like(template, 0.0f32, 1.0f32) }
