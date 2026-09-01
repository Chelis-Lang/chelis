original = sub(reshape(expand(to_tensor([0.0f32]), 0, 6i64), [3i64, 2i64]), expand(to_tensor([1.0f32, 2.0f32]), 0, 3i64))
mirrored = sub(expand(to_tensor([1.0f32, 2.0f32]), 0, 3i64), reshape(expand(to_tensor([0.0f32]), 0, 6i64), [3i64, 2i64]))
