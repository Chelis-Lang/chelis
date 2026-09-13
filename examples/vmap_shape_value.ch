def width(x: tensor[3, f32]) -> f32 = x |> shape(0) |> cast(f32)
out = vmap(width)(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))
