module Examples.ConstructAlternatives
x = to_tensor([[1.0f32, 2.0f32]])
squeezed = reshape(x, [2i64])
unsqueezed = insert(squeezed, 0i32, 1i64)
stacked = concat([insert(squeezed, 0i32, 1i64), insert(to_tensor([3.0f32, 4.0f32]), 0i32, 1i64)], 0i32)
