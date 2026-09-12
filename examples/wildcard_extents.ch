-- Each unknown extent is independent; * is not a shared dimension name.
def widths(x: tensor[*, *, f32]) = (shape(&x, 0i32), shape(x, 1i32))
output = widths(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))
