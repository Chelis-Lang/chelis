def constant_loss[rows, cols](x: tensor[rows, cols, f32]) -> f32 = 1.0f32
out = grad(constant_loss)(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))
