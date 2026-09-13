def loss(x: tensor[f32]) -> f32 = 3.0f32
out = vmap(grad(loss))(to_tensor([2.0f32, 7.0f32]))
