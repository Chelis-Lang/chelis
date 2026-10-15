module Demo.Main
def loss[n](x: tensor[n, f32]) -> tensor[f32] = sum(cumsum(x, 0i32), 0i32)
def prefixes_of[n](x: tensor[n, f32]) -> tensor[n, f32] = cumsum(x, 0i32)
g = grad(loss)(to_tensor([1.0f32, 2.0f32, 3.0f32]))
batched = vmap(grad(loss))(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))
prefixes = vmap(prefixes_of)(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))
