module Demo.Main
def loss[n](x: tensor[n, f32]) -> tensor[f32] = sum(cumsum(x, 0i32), 0i32)
def prefixes_of[n](x: tensor[n, f32]) -> tensor[n, f32] = cumsum(x, 0i32)
g = grad(loss)(to_tensor([1.0f32, 2.0f32, 3.0f32]))
batched = vmap(grad(loss))(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))
prefixes = vmap(prefixes_of)(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))
def integer_prefixes[time](x: tensor[time, i8]) -> tensor[time, i32] = cumsum(x, 0i32)
integer_out = vmap(integer_prefixes)(to_tensor([[127i8, 1i8, -1i8], [2i8, 3i8, 4i8]]))
