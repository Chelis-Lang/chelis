def filled_like(reference: tensor[n, f32], source: tensor[m, f32]) -> tensor[n, f32] = insert(scalar_to_tensor(7.0f32), 0i32, shape(source, 0i32))
def loss(x: tensor[3, f32], z: tensor[3, f32]) -> tensor[f32] = sum(mul(filled_like(x, to_tensor([1.0f32, 2.0f32, 3.0f32])), z), 0i32)
out = grad(loss, wrt=(x, z))(to_tensor([1.0f32, 2.0f32, 3.0f32]), to_tensor([4.0f32, 5.0f32, 6.0f32]))
