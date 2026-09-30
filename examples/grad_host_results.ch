-- Scalar and rank-zero tensor cotangents retain their distinct types.
def loss(x: tensor[f32], scale: f32) -> f32 = x |> mul(scalar_to_tensor(scale)) |> tensor_to_scalar
g = grad(loss, wrt=(scale, x))(scalar_to_tensor(3.0f32), 2.0f32)
out = add(g.0, tensor_to_scalar(g.1))
