weights = to_tensor([10.0f32, 20.0f32])
def dot_weights(x: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(mul(x, weights), 0))
out: tensor[3, f32] = vmap(dot_weights)(to_tensor([[1.0f32, 1.0f32], [2.0f32, 2.0f32], [0.0f32, 1.0f32]]))
