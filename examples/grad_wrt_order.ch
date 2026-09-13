def loss(x: tensor[f32], y: tensor[f32]) -> f32 = tensor_to_scalar(mul(x, y))
def verify(x: tensor[f32], y: tensor[f32]) -> bool = {
  ds = grad(loss, wrt=(y, x))(x, y)
  and(eq(tensor_to_scalar(ds.0), tensor_to_scalar(x)), eq(tensor_to_scalar(ds.1), tensor_to_scalar(y)))
}
out = verify(scalar_to_tensor(2.0f32), scalar_to_tensor(3.0f32))
