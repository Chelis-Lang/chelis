def basis[n](index: i64, scale: f32, reference: &tensor[n, f32]) -> tensor[n, f32] = {
  indices = range(0i64, reference |> to_list |> len)
  weights = map(fn (position: i64) -> if eq(position, index) then scale else 0.0f32, indices)
  to_tensor(weights)
}
def loss[n](parameters: tensor[n, f32]) -> f32 = tensor_to_scalar(sum(mul(parameters, basis(1i64, 3.0f32, parameters)), 0i32))
out = grad(loss)(to_tensor([2.0f32, 4.0f32, 6.0f32]))
