def filled_like[n, m](reference: tensor[n, f32], source: tensor[m, f32]) -> tensor[n, f32] = 7.0f32 |> scalar_to_tensor |> insert(0i32, shape(source, 0i32))
def scaled_loss(scale: f32, reference: tensor[3, f32]) -> tensor[f32] = mul(sum(filled_like(reference, to_tensor([1.0f32, 2.0f32, 3.0f32])), 0i32), scalar_to_tensor(scale))
out = grad(scaled_loss, wrt=(scale, reference))(3.0f32, to_tensor([1.0f32, 2.0f32, 3.0f32]))
