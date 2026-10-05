def step(state: tensor[4, f32], index: i64) -> tensor[4, f32] =
  if gte(index, 1i64) then state else {
    offset = to_tensor([1.0, 1.0, 1.0, 1.0], f32)
    shifted = add(state, offset)
    scaled = mul(shifted, offset)
    step(scaled, add(index, 1i64))
  }
out = [0.0, 0.0, 0.0, 0.0] |> to_tensor(f32) |> step(0i64)
