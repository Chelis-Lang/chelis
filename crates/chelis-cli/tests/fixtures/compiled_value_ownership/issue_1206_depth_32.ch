def step(state: tensor[4, f32], index: int64) -> tensor[4, f32] =
  if gte(index, 32i64) then state else {
    offset = to_tensor([1.0, 1.0, 1.0, 1.0])
    shifted = add(state, offset)
    scaled = mul(shifted, offset)
    step(scaled, add(index, 1i64))
  }
out = step(to_tensor([0.0, 0.0, 0.0, 0.0]), 0i64)
