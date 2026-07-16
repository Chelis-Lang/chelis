def g(x: tensor[4, f32]) -> tensor[f32] = {
  w = cast(abs(to_tensor([cast(-100, int64), cast(200, int64), cast(-300, int64), cast(400, int64)])), f32)
  sum(mul(copy(x), w), 0)
}
out = print(g(to_tensor([0.1, 0.2, 0.3, 0.4])))
