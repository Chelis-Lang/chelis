def fanout(x: tensor[1024, 1024, f32]) -> tensor[1024, 1024, f32] = {
  a = exp(copy(x))
  b = log(copy(x))
  c = sin(copy(x))
  d = neg(copy(x))
  e = sqrt(copy(x))
  add(add(add(a, b), add(c, d)), e)
}
