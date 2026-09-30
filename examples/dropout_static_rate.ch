def draw(k: key, x: tensor[4, f32]) -> tensor[4, f32] = {
  keep = fn (j: key, v: tensor[4, f32], rate: f32) -> dropout(j, v, rate)
  rate = 0.5f32
  keep(k, x, rate)
}
def loss(k: key, x: tensor[4, f32]) -> tensor[f32] = k |> draw(x) |> sum(0i32)
def generic_keep[p: Float](k: key, x: tensor[4, p]) -> tensor[4, p] = dropout(k, x, cast(0.5, p))
result = {
  (k1, rest) = 42i64 |> key_from_seed |> split_key
  (k2, k3) = split_key(rest)
  x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])
  forward = draw(k1, copy(x))
  backward = grad(loss, wrt=x)(k2, x)
  following = generic_keep(k3, x)
  (forward, backward, following, x)
}
