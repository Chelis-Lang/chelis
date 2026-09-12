def draw(x: tensor[4, f32]) -> tensor[4, f32] ! { Random } = {
  keep = fn (v: tensor[4, f32], rate: f32) -> dropout(v, rate)
  rate = 0.5f32
  keep(x, rate)
}
def loss(x: tensor[4, f32]) -> tensor[f32] ! { Random } = sum(draw(x), 0i32)
def generic_keep[p: Float](x: tensor[4, p]) -> tensor[4, p] ! { Random } = dropout(x, cast(0.5, p))
result = with seed(42i64) {
  x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])
  forward = draw(copy(x))
  backward = grad(loss)(copy(x))
  following = generic_keep(x)
  (forward, backward, following, x)
}
