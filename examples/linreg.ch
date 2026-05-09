def predict(x: tensor[64, 64, f32], y: tensor[64, 1, f32], w: tensor[64, 1, f32], b: tensor[1, f32]) -> tensor[64, 1, f32] = {
  linear = matmul(x, w)
  bias = expand(b, 0, 64)
  out = add(linear, bias)
  _ = drop(linear)
  _ = drop(bias)
  out
}
def loss(x: tensor[64, 64, f32], y: tensor[64, 1, f32], w: tensor[64, 1, f32], b: tensor[1, f32]) -> tensor[f32] = {
  linear = matmul(x, w)
  bias = expand(b, 0, 64)
  pred = add(linear, bias)
  neg_y = neg(y)
  err = add(pred, neg_y)
  sq = mul(err, err)
  per_col = sum(sq, 1)
  out = sum(per_col, 0)
  _ = drop(linear)
  _ = drop(bias)
  _ = drop(pred)
  _ = drop(neg_y)
  _ = drop(err)
  _ = drop(sq)
  _ = drop(per_col)
  out
}
