def predict(x: tensor[64, 64, f32], y: tensor[64, 1, f32], w: tensor[64, 1, f32], b: tensor[1, f32]) -> tensor[64, 1, f32] = add(matmul(x, w), expand(b, 0, 64))
def loss(x: tensor[64, 64, f32], y: tensor[64, 1, f32], w: tensor[64, 1, f32], b: tensor[1, f32]) -> tensor[f32] = {
  pred = matmul(x, w) |> add(expand(b, 0, 64))
  pred_copy = copy(pred)
  y_copy = copy(y)
  err = add(pred, neg(y))
  err_copy = add(pred_copy, neg(y_copy))
  sq = mul(err, err_copy)
  sum(sum(sq, 1), 0)
}
