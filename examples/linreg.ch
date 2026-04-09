def predict(x: tensor[64, 64, f32], y: tensor[64, 1, f32], w: tensor[64, 1, f32], b: tensor[1, f32]) -> tensor[64, 1, f32] =
  add(matmul(x, w), expand(b, 0, 64))
def loss(x: tensor[64, 64, f32], y: tensor[64, 1, f32], w: tensor[64, 1, f32], b: tensor[1, f32]) -> tensor[f32] =
  let pred = add(matmul(x, w), expand(b, 0, 64))
  in let pred_copy = copy(pred)
  in let y_copy = copy(y)
  in let err = add(pred, neg(y))
  in let err_copy = add(pred_copy, neg(y_copy))
  in let sq = mul(err, err_copy)
  in sum(sum(sq, 1), 0)