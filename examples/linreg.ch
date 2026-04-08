-- Phase 1e executable benchmark: fixed-batch linear regression.
-- Uses the benchmark batch size directly because Phase 0e lowering still requires a
-- concrete reduced-axis extent for the final batch reduction in the scalar loss.

def predict(
  x: tensor[64, 64, f32],
  y: tensor[64, 1, f32],
  w: tensor[64, 1, f32],
  b: tensor[1, f32]
) -> tensor[64, 1, f32] =
  add(matmul(x, w), expand(b, 0, 64))

def loss(
  x: tensor[64, 64, f32],
  y: tensor[64, 1, f32],
  w: tensor[64, 1, f32],
  b: tensor[1, f32]
) -> tensor[f32] =
  let pred = add(matmul(x, w), expand(b, 0, 64))
  let pred_copy = copy(pred)
  let y_copy = copy(y)
  let err = add(pred, neg(y))
  let err_copy = add(pred_copy, neg(y_copy))
  let sq = mul(err, err_copy)
  in sum(sum(sq, 1), 0)
