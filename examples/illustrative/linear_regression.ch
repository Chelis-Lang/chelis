def predict(w: tensor[features, f32], b: tensor[f32], x: tensor[features, f32]) -> tensor[f32] =
  x
  |> mul(w)
  |> sum(0i32)
  |> add(b)
def squared_error(w: tensor[features, f32], b: tensor[f32], x: tensor[features, f32], y: tensor[f32]) -> tensor[f32] = {
  diff = w |> predict(b, x) |> sub(y)
  mul(diff, diff)
}
def train_step(w: tensor[features, f32], b: tensor[f32], x: tensor[features, f32], y: tensor[f32], lr: tensor[f32]) -> (tensor[features, f32], tensor[f32]) = {
  (dw, db) = grad(squared_error, wrt=(w, b))(w, b, x, y)
  step = insert(lr, 0, shape(w, 0i32))
  (sub(w, mul(step, dw)), sub(b, mul(lr, db)))
}
