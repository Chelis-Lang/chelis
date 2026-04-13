def predict(w: tensor[features, f32], b: f32, x: tensor[features, f32]) -> f32 = add(dot(w, x), b)
def mse_loss(pred: f32, target: f32) -> f32 = {
  diff = sub(pred, target)
  mul(diff, diff)
}
def train_step(w: tensor[features, f32], b: f32, x: tensor[features, f32], y: f32, lr: f32) -> (tensor[features, f32], f32) = {
  loss_fn = fn (w_, b_) -> mse_loss(predict(w_, b_, x), y)
  grads = grad(loss_fn)
  (sub(w, mul(lr, fst(grads(w, b)))), sub(b, mul(lr, snd(grads(w, b)))))
}
