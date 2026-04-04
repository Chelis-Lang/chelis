def predict(w: tensor[features, f32], b: f32, x: tensor[features, f32]): f32 =
  add(dot(w, x), b)

def mse_loss(pred: f32, target: f32): f32 =
  let diff = sub(pred, target)
  in mul(diff, diff)

def train_step(w: tensor[features, f32], b: f32, x: tensor[features, f32], y: f32, lr: f32): (tensor[features, f32], f32) =
  let loss_fn = fn (w_, b_) -> mse_loss(predict(w_, b_, x), y)
  let grads = grad(loss_fn)
  in (sub(w, mul(lr, fst(grads(w, b)))), sub(b, mul(lr, snd(grads(w, b)))))
