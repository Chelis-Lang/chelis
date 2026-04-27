-- ILLUSTRATIVE ONLY — NOT EXECUTABLE.
-- This example sketches a linear-regression update step in a syntax-first style.
-- The `grad(loss_fn)` pattern below uses a locally-bound fn whose `wrt` defaults
-- to ALL parameters, including the scalar bias `b_`. That requires host-lane
-- scalar AD, which is a Phase 5 deferred feature
-- (see `spec/design/phase5_host_scalar_ad.md`).
--
-- The supported tensor-lane pattern restricts `wrt` to a single tensor parameter
-- and closes scalar args over the local fn:
--   target = fn (theta_local: tensor[n, f32]) -> loss(theta_local, x_const, y_const)
--   grad(target, wrt=(theta_local))(theta)
-- See `crates/chelis-cli/tests/cli.rs::build_c_tensor_grad_local_wrapper_over_function_param_builds`
-- for the working version.

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
