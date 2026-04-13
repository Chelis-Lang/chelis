module Std.Optim
export (AdamWConfig, AdamWState, LAMBConfig, LAMBState, adamw_init_like, adamw_step, lamb_init_like, lamb_step)
type AdamWConfig =
  | AdamWConfig { lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32 }
type AdamWState[a] =
  | AdamWState { step: int64, m: a, v: a }
type LAMBConfig =
  | LAMBConfig { lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32 }
type LAMBState[a] =
  | LAMBState { step: int64, m: a, v: a }
def adamw_init_like[n](params: tensor[n, f32]) = AdamWState { step: cast(0, int64), m: zeros_like(copy(params)), v: zeros_like(params) }
def adamw_step[n](params: tensor[n, f32], grads: tensor[n, f32], state: AdamWState[tensor[n, f32]], config: AdamWConfig) = { match config with {
  | AdamWConfig { lr: lr, beta1: beta1, beta2: beta2, eps: eps, weight_decay: weight_decay } => match state with {
  | AdamWState { step: step, m: m, v: v } => {
  next_step = add(step, cast(1, int64))
  one = cast(1.0, f32)
  m_next = tensor_add(tensor_scale(m, beta1), tensor_scale(copy(grads), sub(one, beta1)))
  v_next = tensor_add(tensor_scale(v, beta2), tensor_scale(tensor_mul(copy(grads), grads), sub(one, beta2)))
  bias1 = sub(one, powf(beta1, next_step))
  bias2 = sub(one, powf(beta2, next_step))
  m_hat = tensor_scale(copy(m_next), div(one, bias1))
  v_hat = tensor_scale(copy(v_next), div(one, bias2))
  denom = tensor_add_scalar(tensor_sqrt(v_hat), eps)
  update = tensor_add(tensor_div(m_hat, denom), tensor_scale(copy(params), weight_decay))
  (tensor_sub(params, tensor_scale(update, lr)), AdamWState { step: next_step, m: m_next, v: v_next })
}
}
} }
def lamb_init_like[n](params: tensor[n, f32]) = LAMBState { step: cast(0, int64), m: zeros_like(copy(params)), v: zeros_like(params) }
def lamb_step[n](params: tensor[n, f32], grads: tensor[n, f32], state: LAMBState[tensor[n, f32]], config: LAMBConfig) = { match config with {
  | LAMBConfig { lr: lr, beta1: beta1, beta2: beta2, eps: eps, weight_decay: weight_decay } => match state with {
  | LAMBState { step: step, m: m, v: v } => {
  next_step = add(step, cast(1, int64))
  one = cast(1.0, f32)
  m_next = tensor_add(tensor_scale(m, beta1), tensor_scale(copy(grads), sub(one, beta1)))
  v_next = tensor_add(tensor_scale(v, beta2), tensor_scale(tensor_mul(copy(grads), grads), sub(one, beta2)))
  bias1 = sub(one, powf(beta1, next_step))
  bias2 = sub(one, powf(beta2, next_step))
  m_hat = tensor_scale(copy(m_next), div(one, bias1))
  v_hat = tensor_scale(copy(v_next), div(one, bias2))
  adam_step = tensor_add(tensor_div(m_hat, tensor_add_scalar(tensor_sqrt(v_hat), eps)), tensor_scale(copy(params), weight_decay))
  param_norm = l2_norm(copy(params))
  update_norm = l2_norm(copy(adam_step))
  trust_ratio = if and(gt(param_norm, cast(0.0, f32)), gt(update_norm, cast(0.0, f32))) then div(param_norm, update_norm) else cast(1.0, f32)
  (tensor_sub(params, tensor_scale(adam_step, mul(lr, trust_ratio))), LAMBState { step: next_step, m: m_next, v: v_next })
}
}
} }
def zeros_like[n](value: tensor[n, f32]) -> tensor[n, f32] = to_tensor(map(fn (x) -> cast(0.0, f32), to_list(value)))
def tensor_scale[n](value: tensor[n, f32], factor: f32) -> tensor[n, f32] = to_tensor(map(fn (x: f32) -> mul(x, factor), to_list(value)))
def tensor_add[n](lhs: tensor[n, f32], rhs: tensor[n, f32]) -> tensor[n, f32] = to_tensor(map(fn (pair) -> add(pair.0, pair.1), zip(to_list(lhs), to_list(rhs))))
def tensor_sub[n](lhs: tensor[n, f32], rhs: tensor[n, f32]) -> tensor[n, f32] = to_tensor(map(fn (pair) -> sub(pair.0, pair.1), zip(to_list(lhs), to_list(rhs))))
def tensor_mul[n](lhs: tensor[n, f32], rhs: tensor[n, f32]) -> tensor[n, f32] = to_tensor(map(fn (pair) -> mul(pair.0, pair.1), zip(to_list(lhs), to_list(rhs))))
def tensor_div[n](lhs: tensor[n, f32], rhs: tensor[n, f32]) -> tensor[n, f32] = to_tensor(map(fn (pair) -> div(pair.0, pair.1), zip(to_list(lhs), to_list(rhs))))
def tensor_add_scalar[n](value: tensor[n, f32], scalar: f32) -> tensor[n, f32] = to_tensor(map(fn (x: f32) -> add(x, scalar), to_list(value)))
def tensor_sqrt[n](value: tensor[n, f32]) -> tensor[n, f32] = to_tensor(map(fn (x: f32) -> sqrt(x), to_list(value)))
def l2_norm[n](value: tensor[n, f32]) -> f32 = sqrt(fold(fn (acc: f32, x: f32) -> add(acc, mul(x, x)), cast(0.0, f32), to_list(value)))
def powf(base: f32, exp: int64) -> f32 = if lte(exp, cast(0, int64)) then cast(1.0, f32) else mul(base, powf(base, sub(exp, cast(1, int64))))
