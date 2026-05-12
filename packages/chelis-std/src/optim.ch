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
def adamw_init_like[n](params: &tensor[n, f32]) = AdamWState { step: cast(0, int64), m: zeros_like(params), v: zeros_like(params) }
def adamw_step[n](params: tensor[n, f32], grads: &tensor[n, f32], state: AdamWState[tensor[n, f32]], config: AdamWConfig) = {
  match config with {
    | AdamWConfig { lr: lr, beta1: beta1, beta2: beta2, eps: eps, weight_decay: weight_decay } => match state with {
    | AdamWState { step: step, m: m, v: v } => {
    next_step = add(step, cast(1, int64))
    one = cast(1.0, f32)
    m_old = tensor_scale(m, beta1)
    m_grad = tensor_scale(grads, sub(one, beta1))
    m_next = tensor_add(m_old, m_grad)
    grad_sq = tensor_mul(grads, grads)
    v_old = tensor_scale(v, beta2)
    v_grad = tensor_scale(grad_sq, sub(one, beta2))
    v_next = tensor_add(v_old, v_grad)
    bias1 = sub(one, powf(beta1, next_step))
    bias2 = sub(one, powf(beta2, next_step))
    m_hat = tensor_scale(m_next, div(one, bias1))
    v_hat = tensor_scale(v_next, div(one, bias2))
    sqrt_v = tensor_sqrt(v_hat)
    denom = tensor_add_scalar(sqrt_v, eps)
    ratio = tensor_div(m_hat, denom)
    decay = tensor_scale(params, weight_decay)
    update = tensor_add(ratio, decay)
    scaled_update = tensor_scale(update, lr)
    next_params = tensor_sub(params, scaled_update)
    _ = drop(m)
    _ = drop(v)
    _ = drop(m_old)
    _ = drop(m_grad)
    _ = drop(grad_sq)
    _ = drop(v_old)
    _ = drop(v_grad)
    _ = drop(m_hat)
    _ = drop(v_hat)
    _ = drop(sqrt_v)
    _ = drop(denom)
    _ = drop(ratio)
    _ = drop(decay)
    _ = drop(update)
    _ = drop(scaled_update)
    _ = drop(params)
    (next_params, AdamWState { step: next_step, m: m_next, v: v_next })
  }
  }
  }
}
def lamb_init_like[n](params: &tensor[n, f32]) = LAMBState { step: cast(0, int64), m: zeros_like(params), v: zeros_like(params) }
def lamb_step[n](params: tensor[n, f32], grads: &tensor[n, f32], state: LAMBState[tensor[n, f32]], config: LAMBConfig) = {
  match config with {
    | LAMBConfig { lr: lr, beta1: beta1, beta2: beta2, eps: eps, weight_decay: weight_decay } => match state with {
    | LAMBState { step: step, m: m, v: v } => {
    next_step = add(step, cast(1, int64))
    one = cast(1.0, f32)
    m_old = tensor_scale(m, beta1)
    m_grad = tensor_scale(grads, sub(one, beta1))
    m_next = tensor_add(m_old, m_grad)
    grad_sq = tensor_mul(grads, grads)
    v_old = tensor_scale(v, beta2)
    v_grad = tensor_scale(grad_sq, sub(one, beta2))
    v_next = tensor_add(v_old, v_grad)
    bias1 = sub(one, powf(beta1, next_step))
    bias2 = sub(one, powf(beta2, next_step))
    m_hat = tensor_scale(m_next, div(one, bias1))
    v_hat = tensor_scale(v_next, div(one, bias2))
    sqrt_v = tensor_sqrt(v_hat)
    denom = tensor_add_scalar(sqrt_v, eps)
    ratio = tensor_div(m_hat, denom)
    decay = tensor_scale(params, weight_decay)
    adam_step = tensor_add(ratio, decay)
    param_norm = l2_norm(params)
    update_norm = l2_norm(adam_step)
    trust_ratio = if and(gt(param_norm, cast(0.0, f32)), gt(update_norm, cast(0.0, f32))) then div(param_norm, update_norm) else cast(1.0, f32)
    scaled_step = tensor_scale(adam_step, mul(lr, trust_ratio))
    next_params = tensor_sub(params, scaled_step)
    _ = drop(m)
    _ = drop(v)
    _ = drop(m_old)
    _ = drop(m_grad)
    _ = drop(grad_sq)
    _ = drop(v_old)
    _ = drop(v_grad)
    _ = drop(m_hat)
    _ = drop(v_hat)
    _ = drop(sqrt_v)
    _ = drop(denom)
    _ = drop(ratio)
    _ = drop(decay)
    _ = drop(adam_step)
    _ = drop(scaled_step)
    _ = drop(params)
    (next_params, LAMBState { step: next_step, m: m_next, v: v_next })
  }
  }
  }
}
def zeros_like[n](value: &tensor[n, f32]) -> tensor[n, f32] = to_tensor(map(fn (x) -> cast(0.0, f32), to_list(value)))
def tensor_scale[n](value: &tensor[n, f32], factor: f32) -> tensor[n, f32] = to_tensor(map(fn (x: f32) -> mul(x, factor), to_list(value)))
def tensor_add[n, p](lhs: &tensor[n, p], rhs: &tensor[n, p]) -> tensor[n, p] = add(lhs, rhs)
def tensor_sub[n, p](lhs: &tensor[n, p], rhs: &tensor[n, p]) -> tensor[n, p] = sub(lhs, rhs)
def tensor_mul[n, p](lhs: &tensor[n, p], rhs: &tensor[n, p]) -> tensor[n, p] = mul(lhs, rhs)
def tensor_div[n, p](lhs: &tensor[n, p], rhs: &tensor[n, p]) -> tensor[n, p] = div(lhs, rhs)
def tensor_add_scalar[n](value: &tensor[n, f32], scalar: f32) -> tensor[n, f32] = to_tensor(map(fn (x: f32) -> add(x, scalar), to_list(value)))
def tensor_sqrt[n](value: &tensor[n, f32]) -> tensor[n, f32] = sqrt(value)
def l2_norm[n](value: &tensor[n, f32]) -> f32 = { sqrt(fold(fn (acc: f32, x: f32) -> add(acc, mul(x, x)), cast(0.0, f32), to_list(value))) }
def powf(base: f32, exp: int64) -> f32 = if lte(exp, cast(0, int64)) then cast(1.0, f32) else mul(base, powf(base, sub(exp, cast(1, int64))))
