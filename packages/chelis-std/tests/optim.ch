module Std.Tests.Optim
import Std.Optim (AdamWConfig, AdamWState, LAMBConfig, LAMBState, adamw_init_like, adamw_step, lamb_init_like, lamb_step)
import Std.Test (assert_close, assert_close_tensor, assert_eq_int, assert_true)
def l2_norm_list[n](t: tensor[n, f32]) -> f32 = sqrt(fold(fn (acc: f32, x: f32) -> add(acc, mul(x, x)), cast(0.0, f32), to_list(t)))
def test_adamw_zero_grad_zero_wd_leaves_params_unchanged() -> unit ! { Test } = {
  params = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(-3.0, f32)])
  grads = to_tensor([cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)])
  s0 = adamw_init_like(copy(params))
  cfg = AdamWConfig { lr: cast(0.001, f32), beta1: cast(0.9, f32), beta2: cast(0.999, f32), eps: cast(0.00000001, f32), weight_decay: cast(0.0, f32) }
  result = adamw_step(copy(params), grads, s0, cfg)
  params_next = result.0
  assert_close_tensor(params_next, params, cast(0.0, f32), "adamw_step with grad=0 and weight_decay=0 leaves params exactly unchanged")
}
def test_adamw_zero_grad_decoupled_weight_decay_shrinks_params() -> unit ! { Test } = {
  params = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(-3.0, f32)])
  grads = to_tensor([cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)])
  s0 = adamw_init_like(copy(params))
  cfg = AdamWConfig { lr: cast(0.1, f32), beta1: cast(0.9, f32), beta2: cast(0.999, f32), eps: cast(0.00000001, f32), weight_decay: cast(0.5, f32) }
  result = adamw_step(params, grads, s0, cfg)
  params_next = result.0
  expected = to_tensor([cast(0.95, f32), cast(1.9, f32), cast(-2.85, f32)])
  assert_close_tensor(params_next, expected, cast(0.000001, f32), "adamw_step with grad=0 and wd>0 implements decoupled WD: params' = params * (1 - lr*wd)")
}
def test_adamw_step_counter_increments_from_zero_to_one() -> unit ! { Test } = {
  params = to_tensor([cast(1.0, f32), cast(2.0, f32)])
  grads = to_tensor([cast(0.5, f32), cast(0.25, f32)])
  s0 = adamw_init_like(copy(params))
  cfg = AdamWConfig { lr: cast(0.001, f32), beta1: cast(0.9, f32), beta2: cast(0.999, f32), eps: cast(0.00000001, f32), weight_decay: cast(0.0, f32) }
  result = adamw_step(params, grads, s0, cfg)
  state_next = result.1
  match state_next with {
    | AdamWState { step: step, m: m, v: v } => assert_eq_int(step, cast(1, int64), "adamw_step increments state.step from 0 to 1 after one call")
  }
}
def test_adamw_init_like_has_zero_step_and_zero_moments() -> unit ! { Test } = {
  params = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32)])
  s0 = adamw_init_like(params)
  expected_zeros = to_tensor([cast(0.0, f32), cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)])
  match s0 with {
    | AdamWState { step: step, m: m, v: v } => {
    step_check = assert_eq_int(step, cast(0, int64), "adamw_init_like sets state.step = 0")
    m_check = assert_close_tensor(m, copy(expected_zeros), cast(0.0, f32), "adamw_init_like sets state.m to zeros (shape and values)")
    assert_close_tensor(v, expected_zeros, cast(0.0, f32), "adamw_init_like sets state.v to zeros (shape and values)")
  }
  }
}
def test_lamb_step_norm_identity_first_step() -> unit ! { Test } = {
  params = to_tensor([cast(3.0, f32), cast(4.0, f32)])
  grads = to_tensor([cast(0.5, f32), cast(0.5, f32)])
  s0 = lamb_init_like(copy(params))
  cfg = LAMBConfig { lr: cast(0.01, f32), beta1: cast(0.9, f32), beta2: cast(0.999, f32), eps: cast(0.00000001, f32), weight_decay: cast(0.0, f32) }
  result = lamb_step(copy(params), grads, s0, cfg)
  params_next = result.0
  diffs = to_tensor(map(fn (pair: (f32, f32)) -> sub(pair.0, pair.1), zip(to_list(params), to_list(params_next))))
  step_norm = l2_norm_list(diffs)
  expected = mul(cast(0.01, f32), cast(5.0, f32))
  assert_close(step_norm, expected, cast(0.00001, f32), "lamb_step norm identity: ||params - params'|| == lr * ||params|| on first step (trust_ratio branch)")
}
def test_lamb_step_zero_grad_zero_wd_leaves_params_unchanged() -> unit ! { Test } = {
  params = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(-3.0, f32)])
  grads = to_tensor([cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)])
  s0 = lamb_init_like(copy(params))
  cfg = LAMBConfig { lr: cast(0.01, f32), beta1: cast(0.9, f32), beta2: cast(0.999, f32), eps: cast(0.00000001, f32), weight_decay: cast(0.0, f32) }
  result = lamb_step(copy(params), grads, s0, cfg)
  params_next = result.0
  assert_close_tensor(params_next, params, cast(0.0, f32), "lamb_step with grad=0 and wd=0 takes the trust_ratio=1 fallback branch and leaves params unchanged")
}
