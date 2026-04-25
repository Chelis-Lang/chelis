module Std.Tests.Optim
import Std.Optim (AdamWConfig, AdamWState, LAMBConfig, LAMBState, adamw_init_like, adamw_step, lamb_init_like, lamb_step)
import Std.Test (assert_close, assert_close_tensor, assert_eq_int, assert_true)
-- Acceptance surface for `Std.Optim` at Phase 3t A1 Wave 2.
--
-- The optimizer bodies in `src/optim.ch` are written entirely on top of
-- `to_list` / `to_tensor` / `map` / `zip` / `fold` plus elementwise scalar
-- arithmetic. None of them route through matmul or `expand`, so they
-- reach `chelis test`'s host-runtime evaluator (probed by writing a
-- temporary file invoking each step and observing PASS).
--
-- Every test below pins an algebraic identity derivable from the
-- shipped optimizer math, NEVER a value lifted from PyTorch or SciPy.
-- The identities are derived from the bodies in `src/optim.ch`:
--
--   AdamW first step (m=0, v=0, step=0):
--     m_next  = grads * (1 - beta1)
--     v_next  = grads^2 * (1 - beta2)
--     m_hat   = m_next / (1 - beta1)             = grads
--     v_hat   = v_next / (1 - beta2)             = grads^2
--     denom   = sqrt(v_hat) + eps                = |grads| + eps
--     update  = m_hat / denom + params * weight_decay
--     params' = params - lr * update
--
--   With grads = 0 and weight_decay = 0:  update = 0, so params' = params.
--   With grads = 0 and weight_decay > 0:  update = params * wd, so
--     params' = params * (1 - lr * wd)             [decoupled WD identity]
--
--   LAMB first step shares the AdamW computation up through `adam_step`,
--   then scales by trust_ratio = ||params|| / ||adam_step|| (when both
--   norms are positive, else 1.0). The norm-of-step identity follows:
--     ||params - params'|| = ||lr * trust_ratio * adam_step||
--                          = lr * ||params||           [LAMB norm identity]
--   when both ||params|| > 0 and ||adam_step|| > 0.
--
-- All tolerances below are chosen against the eps + finite-precision
-- residual of the closed-form identity, not against any external numeric
-- oracle.
--
-- Tuple destructuring note: `match result with | (a, b) => ...` does not
-- introduce `a` and `b` into the arm body in the current evaluator path
-- used by `chelis test` (verified by mutation: referencing the bound
-- name fails compile with `unbound variable: a`). The tests below use
-- the documented `.0` / `.1` tuple-field access instead, which is the
-- same form used inside `src/optim.ch::tensor_add`.
def l2_norm_list[n](t: tensor[n, f32]) -> f32 = sqrt(fold(fn (acc: f32, x: f32) -> add(acc, mul(x, x)), cast(0.0, f32), to_list(t)))
def test_adamw_zero_grad_zero_wd_leaves_params_unchanged() -> unit ! { Test } = {
  -- Identity: grads=0 and weight_decay=0 implies update=0, so the
  -- post-step params equal the pre-step params exactly. Tolerance is
  -- 0.0 because every term that could introduce numerical noise is
  -- multiplied by zero before any sqrt or division on a tensor element
  -- would round.
  params = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(-3.0, f32)])
  grads = to_tensor([cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)])
  s0 = adamw_init_like(copy(params))
  cfg = AdamWConfig { lr: cast(0.001, f32), beta1: cast(0.9, f32), beta2: cast(0.999, f32), eps: cast(0.00000001, f32), weight_decay: cast(0.0, f32) }
  result = adamw_step(copy(params), grads, s0, cfg)
  params_next = result.0
  assert_close_tensor(params_next, params, cast(0.0, f32), "adamw_step with grad=0 and weight_decay=0 leaves params exactly unchanged")
}
def test_adamw_zero_grad_decoupled_weight_decay_shrinks_params() -> unit ! { Test } = {
  -- Decoupled-WD identity (grad=0): params' = params * (1 - lr * wd).
  -- With lr=0.1 and wd=0.5, the shrink factor is 1 - 0.05 = 0.95, so
  -- [1, 2, -3] -> [0.95, 1.9, -2.85]. The eps in the denom only ever
  -- meets a numerator of 0/eps = 0, so the eps drift never reaches the
  -- subtraction; tolerance is set to the rounding tolerance of an f32
  -- multiply.
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
  -- State invariant: after one call to adamw_step, the step field of
  -- the returned AdamWState is exactly 1 (init starts it at 0). This
  -- pins the bias-correction time-base shipped by the optimizer, which
  -- is later used by the schedule integrator to drive lr decays.
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
  -- Init invariant: adamw_init_like returns step=0 and m, v that are
  -- elementwise zero with the same shape as the template. Nonzero
  -- moments at step=0 would silently bias the first true step away
  -- from the closed-form first-step identities used in the other
  -- tests in this file.
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
  -- LAMB norm identity (first step, m=0, v=0, weight_decay=0):
  --   adam_step = grads / (|grads| + eps)  (elementwise, after the bias
  --   corrections cancel; magnitude ~= 1 per element when grads != 0)
  --   trust_ratio = ||params|| / ||adam_step||
  --   params' - params = -lr * trust_ratio * adam_step
  -- Therefore ||params - params'|| = lr * ||params||, independent of
  -- the direction of grads, as long as ||params|| > 0 and
  -- ||adam_step|| > 0 (so the trust_ratio takes the non-default branch).
  -- With params = [3, 4] we have ||params|| = 5; with lr = 0.01 the
  -- expected step magnitude is exactly 0.05. The eps drift in
  -- adam_step magnitude is ~eps/|grad_i| per element, so the
  -- ||params-params'|| == lr*||params|| identity holds to a few ulps;
  -- 1e-5 is comfortable.
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
  -- LAMB degenerate branch: with grads=0 and weight_decay=0, the
  -- ||adam_step|| > 0 guard fails (adam_step is the zero tensor) and
  -- the trust_ratio falls back to 1.0 by the optimizer's `else` branch.
  -- The actual update is then lr*1.0*0 = 0, so params is unchanged
  -- exactly. This pins the fallback branch (the "either norm is zero"
  -- guard) of the trust_ratio expression in `src/optim.ch`.
  params = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(-3.0, f32)])
  grads = to_tensor([cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)])
  s0 = lamb_init_like(copy(params))
  cfg = LAMBConfig { lr: cast(0.01, f32), beta1: cast(0.9, f32), beta2: cast(0.999, f32), eps: cast(0.00000001, f32), weight_decay: cast(0.0, f32) }
  result = lamb_step(copy(params), grads, s0, cfg)
  params_next = result.0
  assert_close_tensor(params_next, params, cast(0.0, f32), "lamb_step with grad=0 and wd=0 takes the trust_ratio=1 fallback branch and leaves params unchanged")
}
