module Std.Tests.Schedule
import Std.Schedule (CosineWarmupConfig, LinearWarmupConfig, StepDecayConfig, cosine_with_warmup, linear_warmup, step_decay)
import Std.Test (assert_close)
def test_cosine_warmup_peaks_at_warmup_steps() -> unit ! { Test } = {
  config = CosineWarmupConfig { warmup_steps: cast(10, int64), total_steps: cast(110, int64), min_lr: cast(0.001, f32), max_lr: cast(0.1, f32) }
  lr = cosine_with_warmup(cast(10, int64), config)
  assert_close(lr, cast(0.1, f32), cast(0.000001, f32), "cosine_with_warmup(step=warmup_steps) == max_lr (peak)")
}
def test_cosine_warmup_returns_min_lr_at_total_steps() -> unit ! { Test } = {
  config = CosineWarmupConfig { warmup_steps: cast(10, int64), total_steps: cast(110, int64), min_lr: cast(0.001, f32), max_lr: cast(0.1, f32) }
  lr = cosine_with_warmup(cast(110, int64), config)
  assert_close(lr, cast(0.001, f32), cast(0.0, f32), "cosine_with_warmup(step=total_steps) == min_lr (end of decay)")
}
def test_cosine_warmup_midpoint_of_decay() -> unit ! { Test } = {
  config = CosineWarmupConfig { warmup_steps: cast(10, int64), total_steps: cast(110, int64), min_lr: cast(0.001, f32), max_lr: cast(0.1, f32) }
  lr = cosine_with_warmup(cast(60, int64), config)
  expected = mul(cast(0.5, f32), add(cast(0.001, f32), cast(0.1, f32)))
  assert_close(lr, expected, cast(0.0001, f32), "cosine_with_warmup at half-cycle midpoint == (min_lr + max_lr) / 2")
}
def test_linear_warmup_at_step_zero_is_zero() -> unit ! { Test } = {
  config = LinearWarmupConfig { warmup_steps: cast(100, int64), target_lr: cast(0.01, f32) }
  lr = linear_warmup(cast(0, int64), config)
  assert_close(lr, cast(0.0, f32), cast(0.0, f32), "linear_warmup(step=0) == 0 (no LR before warmup begins)")
}
def test_linear_warmup_reaches_target_at_warmup_steps() -> unit ! { Test } = {
  config = LinearWarmupConfig { warmup_steps: cast(100, int64), target_lr: cast(0.01, f32) }
  lr = linear_warmup(cast(100, int64), config)
  assert_close(lr, cast(0.01, f32), cast(0.0, f32), "linear_warmup(step=warmup_steps) == target_lr")
}
def test_linear_warmup_plateaus_after_warmup() -> unit ! { Test } = {
  config = LinearWarmupConfig { warmup_steps: cast(100, int64), target_lr: cast(0.01, f32) }
  lr = linear_warmup(cast(10000, int64), config)
  assert_close(lr, cast(0.01, f32), cast(0.0, f32), "linear_warmup(step >> warmup_steps) plateaus at target_lr")
}
def test_step_decay_initial_step_no_decay() -> unit ! { Test } = {
  config = StepDecayConfig { initial_lr: cast(0.1, f32), decay_factor: cast(0.5, f32), decay_steps: [cast(10, int64), cast(20, int64)] }
  lr = step_decay(cast(0, int64), config)
  assert_close(lr, cast(0.1, f32), cast(0.0, f32), "step_decay(step=0) == initial_lr (no thresholds crossed)")
}
def test_step_decay_after_second_threshold() -> unit ! { Test } = {
  config = StepDecayConfig { initial_lr: cast(0.1, f32), decay_factor: cast(0.5, f32), decay_steps: [cast(10, int64), cast(20, int64)] }
  lr = step_decay(cast(20, int64), config)
  expected = mul(cast(0.1, f32), mul(cast(0.5, f32), cast(0.5, f32)))
  assert_close(lr, expected, cast(0.0, f32), "step_decay(step=20) == initial_lr * decay_factor^2")
}
