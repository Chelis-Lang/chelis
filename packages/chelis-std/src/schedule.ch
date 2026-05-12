module Std.Schedule
export (CosineWarmupConfig, LinearWarmupConfig, StepDecayConfig, cosine_with_warmup, linear_warmup, step_decay)
type CosineWarmupConfig =
  | CosineWarmupConfig { warmup_steps: int64, total_steps: int64, min_lr: f32, max_lr: f32 }
type LinearWarmupConfig =
  | LinearWarmupConfig { warmup_steps: int64, target_lr: f32 }
type StepDecayConfig =
  | StepDecayConfig { initial_lr: f32, decay_factor: f32, decay_steps: List[int64] }
def cosine_with_warmup(step: int64, config: CosineWarmupConfig) -> f32 = {
  match config with {
    | CosineWarmupConfig { warmup_steps: warmup_steps, total_steps: total_steps, min_lr: min_lr, max_lr: max_lr } => if lte(step, cast(0, int64)) then cast(0.0, f32) else if and(gt(warmup_steps, cast(0, int64)), gt(warmup_steps, step)) then mul(max_lr, div(cast(step, f32), cast(warmup_steps, f32))) else if lte(total_steps, warmup_steps) then max_lr else if gte(step, total_steps) then min_lr else {
    progress = div(cast(sub(step, warmup_steps), f32), cast(sub(total_steps, warmup_steps), f32))
    cosine = sin(sub(cast(1.5707964, f32), mul(progress, cast(3.1415927, f32))))
    mix = mul(cast(0.5, f32), add(cast(1.0, f32), cosine))
    add(min_lr, mul(sub(max_lr, min_lr), mix))
  }
  }
}
def linear_warmup(step: int64, config: LinearWarmupConfig) -> f32 = {
  match config with {
    | LinearWarmupConfig { warmup_steps: warmup_steps, target_lr: target_lr } => if lte(step, cast(0, int64)) then cast(0.0, f32) else if and(gt(warmup_steps, cast(0, int64)), gt(warmup_steps, step)) then mul(target_lr, div(cast(step, f32), cast(warmup_steps, f32))) else target_lr
  }
}
def step_decay(step: int64, config: StepDecayConfig) -> f32 = {
  match config with {
    | StepDecayConfig { initial_lr: initial_lr, decay_factor: decay_factor, decay_steps: decay_steps } => mul(initial_lr, powf(decay_factor, decay_count(step, decay_steps)))
  }
}
def decay_count(step: int64, thresholds: List[int64]) -> int64 = {
  if eq(len(thresholds), cast(0, int64)) then cast(0, int64) else {
    head = index(thresholds, cast(0, int64))
    tail = drop(thresholds, cast(1, int64))
    if gte(step, head) then add(cast(1, int64), decay_count(step, tail)) else decay_count(step, tail)
  }
}
def powf(base: f32, exp: int64) -> f32 = if lte(exp, cast(0, int64)) then cast(1.0, f32) else mul(base, powf(base, sub(exp, cast(1, int64))))
