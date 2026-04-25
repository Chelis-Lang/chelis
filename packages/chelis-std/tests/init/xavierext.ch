module Std.Tests.Init.XavierExt
import Std.Init.XavierExt (xavier_uniform, xavier_normal, trunc_normal)
import Std.Test (assert_close, assert_close_tensor, assert_eq_int, assert_shape, assert_true)
-- Acceptance surface for `Std.Init.XavierExt` at Phase 3t A1 Wave 2.
--
-- XavierExt extends the bare `Std.Init.Xavier` signature stub with three
-- shipped initializers, all `! { Random }`:
--   * `xavier_uniform(template, fan_in, fan_out)` — uniform on
--     [-bound, +bound] with bound = sqrt(6 / (fan_in + fan_out)).
--   * `xavier_normal(template, fan_in, fan_out)` — N(0, std) with
--     std = sqrt(2 / (fan_in + fan_out)).
--   * `trunc_normal(template, mean, std, a, b)` — normal samples with the
--     extreme values clamped to [a, b] (clipped, not resampled).
--
-- Each test handles `Random` via `with seed(...)` so the suite runs under
-- the `! { Test }` envelope expected by `chelis test`.
def long_template() -> tensor[2000, f32] = to_tensor(map(fn (i: int64) -> cast(0.0, f32), range(cast(0, int64), cast(2000, int64))))
def small_template() -> tensor[8, f32] = to_tensor(map(fn (i: int64) -> cast(0.0, f32), range(cast(0, int64), cast(8, int64))))
def sample_mean[n](t: tensor[n, f32]) -> f32 = {
  xs = to_list(t)
  total = fold(fn (acc: f32, x: f32) -> add(acc, x), cast(0.0, f32), xs)
  div(total, cast(len(xs), f32))
}
def sample_std[n](t: tensor[n, f32]) -> f32 = {
  m = sample_mean(copy(t))
  xs = to_list(t)
  sq = map(fn (x: f32) -> mul(sub(x, m), sub(x, m)), xs)
  total = fold(fn (acc: f32, x: f32) -> add(acc, x), cast(0.0, f32), sq)
  sqrt(div(total, cast(len(sq), f32)))
}
def tensor_max[n](t: tensor[n, f32]) -> f32 = fold(fn (acc: f32, x: f32) -> if gt(x, acc) then x else acc, cast(-1000000.0, f32), to_list(t))
def tensor_min[n](t: tensor[n, f32]) -> f32 = fold(fn (acc: f32, x: f32) -> if lt(x, acc) then x else acc, cast(1000000.0, f32), to_list(t))
def first_elem[n](t: tensor[n, f32]) -> f32 = index(to_list(t), cast(0, int64))
-- =========================================================================
-- Determinism under seed: same seed twice must produce identical tensors.
-- Tolerance is 0.0 so `assert_close_tensor` collapses to exact equality.
-- =========================================================================
def test_xavier_uniform_deterministic_under_seed() -> unit ! { Test } = {
  a = with seed(101) { xavier_uniform(long_template(), cast(64.0, f32), cast(64.0, f32)) }
  b = with seed(101) { xavier_uniform(long_template(), cast(64.0, f32), cast(64.0, f32)) }
  assert_close_tensor(a, b, cast(0.0, f32), "xavier_uniform same-seed determinism (exact)")
}
def test_xavier_normal_deterministic_under_seed() -> unit ! { Test } = {
  a = with seed(202) { xavier_normal(long_template(), cast(64.0, f32), cast(64.0, f32)) }
  b = with seed(202) { xavier_normal(long_template(), cast(64.0, f32), cast(64.0, f32)) }
  assert_close_tensor(a, b, cast(0.0, f32), "xavier_normal same-seed determinism (exact)")
}
def test_trunc_normal_deterministic_under_seed() -> unit ! { Test } = {
  a = with seed(303) { trunc_normal(long_template(), cast(0.0, f32), cast(1.0, f32), cast(-0.5, f32), cast(0.5, f32)) }
  b = with seed(303) { trunc_normal(long_template(), cast(0.0, f32), cast(1.0, f32), cast(-0.5, f32), cast(0.5, f32)) }
  assert_close_tensor(a, b, cast(0.0, f32), "trunc_normal same-seed determinism (exact)")
}
-- =========================================================================
-- Seed sensitivity: different seeds must produce different tensors.
-- We pin element 0; if the RNG were ignoring the seed, all seeds would
-- collapse to the same first sample.
-- =========================================================================
def test_xavier_uniform_seed_sensitive() -> unit ! { Test } = {
  a = with seed(11) { xavier_uniform(small_template(), cast(8.0, f32), cast(8.0, f32)) }
  b = with seed(22) { xavier_uniform(small_template(), cast(8.0, f32), cast(8.0, f32)) }
  diff = sub(first_elem(a), first_elem(b))
  abs_diff = if lt(diff, cast(0.0, f32)) then neg(diff) else diff
  assert_true(gt(abs_diff, cast(0.0, f32)), "xavier_uniform seeds 11 vs 22 differ at index 0")
}
def test_xavier_normal_seed_sensitive() -> unit ! { Test } = {
  a = with seed(11) { xavier_normal(small_template(), cast(8.0, f32), cast(8.0, f32)) }
  b = with seed(22) { xavier_normal(small_template(), cast(8.0, f32), cast(8.0, f32)) }
  diff = sub(first_elem(a), first_elem(b))
  abs_diff = if lt(diff, cast(0.0, f32)) then neg(diff) else diff
  assert_true(gt(abs_diff, cast(0.0, f32)), "xavier_normal seeds 11 vs 22 differ at index 0")
}
def test_trunc_normal_seed_sensitive() -> unit ! { Test } = {
  -- Wide bounds [-10, 10] so clipping never collapses two seeds to the
  -- same boundary value — any difference must come from the underlying
  -- normal_like draw, not from saturation.
  a = with seed(11) { trunc_normal(small_template(), cast(0.0, f32), cast(1.0, f32), cast(-10.0, f32), cast(10.0, f32)) }
  b = with seed(22) { trunc_normal(small_template(), cast(0.0, f32), cast(1.0, f32), cast(-10.0, f32), cast(10.0, f32)) }
  diff = sub(first_elem(a), first_elem(b))
  abs_diff = if lt(diff, cast(0.0, f32)) then neg(diff) else diff
  assert_true(gt(abs_diff, cast(0.0, f32)), "trunc_normal seeds 11 vs 22 differ at index 0")
}
-- =========================================================================
-- Shape contract: output length matches the template length, no clipping
-- of trailing samples, no padding to a power of two.
-- =========================================================================
def test_xavier_uniform_shape_matches_template() -> unit ! { Test } = {
  out = with seed(7) { xavier_uniform(small_template(), cast(8.0, f32), cast(8.0, f32)) }
  assert_shape(out, cast(8, int64), "xavier_uniform preserves template length 8")
}
def test_xavier_normal_shape_matches_template() -> unit ! { Test } = {
  out = with seed(7) { xavier_normal(small_template(), cast(8.0, f32), cast(8.0, f32)) }
  assert_shape(out, cast(8, int64), "xavier_normal preserves template length 8")
}
def test_trunc_normal_shape_matches_template() -> unit ! { Test } = {
  out = with seed(7) { trunc_normal(small_template(), cast(0.0, f32), cast(1.0, f32), cast(-1.0, f32), cast(1.0, f32)) }
  assert_shape(out, cast(8, int64), "trunc_normal preserves template length 8")
}
-- =========================================================================
-- Statistical bounds: each initializer's distribution-shape contract.
-- =========================================================================
def test_xavier_uniform_stays_within_bound() -> unit ! { Test } = {
  -- bound = sqrt(6 / (4 + 4)) = sqrt(0.75) ≈ 0.8660254. Every sample in
  -- the closed interval [-bound, +bound]. fan_in=fan_out=4 keeps the
  -- bound large enough that floating-point slack never approaches it.
  out = with seed(45) { xavier_uniform(long_template(), cast(4.0, f32), cast(4.0, f32)) }
  hi = tensor_max(copy(out))
  lo = tensor_min(out)
  bound = cast(0.8660254, f32)
  _ = assert_true(lte(hi, bound), "xavier_uniform max <= sqrt(6/(4+4))")
  assert_true(gte(lo, neg(bound)), "xavier_uniform min >= -sqrt(6/(4+4))")
}
def test_xavier_normal_empirical_std_matches_formula() -> unit ! { Test } = {
  -- std = sqrt(2 / (fan_in + fan_out)) = sqrt(2/8) = 0.5 exactly.
  -- 2000 samples; tolerance 0.1 covers Box–Muller sampling noise.
  out = with seed(46) { xavier_normal(long_template(), cast(4.0, f32), cast(4.0, f32)) }
  s = sample_std(out)
  assert_close(s, cast(0.5, f32), cast(0.1, f32), "xavier_normal empirical std ≈ sqrt(2/(fan_in+fan_out))")
}
def test_trunc_normal_clips_outside_bounds() -> unit ! { Test } = {
  -- Tight clip [-0.5, 0.5] over a unit-variance normal forces real
  -- saturation: ~62% of draws lie outside ±0.5. Every output element
  -- must end up inside the closed interval.
  out = with seed(47) { trunc_normal(long_template(), cast(0.0, f32), cast(1.0, f32), cast(-0.5, f32), cast(0.5, f32)) }
  hi = tensor_max(copy(out))
  lo = tensor_min(out)
  _ = assert_true(lte(hi, cast(0.5, f32)), "trunc_normal max <= upper bound 0.5")
  assert_true(gte(lo, cast(-0.5, f32)), "trunc_normal min >= lower bound -0.5")
}
-- =========================================================================
-- XavierExt-specific contract not covered by the bare `Std.Init.Xavier`
-- signature stub: fan_out actually changes the distribution. If the
-- impl ignored fan_out (e.g., copied Kaiming and only read fan_in), this
-- regression would slip past the existing batch-4 oracle which only
-- checks the symmetric (fan_in == fan_out) case.
-- =========================================================================
def test_xavier_uniform_fan_out_widens_bound() -> unit ! { Test } = {
  -- (fan_in=4, fan_out=4) → bound ≈ 0.866
  -- (fan_in=4, fan_out=64) → bound = sqrt(6/68) ≈ 0.297
  -- An impl that ignored fan_out would give the same bound for both.
  -- We compare empirical std of a uniform on [-b, b], which is b/sqrt(3),
  -- so the symmetric draw should yield ≈ 0.5 and the asymmetric ≈ 0.171.
  sym = with seed(91) { xavier_uniform(long_template(), cast(4.0, f32), cast(4.0, f32)) }
  asym = with seed(91) { xavier_uniform(long_template(), cast(4.0, f32), cast(64.0, f32)) }
  s_sym = sample_std(sym)
  s_asym = sample_std(asym)
  -- Asymmetric must be strictly smaller; tolerance 0.05 keeps the gap
  -- well outside Monte Carlo noise (true gap is ≈ 0.33).
  gap = sub(s_sym, s_asym)
  assert_true(gt(gap, cast(0.05, f32)), "xavier_uniform fan_out widens the denominator (asym std < sym std)")
}
def test_xavier_normal_fan_out_shrinks_std() -> unit ! { Test } = {
  -- (4, 4) → std = 0.5; (4, 64) → std = sqrt(2/68) ≈ 0.1715.
  -- True gap ≈ 0.33, well above sampling noise on 2000 draws.
  sym = with seed(92) { xavier_normal(long_template(), cast(4.0, f32), cast(4.0, f32)) }
  asym = with seed(92) { xavier_normal(long_template(), cast(4.0, f32), cast(64.0, f32)) }
  s_sym = sample_std(sym)
  s_asym = sample_std(asym)
  gap = sub(s_sym, s_asym)
  assert_true(gt(gap, cast(0.1, f32)), "xavier_normal fan_out shrinks std (asym std < sym std)")
}
-- =========================================================================
-- trunc_normal-specific: the mean argument must actually shift the
-- pre-clip distribution. With clip [a, b] = [9.0, 11.0] and std = 0.1,
-- shifting mean from 0 to 10 takes every output from "saturated at 9"
-- to "centred near 10".
-- =========================================================================
def test_trunc_normal_mean_shifts_distribution() -> unit ! { Test } = {
  -- mean=10, std=0.1, clip=[9.0, 11.0]: virtually all draws of N(10, 0.1)
  -- land inside [9, 11], so the empirical mean must sit very close to 10.
  out = with seed(48) { trunc_normal(long_template(), cast(10.0, f32), cast(0.1, f32), cast(9.0, f32), cast(11.0, f32)) }
  m = sample_mean(out)
  assert_close(m, cast(10.0, f32), cast(0.05, f32), "trunc_normal mean=10 -> empirical mean ≈ 10")
}
