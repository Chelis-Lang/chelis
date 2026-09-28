module Std.Tests.Init.XavierExt
import Std.Init.XavierExt (xavier_uniform, xavier_normal, trunc_normal)
import Std.Test (assert_close, assert_close_tensor, assert_eq, assert_shape, assert_true)
def long_template() -> tensor[2000, f32] = to_tensor(map(fn (i: i64) -> cast(0.0, f32), range(cast(0, i64), cast(2000, i64))))
def small_template() -> tensor[8, f32] = to_tensor(map(fn (i: i64) -> cast(0.0, f32), range(cast(0, i64), cast(8, i64))))
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
def first_elem[n](t: tensor[n, f32]) -> f32 = index(to_list(t), cast(0, i64))
def test_xavier_uniform_deterministic_under_seed() -> unit ! { Test } = {
  a = xavier_uniform(key_from_seed(101i64), long_template(), cast(64.0, f32), cast(64.0, f32))
  b = xavier_uniform(key_from_seed(101i64), long_template(), cast(64.0, f32), cast(64.0, f32))
  assert_close_tensor(a, b, cast(0.0, f32), "xavier_uniform same-seed determinism (exact)")
}
def test_xavier_normal_deterministic_under_seed() -> unit ! { Test } = {
  a = xavier_normal(key_from_seed(202i64), long_template(), cast(64.0, f32), cast(64.0, f32))
  b = xavier_normal(key_from_seed(202i64), long_template(), cast(64.0, f32), cast(64.0, f32))
  assert_close_tensor(a, b, cast(0.0, f32), "xavier_normal same-seed determinism (exact)")
}
def test_trunc_normal_deterministic_under_seed() -> unit ! { Test } = {
  a = trunc_normal(key_from_seed(303i64), long_template(), cast(0.0, f32), cast(1.0, f32), cast(-0.5, f32), cast(0.5, f32))
  b = trunc_normal(key_from_seed(303i64), long_template(), cast(0.0, f32), cast(1.0, f32), cast(-0.5, f32), cast(0.5, f32))
  assert_close_tensor(a, b, cast(0.0, f32), "trunc_normal same-seed determinism (exact)")
}
def test_xavier_uniform_seed_sensitive() -> unit ! { Test } = {
  a = xavier_uniform(key_from_seed(11i64), small_template(), cast(8.0, f32), cast(8.0, f32))
  b = xavier_uniform(key_from_seed(22i64), small_template(), cast(8.0, f32), cast(8.0, f32))
  diff = sub(first_elem(a), first_elem(b))
  abs_diff = if lt(diff, cast(0.0, f32)) then neg(diff) else diff
  assert_true(gt(abs_diff, cast(0.001, f32)), "xavier_uniform seeds 11 vs 22 differ at index 0 by > 1e-3 (E ≈ 0.4)")
}
def test_xavier_normal_seed_sensitive() -> unit ! { Test } = {
  a = xavier_normal(key_from_seed(11i64), small_template(), cast(8.0, f32), cast(8.0, f32))
  b = xavier_normal(key_from_seed(22i64), small_template(), cast(8.0, f32), cast(8.0, f32))
  diff = sub(first_elem(a), first_elem(b))
  abs_diff = if lt(diff, cast(0.0, f32)) then neg(diff) else diff
  assert_true(gt(abs_diff, cast(0.001, f32)), "xavier_normal seeds 11 vs 22 differ at index 0 by > 1e-3 (E ≈ 0.4)")
}
def test_trunc_normal_seed_sensitive() -> unit ! { Test } = {
  a = trunc_normal(key_from_seed(11i64), small_template(), cast(0.0, f32), cast(1.0, f32), cast(-10.0, f32), cast(10.0, f32))
  b = trunc_normal(key_from_seed(22i64), small_template(), cast(0.0, f32), cast(1.0, f32), cast(-10.0, f32), cast(10.0, f32))
  diff = sub(first_elem(a), first_elem(b))
  abs_diff = if lt(diff, cast(0.0, f32)) then neg(diff) else diff
  assert_true(gt(abs_diff, cast(0.001, f32)), "trunc_normal seeds 11 vs 22 differ at index 0 by > 1e-3 (E ≈ 1.13)")
}
def test_xavier_uniform_shape_matches_template() -> unit ! { Test } = {
  out = xavier_uniform(key_from_seed(7i64), small_template(), cast(8.0, f32), cast(8.0, f32))
  assert_shape(out, [cast(8, i64)], "xavier_uniform preserves template length 8")
}
def test_xavier_normal_shape_matches_template() -> unit ! { Test } = {
  out = xavier_normal(key_from_seed(7i64), small_template(), cast(8.0, f32), cast(8.0, f32))
  assert_shape(out, [cast(8, i64)], "xavier_normal preserves template length 8")
}
def test_trunc_normal_shape_matches_template() -> unit ! { Test } = {
  out = trunc_normal(key_from_seed(7i64), small_template(), cast(0.0, f32), cast(1.0, f32), cast(-1.0, f32), cast(1.0, f32))
  assert_shape(out, [cast(8, i64)], "trunc_normal preserves template length 8")
}
def test_xavier_uniform_stays_within_bound() -> unit ! { Test } = {
  out = xavier_uniform(key_from_seed(45i64), long_template(), cast(4.0, f32), cast(4.0, f32))
  hi = tensor_max(copy(out))
  lo = tensor_min(out)
  bound = cast(0.8660254, f32)
  _ = assert_true(lte(hi, bound), "xavier_uniform max <= sqrt(6/(4+4))")
  assert_true(gte(lo, neg(bound)), "xavier_uniform min >= -sqrt(6/(4+4))")
}
def test_xavier_normal_empirical_std_matches_formula() -> unit ! { Test } = {
  out = xavier_normal(key_from_seed(46i64), long_template(), cast(4.0, f32), cast(4.0, f32))
  s = sample_std(out)
  assert_close(s, cast(0.5, f32), cast(0.1, f32), "xavier_normal empirical std ≈ sqrt(2/(fan_in+fan_out))")
}
def test_trunc_normal_clips_outside_bounds() -> unit ! { Test } = {
  out = trunc_normal(key_from_seed(47i64), long_template(), cast(0.0, f32), cast(1.0, f32), cast(-0.5, f32), cast(0.5, f32))
  hi = tensor_max(copy(out))
  lo = tensor_min(out)
  _ = assert_true(lte(hi, cast(0.5, f32)), "trunc_normal max <= upper bound 0.5")
  assert_true(gte(lo, cast(-0.5, f32)), "trunc_normal min >= lower bound -0.5")
}
def test_xavier_uniform_fan_out_widens_bound() -> unit ! { Test } = {
  sym = xavier_uniform(key_from_seed(91i64), long_template(), cast(4.0, f32), cast(4.0, f32))
  asym = xavier_uniform(key_from_seed(91i64), long_template(), cast(4.0, f32), cast(64.0, f32))
  s_sym = sample_std(sym)
  s_asym = sample_std(asym)
  gap = sub(s_sym, s_asym)
  assert_true(gt(gap, cast(0.05, f32)), "xavier_uniform fan_out widens the denominator (asym std < sym std)")
}
def test_xavier_normal_fan_out_shrinks_std() -> unit ! { Test } = {
  sym = xavier_normal(key_from_seed(92i64), long_template(), cast(4.0, f32), cast(4.0, f32))
  asym = xavier_normal(key_from_seed(92i64), long_template(), cast(4.0, f32), cast(64.0, f32))
  s_sym = sample_std(sym)
  s_asym = sample_std(asym)
  gap = sub(s_sym, s_asym)
  assert_true(gt(gap, cast(0.1, f32)), "xavier_normal fan_out shrinks std (asym std < sym std)")
}
def test_trunc_normal_mean_shifts_distribution() -> unit ! { Test } = {
  out = trunc_normal(key_from_seed(48i64), long_template(), cast(10.0, f32), cast(0.1, f32), cast(9.0, f32), cast(11.0, f32))
  m = sample_mean(out)
  assert_close(m, cast(10.0, f32), cast(0.05, f32), "trunc_normal mean=10 -> empirical mean ≈ 10")
}
