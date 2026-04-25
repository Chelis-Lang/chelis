module Std.Tests.Init.Kaiming
import Std.Init.Kaiming (kaiming_normal, kaiming_uniform)
import Std.Test (assert_close_tensor, assert_shape, assert_true)
def template1024() -> tensor[1024, f32] = to_tensor(map(fn (i: int64) -> cast(0.0, f32), range(cast(0, int64), cast(1024, int64))))
def template8() -> tensor[8, f32] = to_tensor(map(fn (i: int64) -> cast(0.0, f32), range(cast(0, int64), cast(8, int64))))
def sample_std[n](t: tensor[n, f32]) -> f32 = {
  xs = to_list(t)
  total = fold(fn (acc: f32, x: f32) -> add(acc, x), cast(0.0, f32), xs)
  mean = div(total, cast(len(xs), f32))
  ys = to_list(t)
  sq = map(fn (x: f32) -> mul(sub(x, mean), sub(x, mean)), ys)
  ssq = fold(fn (acc: f32, x: f32) -> add(acc, x), cast(0.0, f32), sq)
  sqrt(div(ssq, cast(len(sq), f32)))
}
def abs_f32(x: f32) -> f32 = if lt(x, cast(0.0, f32)) then sub(cast(0.0, f32), x) else x
def sum_abs_diff[n](a: tensor[n, f32], b: tensor[n, f32]) -> f32 = {
  xs = to_list(a)
  ys = to_list(b)
  diffs = map(fn (pair: (f32, f32)) -> abs_f32(sub(pair.0, pair.1)), zip(xs, ys))
  fold(fn (acc: f32, x: f32) -> add(acc, x), cast(0.0, f32), diffs)
}
def max_abs[n](t: tensor[n, f32]) -> f32 = {
  xs = to_list(t)
  fold(fn (acc: f32, x: f32) -> if gt(abs_f32(x), acc) then abs_f32(x) else acc, cast(0.0, f32), xs)
}
def test_kaiming_uniform_is_deterministic_under_same_seed() -> unit ! { Test } = {
  a = with seed(101) { kaiming_uniform(template8(), cast(4.0, f32)) }
  b = with seed(101) { kaiming_uniform(template8(), cast(4.0, f32)) }
  assert_close_tensor(a, b, cast(0.0, f32), "kaiming_uniform(seed=101) is deterministic")
}
def test_kaiming_normal_is_deterministic_under_same_seed() -> unit ! { Test } = {
  a = with seed(202) { kaiming_normal(template8(), cast(4.0, f32)) }
  b = with seed(202) { kaiming_normal(template8(), cast(4.0, f32)) }
  assert_close_tensor(a, b, cast(0.0, f32), "kaiming_normal(seed=202) is deterministic")
}
def test_kaiming_uniform_distinct_seeds_produce_distinct_samples() -> unit ! { Test } = {
  a = with seed(101) { kaiming_uniform(template8(), cast(4.0, f32)) }
  b = with seed(303) { kaiming_uniform(template8(), cast(4.0, f32)) }
  diff = sum_abs_diff(a, b)
  assert_true(gt(diff, cast(0.000001, f32)), "kaiming_uniform(seed=101) != kaiming_uniform(seed=303)")
}
def test_kaiming_normal_distinct_seeds_produce_distinct_samples() -> unit ! { Test } = {
  a = with seed(202) { kaiming_normal(template8(), cast(4.0, f32)) }
  b = with seed(404) { kaiming_normal(template8(), cast(4.0, f32)) }
  diff = sum_abs_diff(a, b)
  assert_true(gt(diff, cast(0.000001, f32)), "kaiming_normal(seed=202) != kaiming_normal(seed=404)")
}
def test_kaiming_uniform_preserves_template_shape() -> unit ! { Test } = {
  out = with seed(11) { kaiming_uniform(template8(), cast(4.0, f32)) }
  assert_shape(out, cast(8, int64), "kaiming_uniform output rank-1 length matches template (8)")
}
def test_kaiming_normal_preserves_template_shape() -> unit ! { Test } = {
  out = with seed(12) { kaiming_normal(template8(), cast(4.0, f32)) }
  assert_shape(out, cast(8, int64), "kaiming_normal output rank-1 length matches template (8)")
}
def test_kaiming_uniform_respects_bound() -> unit ! { Test } = {
  -- kaiming_uniform with fan_in=4 has bound = sqrt(6/4) ~= 1.2247449.
  -- All samples must lie inside [-bound, bound]; we check max(|x|) <= bound + slack.
  out = with seed(13) { kaiming_uniform(template1024(), cast(4.0, f32)) }
  m = max_abs(out)
  assert_true(lt(m, cast(1.2249, f32)), "kaiming_uniform(fan_in=4) samples lie inside [-sqrt(6/4), sqrt(6/4)]")
}
def test_kaiming_uniform_std_matches_uniform_theory() -> unit ! { Test } = {
  -- kaiming_uniform with fan_in=4: bound = sqrt(6/4); std of Uniform(-bound, bound)
  -- is bound/sqrt(3) = sqrt(6/4)/sqrt(3) = sqrt(1/2) ~= 0.7071068. With n=1024
  -- a 0.1-loose tolerance is well above sampling noise.
  out = with seed(14) { kaiming_uniform(template1024(), cast(4.0, f32)) }
  s = sample_std(out)
  assert_true(lt(abs_f32(sub(s, cast(0.7071, f32))), cast(0.1, f32)), "kaiming_uniform(fan_in=4) sample std ~= 0.7071")
}
def test_kaiming_normal_std_matches_normal_theory() -> unit ! { Test } = {
  -- kaiming_normal with fan_in=4: std = sqrt(2/4) ~= 0.7071068. With n=1024
  -- a 0.1-loose tolerance is well above Box-Muller sampling noise.
  out = with seed(15) { kaiming_normal(template1024(), cast(4.0, f32)) }
  s = sample_std(out)
  assert_true(lt(abs_f32(sub(s, cast(0.7071, f32))), cast(0.1, f32)), "kaiming_normal(fan_in=4) sample std ~= 0.7071")
}
