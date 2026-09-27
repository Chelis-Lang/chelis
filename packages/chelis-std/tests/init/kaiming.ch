module Std.Tests.Init.Kaiming
import Std.Init.Kaiming (kaiming_normal, kaiming_uniform)
import Std.Test (assert_close_tensor, assert_shape, assert_true)
def template1024() -> tensor[1024, f32] = to_tensor(map(fn (i: i64) -> cast(0.0, f32), range(cast(0, i64), cast(1024, i64))))
def template8() -> tensor[8, f32] = to_tensor(map(fn (i: i64) -> cast(0.0, f32), range(cast(0, i64), cast(8, i64))))
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
  a = kaiming_uniform(key_from_seed(101i64), template8(), cast(4.0, f32))
  b = kaiming_uniform(key_from_seed(101i64), template8(), cast(4.0, f32))
  assert_close_tensor(a, b, cast(0.0, f32), "kaiming_uniform(seed=101) is deterministic")
}
def test_kaiming_normal_is_deterministic_under_same_seed() -> unit ! { Test } = {
  a = kaiming_normal(key_from_seed(202i64), template8(), cast(4.0, f32))
  b = kaiming_normal(key_from_seed(202i64), template8(), cast(4.0, f32))
  assert_close_tensor(a, b, cast(0.0, f32), "kaiming_normal(seed=202) is deterministic")
}
def test_kaiming_uniform_distinct_seeds_produce_distinct_samples() -> unit ! { Test } = {
  a = kaiming_uniform(key_from_seed(101i64), template8(), cast(4.0, f32))
  b = kaiming_uniform(key_from_seed(303i64), template8(), cast(4.0, f32))
  diff = sum_abs_diff(a, b)
  assert_true(gt(diff, cast(0.5, f32)), "kaiming_uniform(seed=101) vs (seed=303): sum |a - b| > 0.5 across 8 draws")
}
def test_kaiming_normal_distinct_seeds_produce_distinct_samples() -> unit ! { Test } = {
  a = kaiming_normal(key_from_seed(202i64), template8(), cast(4.0, f32))
  b = kaiming_normal(key_from_seed(404i64), template8(), cast(4.0, f32))
  diff = sum_abs_diff(a, b)
  assert_true(gt(diff, cast(0.5, f32)), "kaiming_normal(seed=202) vs (seed=404): sum |a - b| > 0.5 across 8 draws")
}
def test_kaiming_uniform_preserves_template_shape() -> unit ! { Test } = {
  out = kaiming_uniform(key_from_seed(11i64), template8(), cast(4.0, f32))
  assert_shape(out, [cast(8, i64)], "kaiming_uniform output rank-1 length matches template (8)")
}
def test_kaiming_normal_preserves_template_shape() -> unit ! { Test } = {
  out = kaiming_normal(key_from_seed(12i64), template8(), cast(4.0, f32))
  assert_shape(out, [cast(8, i64)], "kaiming_normal output rank-1 length matches template (8)")
}
def test_kaiming_uniform_respects_bound() -> unit ! { Test } = {
  out = kaiming_uniform(key_from_seed(13i64), template1024(), cast(4.0, f32))
  m = max_abs(out)
  assert_true(lt(m, cast(1.2249, f32)), "kaiming_uniform(fan_in=4) samples lie inside [-sqrt(6/4), sqrt(6/4)]")
}
def test_kaiming_uniform_std_matches_uniform_theory() -> unit ! { Test } = {
  out = kaiming_uniform(key_from_seed(14i64), template1024(), cast(4.0, f32))
  s = sample_std(out)
  assert_true(lt(abs_f32(sub(s, cast(0.7071, f32))), cast(0.1, f32)), "kaiming_uniform(fan_in=4) sample std ~= 0.7071")
}
def test_kaiming_normal_std_matches_normal_theory() -> unit ! { Test } = {
  out = kaiming_normal(key_from_seed(15i64), template1024(), cast(4.0, f32))
  s = sample_std(out)
  assert_true(lt(abs_f32(sub(s, cast(0.7071, f32))), cast(0.1, f32)), "kaiming_normal(fan_in=4) sample std ~= 0.7071")
}
