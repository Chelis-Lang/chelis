module Std.Tests.Init.Random
import Std.Init.Random (normal_like)
import Std.Test (assert_close, assert_close_tensor, assert_eq_int, assert_true)
def abs_f32(x: f32) -> f32 = if lt(x, cast(0.0, f32)) then sub(cast(0.0, f32), x) else x
def make_template(n: int64) -> tensor[n, f32] = to_tensor(map(fn (i: int64) -> cast(0.0, f32), range(cast(0, int64), n)))
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
def max_abs_diff[n](a: tensor[n, f32], b: tensor[n, f32]) -> f32 = {
  diffs = map(fn (pair: (f32, f32)) -> abs_f32(sub(pair.0, pair.1)), zip(to_list(a), to_list(b)))
  fold(fn (acc: f32, x: f32) -> if gt(x, acc) then x else acc, cast(0.0, f32), diffs)
}
def test_normal_like_is_deterministic_under_same_seed() -> unit ! { Test } = {
  template = make_template(cast(64, int64))
  a = with seed(42i64) { normal_like(copy(template), cast(0.0, f32), cast(1.0, f32)) }
  b = with seed(42i64) { normal_like(template, cast(0.0, f32), cast(1.0, f32)) }
  assert_close_tensor(a, b, cast(0.0, f32), "normal_like under same seed produces identical tensors")
}
def test_normal_like_differs_across_seeds() -> unit ! { Test } = {
  template = make_template(cast(64, int64))
  a = with seed(1i64) { normal_like(copy(template), cast(0.0, f32), cast(1.0, f32)) }
  b = with seed(2i64) { normal_like(template, cast(0.0, f32), cast(1.0, f32)) }
  max_diff = max_abs_diff(a, b)
  assert_true(gt(max_diff, cast(0.5, f32)), "normal_like(seed=1) vs (seed=2): max element-wise diff > 0.5 across 64 draws")
}
def test_normal_like_preserves_template_shape() -> unit ! { Test } = {
  template = make_template(cast(64, int64))
  out = with seed(7i64) { normal_like(template, cast(0.0, f32), cast(1.0, f32)) }
  actual_n = cast(len(to_list(out)), int64)
  assert_eq_int(actual_n, cast(64, int64), "normal_like output length matches template length")
}
def test_normal_like_mean_zero_std_one_within_loose_bound() -> unit ! { Test } = {
  template = make_template(cast(1024, int64))
  sample = with seed(42i64) { normal_like(template, cast(0.0, f32), cast(1.0, f32)) }
  m = sample_mean(copy(sample))
  s = sample_std(sample)
  _ = assert_close(m, cast(0.0, f32), cast(0.5, f32), "normal_like(mean=0, std=1) sample mean within 0.5 of 0 (loose; 1024 samples)")
  assert_close(s, cast(1.0, f32), cast(0.5, f32), "normal_like(mean=0, std=1) sample std within 0.5 of 1 (loose; 1024 samples)")
}
def test_normal_like_mean_shift_tracks_requested_mean() -> unit ! { Test } = {
  template = make_template(cast(1024, int64))
  sample = with seed(42i64) { normal_like(template, cast(5.0, f32), cast(1.0, f32)) }
  m = sample_mean(sample)
  assert_close(m, cast(5.0, f32), cast(0.5, f32), "normal_like(mean=5, std=1) sample mean within 0.5 of 5 (loose; 1024 samples)")
}
