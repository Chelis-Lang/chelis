module Std.Tests.Tensor.Construct
import Std.Tensor.Construct (linspace, arange)
import Std.Test (assert_close_tensor, assert_eq, assert_shape)
def test_arange_basic() -> unit ! { Test } = {
  xs = to_list(arange(cast(0, int32), cast(4, int32)))
  _ = assert_eq(len(xs), cast(4, int64), "arange(0,4) length")
  _ = assert_eq(cast(index(xs, cast(0, int64)), int64), cast(0, int64), "arange(0,4)[0]")
  _ = assert_eq(cast(index(xs, cast(1, int64)), int64), cast(1, int64), "arange(0,4)[1]")
  _ = assert_eq(cast(index(xs, cast(2, int64)), int64), cast(2, int64), "arange(0,4)[2]")
  assert_eq(cast(index(xs, cast(3, int64)), int64), cast(3, int64), "arange(0,4)[3]")
}
def test_arange_offset() -> unit ! { Test } = {
  xs = to_list(arange(cast(2, int32), cast(6, int32)))
  _ = assert_eq(len(xs), cast(4, int64), "arange(2,6) length")
  _ = assert_eq(cast(index(xs, cast(0, int64)), int64), cast(2, int64), "arange(2,6)[0]")
  _ = assert_eq(cast(index(xs, cast(1, int64)), int64), cast(3, int64), "arange(2,6)[1]")
  _ = assert_eq(cast(index(xs, cast(2, int64)), int64), cast(4, int64), "arange(2,6)[2]")
  assert_eq(cast(index(xs, cast(3, int64)), int64), cast(5, int64), "arange(2,6)[3]")
}
def test_linspace_degenerate() -> unit ! { Test } = {
  actual = linspace(cast(3.0, f32), cast(7.0, f32), cast(1, int64))
  expected = to_tensor([cast(3.0, f32)])
  _ = assert_shape(&actual, [cast(1, int64)], "linspace count=1 length")
  assert_close_tensor(actual, expected, cast(1e-6, f32), "linspace count=1 value")
}
def test_linspace_endpoints() -> unit ! { Test } = {
  actual = linspace(cast(0.0, f32), cast(1.0, f32), cast(3, int64))
  expected = to_tensor([cast(0.0, f32), cast(0.5, f32), cast(1.0, f32)])
  _ = assert_shape(&actual, [cast(3, int64)], "linspace count=3 length")
  assert_close_tensor(actual, expected, cast(1e-6, f32), "linspace count=3 endpoints")
}
def test_arange_empty_range_produces_empty_tensor() -> unit ! { Test } = {
  xs = to_list(arange(cast(5, int32), cast(5, int32)))
  assert_eq(len(xs), cast(0, int64), "arange(5,5) is empty")
}
def test_linspace_count_zero_does_not_overrun() -> unit ! { Test } = {
  actual = linspace(cast(0.0, f32), cast(1.0, f32), cast(0, int64))
  expected = to_tensor([cast(0.0, f32)])
  assert_close_tensor(actual, expected, cast(1e-6, f32), "linspace count=0 falls into count<=1 branch and returns [start]")
}
