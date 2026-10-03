module Std.Tests.Tensor.Construct
import Std.Tensor.Construct (linspace, arange)
import Std.Test (assert_close_tensor, assert_eq, assert_shape)
def test_arange_basic() -> unit ! { Test } = {
  xs = to_list(arange(cast(0, i32), cast(4, i32)))
  _ = assert_eq(len(xs), cast(4, i64), "arange(0,4) length")
  _ = assert_eq(cast(index(xs, cast(0, i64)), i64), cast(0, i64), "arange(0,4)[0]")
  _ = assert_eq(cast(index(xs, cast(1, i64)), i64), cast(1, i64), "arange(0,4)[1]")
  _ = assert_eq(cast(index(xs, cast(2, i64)), i64), cast(2, i64), "arange(0,4)[2]")
  assert_eq(cast(index(xs, cast(3, i64)), i64), cast(3, i64), "arange(0,4)[3]")
}
def test_arange_offset() -> unit ! { Test } = {
  xs = to_list(arange(cast(2, i32), cast(6, i32)))
  _ = assert_eq(len(xs), cast(4, i64), "arange(2,6) length")
  _ = assert_eq(cast(index(xs, cast(0, i64)), i64), cast(2, i64), "arange(2,6)[0]")
  _ = assert_eq(cast(index(xs, cast(1, i64)), i64), cast(3, i64), "arange(2,6)[1]")
  _ = assert_eq(cast(index(xs, cast(2, i64)), i64), cast(4, i64), "arange(2,6)[2]")
  assert_eq(cast(index(xs, cast(3, i64)), i64), cast(5, i64), "arange(2,6)[3]")
}
def test_linspace_degenerate() -> unit ! { Test } = {
  actual = linspace(cast(3.0, f32), cast(7.0, f32), cast(1, i64))
  expected = to_tensor([cast(3.0, f32)])
  _ = assert_shape(&actual, [cast(1, i64)], "linspace count=1 length")
  assert_close_tensor(actual, expected, cast(1e-6, f32), "linspace count=1 value")
}
def test_linspace_endpoints() -> unit ! { Test } = {
  actual = linspace(cast(0.0, f32), cast(1.0, f32), cast(3, i64))
  expected = to_tensor([cast(0.0, f32), cast(0.5, f32), cast(1.0, f32)])
  _ = assert_shape(&actual, [cast(3, i64)], "linspace count=3 length")
  assert_close_tensor(actual, expected, cast(1e-6, f32), "linspace count=3 endpoints")
}
def test_arange_empty_range_produces_empty_tensor() -> unit ! { Test } = {
  xs = to_list(arange(cast(5, i32), cast(5, i32)))
  assert_eq(len(xs), cast(0, i64), "arange(5,5) is empty")
}

def test_linspace_bf16_exact_rational_weight() -> unit ! { Test } = {
  values = to_list(linspace(0.0bf16, 1.0bf16, 300i64))
  _ = assert_eq(index(values, 0i64), 0.0bf16, "bf16 start endpoint")
  _ = assert_eq(index(values, 257i64), 0.859375bf16, "bf16 exact 257/299 weight")
  assert_eq(index(values, 299i64), 1.0bf16, "bf16 stop endpoint")
}
