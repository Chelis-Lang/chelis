module Std.Tests.Scalar
import Std.Scalar (max, min, abs)
import Std.Test (assert_eq)
def test_max() -> () ! { Test } = {
  _ = assert_eq(max(cast(1.0, f32), cast(2.0, f32)), cast(2.0, f32), "max picks the larger operand")
  _ = assert_eq(max(cast(2.0, f32), cast(1.0, f32)), cast(2.0, f32), "max is order-independent")
  assert_eq(max(cast(-1.0, f32), cast(-2.0, f32)), cast(-1.0, f32), "max over negatives")
}
def test_min() -> () ! { Test } = {
  _ = assert_eq(min(cast(1.0, f32), cast(2.0, f32)), cast(1.0, f32), "min picks the smaller operand")
  _ = assert_eq(min(cast(2.0, f32), cast(1.0, f32)), cast(1.0, f32), "min is order-independent")
  assert_eq(min(cast(-1.0, f32), cast(-2.0, f32)), cast(-2.0, f32), "min over negatives")
}
def test_abs() -> () ! { Test } = {
  _ = assert_eq(abs(cast(3.0, f32)), cast(3.0, f32), "abs of a positive is itself")
  _ = assert_eq(abs(cast(-3.0, f32)), cast(3.0, f32), "abs of a negative is its magnitude")
  assert_eq(abs(cast(0.0, f32)), cast(0.0, f32), "abs of zero is zero")
}
