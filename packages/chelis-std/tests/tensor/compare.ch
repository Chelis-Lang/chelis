module Std.Tests.Tensor.Compare
import Std.Test (assert_eq)
def check_bool_tensor(name: string, t: tensor[3, bool], v0: bool, v1: bool, v2: bool) -> unit ! { Test } = {
  xs = to_list(t)
  _ = assert_eq(index(xs, cast(0, i64)), v0, string_concat(name, "[0]"))
  _ = assert_eq(index(xs, cast(1, i64)), v1, string_concat(name, "[1]"))
  assert_eq(index(xs, cast(2, i64)), v2, string_concat(name, "[2]"))
}
def test_eq_tensor_tensor_elementwise() -> unit ! { Test } = {
  a = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)])
  b = to_tensor([cast(1.0, f32), cast(5.0, f32), cast(3.0, f32)])
  out = eq(a, b)
  check_bool_tensor("eq([1,2,3], [1,5,3])", out, true, false, true)
}
def test_neq_tensor_tensor_elementwise() -> unit ! { Test } = {
  a = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)])
  b = to_tensor([cast(1.0, f32), cast(5.0, f32), cast(3.0, f32)])
  out = neq(a, b)
  check_bool_tensor("neq([1,2,3], [1,5,3])", out, false, true, false)
}
def test_lt_tensor_tensor_elementwise() -> unit ! { Test } = {
  a = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)])
  b = to_tensor([cast(1.0, f32), cast(5.0, f32), cast(2.0, f32)])
  out = lt(a, b)
  check_bool_tensor("lt([1,2,3], [1,5,2])", out, false, true, false)
}
def test_gt_tensor_tensor_elementwise() -> unit ! { Test } = {
  a = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)])
  b = to_tensor([cast(1.0, f32), cast(5.0, f32), cast(2.0, f32)])
  out = gt(a, b)
  check_bool_tensor("gt([1,2,3], [1,5,2])", out, false, false, true)
}
def test_gte_tensor_tensor_elementwise() -> unit ! { Test } = {
  a = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)])
  b = to_tensor([cast(1.0, f32), cast(5.0, f32), cast(2.0, f32)])
  out = gte(a, b)
  check_bool_tensor("gte([1,2,3], [1,5,2])", out, true, false, true)
}
def test_lte_tensor_tensor_elementwise() -> unit ! { Test } = {
  a = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)])
  b = to_tensor([cast(1.0, f32), cast(5.0, f32), cast(2.0, f32)])
  out = lte(a, b)
  check_bool_tensor("lte([1,2,3], [1,5,2])", out, true, true, false)
}
