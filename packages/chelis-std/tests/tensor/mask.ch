module Std.Tests.Tensor.Mask
import Std.Tensor.Mask (where_indices)
import Std.Test (assert_eq)
def test_where_indices_mixed_mask() -> unit ! { Test } = {
  mask = cmplt((to_tensor([0.0, 1.0, 0.0, 1.0, 1.0], f32) : tensor[5, f32]), (to_tensor([0.5, 0.5, 0.5, 0.5, 0.5], f32) : tensor[5, f32]))
  result = where_indices(mask)
  shape0 = cast(shape(result, cast(0, i32)), i64)
  _ = assert_eq(shape0, cast(2, i64), "mixed-shape")
  actual = to_list(result)
  expected = [cast(0, i64), cast(2, i64)]
  pairs = zip(actual, expected)
  hits = fold(fn (acc: i64, p: (i64, i64)) -> if eq(p.0, p.1) then add(acc, cast(1, i64)) else acc, cast(0, i64), pairs)
  assert_eq(hits, cast(2, i64), "mixed-vals")
}
def test_where_indices_all_true_mask() -> unit ! { Test } = {
  mask = cmplt((to_tensor([0.0, 0.0, 0.0], f32) : tensor[3, f32]), (to_tensor([1.0, 1.0, 1.0], f32) : tensor[3, f32]))
  result = where_indices(mask)
  shape0 = cast(shape(result, cast(0, i32)), i64)
  _ = assert_eq(shape0, cast(3, i64), "all-true-shape")
  actual = to_list(result)
  expected = [cast(0, i64), cast(1, i64), cast(2, i64)]
  pairs = zip(actual, expected)
  hits = fold(fn (acc: i64, p: (i64, i64)) -> if eq(p.0, p.1) then add(acc, cast(1, i64)) else acc, cast(0, i64), pairs)
  assert_eq(hits, cast(3, i64), "all-true-vals")
}
def test_where_indices_all_false_mask() -> unit ! { Test } = {
  mask = cmplt((to_tensor([1.0, 1.0, 1.0], f32) : tensor[3, f32]), (to_tensor([0.0, 0.0, 0.0], f32) : tensor[3, f32]))
  result = where_indices(mask)
  shape0 = cast(shape(result, cast(0, i32)), i64)
  assert_eq(shape0, cast(0, i64), "all-false-shape")
}
def test_where_indices_single_true_mask() -> unit ! { Test } = {
  mask = cmplt((to_tensor([0.0], f32) : tensor[1, f32]), (to_tensor([1.0], f32) : tensor[1, f32]))
  result = where_indices(mask)
  shape0 = cast(shape(result, cast(0, i32)), i64)
  _ = assert_eq(shape0, cast(1, i64), "single-true-shape")
  actual = to_list(result)
  expected = [cast(0, i64)]
  pairs = zip(actual, expected)
  hits = fold(fn (acc: i64, p: (i64, i64)) -> if eq(p.0, p.1) then add(acc, cast(1, i64)) else acc, cast(0, i64), pairs)
  assert_eq(hits, cast(1, i64), "single-true-vals")
}
def test_where_indices_single_false_mask() -> unit ! { Test } = {
  mask = cmplt((to_tensor([1.0], f32) : tensor[1, f32]), (to_tensor([0.0], f32) : tensor[1, f32]))
  result = where_indices(mask)
  shape0 = cast(shape(result, cast(0, i32)), i64)
  assert_eq(shape0, cast(0, i64), "single-false-shape")
}
