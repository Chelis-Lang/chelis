module Std.Tests.Sort
import Std.Sort (sort_1d, sort_2d)
import Std.Test (assert_eq_int)
def test_sort_1d_values_and_indices() -> unit ! { Test } = {
  values = to_tensor([cast(3.0, f32), cast(1.0, f32), cast(2.0, f32)])
  pair = sort_1d(values, cast(0, int32))
  sorted_values = pair.0
  sorted_indices = pair.1
  _ = assert_eq_int(cast(index(to_list(sorted_values), cast(0, int64)), int64), cast(1, int64), "sort_1d sorted first value")
  assert_eq_int(index(to_list(sorted_indices), cast(0, int64)), cast(1, int64), "sort_1d original first index")
}
def test_sort_2d_typechecks() -> unit ! { Test } = {
  values = to_tensor([[cast(3.0, f32), cast(1.0, f32)], [cast(4.0, f32), cast(2.0, f32)]])
  pair = sort_2d(values, cast(1, int32))
  sorted_indices = pair.1
  assert_eq_int(numel(sorted_indices), cast(4, int64), "sort_2d preserves element count")
}
