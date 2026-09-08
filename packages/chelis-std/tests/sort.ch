module Std.Tests.Sort
import Std.Sort (sort)
import Std.Test (assert_eq)
def test_sort_rank1_values_and_indices() -> unit ! { Test } = {
  values = to_tensor([cast(3.0, f32), cast(1.0, f32), cast(2.0, f32)])
  pair = sort(values, cast(0, int32))
  sorted_values = pair.0
  sorted_indices = pair.1
  _ = assert_eq(cast(index(to_list(sorted_values), cast(0, int64)), int64), cast(1, int64), "sort rank-1 sorted first value")
  assert_eq(index(to_list(sorted_indices), cast(0, int64)), cast(1, int64), "sort rank-1 original first index")
}
def test_sort_rank2_typechecks() -> unit ! { Test } = {
  values = to_tensor([[cast(3.0, f32), cast(1.0, f32)], [cast(4.0, f32), cast(2.0, f32)]])
  pair = sort(values, cast(1, int32))
  sorted_indices = pair.1
  assert_eq(numel(sorted_indices), cast(4, int64), "sort rank-2 preserves element count")
}
