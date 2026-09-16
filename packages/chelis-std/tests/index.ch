module Std.Tests.Index
import Std.Index (list_index, take_list, drop_list)
import Std.Test (assert_eq)
def test_list_index() -> unit ! { Test } = {
  values = [cast(4, i64), cast(5, i64), cast(6, i64)]
  assert_eq(list_index(values, cast(1, i64)), cast(5, i64), "list_index returns selected value")
}
def test_take_and_drop_list() -> unit ! { Test } = {
  values = [cast(4, i64), cast(5, i64), cast(6, i64)]
  prefix = take_list(values, cast(2, i64))
  suffix = drop_list(values, cast(1, i64))
  _ = assert_eq(len(prefix), cast(2, i64), "take_list length")
  assert_eq(list_index(suffix, cast(0, i64)), cast(5, i64), "drop_list removes prefix")
}
