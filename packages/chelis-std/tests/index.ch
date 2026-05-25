module Std.Tests.Index
import Std.Index (list_index, take_list, drop_list)
import Std.Test (assert_eq_int)
def test_list_index() -> unit ! { Test } = {
  values = [cast(4, int64), cast(5, int64), cast(6, int64)]
  assert_eq_int(list_index(values, cast(1, int64)), cast(5, int64), "list_index returns selected value")
}
def test_take_and_drop_list() -> unit ! { Test } = {
  values = [cast(4, int64), cast(5, int64), cast(6, int64)]
  prefix = take_list(values, cast(2, int64))
  suffix = drop_list(values, cast(1, int64))
  _ = assert_eq_int(len(prefix), cast(2, int64), "take_list length")
  assert_eq_int(list_index(suffix, cast(0, int64)), cast(5, int64), "drop_list removes prefix")
}
