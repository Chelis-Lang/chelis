module Std.Test
export (assert_true, assert_false, assert_eq, assert_close, assert_close_tensor, assert_eq_tensor, assert_shape, fail)
def assert_true(cond: bool, label: string) -> unit ! { Test } = test_assert(cond, label)
def assert_false(cond: bool, label: string) -> unit ! { Test } = test_assert(not(cond), label)
def assert_eq[q](actual: q, expected: q, label: string) -> unit ! { Test } = test_assert_eq(actual, expected, label)
def assert_close[p: Float](actual: p, expected: p, tol: p, label: string) -> unit ! { Test } = test_assert_close_tensor(scalar_to_tensor(actual), scalar_to_tensor(expected), tol, label)
def assert_close_tensor[r, p: Float](actual: &tensor[..r, p], expected: &tensor[..r, p], tol: p, label: string) -> unit ! { Test } = test_assert_close_tensor(actual, expected, tol, label)
def assert_eq_tensor[r, p](actual: &tensor[..r, p], expected: &tensor[..r, p], label: string) -> unit ! { Test } = test_assert_eq_tensor(actual, expected, label)
def assert_shape[r, p](t: &tensor[..r, p], expected: List[i64], label: string) -> unit ! { Test } = {
  actual_rank = cast(rank(t), i64)
  same_rank = eq(actual_rank, len(expected))
  same_extents = if same_rank then shape_matches(t, expected, cast(0, i32), cast(actual_rank, i32)) else false
  test_assert(and(same_rank, same_extents), label)
}
def shape_matches[r, p](t: &tensor[..r, p], expected: List[i64], axis: i32, limit: i32) -> bool = if gte(axis, limit) then true else and(eq(shape(t, axis), index(expected, cast(axis, i64))), shape_matches(t, expected, add(axis, cast(1, i32)), limit))
def fail(msg: string) -> unit ! { Test } = test_assert(false, msg)
