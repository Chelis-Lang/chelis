module Std.Test
export (assert_true, assert_false, assert_eq, assert_close, assert_close_tensor, assert_eq_tensor, assert_shape, fail)
def assert_true(cond: bool, label: string) -> unit ! { Test } = test_assert(cond, label)
def assert_false(cond: bool, label: string) -> unit ! { Test } = test_assert(not(cond), label)
def assert_eq[q](actual: q, expected: q, label: string) -> unit ! { Test } = test_assert_eq(actual, expected, label)
def assert_close[p: Float](actual: p, expected: p, tol: p, label: string) -> unit ! { Test } = {
  zero = sub(tol, tol)
  tol_nan = neq(tol, tol)
  tol_negative = gt(zero, tol)
  if or(tol_nan, tol_negative) then test_assert(false, string_concat("assert_close (", string_concat(label, string_concat("): invalid tolerance ", to_string(tol))))) else {
    actual_nan = neq(actual, actual)
    expected_nan = neq(expected, expected)
    diff = sub(actual, expected)
    abs_diff = if gt(zero, diff) then sub(zero, diff) else diff
    in_tol = if eq(tol, zero) then eq(actual, expected) else not(gt(abs_diff, tol))
    ok = and(not(or(actual_nan, expected_nan)), in_tol)
    test_assert(ok, string_concat("assert_close (", string_concat(label, string_concat("): expected ", string_concat(to_string(expected), string_concat(", got ", string_concat(to_string(actual), string_concat(", tol ", to_string(tol)))))))))
  }
}
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
