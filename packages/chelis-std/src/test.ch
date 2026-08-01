module Std.Test
export (assert_true, assert_false, assert_eq, assert_eq_int, assert_eq_bool, assert_eq_string, assert_close, assert_close_tensor, assert_eq_tensor_int64, assert_shape, fail)
def assert_true(cond: bool, label: string) -> () ! { Test } = test_assert(cond, label)
def assert_false(cond: bool, label: string) -> () ! { Test } = test_assert(not(cond), label)
def assert_eq(actual: f32, expected: f32, label: string) -> () ! { Test } = test_assert_eq_f32(actual, expected, label)
def assert_eq_int(actual: int64, expected: int64, label: string) -> () ! { Test } = test_assert_eq_int(actual, expected, label)
def assert_eq_bool(actual: bool, expected: bool, label: string) -> () ! { Test } = test_assert_eq_bool(actual, expected, label)
def assert_eq_string(actual: string, expected: string, label: string) -> () ! { Test } = test_assert_eq_string(actual, expected, label)
def assert_close(actual: f32, expected: f32, tol: f32, label: string) -> () ! { Test } = {
  tol_nan = neq(tol, tol)
  tol_negative = gt(cast(0.0, f32), tol)
  if or(tol_nan, tol_negative) then test_assert(false, string_concat("assert_close (", string_concat(label, string_concat("): invalid tolerance ", to_string(tol))))) else {
    actual_nan = neq(actual, actual)
    expected_nan = neq(expected, expected)
    diff = sub(actual, expected)
    abs_diff = if gt(cast(0.0, f32), diff) then sub(cast(0.0, f32), diff) else diff
    in_tol = if eq(tol, cast(0.0, f32)) then eq(actual, expected) else not(gt(abs_diff, tol))
    ok = and(not(or(actual_nan, expected_nan)), in_tol)
    test_assert(ok, string_concat("assert_close (", string_concat(label, string_concat("): expected ", string_concat(to_string(expected), string_concat(", got ", string_concat(to_string(actual), string_concat(", tol ", to_string(tol)))))))))
  }
}
def assert_close_tensor[n, p](actual: &tensor[n, p], expected: &tensor[n, p], tol: f32, label: string) -> () ! { Test } = test_assert_close_tensor(actual, expected, tol, label)
def assert_eq_tensor_int64[n](actual: &tensor[n, int64], expected: &tensor[n, int64], label: string) -> () ! { Test } = test_assert_eq_tensor_int64(actual, expected, label)
def assert_shape[n, p](t: &tensor[n, p], expected_n: int64, label: string) -> () ! { Test } = {
  actual_n = cast(shape(t, cast(0, int32)), int64)
  ok = eq(actual_n, expected_n)
  test_assert(ok, string_concat("assert_shape (", string_concat(label, string_concat("): expected ", string_concat(to_string(expected_n), string_concat(", got ", to_string(actual_n)))))))
}
def fail(msg: string) -> () ! { Test } = test_assert(false, msg)
