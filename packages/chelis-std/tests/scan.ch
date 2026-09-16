module Std.Tests.Scan
import Std.Scan (scan_list)
import Std.Test (assert_eq)
def scan_add(acc: i64, x: i64) -> i64 = add(acc, x)
def test_scan_list_prefix_sums() -> unit ! { Test } = {
  values = [cast(1, i64), cast(2, i64), cast(3, i64)]
  out = scan_list(scan_add, cast(0, i64), values)
  _ = assert_eq(index(out, cast(0, i64)), cast(1, i64), "scan_list first prefix")
  assert_eq(index(out, cast(2, i64)), cast(6, i64), "scan_list final prefix")
}
