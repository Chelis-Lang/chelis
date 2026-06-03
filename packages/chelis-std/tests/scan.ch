module Std.Tests.Scan
import Std.Scan (scan_list)
import Std.Test (assert_eq_int)
def scan_add(acc: int64, x: int64) -> int64 = add(acc, x)
def test_scan_list_prefix_sums() -> unit ! { Test } = {
  values = [cast(1, int64), cast(2, int64), cast(3, int64)]
  out = scan_list(scan_add, cast(0, int64), values)
  _ = assert_eq_int(index(out, cast(0, int64)), cast(1, int64), "scan_list first prefix")
  assert_eq_int(index(out, cast(2, int64)), cast(6, int64), "scan_list final prefix")
}
