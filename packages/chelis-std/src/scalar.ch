module Std.Scalar
export (max, min, abs)
def max[p: Numeric](a: p, b: p) -> p = max_elem(a, b)
def min[p: Numeric](a: p, b: p) -> p = min_elem(a, b)
def abs[p: Numeric](x: p) -> p = add(mul(x, if lt(x, cast(0, p)) then cast(-1, p) else if gt(x, cast(0, p)) then cast(1, p) else cast(0, p)), cast(0, p))
