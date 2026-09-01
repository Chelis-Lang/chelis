module Std.Scalar
export (max, min, abs)
def max[p: Numeric](a: p, b: p) -> p = if gt(a, b) then a else b
def min[p: Numeric](a: p, b: p) -> p = if lt(a, b) then a else b
def abs[p: Numeric](x: p) -> p = if lt(x, sub(x, x)) then neg(x) else x
