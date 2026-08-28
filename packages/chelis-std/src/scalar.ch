module Std.Scalar
export (max, min, abs)
def max[p_numeric](a: p_numeric, b: p_numeric) -> p_numeric = if gt(a, b) then a else b
def min[p_numeric](a: p_numeric, b: p_numeric) -> p_numeric = if lt(a, b) then a else b
def abs[p_numeric](x: p_numeric) -> p_numeric = if lt(x, sub(x, x)) then neg(x) else x
