module Std.Scalar
export (max, min, abs)
def max(a: f32, b: f32) -> f32 = if gt(a, b) then a else b
def min(a: f32, b: f32) -> f32 = if lt(a, b) then a else b
def abs(x: f32) -> f32 = if lt(x, cast(0.0, f32)) then neg(x) else x
