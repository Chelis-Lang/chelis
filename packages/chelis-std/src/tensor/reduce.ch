module Std.Tensor.Reduce
export (min, prod, argmax, argmin)
def min[a, b](x: tensor[a, b, f32], axis: int32) -> tensor[b, f32] = min_reduce(x, axis)
def prod[a, b](x: tensor[a, b, f32], axis: int32) -> tensor[b, f32] = prod_reduce(x, axis)
def argmax[a, b](x: tensor[a, b, f32], axis: int32) -> tensor[b, f32] = argmax_reduce(x, axis)
def argmin[a, b](x: tensor[a, b, f32], axis: int32) -> tensor[b, f32] = argmin_reduce(x, axis)
