module Std.Tensor.Reduce
export (min, prod, argmax, argmin)
sig min: &tensor[a, b, p] -> int32 -> tensor[b, p]
sig prod: &tensor[a, b, p] -> int32 -> tensor[b, p]
sig argmax: &tensor[a, b, p] -> int32 -> tensor[b, int64]
sig argmin: &tensor[a, b, p] -> int32 -> tensor[b, int64]
