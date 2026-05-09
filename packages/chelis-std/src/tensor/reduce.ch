module Std.Tensor.Reduce
export (min, prod, argmax, argmin)
sig min: &tensor[a, b, f32] -> int32 -> tensor[b, f32]
sig prod: &tensor[a, b, f32] -> int32 -> tensor[b, f32]
sig argmax: &tensor[a, b, f32] -> int32 -> tensor[b, f32]
sig argmin: &tensor[a, b, f32] -> int32 -> tensor[b, f32]
