module Std.Sort
export (sort)
def sort[p: Numeric](values: &tensor[..r, p], axis: int32) -> (tensor[..r, p], tensor[..r, int64]) = sort(values, axis)
