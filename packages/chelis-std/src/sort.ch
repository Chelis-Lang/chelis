module Std.Sort
export (sort)
def sort[p_numeric](values: &tensor[..r, p_numeric], axis: int32) -> (tensor[..r, p_numeric], tensor[..r, int64]) = sort(values, axis)
