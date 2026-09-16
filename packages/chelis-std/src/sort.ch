module Std.Sort
export (sort)
def sort[p: Numeric](values: &tensor[..r, p], axis: i32) -> (tensor[..r, p], tensor[..r, i64]) = sort(values, axis)
