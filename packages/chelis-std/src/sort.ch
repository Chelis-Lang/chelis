module Std.Sort
export (sort_1d, sort_2d)
def sort_1d[n, p](values: &tensor[n, p], axis: int32) -> (tensor[n, p], tensor[n, int64]) = sort(values, axis)
def sort_2d[m, n, p](values: &tensor[m, n, p], axis: int32) -> (tensor[m, n, p], tensor[m, n, int64]) = sort(values, axis)
