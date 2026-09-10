-- Valid-padding windows preserve the leading axis and use the declared stride.
x: tensor[2, 5, f32] = [[1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32], [6.0f32, 7.0f32, 8.0f32, 9.0f32, 10.0f32]]
totals: tensor[2, 2, f32] = reduce_window_sum(&x, [2i64], [2i64])
means: tensor[2, 2, f32] = reduce_window_mean(&x, [2i64], [2i64])
maxima: tensor[2, 2, f32] = reduce_window_max(&x, [2i64], [2i64])
minima: tensor[2, 2, f32] = reduce_window_min(&x, [2i64], [2i64])
