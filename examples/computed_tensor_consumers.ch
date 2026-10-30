def rows() -> tensor[2, 3, i64] = expand(reshape(to_tensor(range(0i64, 3i64), i64), [1i64, 3i64]), 0i32, 2i64)
def average[n](x: tensor[n, f32]) -> f32 = tensor_to_scalar(mean(reshape(x, [numel(x)]), 0i32))
total = tensor_to_scalar(sum(sum(rows(), 1i32), 0i32))
one_third = average(to_tensor([1.0f32, 0.0f32, 0.0f32]))
