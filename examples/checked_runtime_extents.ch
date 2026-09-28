def pair_and_scale[n](b: tensor[unit, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] = reshape(mul(x, expand(b, 0i32, shape(x, 0i32))), [mod(floor_div(shape(x, 0i32), 2i64), 4i64), 2i64])
def main() = pair_and_scale(to_tensor([2.0f32]), to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]))
