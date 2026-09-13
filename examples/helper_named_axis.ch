def aligned[d](x: tensor[d, f32], gain: tensor[fixed, f32]) -> tensor[d, f32] = mul(x, gain)
def repeat_before_fixed(x: &tensor[..pre, fixed, ..post, f32]) = insert(x, fresh, 2i64, fixed)
def main() = repeat_before_fixed(aligned(to_tensor([1.0f32, 2.0f32]), to_tensor([1.0f32, 2.0f32])))
