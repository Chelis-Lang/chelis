-- Both signature relationships apply before the body, including unused b.
def paired(a: tensor[seq, f32], b: tensor[batch, seq, f32], x: tensor[width, f32], y: tensor[height, width, f32]) -> (tensor[seq, f32], tensor[width, f32]) = (neg(a), add(x, sum(y, 0i32)))
def main() = paired(to_tensor([1.0f32, 2.0f32]), to_tensor([[3.0f32, 4.0f32]]), to_tensor([5.0f32, 6.0f32]), to_tensor([[7.0f32, 8.0f32]]))
