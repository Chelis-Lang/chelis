def add_pair[n](xs: List[tensor[n, f32]]) -> tensor[n, f32] = index(xs, 0i64) |> add(index(xs, 1i64))
def main() -> tensor[2, f32] = add_pair([to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 4.0f32])])
