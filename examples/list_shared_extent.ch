def add_pair[n](xs: List[tensor[n, f32]]) -> tensor[n, f32] = index(xs, 0i64) |> add(index(xs, 1i64))
def duplicate_first[n](xs: List[tensor[n, f32]]) -> List[tensor[n, f32]] = {
  first = index(xs, 0i64)
  fold(fn (acc: List[tensor[n, f32]], i: i64) -> {
    old = acc
    older = old
    append(older, first)
  }, xs, range(0i64, 2i64))
}
def main() -> tensor[2, f32] = add_pair(duplicate_first([to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 4.0f32])]))
