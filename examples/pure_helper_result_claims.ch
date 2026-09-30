def without_first[n](x: tensor[n, f32]) -> tensor[*, f32] = shrink(x, [[1i64, shape(x, 0i32)]])
def two_items[n](x: tensor[n, f32]) -> tensor[2, f32] ! { IO } = {
  result = without_first(x)
  _ = print("two items ready")
  result
}
def three_items[n](x: tensor[n, f32]) -> tensor[3, f32] ! { IO } = {
  result = without_first(x)
  _ = print("three items ready")
  result
}
pair = [1.0f32, 2.0f32, 3.0f32] |> to_tensor |> two_items
triple = [1.0f32, 2.0f32, 3.0f32, 4.0f32] |> to_tensor |> three_items
