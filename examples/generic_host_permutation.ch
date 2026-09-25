-- Each inserted axis keeps its physical position across host extraction.
def repeat_rows[n](x: tensor[n, f64]) -> tensor[n, 4, f64] ! { IO } = {
  _ = print("before-result")
  a = x
  b = insert(a, 0i32, 4i64)
  result = permute(b, 1i32, 0i32)
  _ = print("after-result")
  result
}
out = [16777217.0f64, 16777217.0f64, 16777217.0f64] |> to_tensor |> repeat_rows
