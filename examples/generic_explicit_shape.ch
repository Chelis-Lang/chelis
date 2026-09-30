-- A bounded scalar dtype does not give a scalar the shape of a tensor.
-- Construct the matching shape explicitly before arithmetic or comparison.
def scale[p: Float](xs: tensor[3, p], factor: p) -> tensor[3, p] = mul(xs, factor |> scalar_to_tensor |> insert(0i32, shape(xs, 0i32)))
def above[p: Float](xs: tensor[3, p], threshold: p) -> tensor[3, bool] = gt(xs, threshold |> scalar_to_tensor |> insert(0i32, shape(xs, 0i32)))
def main() -> tensor[3, bool] = {
  scaled = [1.0f64, 2.0f64, 3.0f64] |> to_tensor |> scale(2.0f64)
  above(scaled, 3.0f64)
}
