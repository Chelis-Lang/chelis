-- A bounded scalar dtype does not give a scalar the shape of a tensor.
-- Construct the matching shape explicitly before arithmetic or comparison.
def scale[p: Float](xs: tensor[3, p], factor: p) -> tensor[3, p] = mul(xs, insert(scalar_to_tensor(factor), 0i32, shape(xs, 0i32)))
def above[p: Float](xs: tensor[3, p], threshold: p) -> tensor[3, bool] = gt(xs, insert(scalar_to_tensor(threshold), 0i32, shape(xs, 0i32)))
def main() -> tensor[3, bool] = {
  scaled = scale(to_tensor([1.0f64, 2.0f64, 3.0f64]), 2.0f64)
  above(scaled, 3.0f64)
}
