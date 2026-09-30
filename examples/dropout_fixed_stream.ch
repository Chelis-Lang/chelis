-- Keyed draws in the evaluator and C: backward reuses the forward mask.
def loss(k: key, x: tensor[4, f32]) -> tensor[f32] = dropout(k, x, 0.5f32) |> sum(0)
def main() = {
  (first, second) = 42i64 |> key_from_seed |> split_key
  x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])
  gradient = grad(loss, wrt=x)(first, x)
  (gradient, dropout(second, x, 0.5f32))
}
