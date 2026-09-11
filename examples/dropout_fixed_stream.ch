-- Evaluator-only fixed stream: backward reuses the forward mask.
def loss(x: tensor[4, f32]) -> tensor[f32] = x |> dropout(0.5f32) |> sum(0)
def main() =
  with seed(42i64) {
    x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])
    gradient = grad(loss)(x)
    (gradient, dropout(x, 0.5f32))
  }
