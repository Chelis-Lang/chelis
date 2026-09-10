-- Evaluator-only: host-produced sizes retain the checked result claim.
def checked(source: tensor[m, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] = {
  first = dropout(source, 0.0f32)
  dropout(reshape(x, [numel(first), 2i64]), 0.5f32)
}
def main() =
  with seed(42i64) {
    source = to_tensor([1.0f32, 1.0f32])
    x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])
    checked(source, x)
  }
