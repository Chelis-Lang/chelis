-- Host-produced sizes retain the checked result claim in eval and in C.
def checked[m, n](k: key, source: tensor[m, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] = {
  (k1, k2) = split_key(k)
  first = dropout(k1, source, 0.0f32)
  dropout(k2, reshape(x, [numel(first), 2i64]), 0.5f32)
}
def main() = {
  source = to_tensor([1.0f32, 1.0f32])
  x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])
  checked(key_from_seed(42i64), source, x)
}
