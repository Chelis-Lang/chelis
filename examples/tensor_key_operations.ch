-- Build independent keys from a tensor of seeds, then derive each element.
def main() = {
  seeds = to_tensor([[1i64, 2i64], [3i64, 4i64]])
  (left, right) = seeds |> key_from_seed |> split_key
  (fold_in(left, seeds), split_keys(right, 2i64))
}
