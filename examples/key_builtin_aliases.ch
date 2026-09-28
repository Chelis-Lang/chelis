def main() = {
  seed = key_from_seed
  halves = split_key
  folded = fold_in
  children = split_keys
  seeds = to_tensor([1i64, 2i64])
  keys = seed(seeds)
  (left, right) = halves(keys)
  (seed(7i64), seed(seeds), folded(left, seeds), children(right, 2i64))
}
