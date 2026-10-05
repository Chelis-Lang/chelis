def main() = {
  seed = key_from_seed
  halves = split_key
  folded = fold_in
  children = fn (k: tensor[2, key]) -> split_keys(k, 2i64)
  seeds = to_tensor([1i64, 2i64])
  keys = seed(seeds)
  (left, right) = halves(keys)
  (seed(7i64), seed(seeds), folded(left, seeds), children(right))
}
