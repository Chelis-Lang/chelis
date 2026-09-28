type State =
  | State { seed: i64, next_draw: i64 }
def sample[a, c, h, w](k: key, x: &tensor[a, c, h, w, f32], rate: f32) -> tensor[a, f32] = {
  ones = scalar_to_tensor(1.0f32) |> insert(0i32, shape(x, 0i32))
  dropout(k, ones, rate)
}
def given[a, c, h, w](x: tensor[a, c, h, w, f32], row: tensor[a, f32]) -> tensor[a, c, h, w, f32] = {
  mask = insert(insert(insert(row, 1i32, shape(x, 1i32)), 2i32, shape(x, 2i32)), 3i32, shape(x, 3i32))
  mul(x, mask)
}
def direct[a, c, h, w](x: tensor[a, c, h, w, f32], rate: f32, seed: i64, next_draw: i64) -> (tensor[a, c, h, w, f32], State) = {
  row = sample(fold_in(key_from_seed(seed), next_draw), &x, rate)
  (given(x, row), State { seed, next_draw: add(next_draw, 1i64) })
}
def wrapped[a, c, h, w](x: tensor[a, c, h, w, f32], rate: f32, state: State) -> (tensor[a, c, h, w, f32], State) =
  match state with {
    | State { seed, next_draw } => direct(x, rate, seed, next_draw)
  }
def main() -> tensor[2, 1, 1, 1, f32] = {
  x = to_tensor([1.0f32, 2.0f32]) |> reshape([2i64, 1i64, 1i64, 1i64])
  (out, _) = wrapped(x, 0.5f32, State { seed: 0i64, next_draw: 0i64 })
  out
}
