type Choice =
  | Existing
  | Fresh
def choose(original: List[int64]) -> List[int64] = {
  choice = Fresh
  match choice with {
    | Existing => original
    | Fresh => [2i64, 3i64]
  }
}
selected = choose([1i64])
