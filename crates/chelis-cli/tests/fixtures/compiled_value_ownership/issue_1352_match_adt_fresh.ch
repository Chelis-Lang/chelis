type Choice =
  | Existing
  | Fresh
def choose(original: List[i64]) -> List[i64] = {
  choice = Fresh
  match choice with {
    | Existing => original
    | Fresh => [2i64, 3i64]
  }
}
selected = choose([1i64])
