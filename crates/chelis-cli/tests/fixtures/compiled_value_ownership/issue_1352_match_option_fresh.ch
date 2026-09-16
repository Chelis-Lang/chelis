def choose(original: List[i64]) -> List[i64] = {
  choice: Option[bool] = Some(false)
  match choice with {
    | None => original
    | Some(flag) => if flag then original else [2i64, 3i64]
  }
}
selected = choose([1i64])
