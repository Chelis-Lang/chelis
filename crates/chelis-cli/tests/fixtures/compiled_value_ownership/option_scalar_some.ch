def option_value() -> i64 = {
  value: Option[i64] = Some(7i64)
  match value with {
    | None => 0i64
    | Some(inner) => inner
  }
}
out = option_value()
