def option_value() -> i64 = {
  value: Option[i64] = None
  match value with {
    | None => 0i64
    | Some(inner) => inner
  }
}
out = option_value()
