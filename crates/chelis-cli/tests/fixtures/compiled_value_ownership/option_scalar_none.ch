def option_value() -> int64 = {
  value: Option[int64] = None
  match value with {
    | None => 0i64
    | Some(inner) => inner
  }
}
out = option_value()
