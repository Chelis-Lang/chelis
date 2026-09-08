def option_value() -> int64 = {
  value: Option[int64] = Some(7i64)
  match value with {
    | None => 0i64
    | Some(inner) => inner
  }
}
out = option_value()
