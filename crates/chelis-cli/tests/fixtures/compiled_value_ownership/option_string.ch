def option_length() -> i64 = {
  value: Option[string] = Some(string_concat("abc", "def"))
  match value with {
    | None => 0i64
    | Some(inner) => string_len(inner)
  }
}
out = option_length()
