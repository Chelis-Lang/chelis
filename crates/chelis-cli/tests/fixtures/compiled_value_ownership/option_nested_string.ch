def option_length() -> int64 = {
  value: Option[Option[string]] = Some(Some(string_concat("abc", "def")))
  match value with {
    | None => 0i64
    | Some(inner) => match inner with {
    | None => 0i64
    | Some(text) => string_len(text)
  }
  }
}
out = option_length()
