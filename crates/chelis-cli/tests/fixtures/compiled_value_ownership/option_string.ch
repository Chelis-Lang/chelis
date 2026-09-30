def option_length() -> i64 = {
  value: Option[string] = "abc" |> string_concat("def") |> Some
  match value with {
    | None => 0i64
    | Some(inner) => string_len(inner)
  }
}
out = option_length()
