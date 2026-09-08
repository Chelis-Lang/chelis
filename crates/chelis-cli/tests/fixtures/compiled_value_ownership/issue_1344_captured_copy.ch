global_values = [1i64, 2i64]
def take_length() -> int64 = {
  local_alias = global_values
  len(local_alias)
}
out = take_length()
