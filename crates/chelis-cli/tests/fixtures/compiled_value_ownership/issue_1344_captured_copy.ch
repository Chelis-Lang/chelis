global_values: List[i64] = [1i64, 2i64]
def take_length() -> i64 = {
  local_alias = global_values
  len(local_alias)
}
out = take_length()
