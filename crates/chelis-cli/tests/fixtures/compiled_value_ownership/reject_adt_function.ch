type CallbackBox =
  | CallbackBox { callback: int8 -> int8 }
value = CallbackBox { callback: fn (item: int8) -> item }
out = "unreachable"
