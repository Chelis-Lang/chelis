type CallbackBox =
  | CallbackBox { callback: int8 -> int8 }
def identity(item: int8) -> int8 = item
value = CallbackBox { callback: identity }
out = "unreachable"
