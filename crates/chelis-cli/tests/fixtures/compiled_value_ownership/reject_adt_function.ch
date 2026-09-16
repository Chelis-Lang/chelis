type CallbackBox =
  | CallbackBox { callback: i8 -> i8 }
def identity(item: i8) -> i8 = item
value = CallbackBox { callback: identity }
out = "unreachable"
