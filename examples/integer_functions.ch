module IntegerFunctions
def anchor() -> i32 = 7
def user() -> i32 = anchor()
def is_zero_i64(x: i64) -> bool = {
  classify = fn (value) -> match value with {
    | 0 => true
    | _ => false
  }
  classify(x)
}
zero_checked = 0i64 |> is_zero_i64
