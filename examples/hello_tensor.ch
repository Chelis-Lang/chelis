module Hellotensor
def main() -> tensor[n, f32] =
  let a = tensor_const(1.0, 2.0, 3.0)
  in let b = tensor_const(4.0, 5.0, 6.0)
  in add(a, b)