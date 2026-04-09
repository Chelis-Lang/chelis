module Hellotensor
def main() -> tensor[n, f32] = {
  a = tensor_const(1.0, 2.0, 3.0)
  b = tensor_const(4.0, 5.0, 6.0)
  add(a, b)
}