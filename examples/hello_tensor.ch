module Hellotensor
def main() -> tensor[n, f32] = {
  a = to_tensor([1.0, 2.0, 3.0])
  b = to_tensor([4.0, 5.0, 6.0])
  add(a, b)
}
