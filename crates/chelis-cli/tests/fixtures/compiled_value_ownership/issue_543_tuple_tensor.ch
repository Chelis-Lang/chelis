def make() -> List[(tensor[2, f32], i64)] = {
  first = to_tensor([1.0, 2.0], f32)
  second = to_tensor([3.0, 4.0], f32)
  [(first, 1i64), (second, 2i64)]
}
out = make()
