def make() -> List[tensor[2, f32]] = {
  first = to_tensor([1.0, 2.0])
  second = to_tensor([3.0, 4.0])
  [first, second]
}
out = make()
