def make() -> List[List[tensor[2, f32]]] = {
  value = to_tensor([1.0, 2.0])
  inner: List[tensor[2, f32]] = [value, value]
  [inner, inner]
}
out = make()
