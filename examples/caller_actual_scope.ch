-- Expected: out = 25. Each actual uses caller scope before formal bindings exist.
def first(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[2, f32] = sub(x, y)
out = {
  x = to_tensor([2.0f32, 3.0f32])
  sum(first(to_tensor([10.0f32, 20.0f32]), x), 0i32)
}
