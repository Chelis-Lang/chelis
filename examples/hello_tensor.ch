module HelloTensor
def activate_scalar(x: f32) -> f32 = gelu(silu(tanh(sigmoid(relu(x)))))
def preserve_integer_scalar(x: int32) -> int32 = round(ceil(floor(x)))
def choose_scalar(a: f32, b: f32) -> f32 = min_elem(max_elem(a, b), cast(1.0, f32))
def main() -> tensor[n, f32] = {
  a = to_tensor([1.0, 2.0, 3.0])
  b = to_tensor([4.0, 5.0, 6.0])
  out = add(a, b)
  out
}
