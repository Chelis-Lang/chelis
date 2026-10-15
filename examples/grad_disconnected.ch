def constant_loss[rows, cols](x: tensor[rows, cols, f32]) -> f32 = 1.0f32
out = grad(constant_loss)(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))
inferred_zero = grad(fn (unused) -> 1.0f32)(to_tensor([1.0f32, 2.0f32]))
type Leaf[n] =
  | Leaf { value: tensor[n, f32] }
type Pair =
  | Pair { first: Leaf[2], second: Leaf[3] }
def pair_constant(p: &Pair) -> f32 = 1.0f32
def pair_gradient(p: Pair) -> Pair = grad(pair_constant)(p)
nested_zero = pair_gradient(Pair { first: Leaf { value: to_tensor([1.0f32, 2.0f32]) }, second: Leaf { value: to_tensor([3.0f32, 4.0f32, 5.0f32]) } })
