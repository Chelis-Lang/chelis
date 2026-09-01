type TensorBox =
  | TensorBox { value: tensor[2, f32] }
out = TensorBox { value: to_tensor([1.0, 2.0]) }
