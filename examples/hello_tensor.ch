module HelloTensor

def main(): unit =
  let a: tensor[n, f32] = tensor_const(1.0, 2.0, 3.0)
  let b: tensor[n, f32] = tensor_const(4.0, 5.0, 6.0)
  in add(a, b)
