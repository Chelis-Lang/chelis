type Activation =
  | Relu
  | Sigmoid
def activate(act: Activation, x: tensor[n, f32]) -> tensor[n, f32] = {
  match act with {
    | Relu => relu(x)
    | Sigmoid => sigmoid(x)
  }
}
def linear(w: tensor[out, inp, f32], b: tensor[out, f32], x: tensor[inp, f32]) -> tensor[out, f32] = add(matmul(w, x), b)
def mlp(w1: tensor[hidden, input, f32], b1: tensor[hidden, f32], w2: tensor[output, hidden, f32], b2: tensor[output, f32], act: Activation, x: tensor[input, f32]) -> tensor[output, f32] = {
  x
  |> fn (v) -> linear(w1, b1, v) |> fn (v) -> activate(act, v) |> fn (v) -> linear(w2, b2, v)
}
