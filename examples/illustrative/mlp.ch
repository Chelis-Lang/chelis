type Activation =
  | Relu
  | Sigmoid
def activate[batch, n](act: Activation, x: tensor[batch, n, f32]) -> tensor[batch, n, f32] =
  match act with {
    | Relu => relu(x)
    | Sigmoid => sigmoid(x)
  }
def linear[batch, inp, out](w: tensor[inp, out, f32], b: tensor[out, f32], x: tensor[batch, inp, f32]) -> tensor[batch, out, f32] = x |> matmul(w) |> add(insert(b, 0, shape(x, 0i32)))
def mlp[batch, input, hidden, output](w1: tensor[input, hidden, f32], b1: tensor[hidden, f32], w2: tensor[hidden, output, f32], b2: tensor[output, f32], act: Activation, x: tensor[batch, input, f32]) -> tensor[batch, output, f32] =
  x
  |> fn (v) -> linear(w1, b1, v) |> fn (v) -> activate(act, v) |> fn (v) -> linear(w2, b2, v)
