def logits(x: tensor[32, 784, f32], labels: tensor[32, 10, f32], w1: tensor[784, 128, f32], b1: tensor[128, f32], w2: tensor[128, 10, f32], b2: tensor[10, f32]) -> tensor[32, 10, f32] = {
  h1 = matmul(x, w1) |> add(expand(b1, 0, 32)) |> relu
  matmul(h1, w2) |> add(expand(b2, 0, 32))
}
def loss(x: tensor[32, 784, f32], labels: tensor[32, 10, f32], w1: tensor[784, 128, f32], b1: tensor[128, f32], w2: tensor[128, 10, f32], b2: tensor[10, f32]) -> tensor[f32] = {
  h1 = matmul(x, w1) |> add(expand(b1, 0, 32)) |> relu
  logits = matmul(h1, w2) |> add(expand(b2, 0, 32))
  softmax(logits, 1)
  |> log
  |> mul(labels)
  |> sum(1)
  |> neg
  |> mean(0)
}
