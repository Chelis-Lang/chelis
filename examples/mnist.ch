def logits(x: tensor[32, 784, f32], labels: tensor[32, 10, f32], w1: tensor[784, 128, f32], b1: tensor[128, f32], w2: tensor[128, 10, f32], b2: tensor[10, f32]) -> tensor[32, 10, f32] =
  let h1 = relu(add(matmul(x, w1), expand(b1, 0, 32)))
  in add(matmul(h1, w2), expand(b2, 0, 32))
def loss(x: tensor[32, 784, f32], labels: tensor[32, 10, f32], w1: tensor[784, 128, f32], b1: tensor[128, f32], w2: tensor[128, 10, f32], b2: tensor[10, f32]) -> tensor[f32] =
  let h1 = relu(add(matmul(x, w1), expand(b1, 0, 32)))
  in let logits = add(matmul(h1, w2), expand(b2, 0, 32))
  in cross_entropy(logits, labels)