-- MNIST MLP: 784 → 128 → 10
-- Two-layer perceptron with ReLU activation and softmax cross-entropy loss.

def forward(x: tensor[batch, features, f32],
            w1: tensor[features, hidden, f32],
            b1: tensor[hidden, f32],
            w2: tensor[hidden, classes, f32],
            b2: tensor[classes, f32]): tensor[batch, classes, f32] =
  let h = relu(add(matmul(x, w1), b1))
  in add(matmul(h, w2), b2)

def cross_entropy_loss(logits: tensor[batch, classes, f32],
                       labels: tensor[batch, classes, f32]): tensor[f32] =
  let probs = softmax(logits)
  let log_probs = log(probs)
  in neg(mean(sum(mul(log_probs, labels))))
