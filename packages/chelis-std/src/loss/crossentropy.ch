module Std.Loss.CrossEntropy

export (loss)

def loss[batch, classes](
  logits: tensor[batch, classes, f32],
  labels: tensor[batch, classes, f32]
) -> tensor[batch, f32] =
  softmax(logits, 1)
  |> log
  |> mul(labels)
  |> sum(1)
  |> neg
