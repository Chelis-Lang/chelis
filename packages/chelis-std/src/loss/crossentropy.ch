module Std.Loss.CrossEntropy
export (loss)
sig loss: &tensor[a, b, f32] -> &tensor[a, b, f32] -> tensor[a, f32]
def loss(logits, labels) = {
  probs = softmax(logits, 1)
  logs = log(probs)
  weighted = mul(logs, labels)
  totals = sum(weighted, 1)
  out = neg(totals)
  _ = drop(probs)
  _ = drop(logs)
  _ = drop(weighted)
  _ = drop(totals)
  out
}
