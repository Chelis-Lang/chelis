-- MNIST MLP for the executable Phase 0 path.
-- Inputs and trainable parameters are surfaced as top-level typed loads.

let x = (x : tensor[32, 784, f32])
let labels = (labels : tensor[32, 10, f32])
let w1 = (w1 : tensor[784, 128, f32])
let b1 = (b1 : tensor[128, f32])
let w2 = (w2 : tensor[128, 10, f32])
let b2 = (b2 : tensor[10, f32])

let mm1 = (matmul(x, w1) : tensor[32, 128, f32])
let b1_exp = (expand(b1, 0, 32) : tensor[32, 128, f32])
let pre_h1 = (add(mm1, b1_exp) : tensor[32, 128, f32])
let h1 = (relu(pre_h1) : tensor[32, 128, f32])
let mm2 = (matmul(h1, w2) : tensor[32, 10, f32])
let b2_exp = (expand(b2, 0, 32) : tensor[32, 10, f32])
let logits = (add(mm2, b2_exp) : tensor[32, 10, f32])

let probs = (softmax(logits, 1) : tensor[32, 10, f32])
let log_probs = (log(probs) : tensor[32, 10, f32])
let selected = (mul(log_probs, labels) : tensor[32, 10, f32])
let per_sample = (neg((sum(selected, 1) : tensor[32, f32])) : tensor[32, f32])
let loss = (mean(per_sample, 0) : tensor[f32])
