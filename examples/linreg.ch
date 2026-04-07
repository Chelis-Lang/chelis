-- Phase 1e executable benchmark: fixed-shape linear regression.

let x = (x : tensor[64, 64, f32])
let y = (y : tensor[64, 1, f32])
let w = (w : tensor[64, 1, f32])
let b = (b : tensor[1, f32])

let mm = (matmul(x, w) : tensor[64, 1, f32])
let b_exp = (expand(b, 0, 64) : tensor[64, 1, f32])
let pred = (add(mm, b_exp) : tensor[64, 1, f32])
let y_neg = (neg(y) : tensor[64, 1, f32])
let err = (add(pred, y_neg) : tensor[64, 1, f32])
let sq = (mul(err, err) : tensor[64, 1, f32])
let per_sample = (mean(sq, 1) : tensor[64, f32])
let loss = (mean(per_sample, 0) : tensor[f32])
