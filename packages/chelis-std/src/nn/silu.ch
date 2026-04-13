module Std.Nn.Silu
export (forward, sigmoid_scalar)
-- SiLU / Swish activation. Element-wise: silu(x) = x * sigmoid(x),
-- with sigmoid(x) = 1 / (1 + exp(-x)). Implemented with a scalar
-- helper and to_list/map/to_tensor because the `sigmoid` tensor
-- primitive is not supported by the host runtime lowering. Rank-1
-- variant; callers with multi-dim tensors should flatten first.
def forward[n](x: tensor[n, f32]) -> tensor[n, f32] = to_tensor(map(fn (v: f32) -> mul(v, sigmoid_scalar(v)), to_list(x)))
def sigmoid_scalar(v: f32) -> f32 = div(cast(1.0, f32), add(cast(1.0, f32), exp(neg(v))))
