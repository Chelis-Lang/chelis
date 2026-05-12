module Std.Nn.Silu
export (forward, sigmoid_scalar)
sig forward: &tensor[n, f32] -> tensor[n, f32]
def forward(x) = to_tensor(map(fn (v: f32) -> mul(v, sigmoid_scalar(v)), to_list(x)))
sig sigmoid_scalar: f32 -> f32
def sigmoid_scalar(v) = div(cast(1.0, f32), add(cast(1.0, f32), exp(neg(v))))
