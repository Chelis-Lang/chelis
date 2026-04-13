module Std.Nn.Silu
export (forward, sigmoid_scalar)
def forward[n](x: tensor[n, f32]) -> tensor[n, f32] = to_tensor(map(fn (v: f32) -> mul(v, sigmoid_scalar(v)), to_list(x)))
def sigmoid_scalar(v: f32) -> f32 = div(cast(1.0, f32), add(cast(1.0, f32), exp(neg(v))))
