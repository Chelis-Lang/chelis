module Std.Nn.Linear
export (forward)
def forward[batch, in_dim, out_dim](x: tensor[batch, in_dim, f32], w: tensor[in_dim, out_dim, f32], b: tensor[out_dim, f32]) -> tensor[batch, out_dim, f32] = matmul(x, w) |> add(expand(b, 0, batch))
