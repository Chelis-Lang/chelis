module Std.Nn.Linear
export (forward)
sig forward: &tensor[batch, in_dim, p] -> &tensor[in_dim, out_dim, p] -> &tensor[out_dim, p] -> tensor[batch, out_dim, p]
def forward[batch, in_dim, out_dim, p](x, w, b) = {
  bias = expand(b, 0, batch)
  wx = matmul(x, w)
  out = add(wx, bias)
  _ = drop(bias)
  _ = drop(wx)
  out
}
