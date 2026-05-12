module Std.Nn.Linear
export (forward)
sig forward: &tensor[a, b, p] -> &tensor[b, c, p] -> &tensor[c, p] -> tensor[a, c, p]
def forward(x, w, b) = {
  bias = expand(b, 0, shape(x, cast(0, int32)))
  wx = matmul(x, w)
  out = add(wx, bias)
  _ = drop(bias)
  _ = drop(wx)
  out
}
