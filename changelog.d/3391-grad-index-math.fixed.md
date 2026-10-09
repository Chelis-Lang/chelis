`grad` no longer rejects index math that carries no cotangent. An integer
`floor_div` whose result only sizes an `insert` (an `expand` extent) is outside
the gradient path, as reshape and movement bounds already were. Before, it was
refused as `floor_div is non-differentiable (piecewise constant)`; a
`floor_div` whose value reaches the loss is still refused. A comparison whose
operands have runtime extents from `to_tensor(range(n))` or a wildcard `*`
parameter now differentiates with its zero cotangent. Before, it failed with
`comparison ... output shape must match its operands`. See
[#3382](https://github.com/Chelis-Lang/chelis/issues/3382) and
[#3391](https://github.com/Chelis-Lang/chelis/issues/3391).
