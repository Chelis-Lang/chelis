`grad` no longer rejects index math that carries no cotangent. An integer
`floor_div` whose result only sizes an `insert` (an `expand` extent) is outside
the gradient path, as reshape and movement bounds already were. Before, it was
refused as `floor_div is non-differentiable (piecewise constant)`; a
`floor_div` whose value reaches the loss is still refused. A comparison or
logical operation over a runtime extent from `to_tensor(range(n))` or a
wildcard `*` parameter now differentiates, including the masks the `abs` and
`pow` adjoints build. Before, these failed with `comparison ... output shape
must match its operands`, and `pow` over a wildcard extent also failed with
`0 extent source(s) ... (op const)`. See
[#3382](https://github.com/Chelis-Lang/chelis/issues/3382) and
[#3391](https://github.com/Chelis-Lang/chelis/issues/3391).
