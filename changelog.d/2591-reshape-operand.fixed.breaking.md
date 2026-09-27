`reshape` admits a tensor operand, or a scalar of an active data element dtype
read as its rank-0 tensor ([05-OP-49]), and rejects any other operand with its
own diagnostic, however the operand reaches the call
([#2591](https://github.com/Chelis-Lang/chelis/issues/2591)). Previously any
scalar was admitted, so `reshape("s", [3i64])` checked with score 1 and failed
in `eval`, directly, through a lambda parameter, or through a recursive group
member's omitted result.
