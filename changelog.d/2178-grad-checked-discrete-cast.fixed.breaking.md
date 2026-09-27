`grad` over a checked `cast` from a float source to an integer or bool target
now rejects structurally with `AdRejectionReason::PiecewiseConstant` instead of
contributing a silent zero gradient. `spec/04-type-system.md` [04-NUM-14] has
always required this — "it never contributes a silent zero" — and `cast_trunc`,
`floor`, `ceil` and `round` already rejected in the same position; the checked
default was the one operation in its class that did not, so a float-to-integer
round trip inside a differentiated body used to stall training quietly rather
than fail. The rejection reaches the evaluator and `chelis build --target c`
alike; it fires during lowering, so no C is emitted for a rejected program.
A float-to-float cast keeps [04-NUM-14]'s exact backward cast, and a bool or
integer SOURCE still differentiates unchanged, because a discrete source
"carries no cotangent, irrespective of target". Separately, and by a different
mechanism, a float-to-integer cast that only feeds an index — `gather`'s
indices, a movement bound — is still accepted, because index math is a
stop-gradient boundary that carries no cotangent to suppress. [05-OP-63]'s
`Adjoint:` restatement, which read as rejecting an integer-to-float cast too,
now matches the atom it defers to. A rejection reached through a comparison
operand still reports an unstructured reason rather than the atom's
([#730](https://github.com/Chelis-Lang/chelis/issues/730)). Part of
[#729](https://github.com/Chelis-Lang/chelis/issues/729).
