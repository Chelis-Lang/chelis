`grad` over a checked `cast` from a float source to an integer or bool target
now rejects structurally with `AdRejectionReason::PiecewiseConstant` instead of
contributing a silent zero gradient. `spec/04-type-system.md` [04-NUM-14] has
always required this — "it never contributes a silent zero" — and `cast_trunc`,
`floor`, `ceil` and `round` already rejected in the same position; the checked
default was the one operation in its class that did not, so a float-to-integer
round trip inside a differentiated body used to stall training quietly rather
than fail. The rejection reaches the evaluator and the generated C lane alike.
A float-to-float cast keeps [04-NUM-14]'s exact backward cast, and a bool or
integer SOURCE still differentiates unchanged, because a discrete source
"carries no cotangent, irrespective of target" — including on an index edge
such as `gather`'s, which remains a stop-gradient boundary. [05-OP-63]'s
`Adjoint:` restatement, which read as rejecting an integer-to-float cast too,
now matches the atom it defers to. A rejection reached through a comparison
operand still reports an unstructured reason rather than the atom's
([#730](https://github.com/Chelis-Lang/chelis/issues/730)). Part of
[#729](https://github.com/Chelis-Lang/chelis/issues/729).
