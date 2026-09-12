A shape-computed route whose operand is still an unresolved type variable now
ties its result to the shape it computes once the operand binds, whatever the
operand's provenance. A `match`-destructured tuple element, a field read on an
unresolved record target, and a deferred `copy` gate's result previously let a
false declared result shape check with no errors. A reducing route over an
operand no declaration can determine, such as `mean(to_tensor([]), 0i32)`, is
now rejected at the declaration boundary instead of failing only at evaluation.
