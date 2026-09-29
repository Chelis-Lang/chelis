A discarded potentially trapping node is no longer eliminated, so its trap
occurs. `spec/06-transformations.md` §5.2 makes every potentially effectful
or trapping node an observable root ("purity alone does not make a possible
trap dead"), but dead-code elimination derived liveness from value
reachability alone, so a discarded integer overflow, division by zero or
out-of-domain cast did not happen. One predicate now makes an [05-OP-68]
guarded abort, a potentially trapping numeric node and a draw that can trap
by itself observable roots, and the evaluator, dead-code elimination and
`grad`'s pruner all read it. The numeric test is dtype-aware: integer
arithmetic and division and a cast into an integer or `bool` width can trap,
float arithmetic cannot, and §5.2's own example of a removable dead float
`add` still holds.

The seeds are scoped to the declarations the evaluation enters, the rule
§5.2 now states: an evaluation enters each selected root's declaration, a
function runs inlined in its caller, and a reference to a value declaration
whose initializer can trap runs that initializer inlined where the reference
is reached, whether or not it is read, so a function's or a value's own
nodes run only when it is selected. The references one declaration makes
under one activation share one copy of the initializer, and a value named
in another's initializer reuses its copy there, so the copies grow linearly
with a chain of such values for each entry. A discarded trapping node in a selected
declaration, or reached through a call or such a reference, therefore traps
in `chelis eval` (the DAG evaluator and the host interpreter) and in compiled
C, while an uncalled function's discarded overflow neither runs nor makes
its parameters required inputs. A node in an `if` arm that is lowered as a selection checks only when its
arm is taken. An untaken arm's integer arithmetic, division or cast, integer
sum, `max_reduce` or `argmax_reduce` over an empty axis, runtime `shrink`, `stride` or
`pad` bound, call or result extent claim, axis restated under another
extent name ([#2512](https://github.com/Chelis-Lang/chelis/issues/2512)),
abort, or dead value reference checks nothing in
the DAG evaluator and in compiled C,
including under `grad`, per `vmap` row, at a `vmap` call site and in the
body of a `vmap` of `grad`
([#2563](https://github.com/Chelis-Lang/chelis/issues/2563)). The same
holds, in the DAG evaluator, the host interpreter and compiled C, for an
untaken arm holding a callee's result claim, a local tensor ascription, or
a broadcast `expand` of a local or a parameter whose axis the arm claims is
1: the claim is not checked, a node sized by it yields zeros, and the `if`
returns the other branch when lowering proves the two arms' extents equal
(one origin outside the `if`, one declared extent, or one extent C names
both arms by). Where it does not, the DAG evaluator and the host interpreter
refuse the `if` with a typed error
([#2583](https://github.com/Chelis-Lang/chelis/issues/2583)), and compiled C
runs it as host control flow or refuses to build it.
`where` no longer reads or shape-checks a branch
its condition selects in no element, so a condition that selects one branch
everywhere returns that branch, which must still be shaped like the
condition. Under `grad`
such an arm also contributes exactly nothing to the gradient, even where the
values it computes are not finite, so a `log` of zero in an untaken arm, or
of the zeros an untaken draw yields, no longer turns the gradient into NaN;
a taken arm's non-finite derivative is returned unchanged
([#2640](https://github.com/Chelis-Lang/chelis/issues/2640)).
`chelis build --target hip` gates its integer arithmetic and division
kernels the same way and refuses any other checking node under an
activation. A dead `let` of an overflowing integer sum or product, an
empty-axis `max_reduce` or `argmax_reduce`, or an out-of-range runtime `shrink`, `stride` or `pad` bound now
traps in every lane.

Not yet covered: an unused float `mean` over an empty axis, and an unused
runtime `reshape` or `expand` target, may still be removed, and those target
checks still run in an untaken arm. The `gather`, `scatter`, `scatter_add`,
`scatter_elements` and `one_hot` index checks are no longer removed, under a
separate entry for the same issue, but they too still run in an untaken arm.
A host-only scalar
operation (such as a shift) discarded inside a `grad` or `vmap` body does
not run, so its trap does not occur in any lane; [05-OP-68] and
spec/03 §4.4 keep a note that their rule is not fully implemented for every
trapping operation. This carries and supersedes
[#2466](https://github.com/Chelis-Lang/chelis/pull/2466). See
[#2440](https://github.com/Chelis-Lang/chelis/issues/2440).
