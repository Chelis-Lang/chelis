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
nodes run only when it is selected. A discarded trapping node in a selected
declaration, or reached through a call or such a reference, therefore traps
in `chelis eval` (the DAG evaluator and the host interpreter) and in compiled
C, while an uncalled function's discarded overflow neither runs nor makes
its parameters required inputs. A node in an `if` arm that is lowered as a
selection checks only when its arm is taken: an untaken arm's integer
arithmetic, division, cast or dead value reference checks nothing in the
DAG evaluator and in compiled C, including under `grad`, per `vmap` row and
at a `vmap` call site
([#2563](https://github.com/Chelis-Lang/chelis/issues/2563)).

Not yet covered: integer reductions and the movement-op domain traps, so
[05-OP-68] keeps a non-normative note that its rule is not fully
implemented for every trapping operation; and in an `if` lowered as a
selection, an untaken arm's shrink or stride bound, integer reduction,
extent witness or shift, or a check in the body of a `vmap` of `grad`
applied there, can still trap in `chelis eval` and compiled C.
This carries and supersedes
[#2466](https://github.com/Chelis-Lang/chelis/pull/2466). See
[#2440](https://github.com/Chelis-Lang/chelis/issues/2440).
