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
§5.2 now states: an evaluation enters each selected root's declaration and
every value declaration that an entered declaration names, whether or not
the name is read, and because a function runs inlined in its caller, a
function's own nodes are entered only when the function is selected. A
discarded trapping node in a selected or entered declaration therefore traps
in `chelis eval` (the DAG evaluator and the host interpreter) and in compiled
C, while an uncalled function's discarded overflow neither runs nor makes
its parameters required inputs.

Not yet covered: integer reductions and the movement-op domain traps, so
[05-OP-68] keeps a non-normative note that its rule is not fully
implemented for every trapping operation; the host interpreter does not trap
on a dead reference to a trapping value declaration inside a taken `if` arm;
and in an `if` lowered as a selection, an operation or a dead value
reference in the untaken arm can still trap in `chelis eval` and compiled
C ([#2563](https://github.com/Chelis-Lang/chelis/issues/2563)).
This carries and supersedes
[#2466](https://github.com/Chelis-Lang/chelis/pull/2466). See
[#2440](https://github.com/Chelis-Lang/chelis/issues/2440).
