`chelis prove --tier auto` checks a solver proof in machine arithmetic instead of
stopping at it. A Tier B decision is made over the reals, so it rules out no
counterexample that exists only in floating point; `auto` now samples the same
`@property` or derived producer obligation afterwards, looking for one. A concrete
machine counterexample replaces the proof with a definite failure that names the verdict
it displaced; a clean search is reported as corroborated without strengthening the proof,
which keeps its `real_arithmetic` qualifier because sampling is not exhaustive; a search
that cannot sample leaves the proof untouched with its reason. The additive
`machine_arithmetic` record field carries that outcome on the CLI and Tide surfaces, and
plain text prints the disclosure in place of the `0/0 passed` ratio a proof does not
have. An explicitly selected tier runs no such check, and neither does a green
conditional on an asserted axiom, whose assumption the concrete evaluator does not
honor. `auto` is therefore never weaker than `--tier fuzz-only` on the same input
([#3634](https://github.com/Chelis-Lang/chelis/issues/3634)).

Tier C validation lowers an `@invariant` predicate from its defining program, so an
integer-typed module constant keeps its declared type. It previously lowered as a real
and compared against an integer-sorted field as false for every value, so a derived
obligation whose invariant did integer arithmetic on a module constant was reported
failed with a counterexample that satisfies it
([#3651](https://github.com/Chelis-Lang/chelis/issues/3651)).
