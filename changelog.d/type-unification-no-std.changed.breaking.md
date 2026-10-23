Move type models, substitutions, unification, and pure scheme operations into allocation-based `chelis-type-core`.
The crate uses unconditional `no_std`, empty default features, and optional portable `serde`.

Rust consumers now import the canonical core owner and specify evidence through `Type<E>`, `NominalArg<E>`, `Scheme<E>`, and `Subst<E>`.
Old hosted model and solver paths do not exist.
`Prim` remains in `chelis-numeric-core`, and `chelis-vocab` remains unchanged.
The type core has no DAG dependency.

The hosted checker directly uses its sealed `ErrorWitness` type.
It retains diagnostic reports, private sinks and checkpoints, source inference, recursion policy, trusted cache boundaries, and the successful-product veto.
The extraction preserves current nominal arguments, dtype restrictions, levels, collection contracts, shared type children, and deferred shape relations.

The sole type acceptance command is `uv run --managed-python --python 3.11 --no-project python scripts/type_unification_core_oracle.py`.
It compares actual host output against immutable pre-cutover source and separates host execution from target compilation.
Accepted programs can have an empty exported type map.
Remove API-shape tests and unreachable ledger implementation tests.
Host behavior and compiler checks retain coverage.

The unresolved-operand census follows pure decisions into the core.
It retains the existing dispositions and source mutation controls.

The portable solver needs allocation and atomics.
It does not support reentrant interrupts that preempt a held solver lock.
Target compilation proves neither device execution nor allocator capacity.
No language or diagnostic authority transfers through this extraction.
