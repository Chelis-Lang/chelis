# Ownership fixture container types

The ownership oracle exercises List identity, aliasing, transfer and destruction.
Its numeric bracket bindings declare `List[i64]` (and nested `List[List[i64]]`)
so the oracle keeps that container contract under either literal default. This
does not decide the literal-default change owned by chelis#3114.

On main `cc00e1320`, the two integration targets ran 72 tests: 62 passed and
ten failed before this correction. Seven shared fixtures and the inline List
bindings now declare their original container type. The same 72 tests pass,
including the unchanged ownership/drop assertions and typed rejection twins.
The malformed-combinator twins also declare their Lists, so they reach the
authored missing-callee rejection rather than failing on container selection.

The acceptance command is `cargo nextest run -p chelis-ir --test
ownership_lowering --test issue_1277_kernel_decision --no-fail-fast`. Both
existing targets join standing PR CI; their required identities and negative
controls are retained. Shared-fixture execution is additionally checked by
`scripts/compiled_value_ownership_oracle.py --phase 2`.
