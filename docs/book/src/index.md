# Chelis guide

Chelis is a numerical computing language for code that agents write and people
supervise. Tensors carry named dimensions and precision in their type, so
`chelis check` rejects a shape or dtype mismatch before anything runs.
Properties you state with `@property` are checked by `chelis prove`, which
tries an SMT solver or seeded random sampling and reports which method
produced each result: a solver result covers every input under the arithmetic
model it reports, and a sampled result covers only the inputs it drew.

Chelis programs are usually written in Surf. Deep is the canonical representation
used by the compiler and tooling; you can inspect it with `chelis deep`.

## Start here

1. [Install](install.md) the Chelis toolchain.
2. [Write and run your first program](first-program.md).
3. [Read and run the examples](examples.md), each shown with its output.

## Keep learning

- [CLI workflow](cli.md) covers formatting, checking, evaluation, and builds.
- [Type system basics](types.md) introduces tensor shapes and precision.
- [Effects](effects.md) explains host I/O, device regions, and random keys.
- [Checking properties](proving.md) covers `chelis prove` and how to read its results.
- [Reef and packages](reef.md) covers package projects.
- [Reference map](reference.md) points to detailed syntax and semantics.
