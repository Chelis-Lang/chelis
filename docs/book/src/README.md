# Chelis Guide

Chelis is a functional language for AI research. This guide starts with a working
program, then introduces tensor types, effects, packages, and the compiler commands
used to work with them.

Chelis programs are usually written in Surf. Deep is the canonical representation
used by the compiler and tooling; you can inspect it with `chelis deep`.

## Start here

1. [Install](install.md) the Chelis toolchain.
2. [Write and run your first program](first-program.md).
3. [Explore examples](examples.md) from a source checkout.

## Keep learning

- [CLI Workflow](cli.md) covers formatting, checking, evaluation, and builds.
- [Type System Basics](types.md) introduces tensor shapes and precision.
- [Effects and Handlers](effects.md) explains host I/O, device regions, and random keys.
- [Reef and Packages](reef.md) covers package projects.
- [Language Reference](reference.md) points to detailed syntax and semantics.
