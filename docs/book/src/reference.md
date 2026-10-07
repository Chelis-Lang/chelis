# Reference map

| Question | Page | What it gives |
|---|---|---|
| How do I write a definition, a pipe, a match, an import? | [Surf syntax reference](surf-reference.md) | Every Surf construct with an example, and how it maps to Deep. |
| Which dtypes exist, how do shapes unify, how do casts behave? | [Type system reference](type-reference.md) | Primitive types, tensor and dimension rules, casts, dtype bounds, effects, and ownership. |
| What does an operation take and return, and when does it fail? | [Runtime and standard library](stdlib.md) | Signatures, argument domains, and failure behavior of the built-in operations and `Std` modules. |
| How do I differentiate or batch a function? | [Transforms](transforms.md) | `grad`, `wrt`, `vmap`, and their limits in the evaluator and C builds. |
| Which effects does a function carry? | [Effects](effects.md) | `IO`, `Test`, random keys, and device regions. |
| What does a build produce and how do I link it? | [Build programs](backends.md) | Output files, compiler flags, and calling a static library from C. |
| How do I add a dependency or lay out a package? | [Reef and packages](reef.md) | `reef.toml`, setup, and dependency resolution. |
| How do I test and check properties? | [Testing](testing.md), [Checking properties](proving.md) | The assertion builtins, `chelis test`, and `chelis prove` with its result fields. |
| What does each command print and return? | [CLI workflow](cli.md) | The command map, the `check` report, and exit codes. |

The [Examples](examples.md) page has complete programs with their
output.
