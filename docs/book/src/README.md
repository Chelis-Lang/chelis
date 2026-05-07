# Chelis User Book

Chelis is a functional programming language for AI research. Surf is the readable syntax
for humans. Deep is the canonical s-expression syntax for machines and the compiler.

This book is the practical starting point for downstream users and shell authors. It
covers local setup, the first program loop, CLI validation, the style gate, Reef package
basics, examples, and the map from user docs to the authoritative specs.

If you are modifying Chelis itself, use the repository `AGENTS.md` and the shared
`agent-skills/` workflows instead. This book is for using Chelis and authoring packages
on top of it.

## What You Should Read First

- [Install](install.md)
- [First Program](first-program.md)
- [CLI Workflow](cli.md)
- [Type System Basics](types.md)
- [Reef and Packages](reef.md)
- [Examples](examples.md)
- [Reference Map](reference.md)

## A Small Surf Program

```chelis-surf
def square(x: tensor[f32]) -> tensor[f32] = mul(x, x)
```

This is a complete, compiler-validated example. In this book:

- `chelis-surf` and `chelis-deep` fences are full programs and are validated in CI.
- `chelis-surf-fragment` and `chelis-deep-fragment` fences are partial snippets used to
  teach one construct in isolation.
