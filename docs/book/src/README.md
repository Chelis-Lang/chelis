# Chelis Developer Book

Chelis is a functional programming language for AI research. Surf is the readable syntax
for humans. Deep is the canonical s-expression syntax for machines and the compiler.

This book is the developer-facing usage guide. It explains how to install Chelis, write
and run programs, use the CLI, and navigate the Reef package workflow.

For language semantics and design rationale, use the numbered specs in `spec/` and the
active design docs in `spec/design/`.

## What You Should Read First

- [Install](install.md)
- [First Program](first-program.md)
- [CLI Workflow](cli.md)
- [Type System Basics](types.md)
- [Reef and Packages](reef.md)

## A Small Surf Program

```chelis-surf
def square(x: tensor[f32]) -> tensor[f32] = mul(x, x)
```

This is a complete, compiler-validated example. In this book:

- `chelis-surf` and `chelis-deep` fences are full programs and are validated in CI.
- `chelis-surf-fragment` and `chelis-deep-fragment` fences are partial snippets used to
  teach one construct in isolation.
