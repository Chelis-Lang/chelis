# CLI Workflow

Chelis ships one CLI with machine-facing and human-facing subcommands.

## Core Commands

- `chelis fmt` canonicalizes Surf or Deep source.
- `chelis check` parses, desugars, type-checks, and reports fitness/errors.
- `chelis deep` prints canonical Deep for a Surf program.
- `chelis surf` decompiles Deep back to Surf.
- `chelis eval` runs the host/runtime evaluator.
- `chelis build` emits C or HIP source plus runtime artifacts and compile flags.
- `chelis tide` exposes the HTTP/MCP tooling surface.

## Typical Loop

```sh
chelis fmt app.ch --check
chelis check app.ch
chelis eval --file app.ch
chelis deep app.ch
chelis build app.ch --target c --output out/
```

When `chelis eval --file` runs from inside a Reef package root, ad hoc snippet files can
import package modules even if the snippet file itself lives outside `src/` and does not
declare a top-level `module`.

## Output Contract

- `check` is machine-facing: perfect score implies an empty error list.
- `deep` defaults to canonical pretty output.
- `build` emits source and runtime artifacts; it does not invoke `gcc` or `hipcc` for you.

For exact CLI semantics, use the numbered specs plus the CLI integration tests in the
repo.
