# CLI Workflow

Chelis ships one CLI with machine-facing and human-facing subcommands.

## Core Commands

- `chelis fmt` canonicalizes Surf or Deep source.
- `chelis lint` enforces naming and style conventions from `spec/01-nomenclature.md`.
- `chelis check` parses, desugars, type-checks, and reports fitness/errors.
- `chelis deep` prints canonical Deep for a Surf program.
- `chelis surf` decompiles Deep back to Surf.
- `chelis eval` runs the host/runtime evaluator.
- `chelis validate` runs the executable-grammar validator on the input.
- `chelis build` emits C or HIP source plus runtime artifacts and compile flags.
- `chelis tide` exposes the HTTP/MCP tooling surface.

## Style Gate (Built-In on Every Build)

`chelis build`, `chelis check`, `chelis validate`, and
`chelis eval --file` enforce a **style gate** on the input file before
the front-end pipeline runs. The gate is two checks in one:

1. **Formatter check** — the file must be byte-identical to its
   canonical re-print (the same comparison `chelis fmt --check` does).
2. **Lint check** — the file must pass every `chelis lint` rule that
   applies to its surface.

A failure prints a one-line-per-issue diagnostic to stderr and exits
non-zero. The error message tells you exactly what to run:

```text
error: app.ch: not canonically formatted; run `chelis fmt --inplace app.ch` to fix
       app.ch:7: surf-value-snake-case (§3.2): function `BadName` is not snake_case
2 issue(s) blocked the build; pass `--allow-style-violations` to bypass (CI must not).
```

The gate runs only on the user-supplied source. Reef-imported library
decls are not re-checked here — they were already gated when the
package was published.

### Bypass flags

| Surface | Bypass | When to use |
|---|---|---|
| `--allow-style-violations` (CLI flag) | Per-command opt-out; logs a stderr warning. | Emergency local builds and one-off migrations. **CI must not pass this flag.** |
| `CHELIS_STYLE_GATE_DISABLE=1` (env var) | Process-wide opt-out. | The integration-test corpus only — used by tests that synthesize ad-hoc Surf to exercise pipeline behavior independently of style. **Not for production builds or end-user scripts.** |

`chelis eval EXPR` (the inline-expression form) is unaffected — there
is no on-disk source to canonicalize, so the gate does not apply.

## Typical Loop

```sh
# Edit, then canonicalize and lint locally:
chelis fmt --inplace app.ch
chelis lint --check

# Then check, eval, build (each step re-runs the gate):
chelis check app.ch
chelis eval --file app.ch
chelis build app.ch --target c --output out/
```

When `chelis eval --file` runs from inside a Reef package root, ad hoc
snippet files can import package modules even if the snippet file
itself lives outside `src/` and does not declare a top-level `module`.

## Output Contract

- `check` is machine-facing: perfect score implies an empty error list.
- `deep` defaults to canonical pretty output.
- `build` emits source and runtime artifacts; it does not invoke
  `gcc` or `hipcc` for you.
- `lint` prints `path:line:col: rule_id (§spec_ref): message`; with
  `--check` it exits non-zero on any violation.

For exact CLI semantics, use the numbered specs plus the CLI
integration tests in the repo (notably
`crates/chelis-cli/tests/style_gate.rs` for the gate itself and
`crates/chelis-cli/tests/cli.rs` for end-to-end pipeline behavior).
