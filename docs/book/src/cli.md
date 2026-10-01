# CLI Workflow

Use this chapter when you have a Chelis source file and want to check, run, or build it. Surf
files use `.ch`; Deep files use `.dp`. The [first program](first-program.md) introduces both
forms.

## Command map

| Command | What it does |
|---|---|
| `chelis fmt FILE` | Prints canonical Surf or Deep source. Use `--inplace` to write it or `--check` to test formatting without writing. |
| `chelis lint --check FILE` | Reports style violations and exits nonzero for blocking ones. |
| `chelis check FILE` | Checks types and effects and prints a JSON report. |
| `chelis deep FILE.ch` | Prints canonical Deep for Surf input. |
| `chelis surf FILE.dp` | Prints canonical Surf for Deep input. |
| `chelis validate --surf FILE.ch` | Checks Surf syntax against the conformance grammar. Use `--deep` for Deep or `--desugar` to validate the Deep form of Surf input. |
| `chelis eval --file FILE` | Evaluates a `.ch` or `.dp` file. `chelis eval 'EXPR'` evaluates an inline expression. |
| `chelis build FILE.ch --target c --output out/` | Writes source and runtime artifacts for a target. |
| `chelis test` | Runs tests in the current Reef package; see [Testing](testing.md). |
| `chelis prove` | Checks properties; see [Checking Properties](proving.md). |
| `chelis reef` | Manages packages; see [Reef and Packages](reef.md). |
| `chelis lane-check PATH` | Runs each program through the evaluator and a compiled C build and compares their complete stdout. |
| `chelis cost FILE` | Reports the copy cost of the lowered IR. |
| `chelis runtime export DIR` | Writes the runtime archive, public headers, and staging receipt this compiler carries. |
| `chelis migrate surf --from 0.18 PATH...` | Prints Surf written in the 0.18 grammar in the current grammar; `migrate deep` does the same for Deep. `--check` verifies the files are already migrated, and `--inplace` rewrites them. |
| `chelis cove` | Opens the Cove terminal UI. |
| `chelis tide` | Opens the interactive REPL. `chelis tide serve`, `chelis tide mcp`, and `chelis tide lsp` start the HTTP, MCP, and language servers. |

`validate` requires exactly one of `--surf`, `--deep`, or `--desugar`. It checks syntax; use
`check` for type and effect errors. Run `chelis COMMAND --help` for the full options of any
command.

## Format, check, and run

From a directory containing `app.ch`:

```sh
chelis fmt --inplace app.ch
chelis lint --check app.ch
chelis check app.ch
chelis eval --file app.ch
```

`check` prints a JSON report even when checking fails. For one file, an empty `errors` array
means exit `0`; a nonempty one means exit `2`. `chelis check DIRECTORY` checks discovered
`.ch` and `.dp` files and prints one JSON envelope with `files` and `errors`. Directory
failures appear either in the affected file's report or in the envelope. A failing directory
check exits `2`.

For a short calculation without a file, use `chelis eval 'EXPR'`. A file containing
definitions but no expression has nothing to display; with `--json`, a successful evaluation
of such a file prints a result whose `roots` array is empty. `chelis eval --file app.ch --timeout 30` bounds an
evaluation in seconds. A timeout fails with a diagnostic.

For scripts, `chelis eval --json --file app.ch` writes one JSON result to stdout on success. On
failure, it writes no result JSON; diagnostics go to stderr. Effects already completed before
the failure may still have produced output.

## When a style check blocks progress

`check`, `validate`, `build`, and `eval --file` check the input file's canonical formatting
and blocking lint rules before their main work. If one fails, run `chelis fmt --inplace FILE`,
then `chelis lint --check FILE` and address any remaining blocking diagnostics. Advisory lint
warnings do not block these commands.

Each of those commands accepts `--allow-style-violations` for an exceptional local run. It
prints a warning and skips the style check for that command. Parse, type, evaluation, and build
errors still fail.

## Build for a target

```sh
chelis build app.ch --target c --output out/
```

`c` is the default target. `hip` and `metal` are also available. `build` writes generated
source and the runtime support and compile flags needed for the selected target; compiling that
source with a platform toolchain is a separate step. A `.dp` input takes the Deep path
automatically. See [Backends](backends.md) for target outputs and platform requirements.
