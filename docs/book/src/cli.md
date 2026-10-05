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
| `chelis build FILE.ch --target c --output out/` | Builds a native executable or static library for a target. |
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

File reports carry fitness components, checked-node counters, unresolved names,
and diagnostics. `--show-inferred` requests callable signatures. Annotated Deep
is available through the compiler checked-program API; the JSON report has no
`typed_ast` member. A failure report still carries diagnostics even when no
checked program can be produced.

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

`c` is the default target. `hip` and `metal` are also available. `build` invokes
its native toolchain and produces an executable for observable programs, or a
static library for definitions-only modules. It retains generated sources and
runtime support. Run `./out/app` after the example above.

Add `--emit-c` to stop after source emission and runtime staging, without requiring
a native compiler. This flag also applies to HIP and Metal source. A `.dp` input
takes the Deep path automatically. See [Backends](backends.md) for artifact names,
compiler overrides, library linking, and platform requirements.

## Migrating pipes

Surf retains `|>`, and `fmt` preserves it. Deep 0.20 represents a pipe as ordinary
applications. `chelis deep` (with or without `--annotate`) and `chelis surf` therefore
emit calls. Stored Deep and compiled caches containing the previous pipe node must
be regenerated. Compiler caches and shell packages reject previous format versions.

Folding before literal typing makes `0.1 |> cast(f64)` equal to `cast(0.1, f64)`.
To retain an existing program's previous f32-rounded value, migrate using a compiler
from before this change:

```console
chelis migrate pipes --baseline-compiler /path/to/previous/chelis --inplace file.ch other.ch
chelis migrate pipes --baseline-compiler /path/to/previous/chelis --check file.ch other.ch
```

With neither flag, the command prints one migrated file. It uses the previous
compiler's expanded Deep to suffix unsuffixed numeric literals in pipe seeds and
adds grouping with the previous grammar's reading, such as `(2.0f32 * x) |> f`.
Every file must preserve normalized expanded Deep and reach a formatter fixed point.
The whole batch is checked before any file is replaced; missing dtype evidence,
changed Deep, or comments the formatter cannot preserve reject the migration.
Use the command on shell and downstream sources before switching compiler versions.
A file rejected by the previous compiler needs separate diagnosis, rather than an
assumed default dtype.

Retain **both** compilers for this migration, even when A1 (#3164) lands in the
same release. The baseline must precede pipe normalization and A1; the compiler
running `migrate pipes` must include pipe normalization and precede A1. A later
compiler cannot prove an unchanged whole-file graph after unrelated A1 typing
changes. Pin source revisions and retain the corresponding binaries together.
The tested pair is baseline
`c5e4d116c3852c318bdb415fb052add73efbff6d` and migration tool
`f633ca513bf4b1002525f3490f72f66a3c4f80a7`; build each revision's `chelis-cli`
in its own checkout and preserve those executables rather than a moving default.
The historical reader is available only through the non-default
`pre-020-pipe-migration` feature; the CLI opts in for this explicit command.
Normal Deep parsing and stamping still reject retired pipe receipts.
