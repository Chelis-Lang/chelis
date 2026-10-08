# CLI workflow

Surf files use `.ch`; Deep files use `.dp`. The commands below accept source
files for formatting, checking, evaluation, and native builds. For a complete
example, start with the [first program](first-program.md).

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
| `chelis build FILE.ch --target c --output out/` | Builds a native executable or static library. C is the default target. |
| `chelis test` | Runs tests in the current Reef package; see [Testing](testing.md). |
| `chelis prove` | Checks properties; see [Checking Properties](proving.md). |
| `chelis reef` | Manages packages; see [Reef and Packages](reef.md). |
| `chelis cost FILE` | Counts the tensor copies the compiled program makes and the bytes they copy, for example `total_copy_count=0, total_bytes_copied=0`. |
| `chelis runtime export DIR` | Writes the compiler's bundled runtime archive and public headers, plus a JSON record of the exported files. |
| `chelis tide` | Opens the interactive REPL. `chelis tide serve` and `chelis tide lsp` start the HTTP and language servers. |

`validate` requires exactly one of `--surf`, `--deep`, or `--desugar`. It checks syntax; use
`check` for type and effect errors. Run `chelis COMMAND --help` for the full options of any
command.

`chelis tide mcp` serves the compiler to a coding agent over the Model Context
Protocol (MCP), the standard interface agent clients use to call external
tools: the agent gets check, eval, prove, and structural edits as tools. The
[agent workflow](https://chelis.ch/docs/chelis/agent-workflow/) shows how to connect a client
and test the connection.

## Format, check, and run

From a directory containing `app.ch`:

```sh
chelis fmt --inplace app.ch
chelis lint --check app.ch
chelis check app.ch
chelis eval --file app.ch
```

`check` writes a JSON report to stdout even when checking fails, and nothing
else: style-gate errors and lint warnings go to stderr. For one file, an empty
`errors` array means exit `0`; a nonempty one means exit `2`. Here a function
of one argument is called with two:

```chelis-surf-fragment
def f(x: f32) -> f32 = (x * 2.0f32)
y = f(1.0f32, 2.0f32)
```

```json
{
  "score": 0.925,
  "components": {
    "parse": 1,
    "structure": 1,
    "names": 1,
    "types": 0.875
  },
  "typed_nodes": 7,
  "untyped_nodes": 1,
  "total_nodes": 8,
  "unresolved_names": [],
  "errors": [{"kind":"ArityMismatch","message":"call `f`: expected 1 argument, got 2 arguments","severity":0.7,"expected":"1 argument","got":"2 arguments","span":{"span":"point","offset":40},"span_id":"surf:40..57"}]
}
```

`score` is the weighted sum of four components, each between 0 and 1: parse
(weight 0.1), structure (0.1), names (0.2), and types (0.6, the fraction of
subexpressions that type-check). Here 7 of 8 nodes typed, so `types` is
0.875 and the score is `0.4 + 0.6 * 0.875 = 0.925`. Each error has a `kind`, a `message`, and a `severity` between 0
and 1. When the failing node has a source location, the error also carries
`span` and `span_id`: `"surf:40..57"` is the byte range `[40, 57)` of the Surf
file, here the call `f(1.0f32, 2.0f32)`. An error without a location, such as
a formatting failure, has neither field. `--show-inferred` adds the inferred
signature of each callable to the report.

The style gate runs first. An unformatted file fails before type checking,
with `score` 0 and one error whose message says to run
`chelis fmt --inplace FILE`.

`chelis check DIRECTORY` checks every `.ch` and `.dp` file under it and prints
one JSON object with a `files` array of per-file reports and an `errors`
array for failures that belong to no single file. Any failure makes it exit
`2`.

For a short calculation without a file, use `chelis eval 'EXPR'`. A file containing
definitions but no expression has nothing to display; with `--json`, a successful evaluation
of such a file prints a result whose `roots` array is empty. `chelis eval --file app.ch --timeout 30` bounds an
evaluation in seconds. A timeout fails with a diagnostic.

For scripts, `chelis eval --json --file app.ch` writes one JSON object to stdout on success.
For the [first program](first-program.md):

```json
{"schema_version":4,"roots":[{"node_id":0,"name":"result","value":{"type":"tensor","value":{"shape":[3],"data":{"dtype":"f32","bits":["3e590735","3e590735","3f137c65"]}}}}],"manifest":{"target":"Eval","entries":[{"name":"result","lane":"Host","required_inputs":[]}],"requires_main":true}}
```

`roots` has one entry per printed value, in print order, with its `name` and
`value`. A tuple root is split into `pair.0`, `pair.1`, and so on. Values are
exact, never decimal text:

| `value.type` | Payload |
|---|---|
| `scalar` | `value` is `{"dtype":"f32","bits":"40d00000"}` for a float (the IEEE bits as lowercase hex: 8 digits for `f32`, 16 for `f64`, 4 for `f16` and `bf16`) or `{"dtype":"int64","value":4}` for an integer. Integer dtypes are spelled `int8` to `int64` here. |
| `bool` | `value` is a JSON boolean. |
| `string` | `value` is the string. |
| `unit` | no payload. |
| `tensor` | `value` is `{"shape":[...],"data":{"dtype":...,"bits":[...]}}` for floats, or `"values":[...]` for integers and `bool`, in row-major order. |
| `dict` | `entries` is a list of `{"key":...,"value":...}`. |
| `adt` | `ctor` names the constructor and `fields` holds its payload values, as in `{"type":"adt","ctor":"Some","fields":[...]}`. |

Text printed by `print` during evaluation appears in a `transcript` list of
strings. A file with only definitions gives `"roots":[]`. On failure, no JSON
is written; diagnostics go to stderr and the exit status is nonzero. Effects
completed before the failure may already have produced output.

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

`c` is the default target. `build` invokes its native toolchain and produces
an executable for observable programs, or a static library for
definitions-only modules. It retains generated sources and runtime support.
Run `./out/app` after the example above.

Compiled C programs check the current thread's stack before a recursive user
function call. When the reserved stack budget is exhausted, the executable
prints `RuntimeStackBudgetExhausted` with the call's source span to stderr and
exits with status 1. Output produced by earlier effects remains available.

Add `--emit-c` to stop after source emission and runtime staging, without
requiring a native compiler. A `.dp` input takes the Deep path automatically.
See [Build programs](backends.md) for artifact names, compiler overrides,
library linking, and platform requirements.

## Upgrading source from 0.18

Source written for the 0.18 grammar can be rewritten into the current grammar with `chelis migrate surf --from 0.18 PATH...` (or `migrate deep` for Deep files). It prints the result; `--inplace` writes it, and `--check` exits nonzero if a file still needs migrating.
