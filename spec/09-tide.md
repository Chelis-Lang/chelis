# Tide: Interactive And Agent Mode

## 1. Scope

Tide is Chelis's interactive, editor, and agent-facing compiler surface. It provides:

- an evaluator-backed REPL and one-shot evaluation;
- Surf/Deep inspection, decompilation, and formatting;
- HTTP/JSON and MCP compiler services;
- a Language Server Protocol endpoint; and
- the Cove terminal interface.

All Tide surfaces embed compiler libraries directly. They do not define a second parser,
checker, evaluator, proof engine, or edit semantics.

## 2. Interactive Execution

The REPL and `chelis eval` parse, type-check, lower to the RISC DAG, and execute
through the IR evaluator. They do not emit C or spawn an external compiler. Larger
compiled workloads use `chelis build`.

Interactive latency optimization follows a fixed order:

1. IR evaluator;
2. cached C artifacts;
3. a persistent compiler helper process; and
4. JIT only when the first three fail on measured workloads.

## 3. Inspection And Formatting Commands

### `chelis tide`

Launches the interactive REPL. It accepts Deep directly and Surf through the normal
front-end path.

### `chelis eval expr`

Evaluates one expression through the IR evaluator with no native-code emission.

### `chelis deep file.ch`

Prints the canonical Deep form of Surf input. The default is width-aware pretty Deep;
`--flat` prints flat, separately delimited top-level forms.

### `chelis surf file.dp`

Constructs a Surf AST from representable Deep and prints formatter-canonical Surf.
`--verbose` is the explicit best-effort diagnostic form. The default path must never
emit handwritten pseudo-Surf that the Surf parser rejects.

### `chelis fmt file`

Applies the compiler-owned canonical formatter for Surf or Deep. `--check` reports
non-canonical input without rewriting it.

## 4. HTTP And MCP Services

`chelis tide serve` launches the HTTP/JSON service. `chelis tide mcp` launches the
MCP service on stdio. Both use explicit versioned wire models rather than serialized
compiler-internal Rust types.

The HTTP surface includes:

- `/parse`, `/desugar`, `/check`, `/lower`, `/compile`, `/eval`,
  `/grad`, `/validate`, and `/decompile`;
- `/replace_function_body`, `/add_function`, `/replace_function`,
  `/add_property`, `/rename`, and `/change_signature`;
- `/deep_outline`, `/deep_references`, and `/deep_call_graph`; and
- `/batch`.

The MCP surface exposes the corresponding `chelis_*` tools, including
`chelis_prove`.

`/eval` accepts named input bindings resolved by `Load.name`. `/grad` returns
differentiated DAG JSON plus node mappings; it does not claim to produce a
source-level differentiated Surf or Deep program.

## 5. Structural Edit Contract

`chelis_replace_function_body` accepts Deep strings for
`{module, function_name, new_body}`. It returns canonical Deep
`{changed_def_deep, module_deep}` only after the post-splice whole-module check is
clean.

`chelis_add_function` accepts
`{module, new_decls, insert_after_function?}`, where `new_decls` contains exactly
one `(def ...)` and an optional matching `(defsig ...)`. It preserves non-target
exports and returns
`{added_def_deep, added_defsig_deep?, module_deep}` only after whole-module
validation.

Parse, request-shape, splice, insertion-target, type, effect, and linearity failures
return `ok:false` with a stage and typed diagnostics. A failed edit carries no result
payload.

`chelis_deep_outline` returns a `preimage_sha256` computed over each canonical
function `(def ...)` node. An edit carrying a stale preimage fails at
`stage:"preimage"` with no result. Rename and signature changes must prove cascade
completeness over the structural query substrate before the final whole-module
validation gate.

## 6. Proof Contract

`chelis_prove` over MCP runs the same property and producer-obligation collection,
synthesis, assumption injection, and tiered dispatch as CLI `chelis prove`. The
response includes derived obligation records and their summary count.

An SMT proof is over the reals while runtime arithmetic is IEEE floating-point. Every
such artifact carries `proof_tier:"smt"` and `arith_model:"real"`; it claims no
floating-point soundness.

Induction reports success only after the same sound SMT engine discharges both a
concrete base case and a symbolic step case. Unsupported, missing, vacuous, timed-out,
sampled, or assumed evidence cannot become a proof. Deep input under
`tier:"induction-only"` is `unsupported` with zero samples rather than being
reinterpreted as fuzzing.

## 7. Language Server

`chelis tide lsp` runs the Tide LSP server on stdio. It analyzes the complete document
snapshot on open and change and provides:

- diagnostics;
- completion;
- hover;
- go-to-definition; and
- commands for canonical Deep view and fitness information.

Surf is the rich editor surface. Deep receives diagnostics and canonical inspection.
The Deep view is read-only and does not imply bidirectional text editing.

The VS Code-compatible extension supplies Surf and Deep TextMate grammars so
highlighting is independent of language-server availability.

## 8. Cove

`chelis cove [--file PATH]` launches the terminal interface. Cove embeds Tide compiler
helpers in process and presents:

- a Surf editor;
- a read-only canonical Deep view;
- diagnostics and fitness information; and
- compile-preview and evaluator output.

Evaluation may synthesize zero-filled bindings only for unresolved `Load` nodes whose
shapes are known. Surf and Deep highlighting use the bundled tree-sitter grammars.

## 9. Output And Agreement

Interactive tooling surfaces values, inferred types where useful, and structured
fitness diagnostics. `chelis check` remains the explicit machine-facing fitness
command.

Evaluation through Tide and the C backend must each conform to the normative numeric
rules across the shared specification corpus. A divergence means at least one lane is
defective; neither lane is permitted to define alternate program semantics.
