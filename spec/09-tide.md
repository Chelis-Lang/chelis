# Tide: Interactive And Agent Mode

**Status:** Phase 0i REPL shipped; Phase 2e agent API + MCP shipped.
This document records the intended architecture so the rest of the docs stay aligned
while implementation catches up.

## 1. Scope

Phase 0i covers the first interactive developer workflow:

- `chelis tide` for a REPL
- `chelis deep` for Surf to Deep inspection
- `chelis surf` for best-effort Deep to Surf decompilation
- `chelis fmt` for Surf formatting
- `chelis eval expr` for one-shot evaluation

Later phases extend Tide into language-server support and the Cove TUI.

## 2. Interactive Execution Strategy

The REPL and `chelis eval` use the IR evaluator in `chelis-ir/src/eval.rs`.
That means:

- parse input
- type-check it
- lower it to the RISC DAG
- evaluate the DAG directly

This avoids spawning an external C compiler for small interactive workloads.

For larger workloads, `chelis build` continues to use the production C backend.

## 3. Latency Policy

If interactive latency later becomes a problem, the escalation order is fixed:

1. IR evaluator
2. cached C artifacts
3. persistent compiler helper process
4. JIT only if the first three fail on measured workloads

This is a policy decision, not an open exploration track.

## 4. Command Semantics

### `chelis tide`

Launch an interactive REPL.
It should accept Deep expressions directly and may accept Surf input when the front-end
path is available.

### `chelis tide serve`

Launch the HTTP/JSON compiler service.
The shipped surface includes `/parse`, `/desugar`, `/check`, `/lower`, `/compile`,
`/eval`, `/grad`, `/validate`, `/decompile`, and `/batch`.
The public contract uses explicit wire-model types rather than serialized compiler
internals.

### `chelis tide mcp`

Launch the MCP server on stdio.
The shipped MCP tool surface is:

- `chelis_check`
- `chelis_compile`
- `chelis_desugar`
- `chelis_decompile`
- `chelis_eval`
- `chelis_grad`
- `chelis_validate`

### `chelis eval expr`

Evaluate a single expression through the evaluator fast path.
No C emission or external compiler process is involved.

### `chelis deep file.ch`

Show the canonical Deep form of Surf input.

### `chelis surf file.dp`

Best-effort decompile Deep into readable Surf.

### `chelis fmt file.ch`

Format Surf using the compiler-owned canonical style.

## 5. Output Expectations

Interactive tooling should surface:

- values from evaluation
- inferred types where useful
- fitness-oriented diagnostics when programs fail to type-check

`chelis check` remains the machine-friendly command for explicit structured fitness
output.

## 6. Agreement Testing

Tide-related evaluation must agree numerically with the production backend.
The Phase 0i test strategy includes evaluator-agreement tests:

- evaluate via the IR evaluator
- evaluate via the C backend
- compare results across the shared spec test corpus

## 7. Later Tide Work

Phase 2 extends Tide beyond the REPL:

- 2e: HTTP / JSON compiler API plus MCP server for coding agents
- 2f: LSP support with diagnostics, hover, completion, go-to-definition, and Surf/Deep
  visibility
- 2g: `chelis cove` terminal UI with live checking and Surf/Deep toggling
- batch interfaces for agent loops and corpus collection

## 8. Phase 2e Contract Notes

- `/eval` takes named input bindings and resolves them by `Load.name`; the evaluator
  itself remains unchanged.
- `/grad` is DAG-level in the shipped Phase 2e surface and returns differentiated DAG
  JSON plus node mappings, not a source-level differentiated Surf or Deep program.
- The 2e automated acceptance oracle is `cargo test -p chelis-tide --test api`.
- MCP-agent end-to-end validation remains a documented manual gate rather than part of
  the default workspace run.
