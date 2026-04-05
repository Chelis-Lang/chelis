# Tide: Interactive Mode

**Status:** Planned for Phase 0i.
This document records the intended architecture so the rest of the docs stay aligned
while implementation catches up.

## 1. Scope

Phase 0i covers the first interactive developer workflow:

- `chelis tide` for a REPL
- `chelis deep` for Surf to Deep inspection
- `chelis surf` for best-effort Deep to Surf decompilation
- `chelis fmt` for Surf formatting
- `chelis eval expr` for one-shot evaluation

Later phases extend Tide into machine-facing services such as the agent API, MCP,
language server support, and the Cove TUI.

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

- HTTP / JSON API
- MCP server
- LSP support
- `chelis cove` terminal UI
