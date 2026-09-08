## Why

`spec/09-tide.md` records Tide, the interactive and agent-mode surface: the REPL and one-shot
eval, the interactive-execution strategy and latency policy, the HTTP/JSON and MCP compiler
services, the LSP server, the Cove TUI, and the evaluator-agreement invariant. No OpenSpec
capability records these as testable requirements.

## What Changes

- Introduce a `tide` capability recording the interactive-execution strategy (evaluator-first),
  the fixed latency-escalation policy, the command semantics (`tide`, `deep`, `surf`, `fmt`,
  `eval`), the `tide serve`/`tide mcp`/`tide lsp` surfaces and their structured wire contracts,
  and the evaluator-agreement testing requirement.
- Capture the failure contracts (stale-preimage edit rejection, whole-module-check gate on
  structural edits, structured failure envelopes) as negative-parity scenarios.

## Capabilities

### New Capabilities
- `tide`: interactive execution and latency policy, the interactive/decompile/format command
  semantics, the HTTP/JSON and MCP agent services and their structured edit contracts, the LSP
  and Cove surfaces, and evaluator-agreement testing.

### Modified Capabilities

## Impact

- Source: `spec/09-tide.md` (read-only) is the authority for this capability.
- Surface: `chelis-tide` (compiler service, MCP, LSP), `chelis cove`, and the IR evaluator.
- No code changes; this change records current behavior as an OpenSpec capability spec.
