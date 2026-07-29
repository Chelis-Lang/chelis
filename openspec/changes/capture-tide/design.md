## Context

`spec/09-tide.md` records Tide across shipped phases: the Phase-0i REPL and inspection commands,
the interactive-execution strategy and fixed latency policy, the Phase-2e HTTP/JSON and MCP
compiler services with structured wire contracts, the Phase-2f LSP, the Phase-2g Cove TUI, and
the evaluator-agreement invariant. This change records that content as a `tide` capability.

## Goals / Non-Goals

**Goals:**
- Capture the interactive-execution strategy, latency policy, command semantics, agent-service
  edit contracts, prove parity, and evaluator-agreement as SHALL requirements with positive and
  failure scenarios.

**Non-Goals:**
- Enumerating every HTTP endpoint and MCP tool name; the requirement captures the structured-wire
  contract and the edit-gate behavior, not the full endpoint list.
- Specifying LSP/Cove editor-host behavior beyond the shipped library-level contract, which the
  source itself marks as manual-gate territory.

## Decisions

- Split the agent surface into two requirements — the structured edit contract (whole-module
  check gate, failure envelope) and the preimage/cascade guard — because they have distinct
  failure stages (`ok:false` vs `stage:"preimage"`).
- Keep prove parity and the SMT real-arithmetic disclosure as one requirement since both concern
  the `chelis_prove` cross-surface contract.

## Risks / Trade-offs

- [Manual-gate editor behavior] → LSP (2f) and Cove (2g) full editor-host behavior are manual
  gates per the source. The requirements capture the automated library-level contract and the
  structured-wire promises, leaving editor-host behavior to the documented manual oracles.

## Open Questions

- `/grad` over the service is DAG-level (differentiated DAG JSON), not a source-level
  differentiated program; the requirement set records the evaluator-agreement and edit contracts
  rather than asserting source-level grad over the service, matching the source's Phase-2e note.
