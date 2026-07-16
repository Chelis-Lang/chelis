## Why

Chelis correctly tracks `IO` in the language type system, but the host evaluator directly reads and writes files and starts processes while interpreting expressions. This hardwires machine effects into semantic evaluation, prevents deterministic embedding, and makes capability restriction an all-or-nothing process concern.

## What Changes

- Define an explicit evaluator protocol covering filesystem reads and writes, directory queries, memory mapping, subprocess execution, and deterministic captured output events.
- Make evaluator semantics a resumable state machine depending only on the checked program, runtime values, deterministic limits, explicit semantic policy, and typed host observations; resolved host execution and rendering configuration stays in adapters.
- Provide a production operating-system handler that preserves current CLI behavior.
- Provide deterministic in-memory and denying handlers for tests, Tide, Python, and restricted embeddings.
- Require every effect request to carry a deterministic invocation-local correlation identity, builtin/source identity, bounded structured arguments, and non-secret policy context needed for diagnostics; suspensions are non-cloneable and consumed exactly once.
- Preserve Chelis `IO` effect typing and existing user-visible success and failure behavior under the production handler.
- Add positive and negative parity tests for allowed, denied, failed, and malformed host operations.

## Capabilities

### New Capabilities

- `evaluator-effect-handlers`: Defines how evaluator semantics request host effects and how outer adapters execute, deny, or simulate those requests.

### Modified Capabilities

None.

## Impact

The main changes affect `chelis-compiler-api/src/runtime`, evaluator entry points, CLI eval/test orchestration, Python bindings, and Tide. Direct callers of evaluation APIs will need an explicit driver/policy or a clearly named compatibility entry point that selects the production adapter at the outer boundary. Tide is fail-closed immediately with a pre-migration IO rejection guard, then remains deny-all by default under the request protocol. Python's ordinary `eval_json` becomes deny-by-default; explicit `eval_json_with_policy` and `eval_json_unrestricted` entry points provide configured and trusted-local behavior. Trusted CLI compatibility selects the production policy explicitly.
