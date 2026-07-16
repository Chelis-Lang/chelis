## Why

Chelis correctly tracks `IO` in the language type system, but the host evaluator directly reads and writes files and starts processes while interpreting expressions. This hardwires machine effects into semantic evaluation, prevents deterministic embedding, and makes capability restriction an all-or-nothing process concern.

## What Changes

- Define an explicit evaluator protocol covering filesystem reads and writes, directory queries, bounded immutable mapped-file snapshots, subprocess execution, and deterministic captured output events, with policy coverage keyed to host-capable builtin classifications rather than permanently hardcoded to today's single `IO` effect.
- Make evaluator semantics a resumable state machine depending only on the checked program, runtime values, deterministic limits, explicit semantic policy, and typed host observations; resolved host execution and rendering configuration stays in adapters.
- Provide a production operating-system handler that preserves current CLI behavior.
- Provide deterministic in-memory and denying handlers for tests, Tide, Python, and restricted embeddings.
- Require every effect request to carry a deterministic invocation-local correlation identity, builtin/source identity, bounded structured arguments, and non-secret policy context needed for diagnostics; suspensions are non-cloneable and consumed exactly once. Sensitive live payloads may be consumed transiently but disable persistent transcript caching and never enter identities, diagnostics, logs, or replay records.
- Preserve Chelis `IO` effect typing and existing user-visible success and failure behavior under the production handler.
- Add positive and negative parity tests for allowed, denied, failed, and malformed host operations.

## Capabilities

### New Capabilities

- `evaluator-effect-handlers`: Defines how evaluator semantics request host effects and how outer adapters execute, deny, or simulate those requests.

### Modified Capabilities

None.

## Impact

The main changes add dependency-minimal `chelis-eval-core` and affect `chelis-compiler-api/src/runtime` adapters, evaluator entry points, CLI eval/test orchestration, Python bindings, and Tide. Direct callers of evaluation APIs will need an explicit driver/policy or a clearly named compatibility entry point that selects the production adapter at the outer boundary. The independent prerequisite `deny-tide-evaluator-host-effects` makes Tide fail closed with a pre-migration runtime guard at external builtin dispatch—captured `print`/`debug` events remain available—and this change replaces that guard with deny-all request handling. Python's ordinary `eval_json` becomes deny-by-default; explicit `eval_json_with_policy` and `eval_json_unrestricted` entry points provide configured and trusted-local behavior. Trusted CLI compatibility selects the production policy explicitly. Persistent evaluation transcripts, stable evaluation digests, and final-result caching are deferred. After the standalone `establish-dylint-tooling` oracle passes, `establish-fcis-contract-mechanics` registers stable coverage IDs, exact protocol bounds/result projection, the typed builtin-policy registry, prerequisite edge, boundary/threat model, and fail-closed slice/final oracles.
