## Why

Tide currently reaches evaluator code that directly performs filesystem and subprocess operations. The larger evaluator FCIS migration will replace those calls with typed requests, but the secure service default must not wait for that rewrite.

## What Changes

- Add an explicit deny-external evaluation mode at the evaluator builtin-dispatch boundary used by Tide.
- Reject file, directory, mapped-file-open, and subprocess builtins before any host call.
- Preserve pure evaluation and deterministic captured `print`/`debug` events.
- Cover nested functions, callbacks, transforms, imports, and test bodies so the guard cannot be bypassed through another evaluator call form.
- Route Tide through deny-external mode by default; no deployment capability configuration is introduced in this narrow change.

## Capabilities

### New Capabilities

- `tide-evaluator-denial`: Unconfigured Tide evaluation cannot perform external filesystem or subprocess actions.

### Modified Capabilities

None.

## Impact

This affects Tide evaluator entry points and the existing evaluator builtin dispatch. It intentionally changes unsafe implicit Tide capability behavior. It does not implement the future request/observation protocol, Python policy APIs, contained directory capabilities, or production OS adapters. `isolate-evaluator-host-effects` treats this correction as a prerequisite and later replaces the temporary mode with deny-all request handling. Before implementation, `establish-fcis-contract-mechanics` registers this change's stable coverage IDs, typed host-builtin denial owner, exact boundary, and fail-closed oracle; planned or missing evidence cannot be counted green.
