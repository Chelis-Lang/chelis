## Context

The evaluator dispatches external builtins from recursive evaluation paths and currently calls host filesystem/process APIs directly. Tide must deny those operations before the larger continuation-machine migration can land.

## Goals / Non-Goals

**Goals:**

- Add one runtime guard at the common external-builtin dispatch boundary.
- Make Tide select deny-external mode explicitly by default.
- Prove every supported nested call form reaches the guard before a host action.
- Preserve pure expressions and captured `print`/`debug` events.

**Non-Goals:**

- Implementing typed evaluator requests, resumable continuations, replay, or policy-configured capabilities.
- Changing Chelis effect inference or CLI/Python defaults.
- Treating `print` and `debug` as terminal host actions; they remain returned captured events.

## Decisions

### 1. The temporary guard is explicit evaluator input

An invocation-owned evaluator option distinguishes trusted-local compatibility from deny-external execution. It is supplied by the Tide caller and propagated through every evaluator entry path. It is not read from process environment or mutable global state.

### 2. Enforcement occurs at common builtin dispatch

Every file read/write/existence/listing, mapped-file open, and `process_run` branch checks deny-external mode immediately before any host API call. Nested functions, callbacks, transforms, imported definitions, and test bodies all converge on this check. Pure mapped-snapshot reads after a snapshot already exists are unaffected, but Tide cannot open a host-backed mapping in deny mode.

### 3. Denial is structured and observable

The evaluator returns the stable policy-denied diagnostic selected for this migration. Tide renders/serializes it through its normal error surface. Denial is not converted to not-found, `false`, empty output, or success.

### 4. Denial coverage has one typed owner and registered evidence

A closed `HostBuiltinId`/`HostBuiltinPolicy` registry is the single owner of evaluator host-capable builtin classification. The Tide guard dispatches through that classification rather than maintaining an independent string list. Builtin registration, declared effect, denial policy, nested-call coverage, and the future evaluator request mapping are generated projections or tripwire-checked consumers. Adding or reclassifying a host-capable builtin without a denial classification fails the consistency gate.

Before guard implementation, this change registers stable requirement/scenario IDs, positive/negative fixtures, the exact evaluator/Tide boundary, and `tide-evaluator-denial` in the FCIS contract manifest. The fail-closed runner must exist and report the still-failing denial fixtures; missing evidence is not a skip. The later protocol migration consumes the same registry and may remove the temporary mode only after equivalent typed deny-all mapping passes.

## Risks / Trade-offs

- A guard added only at Tide's outer route could be bypassed by nested evaluator paths; the common dispatch check and nested negative corpus are mandatory.
- This is temporary duplication. The later protocol migration must delete the mode only after deny-all request handling has equivalent coverage.

## Migration Plan

1. Add pure/captured-event positive tests and direct/nested external negative tests.
2. Verify external cases perform host actions or otherwise fail the intended precondition before the guard.
3. Add invocation-owned deny-external mode and common dispatch checks.
4. Route all Tide evaluation entry points through denial by default.
5. Run the acceptance oracle.

## Acceptance Oracle

The authoritative completion oracle is:

```text
.venv/bin/python scripts/fcis_gate.py tide-evaluator-denial
```

The runner must cover pure evaluation, captured events, every external builtin family, and nested function/callback/transform/import/test forms. Success means exit status 0, an empty error list, zero observed filesystem or subprocess actions in every Tide denial fixture, and unchanged deterministic captured `print`/`debug` output.
