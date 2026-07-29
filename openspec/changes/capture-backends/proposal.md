## Why

`spec/08-backends.md` records the backend strategy and the C, HIP, and Metal backend contracts:
the sequential C-first strategy, the shared `chelis_tensor **` ABI, the source-to-source GPU
model, per-backend dtype gating, and the cross-backend numerical-agreement invariant. No
OpenSpec capability records these as testable requirements with negative parity.

## What Changes

- Introduce a `backends` capability recording the backend strategy and separation from the
  front end, the C reference backend, the HIP and Metal GPU backends (source-to-source model,
  shared ABI, runtime compilation), the Phase-2a backend-boundary checks, interactive-execution
  escalation, backend selection, and the all-backend invariants.
- Capture the rejection cases (build reject of resource regions/dropout, f64-on-Metal hardware
  rejection, unsupported-op rejection) as negative-parity scenarios.

## Capabilities

### New Capabilities
- `backends`: the C reference backend and cross-backend oracle role, the HIP and Metal GPU
  backends and their shared ABI and source-to-source model, backend-boundary effect checks,
  backend selection, interactive-execution escalation, and the numerical-correctness and
  reference-agreement invariants.

### Modified Capabilities

## Impact

- Source: `spec/08-backends.md` (read-only) is the authority for this capability.
- Surface: `chelis-backend-c`, `chelis-backend-hip`, `chelis-backend-metal`, and the CLI
  `--target` selection.
- No code changes; this change records current behavior as an OpenSpec capability spec.
