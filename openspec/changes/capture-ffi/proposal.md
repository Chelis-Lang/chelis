## Why

`spec/11-ffi.md` records the foreign-function-interface direction: the phased Python interop
(3b interop core, 3b-ii direct execution + NumPy guarantee, 5a JAX), C interop via generated
headers and the Rust static runtime, and compiler embedding. No OpenSpec capability records these
as testable requirements.

## What Changes

- Introduce an `ffi` capability recording the Python interop cuts and their tested guarantees,
  the C interop surface and the `def main` symbol-rename rule, and the compiler-embedding
  direction.
- Capture the boundary rejection cases (GPU tensors as `ValueError` in 3b, source-changed
  `load()` warning) as negative-parity scenarios.

## Capabilities

### New Capabilities
- `ffi`: the phased Python interop (interop core, direct execution, JAX), C interop via generated
  headers plus the Rust static runtime, and compiler-crate embedding.

### Modified Capabilities

## Impact

- Source: `spec/11-ffi.md` (read-only) is the authority for this capability.
- Surface: the `chelis-python` PyO3 bindings, `chelis-compiler-api`, generated C headers, and the
  Rust static runtime.
- No code changes; this change records current and directional behavior as an OpenSpec spec.
