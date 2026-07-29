## Context

`spec/11-ffi.md` is an outline for later phases, but with a clear direction already recorded: the
Phase-3b/3b-ii/5a Python interop cuts and their tested guarantees, C interop via generated
headers plus the Rust static runtime (post-3m), and compiler-crate embedding. Phase 0 does not
depend on FFI. This change records that content as an `ffi` capability.

## Goals / Non-Goals

**Goals:**
- Capture the Python interop cuts, the C interop surface and the `def main` rename rule, and the
  compiler-embedding direction as SHALL requirements with positive and boundary-rejection
  scenarios.

**Non-Goals:**
- Asserting Phase-0 execution of FFI; the source is explicit that FFI is a Phase-3
  interoperability feature, not a Phase-0 requirement.
- Specifying the JAX/StableHLO (Phase 5a) path beyond its named guarantee.

## Decisions

- Record each Python cut (3b, 3b-ii) as its own requirement because they have distinct tested
  guarantees (CPU DLPack + GIL release vs direct execution + NumPy + f32-only) and distinct
  boundary rejections (GPU as `ValueError`, source-changed `load()` warning).

## Risks / Trade-offs

- [Directional/outline source] → The chapter is an outline. The requirements capture the recorded
  tested guarantees and boundary rules (the concrete, testable parts) and avoid over-specifying
  the unshipped 5a/JAX path.

## Open Questions

- The JAX DLPack guarantee (Phase 5a) and the StableHLO backend are directional and unshipped;
  they are recorded as roadmap items rather than current requirements.
- Full GPU direct execution via the HIP device ABI (3b-ii) is layered on that ABI; the requirement
  records the tested CPU guarantees and the f32-only compiled-execution limit as the current
  boundary.
